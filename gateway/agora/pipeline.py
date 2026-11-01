"""Pipeline do gateway: domínio → fila do nó → stream do Ollama → eventos.

É o mesmo caminho para quem chama `POST /v1/chat` (cliente externo, via SSE)
e para o jogo (a IA impostora e o Mestre). Assim, toda mensagem gerada por um
LLM passa pela mesma fila com limite de concorrência e é medida igual
(TTFT, tokens/s, latência fim a fim, espera na fila).

Uso em duas etapas:
    job = gateway.admit("impostor", messages)   # reserva lugar na fila (QueueFull → 503)
    async for event, data in gateway.run(job):  # meta, token..., done | error
        ...
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import DomainConfig, GatewayConfig, NodeConfig
from .metrics import MetricsStore, RequestRecord
from .node_queue import NodeQueue, QueueFull, Ticket
from . import anthropic
from .ollama import NodeError, stream_chat

__all__ = ["Gateway", "ChatJob", "UnknownDomain", "QueueFull", "Completion"]


class UnknownDomain(KeyError):
    pass


class _SafeDict(dict):
    """format_map que deixa {chaves_desconhecidas} intactas."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def render_prompt(template: str, variables: dict[str, Any] | None) -> str:
    template = template.strip()
    if not variables:
        return template
    try:
        return template.format_map(_SafeDict({k: str(v) for k, v in variables.items()}))
    except (ValueError, IndexError):  # chaves mal formadas no YAML: usa o texto como está
        return template


@dataclass
class ChatJob:
    rec: RequestRecord
    ticket: Ticket
    domain: DomainConfig
    node: NodeConfig
    messages: list[dict[str, str]]
    options: dict[str, Any]
    t0: float = field(default_factory=time.perf_counter)

    def ms_since_start(self) -> float:
        return (time.perf_counter() - self.t0) * 1000


@dataclass
class Completion:
    """Resposta inteira (para quem não precisa repassar token a token)."""

    text: str
    record: RequestRecord | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class Gateway:
    def __init__(self, cfg: GatewayConfig, transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg
        self.queues = {
            name: NodeQueue(name, node.max_concurrency, node.max_queue)
            for name, node in cfg.nodes.items()
        }
        self.metrics = MetricsStore(log_path=cfg.metrics_log)
        self.http = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(connect=5.0, read=cfg.read_timeout_s, write=30.0, pool=None),
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=64),
        )

    async def aclose(self) -> None:
        await self.http.aclose()

    # ------------------------------------------------------------------ admissão
    def admit(
        self,
        domain_name: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        variables: dict[str, Any] | None = None,
        source: str = "api",
        room: str | None = None,
        priority: float = 0.0,
        extra_options: dict[str, Any] | None = None,
    ) -> ChatJob:
        """Monta a requisição e reserva o lugar na fila do nó, sem bloquear.

        Levanta UnknownDomain ou QueueFull (fila lotada) antes de qualquer stream.
        """
        domain = self.cfg.domains.get(domain_name)
        if domain is None:
            raise UnknownDomain(domain_name)
        node = self.cfg.nodes[domain.node]
        t0 = time.perf_counter()
        ticket = self.queues[domain.node].enqueue(priority)

        full: list[dict[str, str]] = []
        if domain.system_prompt:
            full.append({"role": "system", "content": render_prompt(domain.system_prompt, variables)})
        full += messages
        options = dict(domain.options)
        if extra_options:
            options.update(extra_options)
        if max_tokens:
            options["num_predict"] = max_tokens

        rec = RequestRecord(
            request_id=uuid.uuid4().hex[:12],
            domain=domain_name,
            node=domain.node,
            model=domain.model,
            started_at=time.time(),
            queue_position=ticket.position,
            source=source,
            room=room,
        )
        return ChatJob(rec, ticket, domain, node, full, options, t0=t0)

    # ------------------------------------------------------------------ execução
    async def run(self, job: ChatJob) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Executa o job e gera eventos ("meta" | "token" | "done" | "error", dados).

        A vaga do nó é liberada no fim, inclusive se quem consome desistir no meio.
        """
        rec = job.rec
        first_token_at: float | None = None
        try:
            yield "meta", {
                "request_id": rec.request_id, "domain": rec.domain, "node": rec.node,
                "model": rec.model, "queue_position": job.ticket.position,
            }
            await job.ticket.wait()
            rec.queue_wait_ms = job.ms_since_start()

            final: dict[str, Any] = {}
            if job.node.kind == "anthropic":
                chunks = anthropic.stream_chat(
                    self.http, job.node.url, job.domain.model, job.messages, job.options,
                    job.node.api_key_env,
                )
            else:
                chunks = stream_chat(
                    self.http, job.node.url, job.domain.model, job.messages, job.options,
                    self.cfg.keep_alive,
                )
            async with aclosing(chunks):
                async for chunk in chunks:
                    text = chunk.get("message", {}).get("content", "")
                    if text:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                            rec.ttft_ms = job.ms_since_start()
                        rec.output_tokens += 1
                        self.metrics.count_token()
                        yield "token", {"text": text}
                    if chunk.get("done"):
                        final = chunk

            rec.e2e_ms = job.ms_since_start()
            _fill_node_stats(rec, final, first_token_at)
            rec.status = "ok"
            yield "done", rec.to_dict()
        except NodeError as exc:
            rec.status, rec.error = "error", str(exc)
            rec.e2e_ms = job.ms_since_start()
            yield "error", {"request_id": rec.request_id, "message": str(exc)}
        except (asyncio.CancelledError, GeneratorExit):
            rec.status = "cancelled"
            rec.e2e_ms = job.ms_since_start()
            raise
        finally:
            job.ticket.release()
            self.metrics.record(rec)

    async def complete(
        self,
        domain_name: str,
        messages: list[dict[str, str]],
        *,
        on_token: Callable[[str], Any] | None = None,
        on_start: Callable[[], Any] | None = None,
        timeout_s: float | None = None,
        **admit_kwargs: Any,
    ) -> Completion:
        """Atalho para o jogo: roda o job e devolve o texto inteiro.

        `on_start` é chamado quando o primeiro token chega; `on_token` a cada token.
        Nunca levanta exceção: falhas voltam em `Completion.error`.
        """
        try:
            job = self.admit(domain_name, messages, **admit_kwargs)
        except (QueueFull, UnknownDomain) as exc:
            return Completion("", None, f"{type(exc).__name__}: {exc}")

        parts: list[str] = []
        error: str | None = None

        async def consume() -> None:
            nonlocal error
            started = False
            async with aclosing(self.run(job)) as events:
                async for event, data in events:
                    if event == "token":
                        if not started:
                            started = True
                            if on_start:
                                on_start()
                        parts.append(data["text"])
                        if on_token:
                            on_token(data["text"])
                    elif event == "error":
                        error = data["message"]

        try:
            await asyncio.wait_for(consume(), timeout_s)
        except asyncio.TimeoutError:
            error = f"timeout de {timeout_s}s"
        return Completion("".join(parts), job.rec, error)


def _fill_node_stats(rec: RequestRecord, final: dict[str, Any], first_token_at: float | None) -> None:
    """Completa o registro com as contagens que o Ollama manda no último pedaço."""
    if final.get("eval_count"):
        rec.output_tokens = int(final["eval_count"])
        if final.get("eval_duration") and final["eval_count"] > 1:
            rec.node_tps = final["eval_count"] / (final["eval_duration"] / 1e9)
    if final.get("prompt_eval_count") is not None:
        rec.prompt_tokens = int(final["prompt_eval_count"])
    if final.get("load_duration") is not None:
        rec.node_load_ms = final["load_duration"] / 1e6
    if final.get("prompt_eval_duration") is not None:
        rec.node_prompt_eval_ms = final["prompt_eval_duration"] / 1e6
    if first_token_at is not None and rec.e2e_ms is not None and rec.ttft_ms is not None:
        gen_s = (rec.e2e_ms - rec.ttft_ms) / 1000
        # O 1º token já saiu no TTFT; a taxa de geração conta os demais.
        if gen_s > 0 and rec.output_tokens > 1:
            rec.gateway_tps = (rec.output_tokens - 1) / gen_s
