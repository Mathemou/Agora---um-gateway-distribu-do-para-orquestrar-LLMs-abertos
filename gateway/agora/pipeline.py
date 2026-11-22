"""Pipeline do gateway: domínio → escolha do nó → fila do nó → stream → eventos.

É o mesmo caminho para quem chama `POST /v1/chat` (cliente externo, via SSE)
e para o jogo (a IA impostora e o Mestre). Assim, toda mensagem gerada por um
LLM passa pela mesma fila com limite de concorrência e é medida igual
(TTFT, tokens/s, latência fim a fim, espera na fila).

Uso em duas etapas:
    job = gateway.admit("impostor", messages)   # escolhe o nó e reserva lugar na fila (QueueFull → 503)
    async for event, data in gateway.run(job):  # meta, token..., [failover, token...], done | error
        ...

Trilha Desempenho (ver balancer.py):
- `admit` pergunta ao Balancer a ordem dos nós do pool do domínio e reserva lugar
  no primeiro que tiver fila livre (se um nó lotou, tenta o próximo antes do 503).
- `run` refaz a requisição em outro nó se o nó falhar, inclusive NO MEIO da
  resposta. A resposta recomeça do zero no nó novo: o gateway emite o evento
  `failover` com quantos tokens o cliente deve descartar. Também detecta nó
  "travado" (stream parado por `stall_timeout_s`).
- A chave da fila vem de `priority_key` (FIFO ou prioridade com envelhecimento).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing, suppress
from dataclasses import dataclass, field
from typing import Any

import httpx

from .balancer import Balancer, priority_key
from .config import DomainConfig, GatewayConfig, NodeConfig
from .metrics import MetricsStore, RequestRecord
from .node_queue import NodeQueue, QueueFull, Ticket
from . import anthropic, openai_compat
from .ollama import NodeError, list_models, stream_chat

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
    domain_name: str = ""
    node_name: str = ""
    priority: float = 0.0
    pinned: bool = False  # nó escolhido por quem chamou (ex.: aquecimento): sem failover
    tried: list[str] = field(default_factory=list)
    attempt_t0: float = field(default_factory=time.perf_counter)

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
        self.balancer = Balancer(cfg, self.queues)
        self.metrics = MetricsStore(log_path=cfg.metrics_log)
        self.http = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(connect=5.0, read=cfg.read_timeout_s, write=30.0, pool=None),
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=64),
        )
        self._health_task: asyncio.Task | None = None

    def start(self) -> None:
        """Liga o health check periódico dos nós (chamado no início do app)."""
        if self._health_task is None and self.cfg.balance.health_interval_s > 0:
            self._health_task = asyncio.create_task(self.balancer.health_loop(self.probe_node))

    async def aclose(self) -> None:
        if self._health_task is not None:
            self._health_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._health_task
            self._health_task = None
        await self.http.aclose()

    async def probe_node(self, name: str) -> list[str]:
        """Modelos disponíveis no nó; levanta NodeError se ele não responder."""
        node = self.cfg.nodes[name]
        timeout = self.cfg.balance.health_timeout_s
        if node.kind == "openai":
            return await openai_compat.list_models(self.http, node.url, node.api_key_env, timeout=timeout)
        if node.kind == "anthropic":
            return await anthropic.list_models(self.http, node.url, node.api_key_env, timeout=timeout)
        return await list_models(self.http, node.url, timeout=timeout)

    # ------------------------------------------------------------------ admissão
    def _enqueue(
        self, domain_name: str, domain: DomainConfig, priority: float,
        exclude: list[str], node: str | None = None, policy: str | None = None,
    ) -> tuple[str, Ticket]:
        """Reserva lugar na fila do melhor nó disponível (ou do nó fixado)."""
        candidates = [node] if node else self.balancer.order(domain_name, domain, exclude, policy)
        last: QueueFull | None = None
        for name in candidates:
            try:
                ticket = self.queues[name].enqueue(priority)
            except QueueFull as exc:
                last = exc  # este nó lotou: tenta o próximo do pool
                continue
            self.balancer.on_selected(name)
            return name, ticket
        raise last or QueueFull(",".join(domain.pool))

    def admit(
        self,
        domain_name: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        variables: dict[str, Any] | None = None,
        source: str = "api",
        room: str | None = None,
        priority: float | None = None,
        extra_options: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        node: str | None = None,
    ) -> ChatJob:
        """Monta a requisição, escolhe o nó e reserva o lugar na fila, sem bloquear.

        Levanta UnknownDomain ou QueueFull (todas as filas do pool lotadas) antes de
        qualquer stream. `node` fixa o nó (precisa estar no pool do domínio).
        """
        domain = self.cfg.domains.get(domain_name)
        if domain is None:
            raise UnknownDomain(domain_name)
        if node is not None and node not in domain.pool:
            raise UnknownDomain(f"{domain_name}@{node}")
        t0 = time.perf_counter()
        key = priority if priority is not None else priority_key(self.cfg, source)
        node_name, ticket = self._enqueue(domain_name, domain, key, [], node)

        full: list[dict[str, str]] = []
        system = domain.system_prompt if system_prompt is None else system_prompt
        if system:
            full.append({"role": "system", "content": render_prompt(system, variables)})
        full += messages
        options = dict(domain.options)
        if extra_options:
            options.update(extra_options)
        if max_tokens:
            options["num_predict"] = max_tokens

        rec = RequestRecord(
            request_id=uuid.uuid4().hex[:12],
            domain=domain_name,
            node=node_name,
            model=domain.model,
            started_at=time.time(),
            queue_position=ticket.position,
            source=source,
            room=room,
            nodes_path=node_name,
            priority=round(key - time.monotonic(), 3) if self.cfg.queue.policy == "priority" else None,
        )
        return ChatJob(
            rec, ticket, domain, self.cfg.nodes[node_name], full, options, t0=t0,
            domain_name=domain_name, node_name=node_name, priority=key,
            pinned=node is not None, tried=[node_name], attempt_t0=t0,
        )

    # ------------------------------------------------------------------ execução
    def _chunks(self, job: ChatJob):
        n = job.node
        if n.kind == "openai":
            return openai_compat.stream_chat(
                self.http, n.url, job.domain.model, job.messages, job.options, n.api_key_env,
                session_id=job.rec.room or job.rec.request_id,
            )
        if n.kind == "anthropic":
            return anthropic.stream_chat(
                self.http, n.url, job.domain.model, job.messages, job.options, n.api_key_env,
            )
        return stream_chat(self.http, n.url, job.domain.model, job.messages, job.options,
                           self.cfg.keep_alive)

    async def _guarded(self, chunks) -> AsyncIterator[dict[str, Any]]:
        """Repassa os pedaços do stream, transformando "nó travado" em NodeError."""
        b = self.cfg.balance
        got_first = False
        it = chunks.__aiter__()
        while True:
            timeout = b.stall_timeout_s if got_first else b.ttft_timeout_s
            try:
                chunk = await _next_with_timeout(it, timeout)
            except StopAsyncIteration:
                return
            except (asyncio.TimeoutError, TimeoutError):
                what = "parou de enviar tokens" if got_first else "não mandou o 1º token"
                raise NodeError(f"nó {what} em {timeout:g} s (travado?)", fault=True) from None
            if chunk.get("message", {}).get("content"):
                got_first = True
            yield chunk

    def _next_node(self, job: ChatJob, exc: NodeError) -> tuple[str, Ticket] | None:
        """Para onde refazer a requisição depois de uma falha (ou None: desiste)."""
        b = self.cfg.balance
        if not (b.failover and exc.fault and not job.pinned):
            return None
        if job.rec.failovers >= b.max_failovers:
            return None
        try:
            # Mantém a chave de prioridade original: quem já esperava não volta pro fim da
            # fila. E vai para o nó com vaga livre agora, seja qual for a política.
            return self._enqueue(job.domain_name, job.domain, job.priority, job.tried,
                                 policy="least_queue")
        except QueueFull:
            return None

    async def run(self, job: ChatJob) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Executa o job e gera eventos ("meta" | "token" | "failover" | "done" | "error", dados).

        A vaga do nó é liberada no fim, inclusive se quem consome desistir no meio.
        """
        rec = job.rec
        failed_at: float | None = None
        try:
            yield "meta", {
                "request_id": rec.request_id, "domain": rec.domain, "node": rec.node,
                "model": rec.model, "queue_position": job.ticket.position,
            }
            while True:
                attempt_tokens = 0
                attempt_first: float | None = None
                final: dict[str, Any] = {}
                try:
                    await job.ticket.wait()
                    granted_at = time.perf_counter()
                    if rec.queue_wait_ms is None:
                        rec.queue_wait_ms = job.ms_since_start()
                    chunks = self._chunks(job)
                    async with aclosing(chunks):
                        async for chunk in self._guarded(chunks):
                            text = chunk.get("message", {}).get("content", "")
                            if text:
                                now = time.perf_counter()
                                if attempt_first is None:
                                    attempt_first = now
                                    if rec.ttft_ms is None:
                                        rec.ttft_ms = job.ms_since_start()
                                    if failed_at is not None:
                                        rec.recovery_ms = (now - failed_at) * 1000
                                attempt_tokens += 1
                                rec.output_tokens += 1
                                self.metrics.count_token()
                                yield "token", {"text": text}
                            if chunk.get("done"):
                                final = chunk
                except NodeError as exc:
                    if exc.fault:
                        self.balancer.on_failure(job.node_name, str(exc))
                    nxt = self._next_node(job, exc)
                    if nxt is None:
                        raise
                    old = job.node_name
                    job.ticket.release()
                    job.node_name, job.ticket = nxt
                    job.node = self.cfg.nodes[job.node_name]
                    job.tried.append(job.node_name)
                    job.attempt_t0 = time.perf_counter()
                    failed_at = failed_at or job.attempt_t0
                    self.balancer.health[old].failovers_from += 1
                    rec.failovers += 1
                    rec.wasted_tokens += attempt_tokens
                    rec.output_tokens -= attempt_tokens
                    rec.node = job.node_name
                    rec.nodes_path = ">".join(job.tried)
                    yield "failover", {
                        "request_id": rec.request_id, "from": old, "to": job.node_name,
                        "error": str(exc), "discard_tokens": attempt_tokens,
                        "elapsed_ms": round(job.ms_since_start(), 1),
                    }
                    continue

                rec.e2e_ms = job.ms_since_start()
                # Amostra para o balanceador: TTFT de SERVIÇO do nó (sem a fila do gateway).
                ttft_service = (attempt_first - granted_at) * 1000 if attempt_first else None
                self.balancer.on_success(job.node_name, ttft_service)
                _fill_node_stats(rec, final, attempt_first)
                rec.status = "ok"
                yield "done", rec.to_dict()
                return
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
        on_reset: Callable[[], Any] | None = None,
        timeout_s: float | None = None,
        **admit_kwargs: Any,
    ) -> Completion:
        """Atalho para o jogo: roda o job e devolve o texto inteiro.

        `on_start` é chamado quando o primeiro token chega; `on_token` a cada token;
        `on_reset` quando um failover descarta o texto parcial. Nunca levanta exceção:
        falhas voltam em `Completion.error`.
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
                    elif event == "failover":
                        parts.clear()  # o nó novo recomeça a resposta do zero
                        if on_reset:
                            on_reset()
                    elif event == "error":
                        error = data["message"]

        try:
            await asyncio.wait_for(consume(), timeout_s)
        except asyncio.TimeoutError:
            error = f"timeout de {timeout_s}s"
        return Completion("".join(parts), job.rec, error)


async def _next_with_timeout(it, timeout: float | None):
    """Próximo pedaço do stream com prazo, na MESMA task (o stream do httpx não gosta
    de ser lido de tasks diferentes). Python 3.11+ tem asyncio.timeout; no 3.10 cai
    no wait_for."""
    if timeout is None:
        return await it.__anext__()
    if hasattr(asyncio, "timeout"):
        async with asyncio.timeout(timeout):
            return await it.__anext__()
    return await asyncio.wait_for(it.__anext__(), timeout)


def _fill_node_stats(rec: RequestRecord, final: dict[str, Any], first_token_at: float | None) -> None:
    """Completa o registro com as contagens que o nó manda no último pedaço.

    `first_token_at` é o 1º token da tentativa que deu certo (depois de um failover,
    a taxa de geração é a do nó novo).
    """
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
    if first_token_at is not None:
        gen_s = time.perf_counter() - first_token_at
        # O 1º token já saiu no TTFT; a taxa de geração conta os demais.
        if gen_s > 0 and rec.output_tokens > 1:
            rec.gateway_tps = (rec.output_tokens - 1) / gen_s
