"""Gateway HTTP: recebe perguntas, roteia pelo domínio e repassa a resposta
token a token via Server-Sent Events (SSE).

Executar:  uvicorn agora.main:create_app --factory --app-dir gateway --port 8000
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import aclosing, asynccontextmanager
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from . import __version__
from .config import GatewayConfig, load_config
from .metrics import MetricsStore, RequestRecord
from .node_queue import NodeQueue, QueueFull
from .ollama import NodeError, list_models, stream_chat


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    domain: str
    # Use `prompt` para uma pergunta avulsa ou `messages` para um histórico.
    prompt: str | None = None
    messages: list[Message] | None = None
    max_tokens: int | None = Field(default=None, ge=1, le=4096)

    @model_validator(mode="after")
    def _need_content(self) -> "ChatRequest":
        if not self.prompt and not self.messages:
            raise ValueError("informe 'prompt' ou 'messages'")
        return self


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(
    config: GatewayConfig | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    cfg = config or load_config()
    queues = {
        name: NodeQueue(name, node.max_concurrency, node.max_queue)
        for name, node in cfg.nodes.items()
    }
    metrics = MetricsStore(log_path=cfg.metrics_log)
    http = httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(connect=5.0, read=cfg.read_timeout_s, write=30.0, pool=None),
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=64),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await http.aclose()

    app = FastAPI(title="Ágora – gateway de LLMs", version=__version__, lifespan=lifespan)
    app.state.config = cfg
    app.state.queues = queues
    app.state.metrics = metrics

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Estado do gateway e de cada nó (acessível? modelos baixados?)."""
        nodes: dict[str, Any] = {}
        for name, node in cfg.nodes.items():
            wanted = sorted({d.model for d in cfg.domains.values() if d.node == name})
            try:
                available = await list_models(http, node.url)
                missing = [m for m in wanted if m not in available and f"{m}:latest" not in available]
                nodes[name] = {"up": True, "models": available, "missing": missing}
            except NodeError as exc:
                nodes[name] = {"up": False, "error": str(exc)}
        all_ok = all(n["up"] and not n["missing"] for n in nodes.values())
        return {"status": "ok" if all_ok else "degraded", "nodes": nodes}

    @app.get("/v1/domains")
    async def domains() -> dict[str, Any]:
        return {
            name: {"description": d.description, "node": d.node, "model": d.model}
            for name, d in cfg.domains.items()
        }

    @app.get("/v1/nodes")
    async def nodes() -> list[dict[str, Any]]:
        return [q.snapshot() for q in queues.values()]

    @app.get("/metrics")
    async def get_metrics(window_s: float = 60.0) -> dict[str, Any]:
        return {"summary": metrics.summary(window_s), "nodes": [q.snapshot() for q in queues.values()]}

    @app.get("/metrics/requests")
    async def recent_requests(limit: int = 50) -> list[dict[str, Any]]:
        return [r.to_dict() for r in list(metrics.records)[-limit:]]

    @app.post("/v1/chat")
    async def chat(req: ChatRequest) -> StreamingResponse:
        t0 = time.perf_counter()
        domain = cfg.domains.get(req.domain)
        if domain is None:
            raise HTTPException(404, f"domínio desconhecido '{req.domain}'. Veja GET /v1/domains")
        node = cfg.nodes[domain.node]
        queue = queues[domain.node]

        # Reserva o lugar na fila já aqui: se lotada, responde 503 antes do stream.
        try:
            ticket = queue.enqueue()
        except QueueFull:
            raise HTTPException(
                503, f"fila do nó '{domain.node}' cheia", headers={"Retry-After": "2"}
            ) from None

        messages: list[dict[str, str]] = []
        if domain.system_prompt:
            messages.append({"role": "system", "content": domain.system_prompt.strip()})
        if req.messages:
            messages += [m.model_dump() for m in req.messages]
        if req.prompt:
            messages.append({"role": "user", "content": req.prompt})
        options = dict(domain.options)
        if req.max_tokens:
            options["num_predict"] = req.max_tokens

        rec = RequestRecord(
            request_id=uuid.uuid4().hex[:12],
            domain=req.domain,
            node=domain.node,
            model=domain.model,
            started_at=time.time(),
            queue_position=ticket.position,
        )

        def ms_since_start() -> float:
            return (time.perf_counter() - t0) * 1000

        async def event_stream():
            first_token_at: float | None = None
            try:
                yield sse("meta", {
                    "request_id": rec.request_id, "domain": rec.domain, "node": rec.node,
                    "model": rec.model, "queue_position": ticket.position,
                })
                await ticket.wait()
                rec.queue_wait_ms = ms_since_start()

                final: dict[str, Any] = {}
                chunks = stream_chat(http, node.url, domain.model, messages, options, cfg.keep_alive)
                async with aclosing(chunks):
                    async for chunk in chunks:
                        text = chunk.get("message", {}).get("content", "")
                        if text:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                                rec.ttft_ms = ms_since_start()
                            rec.output_tokens += 1
                            metrics.count_token()
                            yield sse("token", {"text": text})
                        if chunk.get("done"):
                            final = chunk

                rec.e2e_ms = ms_since_start()
                _fill_node_stats(rec, final, first_token_at)
                rec.status = "ok"
                yield sse("done", rec.to_dict())
            except NodeError as exc:
                rec.status, rec.error = "error", str(exc)
                rec.e2e_ms = ms_since_start()
                yield sse("error", {"request_id": rec.request_id, "message": str(exc)})
            except (asyncio.CancelledError, GeneratorExit):
                # Cliente desconectou: o `finally` libera a vaga para o próximo.
                rec.status = "cancelled"
                rec.e2e_ms = ms_since_start()
                raise
            finally:
                ticket.release()
                metrics.record(rec)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _fill_node_stats(rec: RequestRecord, final: dict[str, Any], first_token_at: float | None) -> None:
    """Completa o registro com as contagens que o Ollama manda no último pedaço."""
    if final.get("eval_count"):
        rec.output_tokens = int(final["eval_count"])
        if final.get("eval_duration"):
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
