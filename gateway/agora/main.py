"""Ágora: gateway de LLMs + servidor do Jogo da Imitação.

Duas portas de entrada no mesmo processo:

- API do gateway (HTTP + SSE): `POST /v1/chat` roteia a pergunta pelo domínio,
  passa pela fila do nó e devolve a resposta token a token.
- Jogo (HTTP + WebSocket): `POST /v1/rooms` cria uma sala e cada jogador abre
  `WS /ws/rooms/{código}`. A IA impostora e o Mestre usam o mesmo pipeline do
  gateway (mesmas filas e métricas) que um cliente do /v1/chat usaria.

Executar:  uvicorn agora.main:create_app --factory --app-dir gateway --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from . import __version__
from .config import GatewayConfig, load_config
from .game import GameError, RoomManager
from .personas import ARCHETYPES
from . import anthropic, openai_compat
from .ollama import NodeError, list_models
from .pipeline import Gateway, QueueFull, UnknownDomain

log = logging.getLogger("agora")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    domain: str
    # Use `prompt` para uma pergunta avulsa ou `messages` para um histórico.
    prompt: str | None = None
    messages: list[Message] | None = None
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    # Preenche {chaves} do prompt de sistema do domínio (ex.: {alias} do impostor).
    variables: dict[str, str] | None = None

    @model_validator(mode="after")
    def _need_content(self) -> "ChatRequest":
        if not self.prompt and not self.messages:
            raise ValueError("informe 'prompt' ou 'messages'")
        return self


class RoomCreate(BaseModel):
    discussion_s: float | None = None
    vote_s: float | None = None
    max_rounds: int | None = None
    ai_count: int | None = None
    min_players: int | None = None
    # Qual domínio faz o papel da IA (ex.: "impostor" local ou "impostor-claude").
    impostor_domain: str | None = None
    # Personalidade da IA (ver GET /v1/personas); vazio = sorteada a cada partida.
    ai_persona: str | None = None


STATIC = Path(__file__).parent / "static"


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(
    config: GatewayConfig | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    cfg = config or load_config()
    gateway = Gateway(cfg, transport)
    rooms = RoomManager(cfg.game, gateway)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        async def sweeper():
            while True:
                await asyncio.sleep(30)
                rooms.sweep()

        task = asyncio.create_task(sweeper())
        yield
        task.cancel()
        rooms.close_all()
        await gateway.aclose()

    app = FastAPI(title="Ágora – Jogo da Imitação", version=__version__, lifespan=lifespan)
    app.state.config = cfg
    app.state.gateway = gateway
    app.state.queues = gateway.queues
    app.state.metrics = gateway.metrics
    app.state.rooms = rooms

    # ------------------------------------------------------------------ páginas
    @app.get("/", include_in_schema=False)
    async def ui() -> FileResponse:
        """O jogo: abra http://localhost:8000 no navegador."""
        return FileResponse(STATIC / "index.html", media_type="text/html")

    @app.get("/painel", include_in_schema=False)
    async def painel_ui() -> FileResponse:
        """Painel ao vivo das filas e métricas (para a demonstração)."""
        return FileResponse(STATIC / "painel.html", media_type="text/html")

    @app.get("/chat", include_in_schema=False)
    async def chat_ui() -> FileResponse:
        """Chat direto com os domínios do gateway (útil para depurar)."""
        return FileResponse(STATIC / "chat.html", media_type="text/html")

    # ------------------------------------------------------------------ gateway
    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Estado do gateway e de cada nó (acessível? modelos baixados?)."""
        nodes: dict[str, Any] = {}
        for name, node in cfg.nodes.items():
            wanted = sorted({d.model for d in cfg.domains.values() if d.node == name})
            try:
                if node.kind == "openai":
                    available = await openai_compat.list_models(gateway.http, node.url, node.api_key_env)
                elif node.kind == "anthropic":
                    available = await anthropic.list_models(gateway.http, node.url, node.api_key_env)
                else:
                    available = await list_models(gateway.http, node.url)
                if node.kind == "ollama":
                    missing = [m for m in wanted if m not in available and f"{m}:latest" not in available]
                else:
                    # Nó remoto (nuvem) não tem modelo a "baixar"; /models pode nem existir.
                    missing = []
                nodes[name] = {"up": True, "kind": node.kind, "models": available, "missing": missing}
            except NodeError as exc:
                nodes[name] = {"up": False, "kind": node.kind, "error": str(exc)}
        all_ok = all(n["up"] and not n["missing"] for n in nodes.values())
        return {"status": "ok" if all_ok else "degraded", "nodes": nodes, "rooms": len(rooms.rooms)}

    @app.get("/v1/domains")
    async def domains() -> dict[str, Any]:
        return {
            name: {"description": d.description, "node": d.node, "model": d.model,
                   "kind": cfg.nodes[d.node].kind, "impostor": rooms.is_impostor_domain(name)}
            for name, d in cfg.domains.items()
        }

    @app.get("/v1/personas")
    async def personas() -> dict[str, str]:
        """A roda de personalidades da IA impostora."""
        return {k: a.label for k, a in ARCHETYPES.items()}

    @app.get("/v1/nodes")
    async def nodes() -> list[dict[str, Any]]:
        return [q.snapshot() for q in gateway.queues.values()]

    @app.get("/metrics")
    async def get_metrics(window_s: float = 60.0, source: str | None = None) -> dict[str, Any]:
        return {
            "summary": gateway.metrics.summary(window_s, source),
            "nodes": [q.snapshot() for q in gateway.queues.values()],
            "rooms": {"open": len(rooms.rooms),
                      "playing": sum(1 for r in rooms.rooms.values() if r.in_game)},
        }

    @app.get("/metrics/requests")
    async def recent_requests(limit: int = 50, source: str | None = None) -> list[dict[str, Any]]:
        recs = [r for r in gateway.metrics.records if source is None or r.source == source]
        return [r.to_dict() for r in recs[-limit:]]

    @app.post("/v1/chat")
    async def chat(req: ChatRequest) -> StreamingResponse:
        try:
            job = gateway.admit(
                req.domain,
                ([m.model_dump() for m in req.messages] if req.messages else [])
                + ([{"role": "user", "content": req.prompt}] if req.prompt else []),
                max_tokens=req.max_tokens,
                variables=req.variables,
            )
        except UnknownDomain:
            raise HTTPException(404, f"domínio desconhecido '{req.domain}'. Veja GET /v1/domains") from None
        except QueueFull as exc:
            # Fila lotada: recusa antes de abrir o stream.
            raise HTTPException(503, f"fila do nó '{exc}' cheia", headers={"Retry-After": "2"}) from None

        async def event_stream():
            async for event, data in gateway.run(job):
                yield sse(event, data)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------ jogo
    @app.post("/v1/rooms", status_code=201)
    async def create_room(body: RoomCreate | None = None) -> dict[str, Any]:
        try:
            room = rooms.create(body.model_dump() if body else None)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        return room.public_summary()

    @app.get("/v1/rooms")
    async def list_rooms() -> list[dict[str, Any]]:
        """Salas esperando jogadores (para a tela inicial)."""
        return [r.public_summary() for r in rooms.rooms.values() if r.phase.value == "lobby" and r.humans()]

    @app.get("/v1/rooms/{code}")
    async def get_room(code: str) -> dict[str, Any]:
        room = rooms.get(code)
        if room is None:
            raise HTTPException(404, "sala não existe")
        return room.public_summary()

    @app.websocket("/ws/rooms/{code}")
    async def room_socket(ws: WebSocket, code: str, name: str = "", token: str | None = None):
        await ws.accept()
        room = rooms.get(code)
        if room is None:
            await ws.send_json({"type": "fatal", "message": "sala não existe (ou já fechou)"})
            await ws.close(code=4404)
            return
        try:
            player = room.join(name, token)
        except GameError as exc:
            await ws.send_json({"type": "fatal", "message": str(exc)})
            await ws.close(code=4400)
            return

        outbox = room.connect(player)

        async def sender():
            while True:
                msg = await outbox.get()
                if msg is None:  # sala fechada ou conexão substituída
                    with suppress(Exception):
                        await ws.close(code=4000)
                    return
                await ws.send_json(msg)

        send_task = asyncio.create_task(sender())
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    data = json.loads(raw)
                    if not isinstance(data, dict):
                        raise ValueError
                    room.handle(player, data)
                except GameError as exc:
                    outbox.put_nowait({"type": "error", "message": str(exc)})
                except ValueError:
                    outbox.put_nowait({"type": "error", "message": "JSON inválido"})
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            send_task.cancel()
            room.disconnect(player, outbox)

    return app
