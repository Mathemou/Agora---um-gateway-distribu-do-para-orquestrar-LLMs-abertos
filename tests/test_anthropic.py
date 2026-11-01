import json

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from agora.config import GatewayConfig
from agora.game import RoomManager
from agora.main import create_app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def fake_claude(seen: dict) -> FastAPI:
    """Imita POST /v1/messages (stream SSE) e GET /v1/models da API do Claude."""
    app = FastAPI()

    @app.get("/v1/models")
    async def models(request: Request):
        seen["models_key"] = request.headers.get("x-api-key")
        return {"data": [{"id": "claude-haiku-5-5"}]}

    @app.post("/v1/messages")
    async def messages(request: Request):
        seen["headers"] = dict(request.headers)
        seen["body"] = await request.json()

        def ev(name, data):
            return f"event: {name}\ndata: {json.dumps(data)}\n\n"

        async def gen():
            yield ev("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 42, "output_tokens": 1}}})
            yield ev("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
            yield ev("ping", {"type": "ping"})
            for t in ["kkk ", "sei ", "la"]:
                yield ev("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": t}})
            yield ev("content_block_stop", {"type": "content_block_stop", "index": 0})
            yield ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}})
            yield ev("message_stop", {"type": "message_stop"})

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def make_config():
    return GatewayConfig.model_validate({
        "metrics_log": None,
        "nodes": {"local": {"url": "http://fake"},
                  "claude": {"url": "http://claude", "kind": "anthropic", "api_key_env": "TEST_CLAUDE_KEY"}},
        "domains": {
            "impostor": {"node": "local", "model": "qwen2.5:1.5b"},
            "impostor-claude": {"node": "claude", "model": "claude-haiku-5-5",
                                "system_prompt": "Você é {alias}.",
                                "options": {"temperature": 1.3, "num_predict": 60, "top_p": 0.9}},
            "mestre": {"node": "local", "model": "qwen2.5:0.5b"},
        },
        "game": {"log_path": None},
    })


def parse_sse(text):
    out = []
    for block in text.strip().split("\n\n"):
        name = data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:])
        out.append((name, data))
    return out


async def test_claude_via_gateway_token_a_token(monkeypatch):
    monkeypatch.setenv("TEST_CLAUDE_KEY", "sk-teste")
    seen = {}
    app = create_app(make_config(), transport=httpx.ASGITransport(app=fake_claude(seen)))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        r = await c.post("/v1/chat", json={"domain": "impostor-claude", "prompt": "oi",
                                           "variables": {"alias": "Laranja"}})
        events = parse_sse(r.text)
        assert [e for e, _ in events].count("token") == 3
        done = events[-1][1]
        assert events[-1][0] == "done" and done["node"] == "claude"
        assert done["output_tokens"] == 3 and done["prompt_tokens"] == 42
        assert done["ttft_ms"] is not None and done["node_tps"] > 0

        h = (await c.get("/health")).json()["nodes"]["claude"]
        assert h["up"] and h["kind"] == "anthropic" and h["missing"] == []

    body = seen["body"]
    assert body["system"] == "Você é Laranja."           # system vai fora de messages
    assert body["messages"] == [{"role": "user", "content": "oi"}]
    assert body["temperature"] == 1.0                      # limitado a [0, 1]
    assert "top_p" not in body and body["max_tokens"] == 60 and body["stream"] is True
    assert seen["headers"]["x-api-key"] == "sk-teste"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"


async def test_sem_chave_vira_erro_claro(monkeypatch):
    monkeypatch.delenv("TEST_CLAUDE_KEY", raising=False)
    app = create_app(make_config(), transport=httpx.ASGITransport(app=fake_claude({})))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        r = await c.post("/v1/chat", json={"domain": "impostor-claude", "prompt": "oi"})
        name, data = parse_sse(r.text)[-1]
        assert name == "error" and "TEST_CLAUDE_KEY" in data["message"]
        h = (await c.get("/health")).json()["nodes"]["claude"]
        assert h["up"] is False
        nodes = (await c.get("/v1/nodes")).json()
        assert all(n["active"] == 0 for n in nodes)  # a vaga foi devolvida


def test_sala_escolhe_quem_joga_como_ia():
    cfg = make_config()

    class G:  # basta o cfg do gateway para validar o nome do domínio
        pass
    g = G(); g.cfg = cfg
    mgr = RoomManager(cfg.game, g)
    assert mgr.create({"impostor_domain": "impostor-claude"}).cfg.impostor_domain == "impostor-claude"
    assert mgr.create({}).cfg.impostor_domain == "impostor"
    with pytest.raises(ValueError):
        mgr.create({"impostor_domain": "mestre"})
