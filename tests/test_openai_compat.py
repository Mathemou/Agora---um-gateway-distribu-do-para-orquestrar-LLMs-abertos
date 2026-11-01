import asyncio
import json
import random

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from agora.config import GatewayConfig
from agora.fake_node import create_fake_app
from agora.game import Phase, Room
from agora.main import create_app
from agora.openai_compat import _ThinkFilter, build_payload
from agora.pipeline import Gateway

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def fake_groq(seen: dict, pieces=("kkk ", "sei ", "la"), status=200, think=False) -> FastAPI:
    """Imita POST /chat/completions (SSE no formato da OpenAI, com x_groq.usage) e GET /models."""
    app = FastAPI()

    @app.get("/openai/v1/models")
    async def models(request: Request):
        seen["auth"] = request.headers.get("authorization")
        return {"data": [{"id": "qwen/qwen3.8-27b"}, {"id": "openai/gpt-oss-120b"}]}

    @app.post("/openai/v1/chat/completions")
    async def chat(request: Request):
        seen["body"] = await request.json()
        seen["auth"] = request.headers.get("authorization")
        if status != 200:
            return JSONResponse({"error": {"message": "Rate limit reached"}}, status_code=status,
                                headers={"retry-after": "7"})
        parts = (["<thi", "nk>hmm, deixa eu pensar</th", "ink>"] if think else []) + list(pieces)

        async def gen():
            for p in parts:
                c = {"choices": [{"index": 0, "delta": {"content": p}, "finish_reason": None}]}
                yield f"data: {json.dumps(c)}\n\n"
            last = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "x_groq": {"id": "req_1", "usage": {"prompt_tokens": 321, "completion_tokens": 3}}}
            yield f"data: {json.dumps(last)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def config(**game):
    return GatewayConfig.model_validate({
        "metrics_log": None,
        "nodes": {"local": {"url": "http://local"},
                  "groq": {"url": "http://groq/openai/v1", "kind": "openai", "api_key_env": "TEST_GROQ_KEY"}},
        "domains": {
            "impostor": {"node": "local", "model": "qwen2.5:1.5b", "options": {"num_predict": 6}},
            "impostor-groq": {"node": "groq", "model": "qwen/qwen3.8-27b", "system_prompt": "Você é {alias}.",
                              "options": {"temperature": 0.9, "num_predict": 48, "reasoning_effort": "none",
                                          "num_ctx": 2048, "repeat_penalty": 1.1, "stop": ["\n"]}},
            "mestre": {"node": "local", "model": "qwen2.5:0.5b", "options": {"num_predict": 6}},
        },
        "game": {"log_path": None, "discussion_s": 5, "vote_s": 2, "reveal_s": 0.05,
                 "read_delay_s": [0.01, 0.02], "typing_cps": [5000, 5000], "ai_cooldown_s": 0, **game},
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


class Router(httpx.AsyncBaseTransport):
    """Manda http://groq para o Groq falso e o resto para o nó Ollama falso."""

    def __init__(self, groq, local):
        self.groq = httpx.ASGITransport(app=groq)
        self.local = httpx.ASGITransport(app=local)

    async def handle_async_request(self, request):
        t = self.groq if request.url.host == "groq" else self.local
        return await t.handle_async_request(request)


async def test_groq_via_gateway(monkeypatch):
    monkeypatch.setenv("TEST_GROQ_KEY", "gsk-teste")
    seen = {}
    app = create_app(config(), transport=Router(fake_groq(seen), create_fake_app(tokens=3, token_ms=1, prefill_ms=1)))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        r = await c.post("/v1/chat", json={"domain": "impostor-groq", "prompt": "oi",
                                           "variables": {"alias": "Laranja"}})
        ev = parse_sse(r.text)
        assert "".join(d["text"] for e, d in ev if e == "token") == "kkk sei la"
        done = ev[-1][1]
        assert ev[-1][0] == "done" and done["prompt_tokens"] == 321 and done["output_tokens"] == 3
        h = (await c.get("/health")).json()["nodes"]["groq"]
        assert h["up"] and h["kind"] == "openai" and h["missing"] == []
    b = seen["body"]
    assert seen["auth"] == "Bearer gsk-teste"
    assert b["messages"][0] == {"role": "system", "content": "Você é Laranja."}
    assert b["max_tokens"] == 48 and b["reasoning_effort"] == "none" and b["stream"] is True
    assert "num_ctx" not in b and "repeat_penalty" not in b and "num_predict" not in b


def test_payload_gpt_oss_ajusta_raciocinio():
    b = build_payload("openai/gpt-oss-120b", [], {"num_predict": 48, "reasoning_effort": "none",
                                                   "reasoning_format": "hidden", "stop": ["\n", "a", "b", "c", "d"]})
    assert b["reasoning_effort"] == "low" and b["include_reasoning"] is False
    assert "reasoning_format" not in b and b["max_tokens"] >= 512 and b["stop"] == ["\n", "a", "b", "c"]


def test_filtro_think_picado():
    f = _ThinkFilter()
    out = "".join(f.feed(p) for p in ["oi <thi", "nk>pensando...</thi", "nk> tudo", " bem"]) + f.flush()
    assert out == "oi  tudo bem"
    g = _ThinkFilter()
    assert g.feed("a <") + g.feed("b") + g.flush() == "a <b"


async def test_think_removido_no_stream(monkeypatch):
    monkeypatch.setenv("TEST_GROQ_KEY", "k")
    app = create_app(config(), transport=Router(fake_groq({}, think=True), create_fake_app()))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        r = await c.post("/v1/chat", json={"domain": "impostor-groq", "prompt": "oi"})
        assert "".join(d["text"] for e, d in parse_sse(r.text) if e == "token") == "kkk sei la"


async def test_limite_de_taxa_vira_erro_claro(monkeypatch):
    monkeypatch.setenv("TEST_GROQ_KEY", "k")
    app = create_app(config(), transport=Router(fake_groq({}, status=429), create_fake_app()))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        r = await c.post("/v1/chat", json={"domain": "impostor-groq", "prompt": "oi"})
        name, data = parse_sse(r.text)[-1]
        assert name == "error" and "429" in data["message"] and "7 s" in data["message"]


async def test_ia_cai_para_o_modelo_local_se_o_groq_falhar(monkeypatch):
    monkeypatch.setenv("TEST_GROQ_KEY", "k")
    cfg = config(impostor_domain="impostor-groq")
    gw = Gateway(cfg, transport=Router(fake_groq({}, status=429),
                                       create_fake_app(tokens=6, token_ms=1, prefill_ms=1)))
    room = Room("TEST", cfg.game, gw, rng=random.Random(3))
    a, b = room.join("A"), room.join("B")
    room.connect(a), room.connect(b)
    room.start(a)
    for _ in range(300):
        if room.phase == Phase.DISCUSSION and room.topic and not room.preparing:
            break
        await asyncio.sleep(0.02)
    ai = room.ais[0]
    room.post_chat(a, f"{ai.player.alias} vc é robô?")
    for _ in range(300):
        if ai.stats:
            break
        await asyncio.sleep(0.02)
    st = ai.stats[0]
    assert st["failover"]["from"] == "impostor-groq" and st["failover"]["to"] == "impostor"
    assert not st.get("fallback")  # o modelo local respondeu: não precisou de frase pronta
    room.close()
