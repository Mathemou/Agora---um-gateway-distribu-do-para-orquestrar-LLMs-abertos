import json

import httpx
import pytest

from agora.config import GatewayConfig
from agora.fake_node import create_fake_app
from agora.main import create_app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def make_config(tmp_path, max_queue=8):
    return GatewayConfig.model_validate({
        "metrics_log": str(tmp_path / "req.jsonl"),
        "nodes": {
            "n1": {"url": "http://fake", "max_concurrency": 1, "max_queue": max_queue},
            "n2": {"url": "http://fake", "max_concurrency": 1, "max_queue": max_queue},
        },
        "domains": {
            "impostor": {"node": "n1", "model": "qwen2.5:1.5b", "system_prompt": "Você é {alias}."},
            "mestre": {"node": "n2", "model": "qwen2.5:0.5b"},
            "quebrado": {"node": "n2", "model": "nao-existe"},
        },
        "game": {"log_path": None},
    })


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:])
        events.append((name, data))
    return events


@pytest.fixture
def gateway(tmp_path):
    fake = create_fake_app(tokens=5, token_ms=1, prefill_ms=1)
    app = create_app(make_config(tmp_path), transport=httpx.ASGITransport(app=fake))
    return app


async def client_for(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")


async def test_stream_token_a_token_e_metricas(gateway, tmp_path):
    async with await client_for(gateway) as c:
        r = await c.post("/v1/chat", json={"domain": "impostor", "prompt": "oi"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(r.text)

    names = [e[0] for e in events]
    assert names[0] == "meta" and names[-1] == "done"
    assert names.count("token") == 5
    done = events[-1][1]
    assert done["status"] == "ok" and done["node"] == "n1"
    assert done["output_tokens"] == 5
    for k in ("ttft_ms", "e2e_ms", "queue_wait_ms", "node_tps"):
        assert done[k] is not None
    assert done["ttft_ms"] <= done["e2e_ms"]
    log = (tmp_path / "req.jsonl").read_text().splitlines()
    assert json.loads(log[-1])["request_id"] == done["request_id"]


async def test_roteia_pelo_dominio(gateway):
    async with await client_for(gateway) as c:
        r = await c.post("/v1/chat", json={"domain": "mestre", "prompt": "oi"})
        done = parse_sse(r.text)[-1][1]
        assert done["node"] == "n2" and done["model"] == "qwen2.5:0.5b"


async def test_dominio_desconhecido_404(gateway):
    async with await client_for(gateway) as c:
        r = await c.post("/v1/chat", json={"domain": "culinaria", "prompt": "oi"})
        assert r.status_code == 404


async def test_requisicao_sem_conteudo_422(gateway):
    async with await client_for(gateway) as c:
        r = await c.post("/v1/chat", json={"domain": "impostor"})
        assert r.status_code == 422


async def test_erro_do_no_vira_evento_error(gateway):
    async with await client_for(gateway) as c:
        r = await c.post("/v1/chat", json={"domain": "quebrado", "prompt": "oi"})
        events = parse_sse(r.text)
        assert events[-1][0] == "error"
        nodes = (await c.get("/v1/nodes")).json()
        assert all(n["active"] == 0 for n in nodes)


async def test_fila_cheia_503(tmp_path):
    fake = create_fake_app(tokens=1, token_ms=1, prefill_ms=1)
    app = create_app(make_config(tmp_path, max_queue=0), transport=httpx.ASGITransport(app=fake))
    app.state.queues["n1"].enqueue()  # ocupa a única vaga
    async with await client_for(app) as c:
        r = await c.post("/v1/chat", json={"domain": "impostor", "prompt": "oi"})
        assert r.status_code == 503
        assert r.headers["retry-after"] == "2"


async def test_health_e_metrics(gateway):
    async with await client_for(gateway) as c:
        await c.post("/v1/chat", json={"domain": "impostor", "prompt": "oi"})
        h = (await c.get("/health")).json()
        assert h["nodes"]["n1"]["up"] is True
        assert h["nodes"]["n2"]["missing"] == ["nao-existe"]
        m = (await c.get("/metrics")).json()
        assert m["summary"]["by_status"]["ok"] == 1
        assert m["summary"]["total_tokens"] == 5


def test_keep_alive_numerico_vira_inteiro(tmp_path):
    for raw, expected in (("-1", -1), (-1, -1), ("30m", "30m"), ("300", 300)):
        cfg = make_config(tmp_path).model_dump()
        cfg["keep_alive"] = raw
        assert GatewayConfig.model_validate(cfg).keep_alive == expected


async def test_variaveis_no_prompt_de_sistema(tmp_path):
    seen = {}
    fake = create_fake_app(tokens=1, token_ms=1, prefill_ms=1)

    class Spy(httpx.ASGITransport):
        async def handle_async_request(self, request):
            if request.url.path == "/api/chat":
                seen.update(json.loads(request.content))
            return await super().handle_async_request(request)

    app = create_app(make_config(tmp_path), transport=Spy(app=fake))
    async with await client_for(app) as c:
        r = await c.post("/v1/chat", json={"domain": "impostor", "prompt": "oi",
                                           "variables": {"alias": "Laranja"}})
        assert parse_sse(r.text)[-1][0] == "done"
    assert seen["messages"][0] == {"role": "system", "content": "Você é Laranja."}


async def test_metricas_separadas_por_origem(gateway):
    async with await client_for(gateway) as c:
        await c.post("/v1/chat", json={"domain": "impostor", "prompt": "oi"})
        m = (await c.get("/metrics", params={"source": "api"})).json()
        assert m["summary"]["by_source"] == {"api": 1}
        assert (await c.get("/metrics", params={"source": "impostor"})).json()["summary"]["requests"] == 0


async def test_paginas_estaticas(gateway):
    async with await client_for(gateway) as c:
        for path in ("/", "/chat", "/painel"):
            r = await c.get(path)
            assert r.status_code == 200 and "<html" in r.text
