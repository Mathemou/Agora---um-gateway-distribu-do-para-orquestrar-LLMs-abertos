"""Versão na nuvem: token na API e a configuração gateway.cloud.yaml."""

import httpx
import pytest

from agora.config import GatewayConfig, load_config
from agora.fake_node import create_fake_app
from agora.main import create_app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def make_app(token_env="AGORA_API_TOKEN"):
    cfg = GatewayConfig.model_validate({
        "metrics_log": None, "api_token_env": token_env,
        "nodes": {"n1": {"url": "http://fake"}},
        "domains": {"impostor": {"node": "n1", "model": "qwen2.5:1.5b"},
                    "mestre": {"node": "n1", "model": "qwen2.5:0.5b"}},
        "game": {"log_path": None},
    })
    fake = create_fake_app(tokens=3, token_ms=1, prefill_ms=1)
    return create_app(cfg, transport=httpx.ASGITransport(app=fake))


async def test_api_exige_token_quando_configurado(monkeypatch):
    monkeypatch.setenv("AGORA_API_TOKEN", "s3gredo")
    app = make_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        body = {"domain": "impostor", "prompt": "oi"}
        assert (await c.post("/v1/chat", json=body)).status_code == 401
        assert (await c.post("/v1/chat", json=body, headers={"Authorization": "Bearer errado"})).status_code == 401
        r = await c.post("/v1/chat", json=body, headers={"Authorization": "Bearer s3gredo"})
        assert r.status_code == 200 and "event: done" in r.text
        assert (await c.post("/v1/chat", json=body, headers={"X-Agora-Token": "s3gredo"})).status_code == 200
        assert (await c.post("/v1/balancer", json={"policy": "round_robin"})).status_code == 401
        # o jogo continua aberto: criar sala e ler o estado não pedem token
        assert (await c.post("/v1/rooms")).status_code == 201
        assert (await c.get("/health")).status_code == 200


async def test_sem_token_definido_a_api_fica_aberta(monkeypatch):
    monkeypatch.delenv("AGORA_API_TOKEN", raising=False)
    app = make_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        r = await c.post("/v1/chat", json={"domain": "impostor", "prompt": "oi"})
        assert r.status_code == 200


def test_config_da_nuvem_carrega():
    cfg = load_config("config/gateway.cloud.yaml")
    assert cfg.game.impostor_domain in cfg.domains
    assert cfg.game.impostor_fallback_domain in cfg.domains
    assert cfg.game.impostor_fallback_domain != cfg.game.impostor_domain
    assert all(n.kind != "ollama" for n in cfg.nodes.values())   # nada de Ollama na VM
    assert cfg.api_token_env == "AGORA_API_TOKEN"
