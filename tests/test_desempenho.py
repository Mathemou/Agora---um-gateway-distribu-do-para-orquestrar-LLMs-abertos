"""Trilha Desempenho: balanceamento, saúde dos nós, failover e prioridade na fila."""

import asyncio
import json

import httpx
import pytest

from agora.balancer import priority_key
from agora.config import GatewayConfig
from agora.fake_node import create_fake_app
from agora.main import create_app
from agora.node_queue import QueueFull
from agora.pipeline import Gateway

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ------------------------------------------------------------------ transportes de teste
class Router(httpx.AsyncBaseTransport):
    """Manda cada host para um transporte diferente (um "nó" por host)."""

    def __init__(self, routes):
        self.routes = routes

    async def handle_async_request(self, request):
        return await self.routes[request.url.host].handle_async_request(request)


class Refuse(httpx.AsyncBaseTransport):
    """Nó fora do ar: conexão recusada."""

    def __init__(self):
        self.calls = 0

    async def handle_async_request(self, request):
        self.calls += 1
        raise httpx.ConnectError("conexão recusada", request=request)


class _Broken(httpx.AsyncByteStream):
    def __init__(self, n, hang):
        self.n, self.hang = n, hang

    async def __aiter__(self):
        for i in range(self.n):
            yield (json.dumps({"message": {"content": f"parcial{i} "}, "done": False}) + "\n").encode()
        if self.hang:
            await asyncio.sleep(30)
        raise httpx.ReadError("conexão cortada pelo nó")


class DiesMidStream(httpx.AsyncBaseTransport):
    """Nó que manda alguns tokens e cai (ou trava) no meio da resposta."""

    def __init__(self, tokens=3, hang=False):
        self.tokens, self.hang, self.calls = tokens, hang, 0

    async def handle_async_request(self, request):
        self.calls += 1
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:1.5b"}]})
        return httpx.Response(200, stream=_Broken(self.tokens, self.hang))


def fake(**kw):
    return httpx.ASGITransport(app=create_fake_app(tokens=kw.pop("tokens", 4), token_ms=1, prefill_ms=1, **kw))


def make_config(nodes=("a", "b", "c"), policy="round_robin", **extra):
    balance = {"policy": policy, "health_interval_s": 0, **extra.pop("balance", {})}
    return GatewayConfig.model_validate({
        "metrics_log": None,
        "nodes": {n: {"url": f"http://{n}", "max_concurrency": 1, "max_queue": 4} for n in nodes},
        "domains": {
            "impostor": {"nodes": list(nodes), "model": "qwen2.5:1.5b"},
            "mestre": {"node": nodes[0], "model": "qwen2.5:0.5b"},
        },
        "game": {"log_path": None},
        "balance": balance,
        **extra,
    })


async def collect(gw, job):
    events = []
    async for ev, data in gw.run(job):
        events.append((ev, data))
    return events


# ------------------------------------------------------------------ configuração
def test_pool_aceita_node_nodes_e_texto():
    cfg = make_config()
    assert cfg.domains["impostor"].pool == ["a", "b", "c"]
    assert cfg.domains["impostor"].node == "a"          # principal = 1º do pool
    assert cfg.domains["mestre"].pool == ["a"]           # `node` vira pool de um
    cfg2 = GatewayConfig.model_validate({
        "nodes": {"a": {"url": "http://a"}, "b": {"url": "http://b"}},
        "domains": {"impostor": {"nodes": "a, b", "model": "m"}},   # vindo do .env
        "game": {"master_domain": None},
    })
    assert cfg2.domains["impostor"].pool == ["a", "b"]
    with pytest.raises(ValueError, match="nó inexistente"):
        GatewayConfig.model_validate({
            "nodes": {"a": {"url": "http://a"}},
            "domains": {"impostor": {"nodes": ["a", "zz"], "model": "m"}},
            "game": {"master_domain": None},
        })


# ------------------------------------------------------------------ políticas
async def test_round_robin_reveza_os_nos():
    gw = Gateway(make_config(policy="round_robin"), transport=fake())
    chosen = []
    for _ in range(6):
        job = gw.admit("impostor", [{"role": "user", "content": "oi"}])
        chosen.append(job.node_name)
        job.ticket.release()
    assert chosen == ["a", "b", "c", "a", "b", "c"]
    await gw.aclose()


async def test_least_queue_escolhe_o_menos_ocupado():
    gw = Gateway(make_config(policy="least_queue"), transport=fake())
    held = [gw.admit("impostor", [{"role": "user", "content": "x"}]) for _ in range(3)]
    assert sorted(j.node_name for j in held) == ["a", "b", "c"]    # um em cada
    held[0].ticket.release()                                        # libera o nó de held[0]
    nxt = gw.admit("impostor", [{"role": "user", "content": "x"}])
    assert nxt.node_name == held[0].node_name
    for j in held[1:] + [nxt]:
        j.ticket.release()
    await gw.aclose()


async def test_least_latency_prefere_o_mais_rapido_e_explora_os_sem_amostra():
    gw = Gateway(make_config(policy="least_latency"), transport=fake())
    b = gw.balancer
    b.on_success("a", 900.0)
    b.on_success("b", 100.0)
    job = gw.admit("impostor", [{"role": "user", "content": "x"}])
    assert job.node_name == "c"            # sem amostra = 0 ms: é tentado primeiro
    job.ticket.release()
    b.on_success("c", 500.0)
    job = gw.admit("impostor", [{"role": "user", "content": "x"}])
    assert job.node_name == "b"
    job.ticket.release()
    # média móvel: b piora aos poucos (alpha 0,3) até perder para c
    for _ in range(5):
        b.on_success("b", 2000.0)
    assert b.health["b"].ewma_ttft_ms > 500
    job = gw.admit("impostor", [{"role": "user", "content": "x"}])
    assert job.node_name == "c"
    job.ticket.release()
    await gw.aclose()


async def test_peak_ewma_pesa_latencia_pela_carga():
    gw = Gateway(make_config(nodes=("a", "b"), policy="peak_ewma"), transport=fake())
    gw.balancer.on_success("a", 100.0)     # a é 3× mais rápido...
    gw.balancer.on_success("b", 300.0)
    held = [gw.admit("impostor", [{"role": "user", "content": "x"}]) for _ in range(4)]
    # ...então recebe enquanto 100 × (carga + 1) ≤ 300: com 3 na conta (100×4 = 400)
    # passa a valer a pena ir para b. least_latency puro mandaria as 4 para a.
    assert [j.node_name for j in held] == ["a", "a", "a", "b"]
    for j in held:
        j.ticket.release()
    # sem amostra nenhuma (início do experimento) se comporta como menor fila
    gw.balancer.reset_stats()
    held = [gw.admit("impostor", [{"role": "user", "content": "x"}]) for _ in range(4)]
    assert sorted(j.node_name for j in held) == ["a", "a", "b", "b"]
    for j in held:
        j.ticket.release()
    await gw.aclose()


async def test_fila_cheia_num_no_transborda_para_o_proximo():
    cfg = make_config(nodes=("a", "b"), policy="round_robin")
    cfg.nodes["a"].max_queue = 0
    cfg.nodes["b"].max_queue = 0
    gw = Gateway(cfg, transport=fake())
    j1 = gw.admit("impostor", [{"role": "user", "content": "x"}])   # a (vaga)
    j2 = gw.admit("impostor", [{"role": "user", "content": "x"}])   # b (vaga)
    assert {j1.node_name, j2.node_name} == {"a", "b"}
    with pytest.raises(QueueFull):                                   # os dois lotados → 503
        gw.admit("impostor", [{"role": "user", "content": "x"}])
    j1.ticket.release()
    j3 = gw.admit("impostor", [{"role": "user", "content": "x"}])   # rodízio cai em b (cheio) → a
    assert j3.node_name == j1.node_name
    for j in (j2, j3):
        j.ticket.release()
    await gw.aclose()


# ------------------------------------------------------------------ saúde
async def test_no_que_falha_sai_do_pool_e_volta_depois_do_castigo():
    now = [100.0]
    cfg = make_config(nodes=("a", "b"), balance={"fail_threshold": 2, "down_cooldown_s": 10})
    gw = Gateway(cfg, transport=fake())
    gw.balancer.clock = lambda: now[0]
    b = gw.balancer
    b.on_failure("a", "caiu")
    assert b.health["a"].healthy                      # 1 falha ainda não tira
    b.on_failure("a", "caiu")
    assert not b.health["a"].healthy
    dom = cfg.domains["impostor"]
    assert [b.order("impostor", dom)[0] for _ in range(4)] == ["b"] * 4
    now[0] += 11                                      # passou o castigo: meio-aberto
    assert "a" in b.order("impostor", dom)
    b.on_success("a", 50.0)
    assert b.health["a"].healthy
    assert [t[1] for t in b.health["a"].transitions] == ["down", "up"]
    await gw.aclose()


async def test_health_check_ativo_tira_e_devolve_o_no():
    cfg = make_config(nodes=("a", "b"), balance={"fail_threshold": 1, "health_interval_s": 5})
    gw = Gateway(cfg, transport=fake())
    up = {"a": True}

    async def probe(name):
        if name == "a" and not up["a"]:
            raise RuntimeError("sem resposta")
        return []

    assert await gw.balancer.check("a", probe)
    up["a"] = False
    assert not await gw.balancer.check("a", probe)
    assert not gw.balancer.available("a")             # com health check, só ele devolve
    up["a"] = True
    await gw.balancer.check("a", probe)
    assert gw.balancer.available("a")
    await gw.aclose()


async def test_todos_fora_tenta_mesmo_assim():
    cfg = make_config(nodes=("a", "b"), balance={"fail_threshold": 1})
    gw = Gateway(cfg, transport=fake())
    gw.balancer.on_failure("a", "x")
    gw.balancer.on_failure("b", "x")
    assert sorted(gw.balancer.order("impostor", cfg.domains["impostor"])) == ["a", "b"]
    await gw.aclose()


# ------------------------------------------------------------------ failover
async def test_failover_antes_do_primeiro_token():
    dead = Refuse()
    gw = Gateway(make_config(nodes=("a", "b")), transport=Router({"a": dead, "b": fake()}))
    job = gw.admit("impostor", [{"role": "user", "content": "x"}])
    assert job.node_name == "a"
    events = await collect(gw, job)
    names = [e for e, _ in events]
    assert names[0] == "meta" and "failover" in names and names[-1] == "done"
    fo = next(d for e, d in events if e == "failover")
    assert fo["from"] == "a" and fo["to"] == "b" and fo["discard_tokens"] == 0
    done = events[-1][1]
    assert done["node"] == "b" and done["failovers"] == 1 and done["nodes_path"] == "a>b"
    assert done["recovery_ms"] is not None
    await gw.aclose()


async def test_failover_no_meio_da_resposta_descarta_o_parcial():
    dying = DiesMidStream(tokens=3)
    gw = Gateway(make_config(nodes=("a", "b")), transport=Router({"a": dying, "b": fake(tokens=4)}))
    comp_resets = []
    comp = await gw.complete("impostor", [{"role": "user", "content": "x"}], on_reset=lambda: comp_resets.append(1))
    assert comp.ok, comp.error
    assert "parcial" not in comp.text                  # só a resposta do nó novo
    assert len(comp.text.split()) == 4
    rec = comp.record
    assert rec.failovers == 1 and rec.wasted_tokens == 3 and rec.output_tokens == 4
    assert rec.node == "b" and comp_resets == [1]
    assert gw.balancer.health["a"].failovers_from == 1
    await gw.aclose()


async def test_no_travado_e_detectado_pelo_stall_timeout():
    hung = DiesMidStream(tokens=2, hang=True)
    cfg = make_config(nodes=("a", "b"), balance={"stall_timeout_s": 0.2})
    gw = Gateway(cfg, transport=Router({"a": hung, "b": fake()}))
    comp = await gw.complete("impostor", [{"role": "user", "content": "x"}], timeout_s=5)
    assert comp.ok, comp.error
    assert comp.record.failovers == 1 and comp.record.wasted_tokens == 2
    assert "travado" in (gw.balancer.health["a"].last_error or "")
    await gw.aclose()


async def test_sem_failover_o_erro_chega_ao_cliente():
    cfg = make_config(nodes=("a", "b"), balance={"failover": False})
    gw = Gateway(cfg, transport=Router({"a": Refuse(), "b": fake()}))
    job = gw.admit("impostor", [{"role": "user", "content": "x"}])
    events = await collect(gw, job)
    assert events[-1][0] == "error" and "failover" not in [e for e, _ in events]
    await gw.aclose()


async def test_erro_da_requisicao_nao_derruba_o_no_nem_faz_failover():
    cfg = make_config(nodes=("a", "b"), balance={"fail_threshold": 1})
    cfg.domains["impostor"].model = "nao-existe"        # 404 do Ollama: culpa do pedido
    gw = Gateway(cfg, transport=fake())
    job = gw.admit("impostor", [{"role": "user", "content": "x"}])
    events = await collect(gw, job)
    assert events[-1][0] == "error" and "404" in events[-1][1]["message"]
    assert all(h.healthy for h in gw.balancer.health.values())
    await gw.aclose()


async def test_aquecimento_fixa_o_no_e_nao_faz_failover():
    gw = Gateway(make_config(nodes=("a", "b")), transport=Router({"a": fake(), "b": Refuse()}))
    comp = await gw.complete("impostor", [{"role": "user", "content": "x"}], node="b", max_tokens=1)
    assert not comp.ok and comp.record.node == "b" and comp.record.failovers == 0
    await gw.aclose()


# ------------------------------------------------------------------ prioridade na fila
async def test_fifo_x_prioridade_com_envelhecimento():
    cfg = make_config(nodes=("a",), queue={"policy": "fifo"})
    # FIFO: a chave é só a hora de chegada
    assert priority_key(cfg, "mestre", now=10) < priority_key(cfg, "impostor", now=11)
    cfg.queue.policy = "priority"
    # Prioridade: a fala da IA que chegou 1 s depois passa na frente do Mestre...
    assert priority_key(cfg, "impostor", now=11) < priority_key(cfg, "mestre", now=10)
    # ...mas não de um Mestre que já espera há mais que a diferença (4 s): sem inanição
    assert priority_key(cfg, "mestre", now=10) < priority_key(cfg, "impostor", now=15)


async def test_fila_com_prioridade_atende_a_fala_da_ia_primeiro():
    cfg = make_config(nodes=("a",), queue={"policy": "priority"})
    gw = Gateway(cfg, transport=fake())
    busy = gw.admit("mestre", [{"role": "user", "content": "x"}])          # ocupa a vaga
    order = []
    jobs = [gw.admit("impostor", [{"role": "user", "content": "x"}], source=src)
            for src in ("mestre", "api", "impostor")]

    async def wait(job, tag):
        await job.ticket.wait()
        order.append(tag)
        job.ticket.release()

    tasks = [asyncio.create_task(wait(j, j.rec.source)) for j in jobs]
    await asyncio.sleep(0)
    busy.ticket.release()
    await asyncio.gather(*tasks)
    assert order == ["impostor", "api", "mestre"]
    await gw.aclose()


# ------------------------------------------------------------------ API
async def test_api_do_balanceador_e_metricas_por_no():
    cfg = make_config(nodes=("a", "b"))
    app = create_app(cfg, transport=Router({"a": Refuse(), "b": fake()}))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        r = await c.post("/v1/balancer", json={"policy": "least_queue", "queue_policy": "priority"})
        assert r.status_code == 200 and r.json()["policy"] == "least_queue"
        assert r.json()["queue_policy"] == "priority"
        r = await c.post("/v1/chat", json={"domain": "impostor", "prompt": "oi", "source": "impostor"})
        body = r.text
        assert "event: failover" in body and "event: done" in body
        nodes = {n["node"]: n for n in (await c.get("/v1/nodes")).json()}
        assert nodes["a"]["failovers_from"] == 1 and nodes["b"]["ewma_ttft_ms"] is not None
        m = (await c.get("/metrics")).json()["summary"]
        assert m["failovers"] == 1 and m["by_node"] == {"b": 1}
        assert (await c.get("/v1/domains")).json()["impostor"]["nodes"] == ["a", "b"]
        h = (await c.get("/health")).json()
        assert h["nodes"]["a"]["up"] is False and h["status"] == "degraded"
