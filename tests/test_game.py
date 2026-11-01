import asyncio
import random

import httpx
import pytest
from starlette.testclient import TestClient

from agora.ai import clean_reply, mentions
from agora.config import GatewayConfig
from agora.fake_node import create_fake_app
from agora.game import GameError, Phase, Room, RoomManager
from agora.main import create_app
from agora.pipeline import Gateway

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


FAST = {
    "discussion_s": 1.0, "vote_s": 1.5, "reveal_s": 0.05, "master_timeout_s": 2,
    "read_delay_s": [0.01, 0.02], "typing_cps": [5000, 5000], "ai_cooldown_s": 0,
}


def make_config(tmp_path, **game):
    return GatewayConfig.model_validate({
        "metrics_log": None,
        "nodes": {"n1": {"url": "http://fake", "max_concurrency": 2},
                  "n2": {"url": "http://fake", "max_concurrency": 2}},
        "domains": {
            "impostor": {"node": "n1", "model": "qwen2.5:1.5b", "system_prompt": "Você é {alias}.",
                         "options": {"num_predict": 6}},
            "mestre": {"node": "n2", "model": "qwen2.5:0.5b", "options": {"num_predict": 6}},
        },
        "game": {**FAST, "log_path": str(tmp_path / "games.jsonl"), **game},
    })


@pytest.fixture
def gateway(tmp_path):
    cfg = make_config(tmp_path)
    fake = create_fake_app(tokens=6, token_ms=1, prefill_ms=1)
    return Gateway(cfg, transport=httpx.ASGITransport(app=fake))


def new_room(gateway, **game):
    cfg = gateway.cfg.game.model_copy(update=game)
    return Room("TEST", cfg, gateway, rng=random.Random(7))


def drain(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


async def wait_for(cond, timeout=10.0):
    t = 0.0
    while not cond():
        await asyncio.sleep(0.02)
        t += 0.02
        if t > timeout:
            raise AssertionError("condição não aconteceu a tempo")


async def setup_game(gateway, humans=2, **game):
    room = new_room(gateway, **game)
    players = [room.join(f"jogador{i}") for i in range(humans)]
    queues = [room.connect(p) for p in players]
    room.start(players[0])
    await wait_for(lambda: room.phase == Phase.DISCUSSION and room.topic)
    return room, players, queues


# ------------------------------------------------------------------ lobby
async def test_lobby_regras_de_entrada(gateway):
    room = new_room(gateway)
    a = room.join("Matheus")
    room.connect(a)
    assert room.host is a
    with pytest.raises(GameError):
        room.join("matheus")  # apelido repetido
    with pytest.raises(GameError):
        room.join("   ")
    assert not room.can_start()  # 1 humano + 1 IA < 3
    with pytest.raises(GameError):
        room.start(a)
    b = room.join("Yan")
    room.connect(b)
    assert room.can_start()
    with pytest.raises(GameError):
        room.start(b)  # só o anfitrião
    # voltar com o token devolve o mesmo jogador
    assert room.join("", token=b.token) is b


async def test_saida_do_anfitriao_no_lobby_passa_a_vez(gateway):
    room = new_room(gateway)
    a, b = room.join("A"), room.join("B")
    qa, _ = room.connect(a), room.connect(b)
    room.disconnect(a, qa)
    assert room.host is b and [p.name for p in room.players] == ["B"]


# ------------------------------------------------------------------ partida
async def test_identidades_escondidas_durante_a_partida(gateway):
    room, (a, b), (qa, qb) = await setup_game(gateway)
    assert len(room.players) == 3 and sum(p.is_ai for p in room.players) == 1
    aliases = {p.alias for p in room.players}
    assert len(aliases) == 3 and None not in aliases
    st = room.state_for(a)
    assert "lobby" not in st
    assert all("name" not in p for p in st["players"])
    assert st["you"]["alias"] == a.alias and st["you"]["name"] == "jogador0"
    assert {p["alias"] for p in st["players"]} == aliases
    room.close()


async def test_mestre_faz_a_pergunta_token_a_token(gateway):
    room, _, (qa, _) = await setup_game(gateway)
    msgs = drain(qa)
    deltas = [m for m in msgs if m["type"] == "narrator_delta"]
    ends = [m for m in msgs if m["type"] == "narrator_end"]
    assert len(deltas) >= 2 and ends and ends[0]["text"] == room.topic
    room.close()


async def test_ia_conversa_e_humano_ve_so_o_codinome(gateway):
    room, (a, b), (qa, qb) = await setup_game(gateway, discussion_s=5)
    ai = next(p for p in room.players if p.is_ai)
    room.post_chat(a, f"oi {ai.alias}, você é robô?")
    await wait_for(lambda: any(m.get("from") == ai.alias and m["kind"] == "player" for m in room.chat))
    msg = next(m for m in room.chat if m.get("from") == ai.alias and m["kind"] == "player")
    assert msg["color"] == ai.color and msg["text"]
    assert any(m.get("from") == a.alias for m in drain(qb) if m["type"] == "chat")
    stats = room.ais[0].stats
    assert stats and stats[0]["ttft_ms"] is not None and stats[0]["response_ms"] > 0
    room.close()


async def test_humanos_acham_a_ia_e_vencem(gateway, tmp_path):
    room, (a, b), _ = await setup_game(gateway)
    ai = next(p for p in room.players if p.is_ai)
    await wait_for(lambda: room.phase == Phase.VOTING)
    room.cast_vote(a, ai.alias)
    room.cast_vote(b, ai.alias)
    await wait_for(lambda: room.phase == Phase.ENDED)
    assert room.ending["winner"] == "humanos"
    assert room.ending["detection"] == {"human_votes": 2, "hits": 2, "rate": 1.0}
    names = {r["alias"]: r["name"] for r in room.ending["reveal"]}
    assert names[a.alias] == "jogador0" and names[ai.alias] is None
    assert (tmp_path / "games.jsonl").read_text().count('"winner": "humanos"') == 1


async def test_expulsar_humano_com_dois_humanos_entrega_o_jogo(gateway):
    room, (a, b), _ = await setup_game(gateway)
    await wait_for(lambda: room.phase == Phase.VOTING)
    room.cast_vote(a, b.alias)
    await wait_for(lambda: room.phase == Phase.ENDED)
    # 1 voto em b contra 0 ou 1 da IA: b sai, ou empata e a IA vence na última rodada
    if not b.alive:
        assert room.ending["winner"] == "ia"


async def test_todos_prontos_encerra_a_discussao_cedo(gateway):
    room, (a, b), _ = await setup_game(gateway, discussion_s=30)
    room.set_ready(a)
    room.set_ready(b)  # a IA acompanha sozinha
    await wait_for(lambda: room.phase == Phase.VOTING, timeout=3)
    room.close()


async def test_regras_de_voto(gateway):
    room, (a, b), _ = await setup_game(gateway, vote_s=30)
    with pytest.raises(GameError):
        room.cast_vote(a, b.alias)  # ainda é discussão
    room.set_ready(a); room.set_ready(b)
    await wait_for(lambda: room.phase == Phase.VOTING)
    with pytest.raises(GameError):
        room.cast_vote(a, a.alias)
    with pytest.raises(GameError):
        room.cast_vote(a, "Ninguém")
    room.cast_vote(a, None)  # pular
    with pytest.raises(GameError):
        room.cast_vote(a, b.alias)  # já votou
    room.close()


async def test_apuracao_empate_e_pular(gateway):
    room = new_room(gateway)
    room.votes = {"A": "B", "B": "A", "C": None}
    t = room.tally()
    assert t["ejected"] is None and t["tie"]
    room.votes = {"A": "C", "B": None, "C": None}
    assert room.tally()["ejected"] is None  # pular empatou com o mais votado
    room.votes = {"A": "C", "B": "C", "C": None}
    assert room.tally()["ejected"] == "C"


async def test_reconectar_no_meio_da_partida(gateway):
    room, (a, b), (qa, qb) = await setup_game(gateway)
    room.disconnect(b, qb)
    assert not b.connected and b in room.players
    with pytest.raises(GameError):
        room.join("intruso")
    again = room.join("", token=b.token)
    assert again is b
    q2 = room.connect(b)
    kinds = [m["type"] for m in drain(q2)]
    assert kinds[:3] == ["welcome", "history", "state"]
    room.close()


async def test_volta_ao_lobby_depois_do_fim(gateway):
    room, (a, b), _ = await setup_game(gateway)
    ai = next(p for p in room.players if p.is_ai)
    await wait_for(lambda: room.phase == Phase.VOTING)
    room.cast_vote(a, ai.alias); room.cast_vote(b, ai.alias)
    await wait_for(lambda: room.phase == Phase.ENDED)
    room.back_to_lobby(a)
    assert room.phase == Phase.LOBBY and all(not p.is_ai for p in room.players)
    assert all(p.alias is None for p in room.players)
    assert room.can_start()


# ------------------------------------------------------------------ IA
def test_limpeza_da_resposta_da_ia():
    rng = random.Random(1)
    aliases = ["Laranja", "Azul"]
    assert clean_reply("Laranja: Eu comeria pizza.", aliases, 280, rng).lower() == "eu comeria pizza"
    assert clean_reply('Laranja (você): "kkk sei la"\noutra linha', aliases, 280, rng) == "kkk sei la"
    assert clean_reply("Como uma IA, não tenho preferências.", aliases, 280, rng) is None
    assert clean_reply("sou um modelo de linguagem", aliases, 280, rng) is None
    assert clean_reply("   \n  ", aliases, 280, rng) is None
    assert len(clean_reply("palavra " * 100, aliases, 50, rng)) <= 50


def test_mencao_ignora_acentos_e_caixa():
    assert mentions("o LARANJA tá estranho", "Laranja")
    assert mentions("e o ciano?", "Ciano")
    assert not mentions("laranjada", "Laranja")


def test_ajustes_da_sala_tem_limites(tmp_path):
    mgr = RoomManager(make_config(tmp_path).game, None)
    r = mgr.create({"discussion_s": 5, "max_rounds": 99, "ai_count": 2, "min_players": 2})
    assert r.cfg.discussion_s == 20 and r.cfg.max_rounds == 10
    assert r.cfg.ai_count == 2 and r.cfg.min_players == 3  # sempre sobra 1 humano
    assert mgr.get(r.code.lower()) is r


# ------------------------------------------------------------------ WebSocket
def test_websocket_partida_ponta_a_ponta(tmp_path):
    fake = create_fake_app(tokens=6, token_ms=1, prefill_ms=1)
    app = create_app(make_config(tmp_path, discussion_s=30), transport=httpx.ASGITransport(app=fake))
    with TestClient(app) as client:
        code = client.post("/v1/rooms", json={}).json()["code"]
        assert client.get("/v1/rooms").json() == []  # sala sem ninguém não aparece
        with client.websocket_connect(f"/ws/rooms/{code}?name=Matheus") as w1, \
             client.websocket_connect(f"/ws/rooms/{code}?name=Yan") as w2:

            def until(ws, pred):
                for _ in range(200):
                    m = ws.receive_json()
                    if pred(m):
                        return m
                raise AssertionError("mensagem esperada não chegou")

            assert until(w1, lambda m: m["type"] == "welcome")["token"]
            st = until(w1, lambda m: m["type"] == "state" and len(m.get("lobby", [])) == 2)
            assert st["can_start"] and st["you"]["is_host"]
            w2.send_json({"type": "start"})
            assert "anfitrião" in until(w2, lambda m: m["type"] == "error")["message"]
            w1.send_json({"type": "start"})
            st = until(w2, lambda m: m["type"] == "state" and m["phase"] == "discussao")
            assert len(st["players"]) == 3
            # espera a estação "ligar" (IA aquecida) e o chat abrir
            until(w1, lambda m: m["type"] == "state" and m.get("topic") and not m.get("preparing"))
            w1.send_json({"type": "chat", "text": "oi gente"})
            msg = until(w2, lambda m: m["type"] == "chat" and m.get("text") == "oi gente")
            assert msg["kind"] == "player" and msg["from"] != "Matheus"
        assert client.get(f"/v1/rooms/{code}").json()["phase"] == "discussao"
        assert client.get("/metrics", params={"source": "mestre"}).json()["summary"]["requests"] >= 1


def test_websocket_sala_inexistente(tmp_path):
    app = create_app(make_config(tmp_path))
    with TestClient(app) as client:
        with client.websocket_connect("/ws/rooms/ZZZZ?name=x") as ws:
            assert ws.receive_json()["type"] == "fatal"


async def test_primeira_rodada_espera_a_ia_esquentar(tmp_path):
    # nó falso com prefill lento, como o 3b em CPU (em escala menor)
    slow = create_fake_app(tokens=6, token_ms=1, prefill_ms=400)
    gateway = Gateway(make_config(tmp_path), transport=httpx.ASGITransport(app=slow))
    room = new_room(gateway)
    a, b = room.join("A"), room.join("B")
    room.connect(a), room.connect(b)
    room.start(a)
    await wait_for(lambda: room.preparing)
    assert room.state_for(a)["preparing"] is True
    with pytest.raises(GameError):
        room.post_chat(a, "oi")  # chat fechado enquanto a estação "liga"
    await wait_for(lambda: room.topic and not room.preparing)
    assert room.warmup_ms is not None
    room.post_chat(a, "agora vai")
    warm = [r for r in gateway.metrics.records if r.source == "impostor-warmup"]
    assert len(warm) == 1 and warm[0].output_tokens == 1 and warm[0].node_tps is None
    room.close()


async def test_sem_aquecimento_quando_desligado(gateway):
    room = new_room(gateway, warmup_timeout_s=0)
    a, b = room.join("A"), room.join("B")
    room.connect(a), room.connect(b)
    room.start(a)
    await wait_for(lambda: room.topic)
    assert not room.preparing and room.warmup_ms is None
    room.close()
