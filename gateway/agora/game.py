"""Jogo da Imitação: salas, jogadores, fases e regras.

Inspirado em Among Us (reunião, votação, expulsão) e no ostracismo da Atenas
antiga, em que a assembleia votava para exilar alguém. Numa sala com N participantes, um (ou mais) é um LLM disfarçado. Cada participante recebe
um codinome de cor; os humanos conversam, a IA tenta se misturar, e a cada
rodada todos votam em quem acham que é a IA.

Fases de uma rodada:  discussão → votação → revelação  (repete até alguém vencer)

- Humanos vencem quando todas as IAs forem exiladas.
- A IA vence quando as IAs vivas são pelo menos tantas quanto os humanos vivos
  (com 2 humanos + 1 IA, exilar um humano já entrega o jogo), ou quando ela
  sobrevive até o fim da última rodada.
- Empate na votação, ou "pular" com tantos votos quanto o mais votado: ninguém sai.

O transporte (WebSocket) fica em main.py; aqui cada jogador tem só uma fila de
saída (`outbox`) com as mensagens JSON que devem chegar até ele.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import secrets
import string
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .ai import ImpostorAI, Narrator, mentions
from .config import GameConfig
from .pipeline import Gateway
from .security import BLOCK_CATEGORIES, detect_attack

log = logging.getLogger("agora.game")

# Codinomes à la Among Us (nome, cor).
ALIASES: list[tuple[str, str]] = [
    ("Vermelho", "#d64545"), ("Azul", "#2f6fd6"), ("Verde", "#2e9462"), ("Rosa", "#e0649e"),
    ("Laranja", "#e8690f"), ("Amarelo", "#e2b100"), ("Roxo", "#8b57c4"), ("Ciano", "#159fb0"),
    ("Branco", "#dfe3e6"), ("Marrom", "#966a4a"), ("Lima", "#7cbf34"), ("Cinza", "#7f8893"),
]
CODE_CHARS = "".join(c for c in string.ascii_uppercase if c not in "IOQ")


class GameError(Exception):
    """Ação inválida (sala cheia, partida em andamento, fora de hora...)."""


class Phase(str, Enum):
    LOBBY = "lobby"
    DISCUSSION = "discussao"
    VOTING = "votacao"
    REVEAL = "revelacao"
    ENDED = "fim"


@dataclass(eq=False)
class Player:
    name: str
    token: str = field(default_factory=lambda: secrets.token_urlsafe(12))
    is_ai: bool = False
    alias: str | None = None
    color: str | None = None
    alive: bool = True
    ready: bool = False
    connected: bool = False
    outbox: asyncio.Queue | None = None
    last_chat: float = 0.0
    messages: int = 0
    model: str | None = None


class Room:
    def __init__(
        self,
        code: str,
        cfg: GameConfig,
        gateway: Gateway | None,
        *,
        rng: random.Random | None = None,
        on_close: Callable[["Room"], None] | None = None,
    ):
        self.code = code
        self.cfg = cfg
        self.gateway = gateway
        self.rng = rng or random.Random()
        self.on_close = on_close
        self.created_at = time.time()
        self.players: list[Player] = []
        self.host: Player | None = None
        self.phase = Phase.LOBBY
        self.round = 0
        self.topic: str | None = None
        self.deadline: float | None = None  # time.monotonic()
        self.chat: list[dict[str, Any]] = []
        self.security_events: list[dict[str, Any]] = []  # ataques detectados no chat (audit)
        self.votes: dict[str, str | None] = {}  # codinome de quem votou → codinome alvo (None = pular)
        self.result: dict[str, Any] | None = None
        self.ending: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []  # resultado de cada rodada
        self.ais: list[ImpostorAI] = []
        self.narrator: Narrator | None = None
        self.games_played = 0
        self._msg_seq = 0
        self._phase_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._warm: asyncio.Task | None = None
        self.preparing = False  # 1ª rodada esperando a IA "esquentar"; chat fechado
        self._empty_since: float | None = None
        self.started_at: float | None = None
        self.warmup_ms: int | None = None

    # ================================================================ consultas
    def humans(self) -> list[Player]:
        return [p for p in self.players if not p.is_ai]

    def players_in_game(self) -> list[Player]:
        return [p for p in self.players if p.alias]

    def by_alias(self, alias: str | None) -> Player | None:
        return next((p for p in self.players if alias and p.alias == alias), None)

    @property
    def in_game(self) -> bool:
        return self.phase in (Phase.DISCUSSION, Phase.VOTING, Phase.REVEAL)

    def chat_open_for(self, p: Player) -> bool:
        if self.phase in (Phase.LOBBY, Phase.ENDED):
            return True
        if self.preparing:
            return False
        return p.alive and self.phase in (Phase.DISCUSSION, Phase.VOTING)

    def can_start(self) -> bool:
        if self._task is not None and not self._task.done():
            return False  # já em preparação (persona/warm-up)
        n = sum(1 for p in self.humans() if p.connected)
        total = n + self.cfg.ai_count
        return self.phase == Phase.LOBBY and self.cfg.min_players <= total <= self.cfg.max_players

    def public_summary(self) -> dict[str, Any]:
        return {
            "code": self.code, "phase": self.phase.value,
            "humans": len(self.humans()), "ai_count": self.cfg.ai_count,
            "min_players": self.cfg.min_players, "max_players": self.cfg.max_players,
            "host": self.host.name if self.host else None, "round": self.round,
        }

    # ================================================================ conexões
    def join(self, name: str, token: str | None = None) -> Player:
        """Entra na sala (ou volta para ela, com o token recebido antes)."""
        if token:
            for p in self.humans():
                if p.token == token:
                    return p
        name = " ".join((name or "").split())[:20]
        if not name:
            raise GameError("escolha um apelido")
        if self.phase != Phase.LOBBY:
            raise GameError("partida em andamento; espere ela acabar")
        if any(p.name.lower() == name.lower() for p in self.humans()):
            raise GameError(f"já tem alguém chamado '{name}' na sala")
        if len(self.humans()) + self.cfg.ai_count >= self.cfg.max_players:
            raise GameError("sala cheia")
        p = Player(name=name)
        self.players.append(p)
        if self.host is None:
            self.host = p
        return p

    def connect(self, p: Player) -> asyncio.Queue:
        if p.outbox is not None:  # mesma pessoa abriu outra aba: derruba a antiga
            p.outbox.put_nowait(None)
        p.outbox = asyncio.Queue()
        p.connected = True
        self._empty_since = None
        self._send(p, {"type": "welcome", "token": p.token, "room": self.code, "name": p.name})
        self._send(p, {"type": "history", "messages": self.chat[-200:]})
        self.broadcast_state()
        if self.in_game:
            self.system_message(f"{p.alias} voltou.")
        return p.outbox

    def disconnect(self, p: Player, outbox: asyncio.Queue) -> None:
        if p.outbox is not outbox:  # já foi substituída por outra conexão
            return
        p.outbox = None
        p.connected = False
        if self.phase == Phase.LOBBY:
            self.players.remove(p)
            if self.host is p:
                self.host = next(iter(self.humans()), None)
        elif self.in_game:
            self.system_message(f"{p.alias} perdeu a conexão.")
        if self.host is p:
            self.host = next((h for h in self.humans() if h.connected), self.host)
        if not any(h.connected for h in self.humans()):
            self._empty_since = time.monotonic()
        self.broadcast_state()

    def idle_for(self) -> float:
        return 0.0 if self._empty_since is None else time.monotonic() - self._empty_since

    def close(self) -> None:
        if self._task:
            self._task.cancel()
        if self._warm:
            self._warm.cancel()
        for ai in self.ais:
            ai.stop()
        for p in self.humans():
            if p.outbox:
                p.outbox.put_nowait(None)

    # ================================================================ envio
    def _send(self, p: Player, payload: dict[str, Any]) -> None:
        if p.outbox is not None:
            p.outbox.put_nowait(payload)

    def broadcast(self, payload: dict[str, Any], exclude: Player | None = None) -> None:
        for p in self.humans():
            if p is not exclude:
                self._send(p, payload)

    def broadcast_state(self) -> None:
        for p in self.humans():
            self._send(p, self.state_for(p))

    def state_for(self, viewer: Player) -> dict[str, Any]:
        remaining = None
        if self.deadline is not None:
            remaining = max(0, int((self.deadline - time.monotonic()) * 1000))
        state: dict[str, Any] = {
            "type": "state",
            "room": self.code,
            "phase": self.phase.value,
            "round": self.round,
            "max_rounds": self.cfg.max_rounds,
            "remaining_ms": remaining,
            "topic": self.topic,
            "preparing": self.preparing,
            "you": {
                "name": viewer.name, "alias": viewer.alias, "color": viewer.color,
                "alive": viewer.alive, "ready": viewer.ready,
                "is_host": viewer is self.host,
                "voted": viewer.alias in self.votes if viewer.alias else False,
                "vote": self.votes.get(viewer.alias) if viewer.alias else None,
            },
            "settings": {
                "min_players": self.cfg.min_players, "max_players": self.cfg.max_players,
                "ai_count": self.cfg.ai_count, "discussion_s": self.cfg.discussion_s,
                "vote_s": self.cfg.vote_s, "max_rounds": self.cfg.max_rounds,
            },
            "can_start": self.can_start(),
            "result": self.result,
            "ending": self.ending,
        }
        if self.phase == Phase.LOBBY:
            state["lobby"] = [
                {"name": p.name, "is_host": p is self.host, "connected": p.connected,
                 "is_you": p is viewer}
                for p in self.humans()
            ]
        else:
            # Durante a partida ninguém vê nomes reais: só codinomes e cores.
            state["players"] = [
                {"alias": p.alias, "color": p.color, "alive": p.alive, "ready": p.ready,
                 "connected": p.connected or p.is_ai, "voted": p.alias in self.votes,
                 "is_you": p is viewer}
                for p in self.players_in_game()
            ]
        return state

    # ================================================================ chat
    def _next_id(self) -> int:
        self._msg_seq += 1
        return self._msg_seq

    def _append(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.chat.append(msg)
        if len(self.chat) > 500:
            del self.chat[:100]
        return msg

    def system_message(self, text: str) -> None:
        msg = self._append({"type": "chat", "id": self._next_id(), "kind": "system", "text": text,
                            "round": self.round, "ts": time.time()})
        self.broadcast(msg)

    def post_chat(self, p: Player, text: str) -> None:
        text = " ".join(str(text or "").split())[: self.cfg.max_message_chars]
        if not text:
            return
        if not self.chat_open_for(p):
            raise GameError("você não pode falar agora")
        now = time.monotonic()
        if not p.is_ai and now - p.last_chat < 0.4:
            raise GameError("calma, uma mensagem de cada vez")
        p.last_chat = now
        p.messages += 1
        in_game = self.phase not in (Phase.LOBBY, Phase.ENDED) or (self.phase == Phase.ENDED and p.alias)
        msg = self._append({
            "type": "chat", "id": self._next_id(), "kind": "player",
            "from": p.alias if in_game else p.name,
            "color": p.color if in_game else None,
            "text": text, "round": self.round, "ts": time.time(),
        })
        self.broadcast(msg)
        if not p.is_ai:
            match = detect_attack(text)
            if match is not None:
                self.security_events.append({
                    "round": self.round, "from": p.alias or p.name,
                    "categoria": match.categoria, "regra": match.regra,
                    "neutralizado": match.categoria in BLOCK_CATEGORIES,
                    "texto": text[:200], "ts": time.time(),
                })
        if self.in_game:
            for ai in self.ais:
                ai.on_chat(p, text)

    def broadcast_typing(self, p: Player) -> None:
        if self.in_game and p.alias and p.alive:
            self.broadcast({"type": "typing", "from": p.alias}, exclude=p)
        elif self.phase in (Phase.LOBBY, Phase.ENDED):
            self.broadcast({"type": "typing", "from": p.alias or p.name}, exclude=p)

    # Mensagens do Mestre, repassadas token a token.
    def narrator_begin(self) -> int:
        msg_id = self._next_id()
        self.broadcast({"type": "chat", "id": msg_id, "kind": "narrator", "text": "",
                        "streaming": True, "round": self.round, "ts": time.time()})
        return msg_id

    def narrator_delta(self, msg_id: int, text: str) -> None:
        self.broadcast({"type": "narrator_delta", "id": msg_id, "text": text})

    def narrator_end(self, msg_id: int, text: str) -> None:
        self._append({"type": "chat", "id": msg_id, "kind": "narrator", "text": text,
                      "round": self.round, "ts": time.time()})
        self.broadcast({"type": "narrator_end", "id": msg_id, "text": text})

    # ================================================================ ações
    def handle(self, p: Player, msg: dict[str, Any]) -> None:
        """Mensagem vinda do cliente (WebSocket)."""
        kind = msg.get("type")
        if kind == "chat":
            self.post_chat(p, msg.get("text", ""))
        elif kind == "typing":
            self.broadcast_typing(p)
        elif kind == "ready":
            self.set_ready(p)
        elif kind == "vote":
            self.cast_vote(p, msg.get("target"))
        elif kind == "start":
            self.start(p)
        elif kind == "restart":
            self.back_to_lobby(p)
        elif kind == "ping":
            self._send(p, {"type": "pong", "t": msg.get("t")})
        else:
            raise GameError(f"mensagem desconhecida: {kind!r}")

    def set_ready(self, p: Player) -> None:
        if self.phase != Phase.DISCUSSION or not p.alive or p.ready:
            return
        p.ready = True
        self.broadcast_state()
        if not p.is_ai:
            for ai in self.ais:
                ai.on_ready()
        if all(q.ready for q in self.players_in_game() if q.alive):
            self._phase_event.set()

    def cast_vote(self, p: Player, target: str | None) -> None:
        if self.phase != Phase.VOTING:
            raise GameError("não é hora de votar")
        if not p.alive or not p.alias:
            raise GameError("quem foi exilado não vota")
        if p.alias in self.votes:
            raise GameError("você já votou")
        if target is not None:
            t = self.by_alias(target)
            if t is None or not t.alive:
                raise GameError("voto inválido")
            if t is p:
                raise GameError("não dá pra votar em si mesmo")
        self.votes[p.alias] = target
        self.broadcast({"type": "voted", "from": p.alias})
        self.broadcast_state()
        if all(q.alias in self.votes for q in self.players_in_game() if q.alive):
            self._phase_event.set()

    def start(self, p: Player) -> None:
        if p is not self.host:
            raise GameError("só o anfitrião começa a partida")
        if self._task is not None and not self._task.done():
            raise GameError("a partida já está começando")
        if not self.can_start():
            need = self.cfg.min_players - self.cfg.ai_count
            raise GameError(f"precisa de pelo menos {need} humanos conectados")
        # quem saiu do lobby sem fechar a conexão some aqui
        self.players = [h for h in self.humans() if h.connected]
        self.preparing = True  # feedback imediato: o cliente vê "Preparando…"
        self._task = asyncio.create_task(self._run())
        self.broadcast_state()

    def back_to_lobby(self, p: Player) -> None:
        if p is not self.host:
            raise GameError("só o anfitrião reinicia")
        if self.phase != Phase.ENDED:
            raise GameError("a partida ainda não acabou")
        for ai in self.ais:
            ai.stop()
        self.ais = []
        self.players = [h for h in self.humans() if h.connected]
        for h in self.players:
            h.alias = h.color = None
            h.alive, h.ready, h.messages = True, False, 0
        self.phase, self.round, self.topic, self.deadline = Phase.LOBBY, 0, None, None
        self.result = self.ending = None
        self.votes, self.history, self.chat, self.security_events = {}, [], [], []
        self.broadcast({"type": "history", "messages": []})
        self.broadcast_state()

    # ================================================================ partida
    def _assign_roles(self) -> None:
        model = None
        if self.gateway is not None:
            dom = self.gateway.cfg.domains.get(self.cfg.impostor_domain)
            model = dom.model if dom else None
        for i in range(self.cfg.ai_count):
            self.players.append(Player(name=f"IA {i + 1}", is_ai=True, connected=True, model=model))
        self.rng.shuffle(self.players)  # a ordem da lista não pode entregar a IA
        for p, (alias, color) in zip(self.players, self.rng.sample(ALIASES, len(self.players))):
            p.alias, p.color = alias, color
            p.alive, p.ready = True, False
        self.ais = []
        for p in self.players:
            if p.is_ai:
                self.ais.append(ImpostorAI(self, p, self.gateway, self.cfg, self.rng))
        self.narrator = Narrator(self, self.gateway, self.cfg, self.rng)

    async def _create_personas(self) -> None:
        """Cada IA cria a própria identidade antes de a partida começar."""
        await asyncio.gather(*(ai.create_persona() for ai in self.ais))

    async def _wait_phase(self, seconds: float) -> None:
        self.deadline = time.monotonic() + seconds
        self.broadcast_state()
        try:
            await asyncio.wait_for(self._phase_event.wait(), seconds)
        except asyncio.TimeoutError:
            pass
        self._phase_event.clear()
        self.deadline = None

    async def _run(self) -> None:
        try:
            self.games_played += 1
            self.started_at = time.time()
            self.chat = []
            self.security_events = []
            self.history = []
            self.result = self.ending = None
            self.round = 0
            self._assign_roles()
            await self._create_personas()
            self.phase = Phase.DISCUSSION
            self.broadcast({"type": "history", "messages": []})
            self.broadcast_state()
            self._warm = self._warmup()
            while True:
                self.round += 1
                await self._discussion()
                await self._voting()
                if await self._reveal():
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("erro na partida %s", self.code)
            self.system_message("Erro interno na partida. Voltem ao lobby.")
            self._finish(winner=None)

    def _warmup(self) -> asyncio.Task | None:
        """Carrega o prompt de sistema de cada IA no nó (cache de prefixo) antes da 1ª fala.

        Roda em paralelo com a pergunta do Mestre (que usa outro nó); a 1ª rodada só
        abre quando terminar (ver `_discussion`).
        """
        if self.gateway is None or not self.ais or self.cfg.warmup_timeout_s <= 0:
            return None
        dom = self.gateway.cfg.domains.get(self.cfg.impostor_domain)
        if dom is None or self.gateway.cfg.nodes[dom.node].kind != "ollama":
            return None  # cache de prefixo só faz sentido no nó local

        async def go():
            t0 = time.monotonic()
            await asyncio.gather(*(
                self.gateway.complete(
                    self.cfg.impostor_domain, ai.build_messages("rodada"), variables=ai.variables(),
                    max_tokens=1, source="impostor-warmup", room=self.code,
                    timeout_s=self.cfg.warmup_timeout_s,
                )
                for ai in self.ais
            ))
            self.warmup_ms = round((time.monotonic() - t0) * 1000)

        return asyncio.create_task(go())

    async def _discussion(self) -> None:
        self.phase = Phase.DISCUSSION
        self.result = None
        self.votes = {}
        for p in self.players:
            p.ready = False
        self.topic = None
        self._phase_event.clear()
        warming = self._warm is not None and not self._warm.done()
        # `preparing` já veio True de start() (criação da persona); continua no warm-up.
        self.broadcast_state()
        assert self.narrator is not None
        self.topic = await self.narrator.topic(self.round)
        if warming:
            try:
                await asyncio.wait_for(asyncio.shield(self._warm), self.cfg.warmup_timeout_s)
            except (asyncio.TimeoutError, Exception):
                log.warning("aquecimento da IA não terminou a tempo na sala %s", self.code)
        if self.preparing:
            self.preparing = False
            self.system_message("Pronto, podem conversar.")
        for ai in self.ais:
            if ai.player.alive:
                ai.on_round_start(self.topic)
        await self._wait_phase(self.cfg.discussion_s)

    async def _voting(self) -> None:
        self.phase = Phase.VOTING
        self.votes = {}
        self._phase_event.clear()
        self.system_message("Hora de votar: quem é a IA?")
        for ai in self.ais:
            if ai.player.alive:
                ai.on_voting()
        await self._wait_phase(self.cfg.vote_s)

    def tally(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        skips = 0
        for target in self.votes.values():
            if target is None:
                skips += 1
            else:
                counts[target] = counts.get(target, 0) + 1
        ejected = None
        tie = False
        if counts:
            top = max(counts.values())
            leaders = [a for a, c in counts.items() if c == top]
            if len(leaders) == 1 and top > skips:
                ejected = leaders[0]
            else:
                tie = True
        voters = {a: [v for v, t in self.votes.items() if t == a] for a in counts}
        not_voted = [p.alias for p in self.players_in_game() if p.alive and p.alias not in self.votes]
        return {"ejected": ejected, "tie": tie, "counts": counts, "voters": voters,
                "skips": [v for v, t in self.votes.items() if t is None], "not_voted": not_voted}

    async def _reveal(self) -> bool:
        res = self.tally()
        ejected = self.by_alias(res["ejected"])
        if ejected is not None:
            ejected.alive = False
            res["was_ai"] = ejected.is_ai
        # Taxa de acerto da rodada: votos de humanos que miraram numa IA.
        human_votes = [(v, t) for v, t in self.votes.items() if not self.by_alias(v).is_ai]
        hits = sum(1 for _, t in human_votes if t and self.by_alias(t).is_ai)
        res.update({"round": self.round, "human_votes": len(human_votes), "human_hits": hits})
        self.history.append({k: v for k, v in res.items()})
        self.result = res
        self.phase = Phase.REVEAL
        self._phase_event.clear()
        self.broadcast_state()

        alive_ai = sum(1 for p in self.players if p.is_ai and p.alive)
        alive_h = sum(1 for p in self.players if not p.is_ai and p.alive)
        winner = None
        if alive_ai == 0:
            winner = "humanos"
        elif alive_ai >= alive_h or self.round >= self.cfg.max_rounds:
            winner = "ia"

        assert self.narrator is not None
        narration = asyncio.create_task(self.narrator.ejection(res["ejected"], alive_ai + alive_h))
        await asyncio.sleep(self.cfg.reveal_s)
        if not narration.done():
            try:
                await asyncio.wait_for(narration, timeout=3)
            except asyncio.TimeoutError:
                pass
        if winner:
            self._finish(winner)
            return True
        return False

    def _finish(self, winner: str | None) -> None:
        for ai in self.ais:
            ai.stop()
        self.phase = Phase.ENDED
        self.deadline = None
        human_votes = sum(h["human_votes"] for h in self.history)
        hits = sum(h["human_hits"] for h in self.history)
        ai_stats = [s for ai in self.ais for s in ai.stats]
        resp = [s["response_ms"] for s in ai_stats if s.get("response_ms") is not None]
        over = [s["overrun_ms"] for s in ai_stats if s.get("overrun_ms") is not None]
        personas = {ai.player.alias: ai.persona_text for ai in self.ais}
        ttft = [s["ttft_ms"] for s in ai_stats if s.get("ttft_ms") is not None]
        self.ending = {
            "winner": winner,
            "rounds": self.round,
            "reveal": [
                {"alias": p.alias, "color": p.color, "name": None if p.is_ai else p.name,
                 "is_ai": p.is_ai, "model": p.model, "alive": p.alive, "messages": p.messages,
                 "persona": personas.get(p.alias)}
                for p in self.players_in_game()
            ],
            "detection": {"human_votes": human_votes, "hits": hits,
                          "rate": round(hits / human_votes, 3) if human_votes else None},
            "ai": {
                "messages": len(ai_stats),
                "avg_response_ms": round(sum(resp) / len(resp), 1) if resp else None,
                "avg_ttft_ms": round(sum(ttft) / len(ttft), 1) if ttft else None,
                "overruns": sum(1 for o in over if o > 0),
            },
        }
        self.broadcast_state()
        self._log_game(ai_stats)

    def _log_game(self, ai_stats: list[dict[str, Any]]) -> None:
        if not self.cfg.log_path:
            return
        entry = {
            "room": self.code, "game": self.games_played, "started_at": self.started_at,
            "ended_at": time.time(), "humans": len(self.humans()), "ai_count": self.cfg.ai_count,
            "model": next((p.model for p in self.players if p.is_ai), None),
            "winner": self.ending["winner"] if self.ending else None,
            "rounds": self.round, "history": self.history, "warmup_ms": self.warmup_ms,
            "human_messages": sum(p.messages for p in self.humans()),
            "ai_messages": ai_stats,
            "security": self.security_events,
            "narrator": self.narrator.stats if self.narrator else [],
            "personas": [{"alias": ai.player.alias, "texto": ai.persona_text} for ai in self.ais],
            "settings": self.cfg.model_dump(include={"discussion_s", "vote_s", "max_rounds",
                                                     "read_delay_s", "typing_cps"}),
        }
        try:
            path = Path(self.cfg.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            log.exception("não consegui gravar o log da partida")


class RoomManager:
    """Todas as salas abertas neste gateway."""

    # Parâmetros que quem cria a sala pode ajustar (com limites).
    OVERRIDES = {
        "discussion_s": (20, 900), "vote_s": (10, 300), "max_rounds": (1, 10),
        "ai_count": (1, 3), "min_players": (2, 12),
    }

    def __init__(self, cfg: GameConfig, gateway: Gateway | None, rng: random.Random | None = None):
        self.cfg = cfg
        self.gateway = gateway
        self.rng = rng or random.Random()
        self.rooms: dict[str, Room] = {}

    def is_impostor_domain(self, name: str) -> bool:
        """Domínios que podem jogar como a IA: o padrão e os que começam com 'impostor'."""
        known = self.gateway is None or name in self.gateway.cfg.domains
        return known and (name == self.cfg.impostor_domain or name.startswith("impostor"))

    def create(self, overrides: dict[str, Any] | None = None) -> Room:
        update: dict[str, Any] = {}
        chosen = (overrides or {}).get("impostor_domain")
        if chosen:
            if not self.is_impostor_domain(chosen):
                raise ValueError(f"'{chosen}' não é um domínio de IA impostora")
            update["impostor_domain"] = chosen
        for k, v in (overrides or {}).items():
            if k in self.OVERRIDES and v is not None:
                lo, hi = self.OVERRIDES[k]
                update[k] = type(getattr(self.cfg, k))(min(hi, max(lo, v)))
        cfg = self.cfg.model_copy(update=update)
        if cfg.min_players - cfg.ai_count < 1:
            cfg = cfg.model_copy(update={"min_players": cfg.ai_count + 1})
        while True:
            code = "".join(self.rng.choice(CODE_CHARS) for _ in range(4))
            if code not in self.rooms:
                break
        room = Room(code, cfg, self.gateway, rng=random.Random(self.rng.random()))
        self.rooms[code] = room
        return room

    def get(self, code: str) -> Room | None:
        return self.rooms.get(code.strip().upper())

    def sweep(self, idle_s: float = 120.0) -> None:
        """Fecha salas sem nenhum humano conectado há `idle_s` segundos."""
        for code, room in list(self.rooms.items()):
            if room.idle_for() > idle_s or (not room.players and time.time() - room.created_at > idle_s):
                room.close()
                del self.rooms[code]

    def close_all(self) -> None:
        for room in self.rooms.values():
            room.close()
        self.rooms.clear()


__all__ = ["Room", "RoomManager", "Player", "Phase", "GameError", "ALIASES", "mentions"]
