"""Os dois papéis de LLM no Jogo da Imitação.

- `ImpostorAI` (a IA impostora): um jogador que tenta se passar por humano. Decide
  quando falar, monta o prompt a partir da conversa, pede a resposta ao gateway
  (domínio `impostor`) e "humaniza" a entrega: espera um tempo de leitura,
  mostra "digitando..." e só publica depois de um tempo compatível com a
  digitação humana. A latência do gateway (fila + TTFT + geração) fica
  escondida dentro desse orçamento; quando o sistema satura, ela estoura o
  orçamento e a IA fica lenta demais, o que a denuncia.

- `Narrator` (o Mestre): gera a pergunta de cada rodada e comenta quem foi exilado
  (domínio `mestre`). O texto do Mestre é repassado aos jogadores token a token,
  ao vivo, como no /v1/chat.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
import unicodedata
from typing import TYPE_CHECKING, Any

from .config import GameConfig
from .humanize import (Style, build_messages, humanize, quick_reaction, repeats_phrase,
                       room_profile, too_similar, word_budget)
from .pipeline import Gateway
from .security import is_breach, is_disclosure

if TYPE_CHECKING:  # evita import circular
    from .game import Player, Room

log = logging.getLogger("agora.ai")

FALLBACK_TOPICS = [
    "Qual foi a última coisa que te fez rir de verdade?",
    "Se você pudesse comer uma única comida pro resto da vida, qual seria?",
    "Conta uma mania estranha que você tem.",
    "Qual o pior presente que você já ganhou?",
    "O que você estava fazendo ontem às 22h?",
    "Qual música não sai da sua cabeça essa semana?",
    "Qual foi o último sonho esquisito que você lembra?",
    "Qual app você mais usa no celular, tirando o WhatsApp?",
    "Qual cheiro te lembra a sua infância?",
    "Qual a coisa mais inútil que você sabe fazer?",
    "Qual foi a melhor coisa que você comeu essa semana?",
    "Qual matéria da faculdade você mais odeia, e por quê?",
    "Qual foi a maior vergonha que você já passou em público?",
    "Se você ganhasse mil reais agora, no que gastaria primeiro?",
]

# Respostas de emergência quando o modelo falha ou se entrega ("sou uma IA").
FALLBACK_LINES = [
    "kkkk q isso", "eu hein", "sei la mano", "nada a ver isso", "pdc kkk", "ah para",
    "to so observando", "hmm suspeito", "oxe", "fala serio", "kkkkk", "nem sei oq dizer",
]

# Identidades prontas, usadas só quando o modelo não consegue criar a própria.
FALLBACK_PERSONAS = [
    "Sou o Lucas, 23 anos, moro em Jardim Camburi e faço administração.",
    "Sou a Ana, 21, de Goiabeiras, estudo direito e corro na orla.",
    "Sou o Pedro, 25, da Praia do Canto, trabalho com suporte de TI.",
]

# Pedido que o modelo recebe uma vez, no início da partida, para criar a persona.
# Formato de frase única, para o stop de quebra de linha do domínio não cortar.
_PERSONA_PROMPT = (
    "Escreva uma única frase se apresentando, no formato: "
    "'Sou o [nome], [idade] anos, moro em [bairro], [o que faz] e gosto de [hobby].' "
    "Invente um nome e use um bairro da Grande Vitória (ES). Responda só essa frase."
)

def fold(text: str) -> str:
    """minúsculas e sem acentos (para achar menções como "laranja" / "LARANJA")."""
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def mentions(text: str, alias: str) -> bool:
    return re.search(rf"\b{re.escape(fold(alias))}\b", fold(text)) is not None


def clean_reply(text: str, aliases: list[str], max_chars: int, rng: random.Random) -> str | None:
    """Transforma a saída crua do modelo numa mensagem de chat crível.

    Devolve None se não sobrar nada aproveitável (ou se o modelo se entregou).
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    msg = lines[0]
    # Tira prefixos "Laranja:", "Laranja (você):", "Eu:", "- " e aspas.
    names = "|".join(re.escape(a) for a in aliases + ["eu", "você", "voce", "resposta"])
    msg = re.sub(rf"^\s*[-*>]*\s*\[?({names})\]?\s*(\((você|voce)\))?\s*:\s*", "", msg, flags=re.IGNORECASE)
    msg = msg.strip().strip('"“”\'`').strip()
    if not msg or is_breach(msg):
        return None
    if len(msg) > max_chars:
        cut = msg[:max_chars]
        msg = cut[: cut.rfind(" ")] if " " in cut else cut
    msg = msg.rstrip(".").rstrip()
    if msg and rng.random() < 0.75:
        msg = msg[0].lower() + msg[1:]
    return msg or None


class ImpostorAI:
    def __init__(self, room: "Room", player: "Player", gateway: Gateway | None, cfg: GameConfig,
                 rng: random.Random):
        self.room = room
        self.player = player
        self.gateway = gateway
        self.cfg = cfg
        self.rng = rng
        self.busy = False
        self.pending = False
        self.last_sent = 0.0
        self.unanswered = 0       # mensagens humanas desde a última fala da IA
        # Identidade criada pelo próprio modelo no início da partida.
        self.persona_text: str | None = None
        self.stats: list[dict[str, Any]] = []
        self._tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------ utilidades
    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def stop(self) -> None:
        for t in list(self._tasks):
            t.cancel()

    @property
    def active(self) -> bool:
        return self.player.alive and self.room.chat_open_for(self.player)

    # ------------------------------------------------------------ eventos do jogo
    def on_round_start(self, topic: str) -> None:
        self.unanswered = 0
        self._spawn(self._reply(trigger="rodada", extra_delay=self.rng.uniform(*self.cfg.read_delay_s)))
        self._spawn(self._idle_loop(self.room.round))

    def on_chat(self, author: "Player", text: str) -> None:
        if author is self.player or not self.active:
            return
        self.unanswered += 1
        called = mentions(text, self.player.alias or "")
        if self.busy:
            self.pending = self.pending or called
            return
        if called:
            p = 0.95
        elif "?" in text:
            p = 0.6
        elif self.unanswered >= 3:
            p = 0.55
        else:
            p = 0.25
        if self.room.phase.value == "votacao":
            p *= 0.6
        if self.rng.random() < p:
            self._spawn(self._reply(trigger="mencao" if called else "conversa", source_text=text))

    def on_ready(self) -> None:
        """Algum humano clicou em "pronto": a IA acompanha depois de um tempinho."""
        if self.player.ready or not self.player.alive:
            return

        async def later():
            await asyncio.sleep(self.rng.uniform(*self.cfg.read_delay_s))
            self.room.set_ready(self.player)

        self._spawn(later())

    def on_voting(self) -> None:
        self._spawn(self._vote())

    # ------------------------------------------------------------ falar
    async def _idle_loop(self, round_no: int) -> None:
        """Se a IA ficar muito tempo calada enquanto os outros falam, ela fala."""
        while self.room.round == round_no and self.active:
            await asyncio.sleep(5)
            quiet = time.monotonic() - self.last_sent
            if not self.busy and self.unanswered >= 1 and quiet > 25:
                await self._reply(trigger="silencio")

    def recent_texts(self, n: int = 8) -> list[str]:
        return [m["text"] for m in self.room.chat[-n:] if m.get("kind") == "player"]

    def own_texts(self) -> list[str]:
        """Tudo o que esta IA já disse na partida (para não se repetir)."""
        return [m["text"] for m in self.room.chat
                if m.get("kind") == "player" and m.get("from") == self.player.alias]

    def is_repeat(self, text: str) -> bool:
        own = self.own_texts()
        return (too_similar(text, self.recent_texts()) or too_similar(text, own, 0.45)
                or repeats_phrase(text, own))

    def build_messages(self, trigger: str) -> list[dict[str, str]]:
        me = self.player.alias or ""
        budget = word_budget(room_profile(self.room.chat, me), Style())
        return build_messages(self.room.chat, me, self.room.topic, trigger,
                              context=self.cfg.ai_context_messages, budget=budget)

    def variables(self) -> dict[str, str]:
        others = [p.alias for p in self.room.players_in_game() if p is not self.player]
        alias = self.player.alias or ""
        others = [o for o in others if o]
        persona = self.persona_text or self.rng.choice(FALLBACK_PERSONAS)
        return {"alias": alias, "alias_lower": alias.lower(), "others": ", ".join(others),
                "persona": persona}

    async def create_persona(self) -> None:
        """Pede ao modelo, uma vez, a identidade que ele vai interpretar."""
        if self.gateway is None:
            self.persona_text = self.rng.choice(FALLBACK_PERSONAS)
            return
        comp = await self.gateway.complete(
            self.cfg.impostor_domain,
            [{"role": "user", "content": _PERSONA_PROMPT}],
            system_prompt="Você é um jogador humano definindo quem é para um bate-papo.",
            source="persona", room=self.room.code, max_tokens=60, timeout_s=60,
            # sem o stop de quebra de linha do domínio (cortava a identidade no meio)
            extra_options={"stop": []},
        )
        text = " ".join(comp.text.strip().split())
        self.persona_text = text[:200] if text and not is_disclosure(text) else self.rng.choice(FALLBACK_PERSONAS)

    async def _generate(self, trigger: str, stat: dict[str, Any]) -> list[str]:
        """Pede a fala ao gateway; humaniza; se sair eco/robótica, tenta de novo uma vez."""
        if self.gateway is None:
            return [self.rng.choice(FALLBACK_LINES)]
        me = self.player.alias or ""
        aliases = [p.alias for p in self.room.players_in_game() if p.alias]
        budget = word_budget(room_profile(self.room.chat, me), Style())
        messages = self.build_messages(trigger)
        # Para assim que o modelo começar a "escrever pelos outros" ("Azul: ...").
        domain = self.cfg.impostor_domain
        dom = self.gateway.cfg.domains.get(domain)
        stops = list((dom.options.get("stop") if dom else None) or [])
        stops += [f"{a}:" for a in aliases if f"{a}:" not in stops]
        for attempt in range(2):
            comp = await self.gateway.complete(
                domain, messages, variables=self.variables(),
                source="impostor", room=self.room.code, timeout_s=90,
                extra_options={"stop": stops, **({"temperature": 1.05, "seed": self.rng.randrange(10**6)}
                                                 if attempt else {})},
            )
            stat["attempts"] = attempt + 1
            stat["error"] = comp.error
            stat["raw"] = comp.text[:200]
            if comp.record is not None:
                stat.update({"ttft_ms": comp.record.ttft_ms, "queue_wait_ms": comp.record.queue_wait_ms,
                             "e2e_ms": comp.record.e2e_ms, "tokens": comp.record.output_tokens,
                             "prompt_tokens": comp.record.prompt_tokens})
            if not comp.ok:
                # Failover: se a API remota falhou (limite de taxa, sem internet...),
                # tenta de novo no modelo local antes de apelar para uma frase pronta.
                fb = self.cfg.impostor_fallback_domain
                if attempt == 0 and fb and fb != domain and fb in self.gateway.cfg.domains:
                    log.warning("IA: %s falhou (%s); usando %s", domain, comp.error, fb)
                    stat["failover"] = {"from": domain, "to": fb, "error": comp.error}
                    domain = fb
                    continue
                break
            texts = humanize(comp.text, aliases, Style(), self.rng,
                             budget=budget, max_chars=self.cfg.max_message_chars)
            if texts and not self.is_repeat(" ".join(texts)):
                stat["_gen_done"] = time.monotonic()
                return texts
            stat["rejected"] = stat.get("rejected", 0) + 1
        stat["fallback"] = True
        stat["_gen_done"] = time.monotonic()
        return [self.rng.choice(FALLBACK_LINES)]

    async def _reply(self, trigger: str, extra_delay: float = 0.0, source_text: str | None = None) -> None:
        if self.busy or not self.active:
            return
        self.busy = True
        t_trigger = time.monotonic()
        stat: dict[str, Any] = {"round": self.room.round, "trigger": trigger}
        gen: asyncio.Task | None = None
        typing: asyncio.Task | None = None
        try:
            cooldown = max(0.0, self.cfg.ai_cooldown_s - (t_trigger - self.last_sent))
            if cooldown:
                await asyncio.sleep(cooldown)

            # Reação rápida, sem LLM ("kkkk", "eita"): humanos fazem muito isso.
            quick = None
            if trigger == "conversa" and source_text and self.rng.random() < 0.3:
                quick = quick_reaction(source_text, Style(), self.rng)
                if quick and quick in self.own_texts()[-3:]:
                    quick = None  # não manda o mesmo "kkkk" três vezes seguidas
            t_gen = time.monotonic()
            if quick:
                stat["quick"] = True
            else:
                # A geração começa JÁ, em paralelo com o tempo de "leitura": a latência
                # do gateway fica escondida dentro do tempo que um humano levaria.
                gen = asyncio.create_task(self._generate(trigger, stat))

            read_s = self.rng.uniform(*self.cfg.read_delay_s) + extra_delay
            await asyncio.sleep(read_s)
            if not self.active:
                return
            typing = asyncio.create_task(self._typing_loop())
            t_typing = time.monotonic()
            texts = [quick] if quick else await gen
            # duração da geração em si (fila + TTFT + tokens + tentativas), não da leitura
            gen_s = (stat.get("_gen_done", t_gen) - t_gen) if not quick else 0.0
            stat.pop("_gen_done", None)

            sent = 0
            budget_s = read_s
            for i, text in enumerate(texts):
                typing_s = len(text) / self.rng.uniform(*self.cfg.typing_cps)
                budget_s += typing_s
                start = t_typing if i == 0 else time.monotonic()
                await asyncio.sleep(max(0.0, typing_s - (time.monotonic() - start)))
                if not self.active:
                    break
                self.room.post_chat(self.player, text)
                sent += 1
                if i + 1 < len(texts):
                    await asyncio.sleep(self.rng.uniform(0.4, 1.2))
            if not sent:
                return
            self.last_sent = time.monotonic()
            self.unanswered = 0
            stat.update({
                "text": " / ".join(texts[:sent]),
                "chars": sum(len(t) for t in texts[:sent]),
                "parts": sent,
                "gen_ms": round(gen_s * 1000, 1),
                "read_ms": round(read_s * 1000, 1),
                "typing_target_ms": round((budget_s - read_s) * 1000, 1),
                # Atraso que o orçamento humano (leitura + digitação) não conseguiu esconder:
                "overrun_ms": round(max(0.0, gen_s - budget_s) * 1000, 1),
                "humanize_wait_ms": round(max(0.0, budget_s - gen_s) * 1000, 1),
                "response_ms": round((self.last_sent - t_trigger) * 1000, 1),
            })
            self.stats.append(stat)
        except asyncio.CancelledError:
            if gen:
                gen.cancel()
            raise
        except Exception:  # nunca derruba a partida por causa da IA
            log.exception("falha na resposta da IA")
        finally:
            if typing:
                typing.cancel()
            if gen and not gen.done():
                gen.cancel()
            self.busy = False
            if self.pending and self.active:
                self.pending = False
                self._spawn(self._reply(trigger="mencao"))

    async def _typing_loop(self) -> None:
        while True:
            self.room.broadcast_typing(self.player)
            await asyncio.sleep(2.5)

    # ------------------------------------------------------------ votar
    def choose_vote(self) -> str | None:
        """Escolhe em quem votar: quem acusou a IA pesa mais; segue a maioria."""
        me = self.player.alias or ""
        humans = [p for p in self.room.players_in_game() if p.alive and not p.is_ai]
        if not humans:
            return None
        weights = {p.alias: 1.0 for p in humans}
        for m in self.room.chat:
            if m["kind"] == "player" and m.get("round") == self.room.round and m["from"] in weights:
                if mentions(m["text"], me):
                    weights[m["from"]] += 2.0
        for target in self.room.votes.values():
            if target in weights:
                weights[target] += 1.5
        if max(weights.values()) == 1.0 and self.rng.random() < 0.15:
            return None  # ninguém se destacou: às vezes pula, como um humano indeciso
        names = list(weights)
        return self.rng.choices(names, weights=[weights[n] for n in names])[0]

    async def _vote(self) -> None:
        await asyncio.sleep(min(15.0, self.cfg.vote_s * self.rng.uniform(0.15, 0.5)))
        if self.player.alive and self.room.phase.value == "votacao":
            self.room.cast_vote(self.player, self.choose_vote())


class Narrator:
    """O Mestre: pergunta da rodada e comentário de quem foi exilado, ao vivo."""

    def __init__(self, room: "Room", gateway: Gateway | None, cfg: GameConfig, rng: random.Random):
        self.room = room
        self.gateway = gateway
        self.cfg = cfg
        self.rng = rng
        self.used: list[str] = []
        self.stats: list[dict[str, Any]] = []

    async def _say(self, prompt: str, fallback: str, max_chars: int = 240) -> str:
        msg_id = self.room.narrator_begin()
        text = ""
        if self.gateway is not None and self.cfg.master_domain:
            comp = await self.gateway.complete(
                self.cfg.master_domain, [{"role": "user", "content": prompt}],
                on_token=lambda t: self.room.narrator_delta(msg_id, t),
                source="mestre", room=self.room.code, timeout_s=self.cfg.master_timeout_s,
            )
            if comp.record is not None:
                self.stats.append({"ttft_ms": comp.record.ttft_ms, "e2e_ms": comp.record.e2e_ms,
                                   "tokens": comp.record.output_tokens, "error": comp.error})
            lines = [ln.strip() for ln in comp.text.strip().splitlines() if ln.strip()]
            text = " ".join(lines[:2]).strip().strip('"“”').strip()
            if len(text) > max_chars:
                text = text[:max_chars].rsplit(" ", 1)[0] + "…"
            if is_breach(text):
                text = ""
        final = text or fallback
        self.room.narrator_end(msg_id, final)
        return final

    async def topic(self, round_no: int) -> str:
        pool = [t for t in FALLBACK_TOPICS if t not in self.used] or FALLBACK_TOPICS
        fallback = self.rng.choice(pool)
        avoid = "; ".join(self.used[-4:]) or "nenhuma"
        prompt = (
            f"Rodada {round_no}. Escreva UMA pergunta curta e pessoal para os jogadores "
            f"responderem no chat, do tipo que um humano responde com uma história real e uma "
            f"IA tem dificuldade (gostos, lembranças, manias, o dia de hoje). "
            f"Não repita estas: {avoid}. Responda só com a pergunta."
        )
        text = await self._say(prompt, fallback, max_chars=200)
        self.used.append(text)
        return text

    async def ejection(self, alias: str | None, remaining: int) -> str:
        if alias is None:
            fallback = "Ninguém foi exilado nesta rodada. A desconfiança continua."
            prompt = ("A votação empatou e ninguém foi exilado. Comente isso em UMA frase curta e "
                      "leve. Não revele quem é a IA.")
        else:
            fallback = f"{alias} deixa a roda. Restam {remaining}."
            prompt = (f"O jogador {alias} foi exilado da roda por votação. "
                      f"Restam {remaining} jogadores. Comente isso em UMA frase curta e leve. "
                      f"NÃO diga se {alias} era humano ou a IA.")
        return await self._say(prompt, fallback)
