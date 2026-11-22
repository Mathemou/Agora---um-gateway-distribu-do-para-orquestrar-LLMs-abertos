"""Trilha Desempenho: balanceamento entre nós, saúde dos nós e prioridade na fila.

Um domínio pode ser servido por vários nós (o "pool", em `DomainConfig.nodes`).
A cada requisição o `Balancer` ordena os nós do pool pela política escolhida e o
gateway tenta, nessa ordem, reservar lugar na fila de um deles:

- round_robin    um de cada vez, sem olhar o estado dos nós (referência);
- least_queue    o nó menos ocupado: (atendendo + esperando) / vagas;
- least_latency  o menor TTFT de serviço observado (média móvel exponencial por nó,
                 medido da vaga concedida até o 1º token, sem a fila do gateway).
                 Nó que ainda não tem amostra é tentado primeiro (exploração).
                 Sozinha, essa política "faz manada": manda tudo para o nó mais
                 rápido até ele lotar, e um nó que parou de receber tráfego nunca
                 tem a média atualizada.
- peak_ewma      latência × carga: média do TTFT × (atendendo + esperando + 1) / vagas,
                 uma estimativa do TTFT que a requisição terá ALI (fila + serviço).
                 É a ideia do "peak EWMA" do Finagle/Linkerd. Nó ainda sem amostra
                 entra com a média dos outros.

Saúde: um nó sai do pool depois de `fail_threshold` falhas seguidas (de requisições,
ou do health check ativo) e volta quando o health check responder de novo. Sem
health check ativo (nós remotos, ou `health_interval_s: 0`), o nó volta a ser
tentado depois de `down_cooldown_s` ("meio-aberto", como um circuit breaker).
Se todos os nós do pool estiverem fora, o gateway tenta mesmo assim (melhor
arriscar do que recusar tudo por causa de um health check atrasado).

Analogia de redes: é o mesmo problema de um balanceador L7 (nginx/HAProxy) ou
do ECMP com detecção de enlace caído; o health check é o "hello" de um protocolo
de roteamento, e o failover é a reconvergência.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .node_queue import NodeQueue

if TYPE_CHECKING:
    from .config import DomainConfig, GatewayConfig

log = logging.getLogger("agora.balancer")

POLICIES = ("round_robin", "least_queue", "least_latency", "peak_ewma")


@dataclass
class NodeHealth:
    name: str
    healthy: bool = True
    consecutive_failures: int = 0
    last_error: str | None = None
    down_since: float | None = None  # relógio monotônico
    last_check: float | None = None  # relógio monotônico
    ewma_ttft_ms: float | None = None
    samples: int = 0
    selected: int = 0
    failures: int = 0
    failovers_from: int = 0
    # Linha do tempo das trocas de estado (epoch, "down" | "up", motivo).
    transitions: list[tuple[float, str, str]] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "ewma_ttft_ms": round(self.ewma_ttft_ms, 1) if self.ewma_ttft_ms is not None else None,
            "samples": self.samples,
            "selected": self.selected,
            "failures": self.failures,
            "failovers_from": self.failovers_from,
            "transitions": [
                {"ts": round(ts, 3), "state": st, "reason": why} for ts, st, why in self.transitions[-10:]
            ],
        }


class Balancer:
    def __init__(
        self,
        cfg: "GatewayConfig",
        queues: dict[str, NodeQueue],
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cfg = cfg
        self.queues = queues
        self.clock = clock
        self.policy: str = cfg.balance.policy
        self.health = {name: NodeHealth(name) for name in cfg.nodes}
        self._rr: dict[str, int] = {}

    # ------------------------------------------------------------------ escolha
    def policy_for(self, domain: "DomainConfig") -> str:
        return domain.balance or self.policy

    def probed(self, name: str) -> bool:
        """Este nó tem health check ativo (e, portanto, volta sozinho quando sobe)?"""
        b = self.cfg.balance
        node = self.cfg.nodes[name]
        return b.health_interval_s > 0 and (node.kind == "ollama" or b.probe_remote)

    def available(self, name: str) -> bool:
        h = self.health[name]
        if h.healthy:
            return True
        if self.probed(name):
            return False  # quem decide a volta é o health check
        return h.down_since is not None and self.clock() - h.down_since >= self.cfg.balance.down_cooldown_s

    def load(self, name: str) -> float:
        q = self.queues[name]
        return (q.active + q.waiting) / q.max_concurrency

    def order(self, domain_name: str, domain: "DomainConfig", exclude: Iterable[str] = (),
              policy: str | None = None) -> list[str]:
        """Nós do pool na ordem em que devem ser tentados (sem os `exclude`).

        `policy` força uma política (o failover usa least_queue: na hora de refazer
        uma resposta interrompida, o que importa é a vaga livre AGORA)."""
        excluded = set(exclude)
        pool = [n for n in domain.pool if n not in excluded]
        if not pool:
            return []
        up = [n for n in pool if self.available(n)]
        candidates = up or pool  # todos fora: tenta assim mesmo
        # Rodízio como desempate (e como a própria política, no round_robin).
        k = self._rr.get(domain_name, 0)
        self._rr[domain_name] = k + 1
        k %= len(candidates)
        rotated = candidates[k:] + candidates[:k]
        policy = policy or self.policy_for(domain)
        if policy == "least_queue":
            rotated.sort(key=self.load)  # sort é estável: empate segue o rodízio
        elif policy == "least_latency":
            rotated.sort(key=lambda n: self.health[n].ewma_ttft_ms or 0.0)
        elif policy == "peak_ewma":
            # Nó sem amostra vale a média dos outros (com 0, o produto ignoraria a carga
            # e todo mundo se amontoaria nele); sem amostra nenhuma, vira least_queue.
            known = [self.health[n].ewma_ttft_ms for n in candidates if self.health[n].ewma_ttft_ms is not None]
            prior = sum(known) / len(known) if known else 1.0
            rotated.sort(key=lambda n: (self.health[n].ewma_ttft_ms or prior) * (self.load(n) + 1))
        return rotated

    # ------------------------------------------------------------------ observações
    def on_selected(self, name: str) -> None:
        self.health[name].selected += 1

    def on_success(self, name: str, ttft_ms: float | None) -> None:
        h = self.health[name]
        h.consecutive_failures = 0
        if ttft_ms is not None:
            a = self.cfg.balance.ewma_alpha
            h.ewma_ttft_ms = ttft_ms if h.ewma_ttft_ms is None else a * ttft_ms + (1 - a) * h.ewma_ttft_ms
            h.samples += 1
        if not h.healthy:
            self._set(h, True, "requisição ok")

    def on_failure(self, name: str, error: str, *, source: str = "requisição") -> None:
        h = self.health[name]
        h.failures += 1
        h.consecutive_failures += 1
        h.last_error = error[:300]
        if h.healthy and h.consecutive_failures >= self.cfg.balance.fail_threshold:
            self._set(h, False, f"{source}: {error[:120]}")
        elif not h.healthy:
            h.down_since = self.clock()  # tentativa no meio-aberto falhou: renova o castigo

    def _set(self, h: NodeHealth, healthy: bool, reason: str) -> None:
        h.healthy = healthy
        h.down_since = None if healthy else self.clock()
        h.transitions.append((time.time(), "up" if healthy else "down", reason))
        if healthy:
            log.warning("nó %s voltou ao pool (%s)", h.name, reason)
        else:
            log.warning("nó %s saiu do pool (%s)", h.name, reason)

    def reset_stats(self) -> None:
        """Zera médias e contadores (não a saúde): cada repetição de um experimento
        começa sem a "memória" da política anterior."""
        for h in self.health.values():
            h.ewma_ttft_ms, h.samples, h.selected, h.failures, h.failovers_from = None, 0, 0, 0, 0
        self._rr.clear()

    # ------------------------------------------------------------------ health check
    def nodes_in_use(self) -> list[str]:
        used: dict[str, None] = {}
        for d in self.cfg.domains.values():
            for n in d.pool:
                used[n] = None
        return list(used)

    async def check(self, name: str, probe: Callable[[str], Awaitable[Any]]) -> bool:
        h = self.health[name]
        h.last_check = self.clock()
        try:
            await asyncio.wait_for(probe(name), self.cfg.balance.health_timeout_s)
        except Exception as exc:  # noqa: BLE001 — qualquer falha conta como nó fora
            msg = str(exc) or type(exc).__name__
            self.on_failure(name, msg, source="health check")
            return False
        h.consecutive_failures = 0
        if not h.healthy:
            self._set(h, True, "health check ok")
        return True

    async def health_loop(self, probe: Callable[[str], Awaitable[Any]]) -> None:
        interval = self.cfg.balance.health_interval_s
        if interval <= 0:
            return
        while True:
            names = [n for n in self.nodes_in_use() if self.probed(n)]
            await asyncio.gather(*(self.check(n, probe) for n in names))
            await asyncio.sleep(interval)

    def snapshot(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "failover": self.cfg.balance.failover,
            "queue_policy": self.cfg.queue.policy,
            "pools": {name: {"nodes": d.pool, "policy": self.policy_for(d)}
                      for name, d in self.cfg.domains.items() if len(d.pool) > 1},
            "nodes": {name: h.snapshot() for name, h in self.health.items()},
        }


def priority_key(cfg: "GatewayConfig", source: str, now: float | None = None) -> float:
    """Chave da fila: menor sai primeiro.

    FIFO: a hora de chegada. Prioridade: chegada + atraso da classe (`queue.offsets_s`),
    um "prazo virtual". Uma fala da IA (atraso 0) passa na frente de uma narração do
    Mestre (atraso 4 s) que chegou até 4 s antes, mas não de uma que espera há mais
    tempo do que isso: é o envelhecimento que evita inanição.
    """
    now = time.monotonic() if now is None else now
    q = cfg.queue
    if q.policy == "fifo":
        return now
    return now + q.offsets_s.get(source, q.default_offset_s)
