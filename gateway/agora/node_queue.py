"""Fila por nó com limite de concorrência.

Cada nó atende no máximo `max_concurrency` requisições ao mesmo tempo; as
demais esperam numa fila de tamanho `max_queue`. A fila é um heap ordenado por
(prioridade, ordem de chegada): com prioridade constante ela é FIFO, e trocar a
política (ex.: "respostas curtas primeiro", para a pergunta-desafio) é só mudar
a prioridade passada em `enqueue`.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools


class QueueFull(Exception):
    """A fila do nó está cheia; o gateway deve rejeitar a requisição (503)."""


class Ticket:
    """Lugar de uma requisição na fila. Sempre chame `release()` ao terminar."""

    def __init__(self, queue: "NodeQueue", future: asyncio.Future[None], position: int):
        self._queue = queue
        self._future = future
        self._released = False
        # Posição na fila no momento da chegada (0 = atendida na hora).
        self.position = position

    @property
    def granted(self) -> bool:
        return self._future.done() and not self._future.cancelled()

    async def wait(self) -> None:
        """Espera até o nó ter uma vaga para esta requisição."""
        await self._future

    def release(self) -> None:
        """Libera a vaga (se já foi concedida) ou sai da fila. Idempotente."""
        if self._released:
            return
        self._released = True
        if self.granted:
            self._queue._release_slot()
        else:
            if not self._future.done():
                self._future.cancel()
            self._queue._waiting -= 1
            self._queue.total_abandoned += 1


class NodeQueue:
    def __init__(self, name: str, max_concurrency: int, max_queue: int):
        self.name = name
        self.max_concurrency = max_concurrency
        self.max_queue = max_queue
        self.active = 0
        self._waiting = 0
        self._heap: list[tuple[float, int, asyncio.Future[None]]] = []
        self._seq = itertools.count()
        self.total_admitted = 0
        self.total_rejected = 0
        self.total_abandoned = 0

    @property
    def waiting(self) -> int:
        return self._waiting

    def enqueue(self, priority: float = 0.0) -> Ticket:
        """Reserva um lugar na fila sem bloquear. Levanta QueueFull se lotada."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        if self.active < self.max_concurrency and self._waiting == 0:
            self.active += 1
            self.total_admitted += 1
            future.set_result(None)
            return Ticket(self, future, position=0)
        if self._waiting >= self.max_queue:
            self.total_rejected += 1
            raise QueueFull(self.name)
        heapq.heappush(self._heap, (priority, next(self._seq), future))
        self._waiting += 1
        self.total_admitted += 1
        return Ticket(self, future, position=self._waiting)

    def _release_slot(self) -> None:
        self.active -= 1
        self._dispatch()

    def _dispatch(self) -> None:
        while self.active < self.max_concurrency and self._heap:
            _, _, future = heapq.heappop(self._heap)
            if future.done():  # quem estava aqui desistiu (já descontado)
                continue
            self._waiting -= 1
            self.active += 1
            future.set_result(None)

    def snapshot(self) -> dict[str, int | str]:
        return {
            "node": self.name,
            "active": self.active,
            "waiting": self._waiting,
            "max_concurrency": self.max_concurrency,
            "max_queue": self.max_queue,
            "total_admitted": self.total_admitted,
            "total_rejected": self.total_rejected,
            "total_abandoned": self.total_abandoned,
        }
