import asyncio

import pytest

from agora.node_queue import NodeQueue, QueueFull

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def test_respeita_limite_de_concorrencia():
    q = NodeQueue("n", max_concurrency=2, max_queue=10)
    peak = 0

    async def job():
        nonlocal peak
        t = q.enqueue()
        try:
            await t.wait()
            peak = max(peak, q.active)
            await asyncio.sleep(0.01)
        finally:
            t.release()

    await asyncio.gather(*(job() for _ in range(8)))
    assert peak == 2
    assert q.active == 0 and q.waiting == 0


async def test_fifo():
    q = NodeQueue("n", max_concurrency=1, max_queue=10)
    order = []
    first = q.enqueue()  # ocupa a vaga
    tickets = [q.enqueue() for _ in range(4)]
    assert [t.position for t in tickets] == [1, 2, 3, 4]

    async def job(i, t):
        await t.wait()
        order.append(i)
        t.release()

    tasks = [asyncio.create_task(job(i, t)) for i, t in enumerate(tickets)]
    await asyncio.sleep(0)
    first.release()
    await asyncio.gather(*tasks)
    assert order == [0, 1, 2, 3]


async def test_fila_cheia_rejeita():
    q = NodeQueue("n", max_concurrency=1, max_queue=1)
    q.enqueue()
    q.enqueue()
    with pytest.raises(QueueFull):
        q.enqueue()
    assert q.total_rejected == 1


async def test_desistencia_na_fila_libera_lugar():
    q = NodeQueue("n", max_concurrency=1, max_queue=5)
    busy = q.enqueue()
    waiting = q.enqueue()
    task = asyncio.create_task(waiting.wait())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    waiting.release()
    assert q.waiting == 0
    busy.release()
    assert q.active == 0
    # A vaga continua utilizável
    t = q.enqueue()
    assert t.granted


async def test_cancelamento_apos_receber_vaga_devolve_vaga():
    q = NodeQueue("n", max_concurrency=1, max_queue=5)
    busy = q.enqueue()
    nxt = q.enqueue()
    busy.release()  # concede a vaga para `nxt`
    assert nxt.granted and q.active == 1
    nxt.release()
    nxt.release()  # idempotente
    assert q.active == 0
