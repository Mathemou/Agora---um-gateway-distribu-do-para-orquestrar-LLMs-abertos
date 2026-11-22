"""Experimentos da trilha Desempenho (balanceamento, failover e prioridade na fila).

Troca as políticas do gateway em tempo de execução (POST /v1/balancer), gera carga
e grava CSVs em results/. Os gráficos saem de client/plot_desempenho.py.

1) balance — compara round_robin × least_queue × least_latency × peak_ewma
   python client/exp_desempenho.py balance -c 6 -r 5 --repeat 3

2) failover — carga contínua; no meio, um nó cai. Compara failover ligado × desligado
   python client/exp_desempenho.py failover --fault-url http://127.0.0.1:11436 --fault-mode kill
   python client/exp_desempenho.py failover --fault-cmd "docker stop ollama-yan" --recover-cmd "docker start ollama-yan"

3) fila — falas curtas da IA (interativas) disputando o nó com pedidos longos
   (API em lote). Compara fifo × priority, medindo quantas falas estouram o
   orçamento de digitação.
   python client/exp_desempenho.py fila -d mestre --interactive 2 --batch 4 --duration 40

Com o nó falso: config/gateway.fake-multi.yaml (3 nós, um 3× mais lento).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import shlex
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, os.path.dirname(__file__))
from sse_client import ClientResult, ask, auth_headers  # noqa: E402

PROMPT = ("Conversa até agora:\nAzul: Laranja, o que você jantou ontem?\n\n"
          "Você é Laranja. Escreva só a sua próxima mensagem.")
VARS = {"alias": "Laranja", "others": "Azul, Verde"}


def pct(values, p):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 1)


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


async def set_policy(client: httpx.AsyncClient, url: str, **body) -> dict:
    r = await client.post(f"{url}/v1/balancer", json=body, headers=auth_headers())
    r.raise_for_status()
    return r.json()


def writer(path: Path, fields: list[str]):
    fh = open(path, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    return fh, w


def body(domain: str, max_tokens: int, source: str = "api") -> dict:
    return {"domain": domain, "prompt": PROMPT, "variables": VARS, "max_tokens": max_tokens, "source": source}


def row_of(r: ClientResult, **extra) -> dict:
    s = r.server
    return {
        **extra, "status": r.status, "ttft_ms": r.ttft_ms and round(r.ttft_ms, 1),
        "e2e_ms": r.e2e_ms and round(r.e2e_ms, 1), "tokens": r.tokens, "chars": len(r.text),
        "node": s.get("node"), "nodes_path": s.get("nodes_path"), "queue_wait_ms": s.get("queue_wait_ms"),
        "failovers": r.failovers, "discarded_tokens": r.discarded_tokens,
        "recovery_ms": s.get("recovery_ms"), "error": (r.error or "")[:120],
    }


# ====================================================================== 1) balance
async def exp_balance(a) -> None:
    out = Path(a.out)
    st = stamp()
    raw_fh, raw = writer(out / f"desempenho-balance-{st}-raw.csv",
                         ["policy", "clients", "rep", "client_id", "seq", "status", "ttft_ms", "e2e_ms",
                          "tokens", "chars", "node", "nodes_path", "queue_wait_ms", "failovers",
                          "discarded_tokens", "recovery_ms", "error"])
    sum_fh, summ = writer(out / f"desempenho-balance-{st}-summary.csv",
                          ["policy", "clients", "rep", "ok", "failed", "wall_s", "throughput_tok_s",
                           "ttft_p50_ms", "ttft_p95_ms", "e2e_p50_ms", "e2e_p95_ms", "per_node"])
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=900.0),
                                 limits=httpx.Limits(max_connections=max(a.clients) + 8)) as client:
        for policy in a.policies:
            for n in a.clients:
                for rep in range(a.repeat):
                    await set_policy(client, a.url, policy=policy, queue_policy="fifo", failover=True, reset=True)
                    results: list[tuple[int, int, ClientResult]] = []

                    async def worker(cid: int):
                        for i in range(a.requests):
                            results.append((cid, i, await ask(client, a.url, body(a.domain, a.max_tokens))))

                    t0 = time.perf_counter()
                    await asyncio.gather(*(worker(c) for c in range(n)))
                    wall = time.perf_counter() - t0
                    ok = [r for _, _, r in results if r.status == "ok"]
                    for cid, seq, r in results:
                        raw.writerow(row_of(r, policy=policy, clients=n, rep=rep, client_id=cid, seq=seq))
                    per_node = Counter(r.server.get("node") for r in ok)
                    s = {
                        "policy": policy, "clients": n, "rep": rep, "ok": len(ok),
                        "failed": len(results) - len(ok), "wall_s": round(wall, 2),
                        "throughput_tok_s": round(sum(r.tokens for r in ok) / wall, 2),
                        "ttft_p50_ms": pct([r.ttft_ms for r in ok], .5),
                        "ttft_p95_ms": pct([r.ttft_ms for r in ok], .95),
                        "e2e_p50_ms": pct([r.e2e_ms for r in ok], .5),
                        "e2e_p95_ms": pct([r.e2e_ms for r in ok], .95),
                        "per_node": json.dumps(dict(sorted(per_node.items()))),
                    }
                    summ.writerow(s)
                    raw_fh.flush()
                    sum_fh.flush()
                    print(f"{policy:<14} clientes={n:<3} rep={rep} ok={s['ok']:<3} "
                          f"TTFT p50={s['ttft_p50_ms']} p95={s['ttft_p95_ms']} ms  "
                          f"vazão={s['throughput_tok_s']} tok/s  nós={s['per_node']}")
    raw_fh.close()
    sum_fh.close()
    print(f"\nCSV: {raw_fh.name}\n     {sum_fh.name}")


# ====================================================================== 2) failover
async def fault(client: httpx.AsyncClient, a, start: bool) -> None:
    cmd = a.fault_cmd if start else a.recover_cmd
    if cmd:
        print(f"  $ {cmd}")
        subprocess.run(shlex.split(cmd) if os.name != "nt" else cmd, shell=os.name == "nt", check=False)
        return
    if a.fault_url:
        mode = a.fault_mode if start else "none"
        try:
            await client.post(f"{a.fault_url}/_fault", json={"mode": mode, "after_tokens": a.fault_after})
        except httpx.HTTPError as exc:  # "crash" derruba o processo: a resposta não volta
            print(f"  (nó falso não respondeu ao /_fault: {type(exc).__name__})")
        print(f"  nó {a.fault_url}: defeito = {mode}")


async def exp_failover(a) -> None:
    out = Path(a.out)
    st = stamp()
    raw_fh, raw = writer(out / f"desempenho-failover-{st}-raw.csv",
                         ["failover", "rep", "client_id", "t_start_s", "t_end_s", "status", "ttft_ms",
                          "e2e_ms", "tokens", "chars", "node", "nodes_path", "queue_wait_ms", "failovers",
                          "discarded_tokens", "recovery_ms", "error"])
    sum_fh, summ = writer(out / f"desempenho-failover-{st}-summary.csv",
                          ["failover", "rep", "requests", "ok", "failed", "with_failover",
                           "recovery_p50_ms", "recovery_max_ms", "discarded_tokens",
                           "e2e_p50_ms", "e2e_p95_ms", "e2e_p95_after_fault_ms", "fail_at_s"])
    modes = [m == "on" for m in a.modes]
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=900.0),
                                 limits=httpx.Limits(max_connections=a.clients + 8)) as client:
        for fo in modes:
            for rep in range(a.repeat):
                await set_policy(client, a.url, policy=a.policy, failover=fo, queue_policy="fifo", reset=True)
                rows: list[dict] = []
                t0 = time.perf_counter()
                stop_at = t0 + a.duration

                async def worker(cid: int):
                    while time.perf_counter() < stop_at:
                        ts = time.perf_counter() - t0
                        r = await ask(client, a.url, body(a.domain, a.max_tokens))
                        rows.append(row_of(r, failover="on" if fo else "off", rep=rep, client_id=cid,
                                           t_start_s=round(ts, 3),
                                           t_end_s=round(time.perf_counter() - t0, 3)))

                async def chaos():
                    await asyncio.sleep(a.fail_at)
                    await fault(client, a, True)
                    if a.recover_at:
                        await asyncio.sleep(max(0.0, a.recover_at - a.fail_at))
                        await fault(client, a, False)

                print(f"failover={'on' if fo else 'off'} rep={rep}: {a.clients} clientes por {a.duration:g} s, "
                      f"defeito em t={a.fail_at:g} s")
                await asyncio.gather(chaos(), *(worker(c) for c in range(a.clients)))
                if not a.recover_at:
                    await fault(client, a, False)  # deixa tudo como estava para a próxima rodada
                if a.settle:
                    await asyncio.sleep(a.settle)  # health check devolve o nó ao pool
                for r in rows:
                    raw.writerow(r)
                ok = [r for r in rows if r["status"] == "ok"]
                rec = [r["recovery_ms"] for r in ok if r["recovery_ms"] is not None]
                after = [r["e2e_ms"] for r in ok if r["t_end_s"] >= a.fail_at]
                s = {
                    "failover": "on" if fo else "off", "rep": rep, "requests": len(rows), "ok": len(ok),
                    "failed": len(rows) - len(ok), "with_failover": sum(1 for r in rows if r["failovers"]),
                    "recovery_p50_ms": pct(rec, .5), "recovery_max_ms": max(rec) if rec else None,
                    "discarded_tokens": sum(r["discarded_tokens"] for r in rows),
                    "e2e_p50_ms": pct([r["e2e_ms"] for r in ok], .5),
                    "e2e_p95_ms": pct([r["e2e_ms"] for r in ok], .95),
                    "e2e_p95_after_fault_ms": pct(after, .95), "fail_at_s": a.fail_at,
                }
                summ.writerow(s)
                raw_fh.flush()
                sum_fh.flush()
                print(f"  ok={s['ok']} falhas={s['failed']} com_failover={s['with_failover']} "
                      f"recuperação p50={s['recovery_p50_ms']} máx={s['recovery_max_ms']} ms "
                      f"tokens_descartados={s['discarded_tokens']}")
    raw_fh.close()
    sum_fh.close()
    print(f"\nCSV: {raw_fh.name}\n     {sum_fh.name}")


# ====================================================================== 3) fila
async def exp_fila(a) -> None:
    out = Path(a.out)
    st = stamp()
    raw_fh, raw = writer(out / f"desempenho-fila-{st}-raw.csv",
                         ["queue_policy", "rep", "classe", "client_id", "t_start_s", "status", "ttft_ms",
                          "e2e_ms", "tokens", "chars", "budget_ms", "overrun_ms", "node", "queue_wait_ms",
                          "error"])
    sum_fh, summ = writer(out / f"desempenho-fila-{st}-summary.csv",
                          ["queue_policy", "rep", "classe", "requests", "ok", "ttft_p50_ms", "ttft_p95_ms",
                           "e2e_p50_ms", "e2e_p95_ms", "queue_wait_p50_ms", "overrun_frac", "tokens_per_s"])
    rng = random.Random(a.seed)
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=900.0),
                                 limits=httpx.Limits(max_connections=a.interactive + a.batch + 8)) as client:
        for qp in a.policies:
            for rep in range(a.repeat):
                await set_policy(client, a.url, queue_policy=qp, reset=True)
                rows: list[dict] = []
                t0 = time.perf_counter()
                stop_at = t0 + a.duration

                async def batch(cid: int):
                    while time.perf_counter() < stop_at:
                        ts = time.perf_counter() - t0
                        r = await ask(client, a.url, body(a.domain, a.batch_tokens, "api"))
                        rows.append(row_of(r, queue_policy=qp, rep=rep, classe="lote", client_id=cid,
                                           t_start_s=round(ts, 3)))

                async def interactive(cid: int):
                    await asyncio.sleep(rng.uniform(0, a.think_s))
                    while time.perf_counter() < stop_at:
                        ts = time.perf_counter() - t0
                        r = await ask(client, a.url, body(a.domain, a.interactive_tokens, "impostor"))
                        # Orçamento de digitação do jogo: "lendo" + digitando a fala a v car/s.
                        budget = (a.read_s + max(len(r.text), 10) / a.cps) * 1000
                        over = max(0.0, (r.e2e_ms or 0) - budget)
                        rows.append(row_of(r, queue_policy=qp, rep=rep, classe="interativa", client_id=cid,
                                           t_start_s=round(ts, 3), budget_ms=round(budget, 1),
                                           overrun_ms=round(over, 1)))
                        await asyncio.sleep(rng.uniform(.5, 1.5) * a.think_s)

                print(f"fila={qp} rep={rep}: {a.interactive} interativos + {a.batch} em lote por {a.duration:g} s")
                await asyncio.gather(*(batch(c) for c in range(a.batch)),
                                     *(interactive(c) for c in range(a.interactive)))
                wall = time.perf_counter() - t0
                for r in rows:
                    raw.writerow(r)
                for classe in ("interativa", "lote"):
                    rs = [r for r in rows if r["classe"] == classe]
                    ok = [r for r in rs if r["status"] == "ok"]
                    s = {
                        "queue_policy": qp, "rep": rep, "classe": classe, "requests": len(rs), "ok": len(ok),
                        "ttft_p50_ms": pct([r["ttft_ms"] for r in ok], .5),
                        "ttft_p95_ms": pct([r["ttft_ms"] for r in ok], .95),
                        "e2e_p50_ms": pct([r["e2e_ms"] for r in ok], .5),
                        "e2e_p95_ms": pct([r["e2e_ms"] for r in ok], .95),
                        "queue_wait_p50_ms": pct([r["queue_wait_ms"] for r in ok], .5),
                        "overrun_frac": (round(statistics.mean(1.0 if r["overrun_ms"] > 0 else 0.0 for r in ok), 3)
                                         if classe == "interativa" and ok else None),
                        "tokens_per_s": round(sum(r["tokens"] for r in ok) / wall, 2),
                    }
                    summ.writerow(s)
                    print(f"  {classe:<10} n={s['ok']:<4} e2e p50={s['e2e_p50_ms']} p95={s['e2e_p95_ms']} ms  "
                          f"fila p50={s['queue_wait_p50_ms']} ms"
                          + (f"  estouro do orçamento={s['overrun_frac']:.0%}" if s["overrun_frac"] is not None else ""))
                raw_fh.flush()
                sum_fh.flush()
        await set_policy(client, a.url, queue_policy="fifo")
    raw_fh.close()
    sum_fh.close()
    print(f"\nCSV: {raw_fh.name}\n     {sum_fh.name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-u", "--url", default=os.environ.get("AGORA_URL", "http://localhost:8000"))
    p.add_argument("-o", "--out", default="results")
    sub = p.add_subparsers(dest="exp", required=True)

    b = sub.add_parser("balance", help="round_robin × least_queue × least_latency × peak_ewma")
    b.add_argument("-d", "--domain", default="impostor")
    b.add_argument("--policies", nargs="+", default=["round_robin", "least_queue", "least_latency", "peak_ewma"])
    b.add_argument("-c", "--clients", type=int, nargs="+", default=[6])
    b.add_argument("-r", "--requests", type=int, default=5, help="pedidos por cliente")
    b.add_argument("--repeat", type=int, default=3)
    b.add_argument("-n", "--max-tokens", type=int, default=32)

    f = sub.add_parser("failover", help="um nó cai no meio da carga: failover on × off")
    f.add_argument("-d", "--domain", default="impostor")
    f.add_argument("--policy", default="round_robin", help="round_robin garante tráfego no nó que cai")
    f.add_argument("--modes", nargs="+", default=["on", "off"], choices=["on", "off"])
    f.add_argument("-c", "--clients", type=int, default=4)
    f.add_argument("--duration", type=float, default=30)
    f.add_argument("--fail-at", type=float, default=10)
    f.add_argument("--recover-at", type=float, default=None, help="quando o nó volta (padrão: no fim)")
    f.add_argument("--settle", type=float, default=5, help="espera entre rodadas (health check)")
    f.add_argument("--fault-url", help="nó falso a quebrar (ex.: http://127.0.0.1:11436)")
    f.add_argument("--fault-mode", default="kill", choices=["kill", "die", "refuse", "hang", "crash"])
    f.add_argument("--fault-after", type=int, default=5, help="tokens antes de cair")
    f.add_argument("--fault-cmd", help='comando que derruba um nó real (ex.: "docker stop ollama-2")')
    f.add_argument("--recover-cmd", help="comando que traz o nó de volta")
    f.add_argument("--repeat", type=int, default=3)
    f.add_argument("-n", "--max-tokens", type=int, default=32)

    q = sub.add_parser("fila", help="fala da IA × lote na mesma fila: fifo × priority")
    q.add_argument("-d", "--domain", default="impostor",
                   help="use um domínio de UM nó para saturar (no fake-multi: mestre)")
    q.add_argument("--policies", nargs="+", default=["fifo", "priority"])
    q.add_argument("--interactive", type=int, default=2, help="clientes imitando a fala da IA")
    q.add_argument("--batch", type=int, default=4, help="clientes de API com respostas longas")
    q.add_argument("--interactive-tokens", type=int, default=16)
    q.add_argument("--batch-tokens", type=int, default=120)
    q.add_argument("--think-s", type=float, default=2.0, help="intervalo médio entre falas da IA")
    q.add_argument("--read-s", type=float, default=2.75, help="'lendo' antes de responder (média do jogo)")
    q.add_argument("--cps", type=float, default=7.0, help="velocidade de digitação simulada (car/s)")
    q.add_argument("--duration", type=float, default=40)
    q.add_argument("--repeat", type=int, default=3)
    q.add_argument("--seed", type=int, default=7)

    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    run = {"balance": exp_balance, "failover": exp_failover, "fila": exp_fila}[a.exp]
    try:
        asyncio.run(run(a))
    except httpx.ConnectError:
        sys.exit(f"gateway inacessível em {a.url}")


if __name__ == "__main__":
    main()
