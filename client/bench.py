"""Gerador de carga: N clientes simultâneos fazendo perguntas ao gateway.

Base do experimento obrigatório (variar 1, 2, 4, 8, 16 clientes e medir TTFT e
vazão). Cada cliente faz `--requests` perguntas em sequência (sistema fechado).

    python client/bench.py -c 1 2 4 8 16 -r 3 --repeat 3 -d impostor

Saídas em results/: um CSV por requisição e um CSV resumido por nível de carga.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, os.path.dirname(__file__))
from sse_client import ClientResult, ask  # noqa: E402

PROMPTS = {
    # Falas típicas de uma partida: a IA responde como um jogador.
    "impostor": [
        "Conversa até agora:\nAzul: oi gente\nVerde: qual foi a última coisa que te fez rir?\n\nVocê é Laranja. Escreva só a sua próxima mensagem.",
        "Conversa até agora:\nAzul: Laranja, você é robô?\n\nVocê é Laranja. Escreva só a sua próxima mensagem.",
        "Conversa até agora:\nVerde: o que vocês jantaram ontem?\n\nVocê é Laranja. Escreva só a sua próxima mensagem.",
        "Conversa até agora:\nAzul: acho que é o Verde\nVerde: nada a ver kkk\n\nVocê é Laranja. Escreva só a sua próxima mensagem.",
    ],
    "mestre": [
        "Rodada 1. Escreva UMA pergunta curta e pessoal para os jogadores.",
        "O jogador Azul foi exilado. Comente em UMA frase curta.",
        "Rodada 2. Escreva UMA pergunta curta sobre lembranças de infância.",
        "A votação empatou. Narre isso em UMA frase curta.",
    ],
}


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


async def run_level(url: str, domain: str, clients: int, requests: int, max_tokens: int | None):
    prompts = PROMPTS.get(domain, PROMPTS["impostor"])
    results: list[tuple[int, int, ClientResult]] = []

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, read=900.0),
        limits=httpx.Limits(max_connections=clients + 4),
    ) as client:

        async def worker(cid: int):
            for i in range(requests):
                body = {"domain": domain, "prompt": prompts[(cid + i) % len(prompts)],
                        "variables": {"alias": "Laranja", "others": "Azul, Verde"}}
                if max_tokens:
                    body["max_tokens"] = max_tokens
                results.append((cid, i, await ask(client, url, body)))

        t0 = time.perf_counter()
        await asyncio.gather(*(worker(c) for c in range(clients)))
        wall = time.perf_counter() - t0
    return results, wall


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-u", "--url", default=os.environ.get("AGORA_URL", "http://localhost:8000"))
    p.add_argument("-d", "--domain", default="impostor")
    p.add_argument("-c", "--clients", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("-r", "--requests", type=int, default=2, help="perguntas por cliente")
    p.add_argument("--repeat", type=int, default=1, help="repetições de cada nível")
    p.add_argument("-n", "--max-tokens", type=int, default=64)
    p.add_argument("-o", "--out", default="results")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    raw_path = out / f"bench-{stamp}-raw.csv"
    sum_path = out / f"bench-{stamp}-summary.csv"

    raw_fields = ["clients", "rep", "client_id", "seq", "status", "ttft_ms", "e2e_ms", "tokens",
                  "client_tps", "queue_wait_ms", "server_ttft_ms", "node_tps", "node", "failovers", "error"]
    sum_fields = ["clients", "rep", "ok", "failed", "wall_s", "throughput_tok_s",
                  "ttft_p50_ms", "ttft_p95_ms", "e2e_p50_ms", "tps_mean"]

    with open(raw_path, "w", newline="", encoding="utf-8") as fr, \
         open(sum_path, "w", newline="", encoding="utf-8") as fs:
        raw_w = csv.DictWriter(fr, fieldnames=raw_fields)
        sum_w = csv.DictWriter(fs, fieldnames=sum_fields)
        raw_w.writeheader()
        sum_w.writeheader()

        for n in a.clients:
            for rep in range(a.repeat):
                results, wall = await run_level(a.url, a.domain, n, a.requests, a.max_tokens)
                ok = [r for _, _, r in results if r.status == "ok"]
                for cid, seq, r in results:
                    raw_w.writerow({
                        "clients": n, "rep": rep, "client_id": cid, "seq": seq, "status": r.status,
                        "ttft_ms": r.ttft_ms, "e2e_ms": r.e2e_ms, "tokens": r.tokens,
                        "client_tps": r.tps, "queue_wait_ms": r.server.get("queue_wait_ms"),
                        "server_ttft_ms": r.server.get("ttft_ms"), "node_tps": r.server.get("node_tps"),
                        "node": r.server.get("node"), "failovers": r.failovers, "error": r.error,
                    })
                ttfts = [r.ttft_ms for r in ok if r.ttft_ms is not None]
                tpss = [r.tps for r in ok if r.tps]
                row = {
                    "clients": n, "rep": rep, "ok": len(ok), "failed": len(results) - len(ok),
                    "wall_s": round(wall, 3),
                    "throughput_tok_s": round(sum(r.tokens for r in ok) / wall, 3),
                    "ttft_p50_ms": pct(ttfts, 0.5), "ttft_p95_ms": pct(ttfts, 0.95),
                    "e2e_p50_ms": pct([r.e2e_ms for r in ok], 0.5),
                    "tps_mean": round(statistics.mean(tpss), 3) if tpss else None,
                }
                sum_w.writerow(row)
                fr.flush()
                fs.flush()
                print(
                    f"clientes={n:<3} rep={rep} ok={row['ok']:<3} falhas={row['failed']:<3} "
                    f"vazão={row['throughput_tok_s']:.1f} tok/s  "
                    f"TTFT p50={row['ttft_p50_ms'] or 0:.0f} ms p95={row['ttft_p95_ms'] or 0:.0f} ms"
                )
    print(f"\nCSV: {raw_path}\n     {sum_path}")


if __name__ == "__main__":
    asyncio.run(main())
