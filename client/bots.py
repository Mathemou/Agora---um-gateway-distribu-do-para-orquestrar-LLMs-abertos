"""Bots "humanos" para o Jogo da Imitação: N salas simultâneas, H bots por sala.

Serve para (1) testar o jogo sem juntar gente e (2) medir quanto a carga de
várias partidas ao mesmo tempo atrasa a IA. Cada bot conversa com frases
prontas, marca "pronto" e vota em alguém aleatório. No fim, o script lê do
gateway as requisições da IA (`/metrics/requests?source=impostor`) e grava CSVs.

    python client/bots.py --rooms 1 2 4 8 --bots 2 --discussion 60 --repeat 2

Pergunta que o experimento responde: a partir de quantas salas simultâneas a IA
passa a demorar mais para responder do que um humano levaria para digitar?
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import statistics
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import httpx
import websockets

LINES = [
    "oi gente", "alguém aí?", "eu comeria lasanha todo dia", "tô no ônibus kkk",
    "isso aí é muito robô", "ontem fiquei jogando até tarde", "sei lá mano",
    "{x} tá muito quieto", "{x}, o que você jantou ontem?", "eu acho que é o {x}",
    "nada a ver", "minha matéria favorita é redes kkkk", "{x} responde rápido demais",
    "quem é de vitória aqui?", "pdc", "alguém mais tá com fome?",
]


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 1)


async def bot(url: str, code: str, name: str, is_host: bool, n_bots: int, talk_s: float,
              rng: random.Random, out: dict) -> None:
    ws_url = url.replace("http", "ws", 1) + f"/ws/rooms/{code}?name={quote(name)}"
    async with websockets.connect(ws_url, max_size=2**22) as ws:
        state: dict = {}
        started = False
        chatter: asyncio.Task | None = None
        chat_round = 0

        async def chat_loop(round_no: int):
            t_end = time.monotonic() + talk_s
            while time.monotonic() < t_end and state.get("round") == round_no:
                await asyncio.sleep(rng.uniform(4, 10))
                if state.get("phase") != "discussao":
                    return
                others = [p["alias"] for p in state.get("players", []) if p["alive"] and not p["is_you"]]
                line = rng.choice(LINES).format(x=rng.choice(others) if others else "alguém")
                await ws.send(json.dumps({"type": "typing"}))
                await asyncio.sleep(len(line) / rng.uniform(5, 9))
                await ws.send(json.dumps({"type": "chat", "text": line}))
            if state.get("phase") == "discussao":
                await ws.send(json.dumps({"type": "ready"}))

        async for raw in ws:
            m = json.loads(raw)
            kind = m.get("type")
            if kind == "fatal":
                raise RuntimeError(m["message"])
            if kind == "chat" and m.get("kind") == "player":
                # tempo entre a última fala de um bot e uma fala de outro participante
                out.setdefault("chat", []).append((time.monotonic(), m["from"], m["text"]))
            if kind != "state":
                continue
            prev_phase = state.get("phase")
            state = m
            phase = m["phase"]
            if phase == "lobby" and is_host and not started and len(m.get("lobby", [])) >= n_bots and m["can_start"]:
                started = True
                await ws.send(json.dumps({"type": "start"}))
            elif phase == "discussao" and m["topic"] and chat_round != m["round"]:
                chat_round = m["round"]
                if chatter:
                    chatter.cancel()
                chatter = asyncio.create_task(chat_loop(m["round"]))
            elif phase == "votacao" and m["you"]["alive"] and not m["you"]["voted"] and prev_phase != "votacao":
                await asyncio.sleep(rng.uniform(1, 4))
                others = [p["alias"] for p in m["players"] if p["alive"] and not p["is_you"]]
                with_skip = others + [None]
                try:
                    await ws.send(json.dumps({"type": "vote", "target": rng.choice(with_skip)}))
                except websockets.ConnectionClosed:
                    return
            elif phase == "fim":
                if chatter:
                    chatter.cancel()
                if is_host:
                    out["ending"] = m["ending"]
                return


async def run_room(url: str, client: httpx.AsyncClient, n_bots: int, discussion: float, rounds: int,
                   talk_s: float, rng: random.Random) -> dict:
    r = await client.post(f"{url}/v1/rooms", json={"discussion_s": discussion, "max_rounds": rounds,
                                                   "vote_s": 20, "min_players": n_bots + 1})
    r.raise_for_status()
    code = r.json()["code"]
    out: dict = {"code": code}
    t0 = time.perf_counter()
    host = asyncio.create_task(bot(url, code, "bot-0", True, n_bots, talk_s, random.Random(rng.random()), out))
    await asyncio.sleep(0.3)
    others = [asyncio.create_task(bot(url, code, f"bot-{i}", False, n_bots, talk_s,
                                      random.Random(rng.random()), {}))
              for i in range(1, n_bots)]
    await asyncio.gather(host, *others)
    out["wall_s"] = time.perf_counter() - t0
    return out


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-u", "--url", default=os.environ.get("AGORA_URL", "http://localhost:8000"))
    p.add_argument("--rooms", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--bots", type=int, default=2, help="humanos (bots) por sala")
    p.add_argument("--discussion", type=float, default=60, help="segundos de discussão")
    p.add_argument("--talk", type=float, default=45, help="segundos que os bots conversam antes do 'pronto'")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-o", "--out", default="results")
    a = p.parse_args()

    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    raw_path, sum_path = out / f"rooms-{stamp}-raw.csv", out / f"rooms-{stamp}-summary.csv"
    raw_fields = ["rooms", "rep", "room", "status", "queue_position", "queue_wait_ms", "ttft_ms",
                  "e2e_ms", "output_tokens", "prompt_tokens", "node_tps"]
    sum_fields = ["rooms", "rep", "ai_requests", "ttft_p50_ms", "ttft_p95_ms", "queue_p50_ms",
                  "e2e_p50_ms", "e2e_p95_ms", "ai_msgs", "overruns", "avg_response_ms", "winner_ai"]

    async with httpx.AsyncClient(timeout=30) as client:
        with open(raw_path, "w", newline="", encoding="utf-8") as fr, \
             open(sum_path, "w", newline="", encoding="utf-8") as fs:
            rw, sw = csv.DictWriter(fr, raw_fields), csv.DictWriter(fs, sum_fields)
            rw.writeheader()
            sw.writeheader()
            for n in a.rooms:
                for rep in range(a.repeat):
                    results = await asyncio.gather(*(
                        run_room(a.url, client, a.bots, a.discussion, a.rounds, a.talk, rng) for _ in range(n)
                    ))
                    codes = {r["code"] for r in results}
                    recs = (await client.get(f"{a.url}/metrics/requests",
                                             params={"limit": 5000, "source": "impostor"})).json()
                    recs = [r for r in recs if r.get("room") in codes]
                    for r in recs:
                        rw.writerow({"rooms": n, "rep": rep, **{k: r.get(k) for k in raw_fields[2:]}})
                    ok = [r for r in recs if r["status"] == "ok"]
                    endings = [r.get("ending") or {} for r in results]
                    ai = [e.get("ai", {}) for e in endings]
                    resp = [x["avg_response_ms"] for x in ai if x.get("avg_response_ms")]
                    row = {
                        "rooms": n, "rep": rep, "ai_requests": len(recs),
                        "ttft_p50_ms": pct([r["ttft_ms"] for r in ok if r["ttft_ms"]], .5),
                        "ttft_p95_ms": pct([r["ttft_ms"] for r in ok if r["ttft_ms"]], .95),
                        "queue_p50_ms": pct([r["queue_wait_ms"] for r in ok if r["queue_wait_ms"] is not None], .5),
                        "e2e_p50_ms": pct([r["e2e_ms"] for r in ok if r["e2e_ms"]], .5),
                        "e2e_p95_ms": pct([r["e2e_ms"] for r in ok if r["e2e_ms"]], .95),
                        "ai_msgs": sum(x.get("messages", 0) for x in ai),
                        "overruns": sum(x.get("overruns", 0) for x in ai),
                        "avg_response_ms": round(statistics.mean(resp), 1) if resp else None,
                        "winner_ai": sum(1 for e in endings if e.get("winner") == "ia"),
                    }
                    sw.writerow(row)
                    fr.flush()
                    fs.flush()
                    print(f"salas={n:<3} rep={rep} req IA={row['ai_requests']:<4} "
                          f"TTFT p50={row['ttft_p50_ms']} ms p95={row['ttft_p95_ms']} ms  "
                          f"e2e p50={row['e2e_p50_ms']} ms  estouros={row['overruns']}/{row['ai_msgs']}")
    print(f"\nCSV: {raw_path}\n     {sum_path}")


if __name__ == "__main__":
    asyncio.run(main())
