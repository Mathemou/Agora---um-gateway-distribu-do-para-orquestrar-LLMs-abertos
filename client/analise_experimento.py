"""Análise do experimento obrigatório: previsão pelo modelo de fila × medição.

Lê a pasta de um experimento (gerada por client/experimento.py) e produz:

  fig-ttft.png        TTFT médio × clientes simultâneos (medição com IC 95% e previsão)
  fig-vazao.png       vazão agregada (tokens/s) × clientes (medição e previsão)
  fig-salas.png       fração de falas da IA que estouram o orçamento de digitação × salas
  fig-tempo-fala.png  onde vai o tempo de uma fala da IA (fila, prefill, geração) × orçamento
  fig-fila.png        fifo × priority: estouro do orçamento e latência dos pedidos em lote
  resumo.md / resumo.json   tabelas prontas para o relatório

    python client/analise_experimento.py results/experimento-20261122-101500
    python client/analise_experimento.py <pasta> --c 2 --games-log logs/games.jsonl

O modelo de fila
----------------
N clientes em laço fechado (cada um faz `r` pedidos seguidos, sem pausa), um nó com
`c` vagas e tempo de serviço S (prefill + geração). Enquanto N ≤ c ninguém espera.
Com N > c, cada pedido novo entra atrás de N − c pedidos e espera ≈ (N/c − 1)·S;
no primeiro "lote" a espera média é metade disso, então em média
W(N) ≈ (N/c − 1)·S·(1 − 1/(2r)), TTFT(N) ≈ TTFT_serviço + W(N) e a vazão satura em
c·tokens/S. A curva da previsão vem de uma simulação determinística desse sistema
(mesmos N, r e c do experimento), que dá o mesmo resultado da fórmula sem
arredondar N/c.

Duas previsões são desenhadas:
  ingênua    S medido com 1 cliente e suposto constante (o que se faz antes de medir)
  calibrada  S medido com c clientes (as c gerações paralelas dividem os mesmos
             núcleos da CPU, então S cresce): mostra quanto do erro vem disso
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

# --------------------------------------------------------------------------- estatística
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}


def t95(df: int) -> float:
    if df <= 0:
        return float("nan")
    for k in sorted(_T95):
        if df <= k:
            return _T95[k]
    return 1.96


def mean_ci(xs: list[float]) -> tuple[float | None, float | None]:
    """Média e meia-largura do IC 95% (t de Student) entre repetições."""
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return None, None
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, None
    return m, t95(len(xs) - 1) * statistics.stdev(xs) / math.sqrt(len(xs))


def wilson(k: int, n: int) -> tuple[float | None, float | None, float | None]:
    """Proporção com IC 95% de Wilson (bom para poucas amostras e 0 %/100 %)."""
    if n == 0:
        return None, None, None
    z, p = 1.96, k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def pct(xs: list[float], p: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def num(v) -> float | None:
    try:
        return None if v in (None, "", "None") else float(v)
    except (TypeError, ValueError):
        return None


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def latest(folder: Path, pattern: str) -> Path | None:
    files = sorted(folder.glob(pattern))
    return files[-1] if files else None


# --------------------------------------------------------------------------- modelo de fila
def simulate_closed(n_clients: int, requests: int, c: int, service_s: float) -> tuple[float, float]:
    """Simulação determinística: N clientes × r pedidos seguidos, c vagas, FIFO.

    Devolve (espera média na fila em s, duração total em s).
    """
    free_at = [0.0] * c          # quando cada vaga fica livre (heap)
    heapq.heapify(free_at)
    arrivals = [(0.0, i, 0) for i in range(n_clients)]  # (chegada, cliente, nº do pedido)
    heapq.heapify(arrivals)
    waits, end = [], 0.0
    while arrivals:
        t, cid, k = heapq.heappop(arrivals)
        slot = heapq.heappop(free_at)
        start = max(t, slot)
        done = start + service_s
        waits.append(start - t)
        heapq.heappush(free_at, done)
        end = max(end, done)
        if k + 1 < requests:
            heapq.heappush(arrivals, (done, cid, k + 1))
    return statistics.mean(waits), end


def predict(levels: list[int], requests: int, c: int, s1: float, ttft1: float, sc: float, ttftc: float,
            tokens: float) -> dict[int, dict]:
    """Previsão por nível de carga (tempos em s)."""
    out = {}
    for n in levels:
        k = min(n, c)
        # com até c pedidos em paralelo, o serviço vai de S1 (sozinho) a Sc (c juntos)
        frac = 0.0 if c == 1 else (k - 1) / (c - 1)
        s = s1 + (sc - s1) * frac
        tt = ttft1 + (ttftc - ttft1) * frac
        wait, wall = simulate_closed(n, requests, c, s)
        out[n] = {"service_s": s, "wait_s": wait, "ttft_s": tt + wait,
                  "throughput_tok_s": n * requests * tokens / wall}
    return out


# --------------------------------------------------------------------------- gráficos
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "font.size": 10, "axes.edgecolor": GRID,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 11,
        "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 2, "lines.markersize": 8,
    })
    return plt


def plot_bench(folder: Path, levels, meas_ttft, meas_tp, naive, calib, c, title_extra=""):
    plt = _style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    xs = levels
    m = [meas_ttft[n][0] for n in xs]
    e = [meas_ttft[n][1] or 0 for n in xs]
    ax.plot(xs, [naive[n]["ttft_s"] for n in xs], color=INK2, linestyle="--", marker=None,
            label="previsão ingênua (S de 1 cliente)")
    ax.plot(xs, [calib[n]["ttft_s"] for n in xs], color=ORANGE, marker="s", markersize=6,
            label=f"previsão calibrada (S com {c} em paralelo)")
    ax.errorbar(xs, m, yerr=e, color=BLUE, marker="o", capsize=4, label="medido (média, IC 95%)")
    ax.axvline(c, color=GRID, linewidth=1.5, zorder=0)
    ax.text(c, 0.04, f" c = {c} vagas", color=INK2, va="bottom", fontsize=9,
            transform=ax.get_xaxis_transform())
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.set_xlabel("clientes simultâneos (N)")
    ax.set_ylabel("TTFT médio (s)")
    ax.set_title("TTFT × carga: o joelho aparece quando N passa de c" + title_extra, loc="left")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(folder / "fig-ttft.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    m = [meas_tp[n][0] for n in xs]
    e = [meas_tp[n][1] or 0 for n in xs]
    ax.plot(xs, [naive[n]["throughput_tok_s"] for n in xs], color=INK2, linestyle="--",
            label="previsão ingênua")
    ax.plot(xs, [calib[n]["throughput_tok_s"] for n in xs], color=ORANGE, marker="s", markersize=6,
            label="previsão calibrada")
    ax.errorbar(xs, m, yerr=e, color=BLUE, marker="o", capsize=4, label="medido (média, IC 95%)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("clientes simultâneos (N)")
    ax.set_ylabel("vazão agregada (tokens/s)")
    ax.set_title("A vazão satura em ≈ c · tokens / S", loc="left")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(folder / "fig-vazao.png")
    plt.close(fig)


def plot_rooms(folder: Path, rows: list[dict], budget_s: float | None):
    plt = _style()
    ks = [r["rooms"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for key, color, marker, label in (("all", BLUE, "o", "todas as falas"),
                                      ("first", ORANGE, "s", "1ª fala da rodada (rajada)")):
        trip = [r[f"over_{key}"] for r in rows]
        p = [t[0] * 100 if t[0] is not None else float("nan") for t in trip]
        lo = [max(0.0, t[0] - t[1]) * 100 if t[0] is not None else 0 for t in trip]
        hi = [max(0.0, t[2] - t[0]) * 100 if t[0] is not None else 0 for t in trip]
        ax.errorbar(ks, p, yerr=[lo, hi],
                    color=color, marker=marker, capsize=4, label=label + " (IC 95%)")
    if rows and rows[0].get("pred_first") is not None:
        ax.plot(ks, [r["pred_first"] * 100 for r in rows], color=INK2, linestyle="--",
                label="previsão (rajada, modelo de fila)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_ylim(-3, 103)
    ax.set_xlabel("salas simultâneas")
    ax.set_ylabel("falas que estouram o orçamento (%)")
    ax.set_title("Quando a IA começa a se denunciar pelo tempo", loc="left")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(folder / "fig-salas.png")
    plt.close(fig)

    # onde vai o tempo (medianas), comparado com o orçamento de digitação
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    import numpy as np
    x = np.arange(len(rows))
    q = np.array([r["med_queue_s"] or 0 for r in rows])
    pre = np.array([r["med_prefill_s"] or 0 for r in rows])
    gen = np.array([r["med_gen_s"] or 0 for r in rows])
    w = 0.55
    ax.bar(x, q, w, color=ORANGE, label="espera na fila", edgecolor="white", linewidth=2)
    ax.bar(x, pre, w, bottom=q, color=BLUE, label="prefill (TTFT sem fila)", edgecolor="white", linewidth=2)
    ax.bar(x, gen, w, bottom=q + pre, color=AQUA, label="geração dos tokens", edgecolor="white", linewidth=2)
    bud = [r["med_budget_s"] for r in rows]
    if any(b is not None for b in bud):
        ax.plot(x, bud, color=INK, linestyle="none", marker="_", markersize=28, markeredgewidth=2.5,
                label="orçamento de digitação (mediana)")
    ax.set_xticks(x, [str(k) for k in ks])
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    top = max([*(q + pre + gen), *[b for b in bud if b is not None], 1.0])
    ax.set_ylim(0, top * 1.5)
    ax.set_xlabel("salas simultâneas")
    ax.set_ylabel("segundos (mediana por fala)")
    ax.set_title("Onde vai o tempo de uma fala da IA", loc="left")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(folder / "fig-tempo-fala.png")
    plt.close(fig)


def plot_fila(folder: Path, table: list[dict]):
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2))
    names = [t["policy"] for t in table]
    for ax, key, title, color, scale in (
        (axes[0], "overrun", "falas da IA que estouram (%)", BLUE, 100),
        (axes[1], "lote_e2e_p50", "pedidos em lote: e2e p50 (s)", ORANGE, 1 / 1000),
    ):
        vals = [(t[key][0] or 0) * scale for t in table]
        errs = [(t[key][1] or 0) * scale for t in table]
        ax.bar(names, vals, 0.55, yerr=errs, capsize=4, color=color, edgecolor="white", linewidth=2)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max([v + e for v, e in zip(vals, errs)] + [5 if scale == 100 else 1]) * 1.2)
        for i, (v, e) in enumerate(zip(vals, errs)):
            ax.text(i, v + e, f"{v:.0f}" if scale == 100 else f"{v:.1f}", ha="center", va="bottom",
                    color=INK, fontsize=9)
    fig.suptitle("Política de fila: fifo × priority (média e IC 95% entre repetições)", x=0.02, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(folder / "fig-fila.png")
    plt.close(fig)


# --------------------------------------------------------------------------- partes da análise
def analyse_bench(folder: Path, c: int, rep_requests: int | None) -> dict | None:
    raw_path = latest(folder, "bench-*-raw.csv")
    if raw_path is None:
        return None
    raw = [r for r in read_csv(raw_path)]
    ok = [r for r in raw if r["status"] == "ok"]
    levels = sorted({int(r["clients"]) for r in raw})
    by = defaultdict(list)
    for r in ok:
        by[(int(r["clients"]), int(r["rep"]))].append(r)
    reps = sorted({int(r["rep"]) for r in raw})
    requests = rep_requests or max(int(r["seq"]) for r in raw) + 1

    summ_path = latest(folder, "bench-*-summary.csv")
    walls = {(int(s["clients"]), int(s["rep"])): num(s["wall_s"]) for s in read_csv(summ_path)} if summ_path else {}

    def ms(r, k):
        v = num(r.get(k))
        return v / 1000 if v is not None else None

    meas_ttft, meas_tp, table = {}, {}, []
    for n in levels:
        per_rep_ttft, per_rep_tp, per_rep_p50, per_rep_p95, per_rep_q = [], [], [], [], []
        for rep in reps:
            rs = by.get((n, rep), [])
            tt = [ms(r, "ttft_ms") for r in rs if ms(r, "ttft_ms") is not None]
            if tt:
                per_rep_ttft.append(statistics.mean(tt))
                per_rep_p50.append(pct(tt, .5))
                per_rep_p95.append(pct(tt, .95))
            qq = [ms(r, "queue_wait_ms") for r in rs if ms(r, "queue_wait_ms") is not None]
            if qq:
                per_rep_q.append(statistics.mean(qq))
            wall = walls.get((n, rep))
            if wall:
                per_rep_tp.append(sum(num(r["tokens"]) or 0 for r in rs) / wall)
        meas_ttft[n] = mean_ci(per_rep_ttft)
        meas_tp[n] = mean_ci(per_rep_tp)
        fails = sum(1 for r in raw if int(r["clients"]) == n and r["status"] != "ok")
        table.append({"clients": n, "reps": len(per_rep_ttft), "ttft_mean": meas_ttft[n],
                      "ttft_p50": mean_ci(per_rep_p50), "ttft_p95": mean_ci(per_rep_p95),
                      "queue_mean": mean_ci(per_rep_q), "throughput": meas_tp[n], "failed": fails})

    # parâmetros do modelo, medidos: S e TTFT de serviço (sem a espera na fila)
    def service(rs):
        s = [ms(r, "e2e_ms") - (ms(r, "queue_wait_ms") or 0) for r in rs if ms(r, "e2e_ms") is not None]
        t = [ms(r, "ttft_ms") - (ms(r, "queue_wait_ms") or 0) for r in rs if ms(r, "ttft_ms") is not None]
        k = [num(r["tokens"]) for r in rs if num(r["tokens"])]
        return (statistics.mean(s) if s else None, statistics.mean(t) if t else None,
                statistics.mean(k) if k else None)

    one = [r for r in ok if int(r["clients"]) == levels[0]]
    s1, ttft1, tokens = service(one)
    cl = min((n for n in levels if n >= c), default=levels[-1])
    sc, ttftc, _ = service([r for r in ok if int(r["clients"]) == cl])
    if c == 1 or sc is None:
        sc, ttftc = s1, ttft1
    naive = predict(levels, requests, c, s1, ttft1, s1, ttft1, tokens)
    calib = predict(levels, requests, c, s1, ttft1, sc, ttftc, tokens)
    plot_bench(folder, levels, meas_ttft, meas_tp, naive, calib, c)

    # erro da previsão (|medido − previsto| / medido) nos níveis acima de c
    def err(pred):
        e = [abs(meas_ttft[n][0] - pred[n]["ttft_s"]) / meas_ttft[n][0]
             for n in levels if n > c and meas_ttft[n][0]]
        return statistics.mean(e) if e else None

    knee = next((n for n in levels if meas_ttft[n][0] and meas_ttft[levels[0]][0]
                 and meas_ttft[n][0] > 2 * meas_ttft[levels[0]][0]), None)
    return {"file": raw_path.name, "levels": levels, "repeats": len(reps), "requests_per_client": requests,
            "c": c, "params": {"S1_s": s1, "TTFT1_s": ttft1, "Sc_s": sc, "TTFTc_s": ttftc,
                               "tokens_per_request": tokens, "slowdown_c": (sc / s1) if s1 else None,
                               "throughput_cap_tok_s": c * tokens / sc if sc else None},
            "table": table, "pred_naive": naive, "pred_calib": calib,
            "err_naive": err(naive), "err_calib": err(calib), "knee_clients": knee}


def analyse_rooms(folder: Path, games_log: Path | None, c: int) -> dict | None:
    raw_path = latest(folder, "rooms-*-raw.csv")
    summ_path = latest(folder, "rooms-*-summary.csv")
    if raw_path is None:
        return None
    codes = defaultdict(set)
    for r in read_csv(raw_path):
        if r.get("room"):
            codes[int(r["rooms"])].add(r["room"])
    games = {}
    if games_log and games_log.exists():
        for line in open(games_log, encoding="utf-8"):
            try:
                g = json.loads(line)
            except json.JSONDecodeError:
                continue
            games.setdefault(g.get("room"), []).append(g)  # códigos podem repetir entre execuções
    summ = read_csv(summ_path) if summ_path else []
    levels = sorted({int(s["rooms"]) for s in summ} | set(codes))
    rows = []
    for k in levels:
        msgs = []
        for code in codes.get(k, ()):
            for g in games.get(code, [])[-1:]:   # a partida mais recente com esse código
                msgs += g.get("ai_messages", [])
        msgs = [m for m in msgs if m.get("gen_ms") is not None]
        first = [m for m in msgs if m.get("trigger") == "rodada"]

        def frac(ms_):
            return wilson(sum(1 for m in ms_ if (m.get("overrun_ms") or 0) > 0), len(ms_))

        def med(key, ms_=msgs):
            v = [m[key] for m in ms_ if m.get(key) is not None]
            return statistics.median(v) / 1000 if v else None

        q = med("queue_wait_ms")
        tt = med("ttft_ms")
        gen = med("gen_ms")
        budgets = [(m.get("read_ms") or 0) + (m.get("typing_target_ms") or 0) for m in msgs
                   if m.get("typing_target_ms") is not None]
        sm = [s for s in summ if int(s["rooms"]) == k]
        if not msgs and sm:  # sem o log das partidas: usa o resumo do bots.py
            over = sum(int(num(s["overruns"]) or 0) for s in sm)
            tot = sum(int(num(s["ai_msgs"]) or 0) for s in sm)
            fa = wilson(over, tot)
        else:
            fa = frac(msgs)
        rows.append({
            "rooms": k, "games": sum(len(games.get(cd, [])[-1:]) for cd in codes.get(k, ())),
            "ai_msgs": len(msgs) or sum(int(num(s["ai_msgs"]) or 0) for s in sm),
            "over_all": fa, "over_first": frac(first), "n_first": len(first),
            "med_queue_s": q, "med_prefill_s": (tt - (q or 0)) if tt is not None else None,
            "med_gen_s": (gen - tt) if gen is not None and tt is not None else None,
            "med_total_s": gen, "p95_total_s": (pct([m["gen_ms"] for m in msgs], .95) or 0) / 1000 if msgs else None,
            "med_budget_s": statistics.median(budgets) / 1000 if budgets else None,
            "ttft_p50_ms": statistics.mean([num(s["ttft_p50_ms"]) for s in sm if num(s["ttft_p50_ms"])])
            if any(num(s["ttft_p50_ms"]) for s in sm) else None,
            "winner_ai": sum(int(num(s["winner_ai"]) or 0) for s in sm),
        })
    # previsão da rajada: k IAs pedem ao mesmo tempo no início da rodada; a i-ésima
    # fica pronta em (⌊i/c⌋ + 1)·S; estoura se isso passar do orçamento B.
    base = rows[0] if rows else None
    s_game = base["med_total_s"] if base else None
    b = base["med_budget_s"] if base else None
    if s_game and b:
        for r in rows:
            k = r["rooms"]
            r["pred_first"] = sum(1 for i in range(k) if (i // c + 1) * s_game > b) / k
        k_star = c * b / s_game
    else:
        k_star = None
    plot_rooms(folder, rows, b)
    return {"file": raw_path.name, "rows": rows, "S_game_s": s_game, "budget_s": b, "k_star": k_star,
            "games_log": str(games_log) if games_log else None}


def analyse_fila(folder: Path) -> dict | None:
    path = latest(folder, "desempenho-fila-*-summary.csv")
    if path is None:
        return None
    rows = read_csv(path)
    out = []
    for pol in dict.fromkeys(r["queue_policy"] for r in rows):
        inter = [r for r in rows if r["queue_policy"] == pol and r["classe"] == "interativa"]
        lote = [r for r in rows if r["queue_policy"] == pol and r["classe"] == "lote"]
        out.append({
            "policy": pol, "reps": len(inter),
            "overrun": mean_ci([num(r["overrun_frac"]) for r in inter]),
            "inter_e2e_p50": mean_ci([num(r["e2e_p50_ms"]) for r in inter]),
            "inter_e2e_p95": mean_ci([num(r["e2e_p95_ms"]) for r in inter]),
            "inter_queue_p50": mean_ci([num(r["queue_wait_p50_ms"]) for r in inter]),
            "lote_e2e_p50": mean_ci([num(r["e2e_p50_ms"]) for r in lote]),
            "lote_tok_s": mean_ci([num(r["tokens_per_s"]) for r in lote]),
        })
    plot_fila(folder, out)
    return {"file": path.name, "table": out}


# --------------------------------------------------------------------------- resumo
def f(v, nd=2, scale=1.0):
    return "—" if v is None else f"{v * scale:.{nd}f}"


def fci(pair, nd=2, scale=1.0):
    m, h = pair if isinstance(pair, tuple) else (pair, None)
    if m is None:
        return "—"
    return f"{m * scale:.{nd}f}" + (f" ± {h * scale:.{nd}f}" if h is not None else "")


def write_summary(folder: Path, env: dict, bench, rooms, fila) -> None:
    L = [f"# Resumo do experimento — {folder.name}", ""]
    if env:
        L += ["## Ambiente", "", "```json", json.dumps(env, ensure_ascii=False, indent=2)[:3000], "```", ""]
    if bench:
        p = bench["params"]
        L += ["## 1. Clientes simultâneos × TTFT e vazão (experimento obrigatório)", "",
              f"Arquivo: `{bench['file']}` · {bench['repeats']} repetições · {bench['requests_per_client']} "
              f"pedidos por cliente · c = {bench['c']} vagas no nó.", "",
              "Parâmetros medidos do modelo de fila:", "",
              f"- S com 1 cliente: **{f(p['S1_s'])} s**; TTFT de serviço: {f(p['TTFT1_s'])} s; "
              f"{f(p['tokens_per_request'], 1)} tokens por resposta",
              f"- S com {bench['c']} em paralelo: **{f(p['Sc_s'])} s** (×{f(p['slowdown_c'])}); "
              f"TTFT de serviço: {f(p['TTFTc_s'])} s",
              f"- teto de vazão previsto: c·tokens/S = **{f(p['throughput_cap_tok_s'], 1)} tokens/s**", "",
              "| N | TTFT médio (s) | previsto ingênuo | previsto calibrado | TTFT p95 (s) | espera na fila (s) "
              "| vazão (tok/s) | prevista | falhas |",
              "|---|---|---|---|---|---|---|---|---|"]
        for row in bench["table"]:
            n = row["clients"]
            L.append(f"| {n} | {fci(row['ttft_mean'])} | {f(bench['pred_naive'][n]['ttft_s'])} | "
                     f"{f(bench['pred_calib'][n]['ttft_s'])} | {fci(row['ttft_p95'])} | {fci(row['queue_mean'])} | "
                     f"{fci(row['throughput'], 1)} | {f(bench['pred_calib'][n]['throughput_tok_s'], 1)} | "
                     f"{row['failed']} |")
        L += ["", f"Erro médio da previsão para N > c: ingênua **{f(bench['err_naive'], 0, 100)} %**, "
                  f"calibrada **{f(bench['err_calib'], 0, 100)} %**. "
                  f"TTFT passa do dobro do valor com 1 cliente a partir de N = **{bench['knee_clients']}**.", "",
              "Figuras: `fig-ttft.png`, `fig-vazao.png`.", ""]
    if rooms:
        L += ["## 2. Versão jogo: salas simultâneas × estouro do orçamento de digitação", "",
              f"Arquivo: `{rooms['file']}` · log das partidas: `{rooms['games_log']}`.", "",
              f"Com 1 sala, a fala da IA leva **{f(rooms['S_game_s'])} s** (mediana, do gatilho ao texto pronto) "
              f"e o orçamento mediano é **{f(rooms['budget_s'])} s**. Previsão para a rajada do início da "
              f"rodada: começa a estourar com k* ≈ c·B/S = **{f(rooms['k_star'], 1)} salas**.", "",
              "| salas | falas da IA | estouram (todas) | estouram (1ª da rodada) | previsto (1ª) | fila (s) "
              "| prefill (s) | geração (s) | total p50 / p95 (s) | orçamento (s) | IA venceu |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rooms["rows"]:
            oa, of = r["over_all"], r["over_first"]
            L.append(
                f"| {r['rooms']} | {r['ai_msgs']} | {f(oa[0], 0, 100)} % [{f(oa[1], 0, 100)}–{f(oa[2], 0, 100)}] | "
                f"{f(of[0], 0, 100)} % (n={r['n_first']}) | {f(r.get('pred_first'), 0, 100)} % | "
                f"{f(r['med_queue_s'])} | {f(r['med_prefill_s'])} | {f(r['med_gen_s'])} | "
                f"{f(r['med_total_s'])} / {f(r['p95_total_s'])} | {f(r['med_budget_s'])} | {r['winner_ai']} |")
        L += ["", "Figuras: `fig-salas.png`, `fig-tempo-fala.png`.", ""]
    if fila:
        L += ["## 3. Política de fila: fifo × priority", "", f"Arquivo: `{fila['file']}`.", "",
              "| política | reps | falas da IA que estouram | fala da IA e2e p50 / p95 (s) | espera da fala (s) "
              "| lote e2e p50 (s) | vazão do lote (tok/s) |",
              "|---|---|---|---|---|---|---|"]
        for t in fila["table"]:
            L.append(f"| {t['policy']} | {t['reps']} | {fci(t['overrun'], 0, 100)} % | "
                     f"{fci(t['inter_e2e_p50'], 2, 1e-3)} / {fci(t['inter_e2e_p95'], 2, 1e-3)} | "
                     f"{fci(t['inter_queue_p50'], 2, 1e-3)} | {fci(t['lote_e2e_p50'], 2, 1e-3)} | "
                     f"{fci(t['lote_tok_s'], 1)} |")
        L += ["", "Figura: `fig-fila.png`.", ""]
    (folder / "resumo.md").write_text("\n".join(L), encoding="utf-8")

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, float):
            return round(o, 4)
        return o
    (folder / "resumo.json").write_text(json.dumps(clean({"ambiente": env, "bench": bench, "salas": rooms,
                                                          "fila": fila}), ensure_ascii=False, indent=2),
                                        encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path, help="pasta do experimento (results/experimento-...)")
    p.add_argument("--c", type=int, help="vagas do nó (padrão: lido de ambiente.json)")
    p.add_argument("--games-log", type=Path, help="log das partidas (padrão: o de ambiente.json ou logs/games.jsonl)")
    a = p.parse_args()

    env_path = a.folder / "ambiente.json"
    env = json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else {}
    c = a.c or env.get("c") or 2
    games_log = a.games_log or (Path(env["games_log"]) if env.get("games_log") else Path("logs/games.jsonl"))

    bench = analyse_bench(a.folder, c, None)
    rooms = analyse_rooms(a.folder, games_log, c)
    fila = analyse_fila(a.folder)
    write_summary(a.folder, env, bench, rooms, fila)
    print((a.folder / "resumo.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
