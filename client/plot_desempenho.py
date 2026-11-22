"""Gráficos dos experimentos da trilha Desempenho (para o relatório).

Lê os CSVs *-raw.csv de client/exp_desempenho.py e grava PNGs ao lado deles.
Sem argumentos, usa o CSV mais recente de cada experimento em results/.

    pip install matplotlib
    python client/plot_desempenho.py
    python client/plot_desempenho.py results/desempenho-balance-20261118-212742-raw.csv
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Paleta categórica (ordem fixa) e tinta de texto/eixos.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
POLICY_LABEL = {"round_robin": "round-robin", "least_queue": "menor\nfila",
                "least_latency": "menor\nlatência", "peak_ewma": "latência\n× carga"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def pct(vals, p):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    k = (len(vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def save(fig, path: Path, suffix: str) -> Path:
    out = path.with_name(path.name.replace("-raw.csv", f"-{suffix}.png"))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(out)
    return out


# ---------------------------------------------------------------- balanceamento
def plot_balance(path: Path) -> None:
    rows = [r for r in read(path) if r["status"] == "ok"]
    policies = list(dict.fromkeys(r["policy"] for r in rows))
    per = defaultdict(lambda: defaultdict(list))  # policy → rep → ttfts
    for r in rows:
        per[r["policy"]][r["rep"]].append(num(r["ttft_ms"]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"wspace": 0.35})
    ax = axes[0]
    w = 0.36
    for j, (p, label) in enumerate([(0.5, "p50"), (0.95, "p95")]):
        means, errs = [], []
        for pol in policies:
            vals = [pct(v, p) / 1000 for v in per[pol].values()]
            means.append(statistics.mean(vals))
            errs.append(statistics.stdev(vals) if len(vals) > 1 else 0)
        xs = [i + (j - 0.5) * w for i in range(len(policies))]
        ax.bar(xs, means, w * 0.94, yerr=errs, color=SERIES[j], label=label,
               error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
        for x, m in zip(xs, means):
            ax.annotate(f"{m:.1f}", (x, m), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(policies)), [POLICY_LABEL.get(p, p) for p in policies])
    ax.set_ylabel("TTFT (s)")
    ax.set_title("Tempo até o 1º token por política")
    ax.legend(loc="upper left", ncol=2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.grid(axis="x", visible=False)

    # Quem atendeu: requisições por nó (mostra a "manada" do least_latency).
    ax = axes[1]
    nodes = sorted({r["node"] for r in rows})
    counts = {pol: Counter(r["node"] for r in rows if r["policy"] == pol) for pol in policies}
    left = [0.0] * len(policies)
    for k, node in enumerate(nodes):
        vals = [100 * counts[pol][node] / max(1, sum(counts[pol].values())) for pol in policies]
        ax.barh(range(len(policies)), vals, left=left, color=SERIES[k % len(SERIES)],
                edgecolor=SURFACE, linewidth=2, label=node, height=0.6)
        for i, (lft, v) in enumerate(zip(left, vals)):
            if v >= 9:
                ax.text(lft + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8, color=INK)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(range(len(policies)), [POLICY_LABEL.get(p, p).replace("\n", " ") for p in policies])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% das requisições")
    ax.set_title("Para onde cada política mandou as requisições")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(nodes))
    ax.grid(axis="y", visible=False)
    save(fig, path, "grafico")


# ---------------------------------------------------------------- failover
def plot_failover(path: Path) -> None:
    rows = read(path)
    sum_path = path.with_name(path.name.replace("-raw", "-summary"))
    fail_at = None
    if sum_path.exists():
        fail_at = num(read(sum_path)[0].get("fail_at_s"))
    modes = list(dict.fromkeys(r["failover"] for r in rows))
    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 4), sharey=True, squeeze=False)
    for ax, mode in zip(axes[0], modes):
        rs = [r for r in rows if r["failover"] == mode and r["rep"] == "0"]
        groups = [
            ("ok", [r for r in rs if r["status"] == "ok" and not int(r["failovers"] or 0)], SERIES[0], "o"),
            ("refeita em outro nó", [r for r in rs if r["status"] == "ok" and int(r["failovers"] or 0)],
             SERIES[1], "^"),
            ("falhou", [r for r in rs if r["status"] != "ok"], INK, "x"),
        ]
        for label, g, color, marker in groups:
            if g:
                ax.scatter([num(r["t_end_s"]) for r in g], [num(r["e2e_ms"]) / 1000 for r in g],
                           s=36, color=color, marker=marker, label=f"{label} ({len(g)})",
                           edgecolors=SURFACE if marker != "x" else None, linewidths=1.2 if marker != "x" else 1.6)
        if fail_at is not None:
            ax.axvline(fail_at, color=INK2, linestyle="--", linewidth=1)
            ax.annotate("nó cai", (fail_at, 1), xycoords=("data", "axes fraction"), xytext=(4, 2),
                        textcoords="offset points", fontsize=8, color=INK2, va="bottom")
        ax.set_title(f"Failover {'ligado' if mode == 'on' else 'desligado'}", pad=14)
        ax.set_xlabel("fim da requisição (s)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=8)
    axes[0][0].set_ylabel("latência fim a fim (s)")
    save(fig, path, "grafico")


# ---------------------------------------------------------------- fila
def plot_fila(path: Path) -> None:
    rows = [r for r in read(path) if r["status"] == "ok"]
    policies = list(dict.fromkeys(r["queue_policy"] for r in rows))
    classes = ["interativa", "lote"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.34
    for j, classe in enumerate(classes):
        data = [[num(r["e2e_ms"]) / 1000 for r in rows if r["queue_policy"] == pol and r["classe"] == classe]
                for pol in policies]
        pos = [i + (j - 0.5) * w * 1.15 for i in range(len(policies))]
        bp = ax.boxplot(data, positions=pos, widths=w, patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "linewidth": 1.5},
                        whiskerprops={"color": INK2}, capprops={"color": INK2})
        for box in bp["boxes"]:
            box.set(facecolor=SERIES[j], edgecolor=SURFACE, linewidth=1.5)
        ax.plot([], [], color=SERIES[j], linewidth=8, label=f"{classe}")
        for x, d in zip(pos, data):
            if d:
                med = statistics.median(d)
                ax.annotate(f"{med:.1f} s", (x, med), xytext=(0, -12), textcoords="offset points",
                            ha="center", fontsize=8, color=INK2)
    over = {}
    for pol in policies:
        rs = [r for r in rows if r["queue_policy"] == pol and r["classe"] == "interativa"]
        if rs:
            over[pol] = sum(1 for r in rs if (num(r["overrun_ms"]) or 0) > 0) / len(rs)
    labels = [f"{pol}\n(estouro do orçamento: {over.get(pol, 0):.0%})" for pol in policies]
    ax.set_xticks(range(len(policies)), labels)
    ax.set_ylabel("latência fim a fim (s)")
    ax.set_title("Fala da IA × pedidos em lote na mesma fila")
    ax.legend(loc="lower left")
    ax.grid(axis="x", visible=False)
    save(fig, path, "grafico")


PLOTS = {"balance": plot_balance, "failover": plot_failover, "fila": plot_fila}


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        res = Path("results")
        for exp in PLOTS:
            found = sorted(res.glob(f"desempenho-{exp}-*-raw.csv"))
            if found:
                paths.append(found[-1])
    if not paths:
        sys.exit("nenhum CSV de desempenho em results/ (rode client/exp_desempenho.py antes)")
    for p in paths:
        exp = p.name.split("-")[1]
        PLOTS[exp](p)


if __name__ == "__main__":
    main()
