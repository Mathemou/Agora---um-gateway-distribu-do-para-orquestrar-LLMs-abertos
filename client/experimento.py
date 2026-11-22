"""Roda o experimento inteiro com um comando e gera a análise.

Fases (na ordem; dá para escolher com --fases):

  bench  experimento obrigatório: 1, 2, 4, 8, 16 clientes simultâneos × TTFT e vazão
  salas  versão jogo: 1, 2, 4, 8 salas com bots, medindo quantas falas da IA estouram
         o orçamento de digitação
  fila   fifo × priority: falas curtas da IA disputando o nó com pedidos longos

Antes de começar, grava o ambiente (CPU, versões, nós, política) em ambiente.json;
no fim, chama client/analise_experimento.py, que faz os gráficos e o resumo.md.

Com o Ollama de verdade (docker compose up -d, gateway em http://localhost:8000):

    python client/experimento.py                 # tudo, ~1 h com qwen2.5:3b em CPU
    python client/experimento.py --fases bench   # só o obrigatório (~25 min)
    python client/experimento.py --rapido        # ensaio curto (~5 min) para conferir

Saída: results/experimento-<data>/ (CSVs, PNGs, resumo.md, log de cada fase).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
CLIENT = ROOT / "client"
sys.path.insert(0, str(CLIENT))
from sse_client import auth_headers  # noqa: E402


def get(client: httpx.Client, url: str, path: str):
    try:
        r = client.get(url + path, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001
        return {"erro": f"{type(exc).__name__}: {exc}"}


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def ollama_version(client: httpx.Client, urls: list[str]) -> dict:
    out = {}
    for u in urls:
        try:
            out[u] = client.get(u.rstrip("/") + "/api/version", timeout=3).json().get("version")
        except Exception:  # noqa: BLE001
            pass
    return out


def run(cmd: list[str], log_path: Path) -> int:
    """Roda uma fase mostrando a saída e guardando uma cópia em log-<fase>.txt."""
    print("$ " + " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            print("  " + line, end="", flush=True)
            log.write(line)
        return proc.wait()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-u", "--url", default=os.environ.get("AGORA_URL", "http://localhost:8000"))
    p.add_argument("-d", "--domain", default="impostor", help="domínio medido (o da IA local)")
    p.add_argument("--fases", nargs="+", default=["bench", "salas", "fila"], choices=["bench", "salas", "fila"])
    p.add_argument("--clients", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--requests", type=int, default=2, help="pedidos seguidos por cliente (bench)")
    p.add_argument("--repeat", type=int, default=5, help="repetições de cada nível (bench)")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--rooms", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--rooms-repeat", type=int, default=2)
    p.add_argument("--discussion", type=float, default=90)
    p.add_argument("--talk", type=float, default=60)
    p.add_argument("--fila-repeat", type=int, default=3)
    p.add_argument("--fila-duration", type=float, default=60)
    p.add_argument("--games-log", default=None,
                   help="log das partidas do gateway (padrão: logs/games.jsonl, o do docker compose)")
    p.add_argument("--rapido", action="store_true", help="ensaio curto para conferir que tudo roda")
    p.add_argument("-o", "--out", default="results")
    a = p.parse_args()
    if a.rapido:
        a.clients, a.repeat, a.requests = [1, 2, 4], 2, 1
        a.rooms, a.rooms_repeat, a.discussion, a.talk = [1, 2], 1, 40, 25
        a.fila_repeat, a.fila_duration = 1, 20

    folder = ROOT / a.out / f"experimento-{datetime.now():%Y%m%d-%H%M%S}"
    folder.mkdir(parents=True, exist_ok=True)

    with httpx.Client() as client:
        health = get(client, a.url, "/health")
        if "erro" in health:
            sys.exit(f"o gateway não respondeu em {a.url}/health ({health['erro']}). "
                     "Suba com `docker compose up -d` e tente de novo.")
        domains = get(client, a.url, "/v1/domains")
        if a.domain not in domains:
            sys.exit(f"domínio '{a.domain}' não existe. Opções: {', '.join(domains)}")
        dom = domains[a.domain]
        nodes = get(client, a.url, "/v1/nodes")
        node_info = {n["node"]: n for n in nodes} if isinstance(nodes, list) else {}
        pool = dom.get("nodes") or [dom["node"]]
        down = [n for n in pool if not health.get("nodes", {}).get(n, {}).get("up")]
        if down:
            sys.exit(f"nó(s) fora do ar: {', '.join(down)}. Veja {a.url}/health")
        if len(pool) > 1:
            print(f"aviso: o domínio {a.domain} tem {len(pool)} nós ({', '.join(pool)}); o modelo de fila "
                  "do experimento supõe UM nó. Para o obrigatório, use IMPOSTOR_NODES=node-impostor.")
        c = sum(int(node_info.get(n, {}).get("max_concurrency", 1)) for n in pool)
        games_log = a.games_log or str(ROOT / "logs" / "games.jsonl")
        env = {
            "data": datetime.now().isoformat(timespec="seconds"),
            "commit": git_commit(),
            "url": a.url, "domain": a.domain, "model": dom.get("model"), "nodes": pool, "c": c,
            "node_snapshots": [node_info.get(n) for n in pool],
            "queue_policy_antes": health.get("queue_policy"), "balance": health.get("balance"),
            "maquina": {"sistema": platform.platform(), "cpu": platform.processor() or platform.machine(),
                        "nucleos_logicos": os.cpu_count(), "python": platform.python_version()},
            "ollama": ollama_version(client, ["http://localhost:21434", "http://localhost:11434"]),
            "parametros": {k: v for k, v in vars(a).items() if k not in ("url",)},
            "games_log": games_log,
        }
        (folder / "ambiente.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"experimento em {folder}\nmodelo {env['model']} · nó(s) {', '.join(pool)} · c = {c} vagas\n")

        # bench e salas medem com a fila em ordem de chegada (a fase "fila" compara as duas)
        try:
            client.post(a.url + "/v1/balancer", json={"queue_policy": "fifo", "reset": True},
                        headers=auth_headers(), timeout=10).raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"aviso: não deu para fixar a fila em fifo ({exc})")

        # aquecimento: carrega o modelo na memória (o 1º pedido inclui o load e não entra na conta)
        print("aquecendo o modelo…", flush=True)
        t0 = time.perf_counter()
        try:
            with client.stream("POST", a.url + "/v1/chat", headers=auth_headers(),
                               json={"domain": a.domain, "prompt": "oi", "max_tokens": 8},
                               timeout=httpx.Timeout(10, read=600)) as r:
                for _ in r.iter_lines():
                    pass
        except Exception as exc:  # noqa: BLE001
            print(f"aviso: aquecimento falhou ({exc})")
        print(f"  pronto em {time.perf_counter() - t0:.1f} s\n")

    py = sys.executable
    t_all = time.perf_counter()
    if "bench" in a.fases:
        print("== fase 1/3: clientes simultâneos (experimento obrigatório)")
        run([py, str(CLIENT / "bench.py"), "-u", a.url, "-d", a.domain, "-c", *map(str, a.clients),
             "-r", str(a.requests), "--repeat", str(a.repeat), "-n", str(a.max_tokens), "-o", str(folder)],
            folder / "log-bench.txt")
    if "salas" in a.fases:
        print("\n== fase 2/3: salas simultâneas (versão jogo)")
        run([py, str(CLIENT / "bots.py"), "-u", a.url, "--rooms", *map(str, a.rooms), "--bots", "2",
             "--discussion", str(a.discussion), "--talk", str(a.talk), "--repeat", str(a.rooms_repeat),
             "--impostor-domain", a.domain, "-o", str(folder)], folder / "log-salas.txt")
    if "fila" in a.fases:
        print("\n== fase 3/3: fifo × priority")
        # pedidos longos de API no MESMO nó da IA (domínio "assistente"), se existir
        extra = ["--batch-domain", "assistente"] if "assistente" in domains and a.domain != "assistente" else []
        run([py, str(CLIENT / "exp_desempenho.py"), "-u", a.url, "-o", str(folder), "fila", "-d", a.domain,
             *extra, "--repeat", str(a.fila_repeat), "--duration", str(a.fila_duration)], folder / "log-fila.txt")
    print(f"\nfases concluídas em {(time.perf_counter() - t_all) / 60:.1f} min\n")

    run([py, str(CLIENT / "analise_experimento.py"), str(folder), "--games-log", games_log],
        folder / "log-analise.txt")
    print(f"\nResultados em {folder}\n  resumo.md e fig-*.png são o que entra no relatório.")


if __name__ == "__main__":
    main()
