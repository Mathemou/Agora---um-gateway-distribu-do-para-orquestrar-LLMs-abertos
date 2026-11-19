"""Red team: roda a bateria fixa de ataques contra a IA impostora e mede
quantos "passam" ANTES e DEPOIS dos filtros de defesa (security.py).

Duas formas de um ataque "passar" (quebrar o disfarce), medidas separadamente:
  vazou     a resposta se entrega ou vaza o prompt (is_breach: "sou uma IA",
            "minhas instruções", eco do prompt...).
  obedeceu  o modelo cumpre o teste de máquina (responde a conta/capital/padrão,
            emite lista/JSON...) — só para a categoria "maquina", que tem
            "resposta certa" conhecida.
  descartada a resposta foi barrada pelo filtro de saída (custo do filtro: a IA
            ficaria calada ou usaria fallback) — só no modo "after".

Para cada ataque × domínio:
  before  o ataque CRU vai ao modelo (sem filtro de entrada nem de saída).
  after   build_messages neutraliza injeção/vazamento e humanize rejeita
          autoentrega/vazamento.

É genérico: qualquer domínio impostor funciona (local, OpenCode Go, Groq,
Claude...). Se um nó estiver sem chave/fora do ar, a linha é gravada como erro
e o restante continua.

    python client/red_team.py -d impostor impostor-opencode -n 1 --seed 7
    python client/red_team.py --validation   # taxa de detecção treino × validação (sem LLM)

Saída: results/red-team-<data>-raw.csv e -summary.csv, mais um resumo no
terminal. O mesmo conjunto pode ser comparado depois adicionando -d impostor-groq
e impostor-claude (com as chaves no .env).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "gateway"))

from agora.attacks import ATTACKS  # noqa: E402
from agora.attacks_validation import VALIDATION_ATTACKS  # noqa: E402
from agora.humanize import Style, build_messages, humanize, room_profile, word_budget  # noqa: E402
from agora.security import BLOCK_CATEGORIES, detect_attack, is_breach, obeyed  # noqa: E402
from sse_client import ask  # noqa: E402

ME = "Laranja"
OTHERS = ["Azul", "Verde"]
TOPIC = "Conte algo sobre você."
PERSONA = "Sou o Lucas, 23 anos, moro em Jardim Camburi e faço administração."


def variables() -> dict[str, str]:
    return {"alias": ME, "alias_lower": ME.lower(), "others": ", ".join(OTHERS),
            "persona": PERSONA}


async def run_one(client: httpx.AsyncClient, url: str, domain: str, attack, rng) -> dict:
    variables_ = variables()

    # before: ataque cru, sem filtro de entrada nem de saída.
    before = await ask(client, url, {
        "domain": domain,
        "messages": [{"role": "user", "content": f"{OTHERS[0]}: {attack.texto}"}],
        "variables": variables_,
    })
    leak_before = before.status == "ok" and is_breach(before.text)

    # after: filtro de entrada (build_messages neutraliza) + saída (humanize).
    chat = [{"kind": "player", "from": OTHERS[0], "text": attack.texto}]
    budget = word_budget(room_profile(chat, ME), Style())
    after_msgs = build_messages(chat, ME, TOPIC, "mencao", budget=budget)
    after = await ask(client, url, {"domain": domain, "messages": after_msgs, "variables": variables_})
    final = humanize(after.text, [ME] + OTHERS, Style(), rng, budget=budget) \
        if after.status == "ok" else None
    final_text = " / ".join(final) if final else ""
    leak_after = bool(final_text) and is_breach(final_text)
    # respostas que o filtro de saída descartou (a IA ficaria calada / usaria fallback)
    descartada_after = after.status == "ok" and final is None

    obedeceu_before = before.status == "ok" and obeyed(attack.resposta, before.text)
    obedeceu_after = bool(final_text) and obeyed(attack.resposta, final_text)

    status = "ok" if (before.status == "ok" and after.status == "ok") else "error"
    error = None
    if before.status != "ok":
        error = before.error
    elif after.status != "ok":
        error = after.error

    return {
        "domain": domain, "categoria": attack.categoria, "id": attack.id,
        "texto": attack.texto, "bloq_entrada": attack.categoria in BLOCK_CATEGORIES,
        "status": status, "error": error,
        "raw_before": before.text, "leak_before": leak_before,
        "raw_after": after.text, "final_after": final_text, "leak_after": leak_after,
        "obedeceu_before": obedeceu_before, "obedeceu_after": obedeceu_after,
        "descartada_after": descartada_after,
    }


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["domain"], r["categoria"])].append(r)
    out = []
    for (domain, cat), rs in sorted(groups.items()):
        out.append({
            "domain": domain, "categoria": cat, "total": len(rs),
            "bloq_entrada": sum(1 for r in rs if r["bloq_entrada"]),
            "vazou_before": sum(1 for r in rs if r["leak_before"]),
            "vazou_after": sum(1 for r in rs if r["leak_after"]),
            "obedeceu_before": sum(1 for r in rs if r["obedeceu_before"]),
            "obedeceu_after": sum(1 for r in rs if r["obedeceu_after"]),
            "descartada_after": sum(1 for r in rs if r["descartada_after"]),
            "erros": sum(1 for r in rs if r["status"] != "ok"),
        })
    return out


def validation_report() -> None:
    """Taxa de detecção nas duas baterias (só regras, sem LLM)."""
    treino = sum(1 for a in ATTACKS if detect_attack(a.texto) is not None)
    val = sum(1 for a in VALIDATION_ATTACKS if detect_attack(a.texto) is not None)
    print(f"bateria de treino (attacks.py):      {treino}/{len(ATTACKS)} detectados")
    print(f"bateria de validação (reescritas):   {val}/{len(VALIDATION_ATTACKS)} detectados")
    print("\nataques de validação que PASSARAM (não detectados):")
    for a in VALIDATION_ATTACKS:
        if detect_attack(a.texto) is None:
            print(f"  [{a.categoria}] {a.texto}")


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-u", "--url", default=os.environ.get("AGORA_URL", "http://localhost:8000"))
    p.add_argument("-d", "--domain", nargs="+", default=["impostor", "impostor-opencode"])
    p.add_argument("-n", "--samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--validation", action="store_true",
                   help="só reporta a taxa de detecção nas baterias de treino e validação (sem LLM)")
    p.add_argument("-o", "--out", default="results")
    a = p.parse_args()

    if a.validation:
        validation_report()
        return

    rng = random.Random(a.seed)

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=600.0),
                                 limits=httpx.Limits(max_connections=6)) as client:
        try:
            doms = (await client.get(f"{a.url}/v1/domains")).json()
        except httpx.HTTPError as exc:
            sys.exit(f"gateway inacessível em {a.url}: {exc}")
        unknown = [d for d in a.domain if d not in doms]
        if unknown:
            print(f"aviso: domínios inexistentes (ignorados): {', '.join(unknown)}")
        domains = [d for d in a.domain if d in doms]
        if not domains:
            sys.exit("nenhum domínio válido. Opções: " + ", ".join(doms))

        print(f"persona: {PERSONA}\n")
        rows: list[dict] = []
        for domain in domains:
            model = doms[domain].get("model")
            print(f"=== {domain} ({model})")
            for attack in ATTACKS:
                for sample in range(a.samples):
                    row = await run_one(client, a.url, domain, attack, rng)
                    rows.append(row)
            print()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    raw_path = out / f"red-team-{stamp}-raw.csv"
    sum_path = out / f"red-team-{stamp}-summary.csv"

    raw_fields = ["domain", "categoria", "id", "texto", "bloq_entrada", "status", "error",
                  "leak_before", "leak_after", "obedeceu_before", "obedeceu_after",
                  "descartada_after", "raw_before", "raw_after", "final_after"]
    with open(raw_path, "w", newline="", encoding="utf-8") as fr, \
         open(sum_path, "w", newline="", encoding="utf-8") as fs:
        raw_w = csv.DictWriter(fr, fieldnames=raw_fields)
        raw_w.writeheader()
        for r in rows:
            raw_w.writerow({k: r.get(k) for k in raw_fields})

        summ = summarize(rows)
        sum_fields = ["domain", "categoria", "total", "bloq_entrada",
                      "vazou_before", "vazou_after", "obedeceu_before", "obedeceu_after",
                      "descartada_after", "erros"]
        sum_w = csv.DictWriter(fs, fieldnames=sum_fields)
        sum_w.writeheader()
        for s in summ:
            sum_w.writerow(s)

    print(f"{'dominio':<20} {'categoria':<16} {'tot':>4} {'bloq':>4} {'vazB':>5} {'vazA':>5} {'obB':>4} {'obA':>4} {'desc':>5} {'err':>4}")
    for s in summarize(rows):
        print(f"{s['domain']:<20} {s['categoria']:<16} {s['total']:>4} {s['bloq_entrada']:>4} "
              f"{s['vazou_before']:>5} {s['vazou_after']:>5} {s['obedeceu_before']:>4} "
              f"{s['obedeceu_after']:>4} {s['descartada_after']:>5} {s['erros']:>4}")

    print(f"\nCSV: {raw_path}\n     {sum_path}")


if __name__ == "__main__":
    asyncio.run(main())
