"""Cliente de linha de comando do Ágora.

Pergunta avulsa:
    python client/chat.py -d redes "O que é o three-way handshake?"
Modo conversa (mantém o histórico; Ctrl+C ou linha vazia para sair):
    python client/chat.py -d programacao
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(__file__))
from sse_client import ask  # noqa: E402

DEFAULT_URL = os.environ.get("AGORA_URL", "http://localhost:8000")


def _print_token(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def _print_stats(res) -> None:
    if res.status != "ok":
        print(f"\n[erro] {res.status}: {res.error}", file=sys.stderr)
        return
    s = res.server
    tps = f"{res.tps:.1f}" if res.tps else "-"
    print(
        f"\n\n[{s.get('node')} · {s.get('model')}] "
        f"TTFT {res.ttft_ms:.0f} ms · {res.tokens} tokens · {tps} tok/s · "
        f"total {res.e2e_ms:.0f} ms · fila {s.get('queue_wait_ms', 0):.0f} ms",
        file=sys.stderr,
    )


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prompt", nargs="?", help="pergunta (sem ela, entra no modo conversa)")
    p.add_argument("-d", "--domain", default="redes")
    p.add_argument("-u", "--url", default=DEFAULT_URL)
    p.add_argument("-n", "--max-tokens", type=int)
    a = p.parse_args()

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=600.0)) as client:
        try:
            domains = (await client.get(f"{a.url}/v1/domains")).json()
        except httpx.HTTPError as exc:
            sys.exit(f"gateway inacessível em {a.url}: {exc}")
        if a.domain not in domains:
            sys.exit(f"domínio '{a.domain}' não existe. Opções: {', '.join(domains)}")

        def payload(messages):
            body = {"domain": a.domain, "messages": messages}
            if a.max_tokens:
                body["max_tokens"] = a.max_tokens
            return body

        if a.prompt:
            res = await ask(client, a.url, payload([{"role": "user", "content": a.prompt}]), _print_token)
            _print_stats(res)
            return

        print(f"Conversando com o domínio '{a.domain}' ({domains[a.domain]['description']}).")
        history: list[dict[str, str]] = []
        while True:
            try:
                question = input("\nvocê> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                break
            history.append({"role": "user", "content": question})
            print("ágora> ", end="")
            res = await ask(client, a.url, payload(history), _print_token)
            _print_stats(res)
            if res.status == "ok":
                history.append({"role": "assistant", "content": res.text})
            else:
                history.pop()


if __name__ == "__main__":
    asyncio.run(main())
