"""Nó de inferência FALSO que imita a API de streaming do Ollama.

Serve para testes automáticos e para validar o modelo de fila sem gastar CPU:
o tempo de serviço é determinístico (prefill + N tokens × atraso), o que deixa
fácil comparar a previsão teórica com a medição.

    python -m agora.fake_node --port 11435 --tokens 40 --token-ms 25 --parallel 2

Para a trilha Desempenho, o nó pode ser mais lento (`--slow 3` = 3× mais lento) e
pode ter defeitos ligados/desligados em tempo de execução, sem reiniciar:

    curl -X POST localhost:11436/_fault -d '{"mode": "die", "after_tokens": 5}'
    curl -X POST localhost:11436/_fault -d '{"mode": "none"}'

  none     funciona normal
  refuse   responde HTTP 503 em tudo (nó sobrecarregado / fora do ar "educado")
  kill     o nó "cai": as respostas em andamento são cortadas depois de `after_tokens`
           tokens e tudo o que chegar depois é recusado (503), inclusive o health
           check, até voltar com {"mode": "none"}. É o cenário principal do failover.
  die      corta a conexão no meio da resposta, depois de `after_tokens` tokens, mas
           continua aceitando pedidos e respondendo ao health check ("falha cinza")
  hang     para de mandar tokens no meio da resposta (nó travado) e não fecha
  crash    o PROCESSO morre no meio da resposta (como um `docker kill`); só na CLI
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Palavras de "bate-papo" (o fake também serve para jogar o Jogo da Imitação sem LLM).
WORDS = (
    "kkk sei la mano acho que eu tava dormindo ontem nessa hora . "
    "o azul ta muito quieto pra mim ein , suspeito demais . "
    "eu comeria pizza todo dia fácil , de calabresa . "
    "tô no ônibus agora indo pra ufes e a internet tá horrível . "
    "nada a ver isso , eu sou humano até demais . "
    "quem falou primeiro foi o vermelho , não fui eu ."
).split()


def create_fake_app(
    tokens: int = 20,
    token_ms: float = 10.0,
    prefill_ms: float = 50.0,
    parallel: int = 1,
    models: tuple[str, ...] = ("qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5-coder:0.5b"),
    slow: float = 1.0,
    fault: str = "none",
    fault_after: int = 5,
) -> FastAPI:
    app = FastAPI(title="Ollama falso")
    slots = asyncio.Semaphore(parallel)  # como o OLLAMA_NUM_PARALLEL
    state = {"mode": fault, "after_tokens": fault_after, "served": 0, "slow": slow}
    app.state.fault = state

    @app.get("/_fault")
    async def get_fault():
        return state

    @app.post("/_fault")
    async def set_fault(request: Request):
        body = await request.json()
        if body.get("mode") not in (None, "none", "refuse", "kill", "die", "hang", "crash"):
            return JSONResponse({"error": "modo inválido"}, status_code=400)
        for k in ("mode", "after_tokens", "slow"):
            if k in body:
                state[k] = body[k]
        return state

    @app.get("/api/tags")
    async def tags():
        if state["mode"] in ("refuse", "kill"):
            return JSONResponse({"error": "nó indisponível"}, status_code=503)
        return {"models": [{"name": m} for m in models]}

    @app.post("/api/chat")
    async def chat(request: Request):
        if state["mode"] in ("refuse", "kill"):
            return JSONResponse({"error": "nó indisponível"}, status_code=503)
        body = await request.json()
        model = body.get("model", "")
        if model not in models:
            return JSONResponse({"error": f"model '{model}' not found"}, status_code=404)
        n = int(body.get("options", {}).get("num_predict") or tokens)

        start = random.randrange(len(WORDS))

        async def gen():
            async with slots:
                state["served"] += 1
                k = float(state["slow"])
                t0 = time.perf_counter_ns()
                await asyncio.sleep(prefill_ms * k / 1000)
                t1 = time.perf_counter_ns()
                for i in range(n):
                    mode = state["mode"]
                    if i >= int(state["after_tokens"]) and mode in ("kill", "die", "hang", "crash"):
                        if mode == "crash":
                            os._exit(1)  # noqa: SLF001 — o processo morre de verdade
                        if mode == "hang":
                            await asyncio.sleep(3600)
                        raise ConnectionResetError("nó falso: conexão cortada no meio da resposta")
                    await asyncio.sleep(token_ms * k / 1000)
                    word = WORDS[(start + i) % len(WORDS)]
                    msg = {"model": model, "message": {"role": "assistant", "content": word + " "}, "done": False}
                    yield json.dumps(msg, ensure_ascii=False) + "\n"
                t2 = time.perf_counter_ns()
                yield json.dumps({
                    "model": model, "message": {"role": "assistant", "content": ""}, "done": True,
                    "total_duration": t2 - t0, "load_duration": 0,
                    "prompt_eval_count": 10, "prompt_eval_duration": t1 - t0,
                    "eval_count": n, "eval_duration": t2 - t1,
                }) + "\n"

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    return app


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=11435)
    p.add_argument("--tokens", type=int, default=40, help="tokens por resposta (padrão)")
    p.add_argument("--token-ms", type=float, default=25.0, help="atraso entre tokens")
    p.add_argument("--prefill-ms", type=float, default=200.0, help="atraso até o 1º token")
    p.add_argument("--parallel", type=int, default=1, help="requisições simultâneas")
    p.add_argument("--slow", type=float, default=1.0, help="fator de lentidão (2 = 2× mais lento)")
    p.add_argument("--fault", default="none", choices=["none", "refuse", "kill", "die", "hang", "crash"],
                   help="defeito inicial (dá para trocar depois com POST /_fault)")
    p.add_argument("--fault-after", type=int, default=5, help="tokens antes do defeito")
    a = p.parse_args()
    app = create_fake_app(a.tokens, a.token_ms, a.prefill_ms, a.parallel,
                          slow=a.slow, fault=a.fault, fault_after=a.fault_after)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
