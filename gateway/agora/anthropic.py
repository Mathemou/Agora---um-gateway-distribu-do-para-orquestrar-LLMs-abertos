"""Nó de inferência na nuvem: a API de Mensagens do Claude (Anthropic).

Traduz o stream SSE da API para o mesmo formato de pedaços que o cliente do
Ollama produz (`message.content` por token e um pedaço final `done: true` com
contagens), então o resto do gateway (fila, métricas, jogo) não muda nada.

Diferença de Redes que vale medir: o nó local fica a ~1 ms na LAN; este fica do
outro lado da internet, com TLS, então o TTFT inclui RTT de WAN + fila do provedor.

A chave vem da variável de ambiente indicada em `api_key_env` (padrão
ANTHROPIC_API_KEY) e nunca é gravada em log.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .ollama import NodeError

API_VERSION = "2023-06-01"


def _headers(api_key_env: str) -> dict[str, str]:
    key = os.environ.get(api_key_env, "").strip()
    if not key:
        raise NodeError(f"defina a variável {api_key_env} com a sua chave da API do Claude")
    return {"x-api-key": key, "anthropic-version": API_VERSION, "content-type": "application/json"}


def _payload(model: str, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    convo = [m for m in messages if m["role"] in ("user", "assistant")]
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": int(options.get("num_predict") or options.get("max_tokens") or 256),
        "messages": convo,
        "stream": True,
    }
    if system:
        body["system"] = system
    # Só a temperatura é repassada (0 a 1 na API); top_p/repeat_penalty são do Ollama.
    if options.get("temperature") is not None:
        body["temperature"] = max(0.0, min(1.0, float(options["temperature"])))
    stops = [s for s in options.get("stop") or [] if s.strip()]  # a API recusa só-espaço
    if stops:
        body["stop_sequences"] = stops[:8]
    return body


async def stream_chat(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    options: dict[str, Any],
    api_key_env: str = "ANTHROPIC_API_KEY",
) -> AsyncIterator[dict[str, Any]]:
    headers = _headers(api_key_env)
    t0 = time.perf_counter_ns()
    t_first: int | None = None
    input_tokens = output_tokens = 0
    try:
        async with client.stream("POST", f"{base_url}/v1/messages", headers=headers,
                                 json=_payload(model, messages, options)) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")
                raise NodeError(f"API do Claude respondeu HTTP {resp.status_code}: {body[:300]}")
            event = ""
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                data = json.loads(line[5:])
                kind = data.get("type", event)
                if kind == "message_start":
                    input_tokens = data["message"].get("usage", {}).get("input_tokens", 0)
                elif kind == "content_block_delta" and data["delta"].get("type") == "text_delta":
                    if t_first is None:
                        t_first = time.perf_counter_ns()
                    yield {"message": {"role": "assistant", "content": data["delta"]["text"]}, "done": False}
                elif kind == "message_delta":
                    output_tokens = data.get("usage", {}).get("output_tokens", output_tokens)
                elif kind == "error":
                    raise NodeError(f"erro da API do Claude: {data.get('error')}")
    except httpx.HTTPError as exc:
        raise NodeError(f"falha de comunicação com {base_url}: {exc!r}") from exc
    t_end = time.perf_counter_ns()
    first = t_first or t_end
    yield {
        "done": True, "message": {"role": "assistant", "content": ""},
        "prompt_eval_count": input_tokens, "eval_count": output_tokens,
        # Não há como separar rede e prefill do lado de cá: "prompt_eval" = até o 1º token.
        "prompt_eval_duration": first - t0, "eval_duration": max(1, t_end - first),
        "load_duration": 0,
    }


async def list_models(client: httpx.AsyncClient, base_url: str, api_key_env: str = "ANTHROPIC_API_KEY",
                      timeout: float = 5.0) -> list[str]:
    try:
        resp = await client.get(f"{base_url}/v1/models", headers=_headers(api_key_env),
                                params={"limit": 100}, timeout=timeout)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise NodeError(f"API do Claude inacessível: {exc!r}") from exc
    return [m.get("id", "") for m in resp.json().get("data", [])]
