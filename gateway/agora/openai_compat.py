"""Nó de inferência em qualquer API compatível com a da OpenAI (Groq, OpenRouter,
Cerebras, LM Studio, vLLM...): `POST {url}/chat/completions` com `stream: true`.

Como o cliente da API do Claude, traduz o stream SSE (`data: {...}` ... `data: [DONE]`)
para o mesmo formato de pedaços do Ollama, então fila, métricas e jogo não mudam.

Detalhes de cada provedor que o adaptador resolve:
- só repassa opções que a API entende (as do Ollama, como `num_ctx`, ficam de fora);
- `num_predict` vira `max_tokens`; no máximo 4 sequências de parada;
- modelos que "raciocinam" (gpt-oss) não aceitam `reasoning_effort: none`: usa
  `low`, esconde o raciocínio e garante tokens suficientes para ele;
- se o modelo mandar o raciocínio no texto (`<think>...</think>`), ele é removido;
- o uso de tokens vem em `usage` ou, no Groq, em `x_groq.usage` no último pedaço.

A chave vem da variável de ambiente indicada em `api_key_env` (ex.: GROQ_API_KEY)
e nunca é gravada em log.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .ollama import NodeError

# Opções que passam direto para a API.
_PASSTHROUGH = ("temperature", "top_p", "seed", "presence_penalty", "frequency_penalty",
                "reasoning_effort", "include_reasoning", "reasoning_format")


def _headers(api_key_env: str) -> dict[str, str]:
    key = os.environ.get(api_key_env, "").strip()
    if not key:
        raise NodeError(f"defina a variável {api_key_env} com a chave da API")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def is_reasoning_model(model: str) -> bool:
    return "gpt-oss" in model


def build_payload(model: str, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
    for k in _PASSTHROUGH:
        if options.get(k) is not None:
            body[k] = options[k]
    max_tokens = int(options.get("num_predict") or options.get("max_tokens") or 256)
    if is_reasoning_model(model):
        # gpt-oss sempre raciocina antes de responder: o mínimo é "low", e os tokens
        # do raciocínio contam no limite, então a folga precisa ser maior.
        if body.get("reasoning_effort") in (None, "none", "default"):
            body["reasoning_effort"] = "low"
        body.pop("reasoning_format", None)  # não suportado no gpt-oss
        body["include_reasoning"] = False
        max_tokens = max(max_tokens, 512)
    elif "include_reasoning" in body and "reasoning_format" in body:
        body.pop("include_reasoning")  # os dois juntos dão erro 400
    body["max_tokens"] = max_tokens
    stops = [s for s in options.get("stop") or [] if s]
    if stops:
        body["stop"] = stops[:4]
    return body


class _ThinkFilter:
    """Remove trechos <think>...</think> que chegam picados no meio do stream."""

    def __init__(self) -> None:
        self.inside = False
        self.buf = ""

    def feed(self, text: str) -> str:
        self.buf += text
        out = []
        while self.buf:
            if self.inside:
                end = self.buf.find("</think>")
                if end < 0:
                    self.buf = self.buf[-8:]  # pode ser o começo de "</think>"
                    return "".join(out)
                self.buf = self.buf[end + len("</think>"):]
                self.inside = False
            else:
                start = self.buf.find("<think>")
                if start < 0:
                    # segura um pedaço que pode ser o começo de "<think>"
                    cut = max((i for i in range(1, min(7, len(self.buf)) + 1) if "<think>".startswith(self.buf[-i:])), default=0)
                    out.append(self.buf[: len(self.buf) - cut])
                    self.buf = self.buf[len(self.buf) - cut:]
                    return "".join(out)
                out.append(self.buf[:start])
                self.buf = self.buf[start + len("<think>"):]
                self.inside = True
        return "".join(out)

    def flush(self) -> str:
        rest, self.buf = ("" if self.inside else self.buf), ""
        return rest


async def stream_chat(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    options: dict[str, Any],
    api_key_env: str,
) -> AsyncIterator[dict[str, Any]]:
    headers = _headers(api_key_env)
    t0 = time.perf_counter_ns()
    t_first: int | None = None
    usage: dict[str, Any] = {}
    pieces = 0
    think = _ThinkFilter()

    def piece(text: str) -> dict[str, Any]:
        return {"message": {"role": "assistant", "content": text}, "done": False}

    try:
        async with client.stream("POST", f"{base_url}/chat/completions", headers=headers,
                                 json=build_payload(model, messages, options)) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")
                hint = ""
                if resp.status_code == 429:
                    hint = f" (limite de taxa; tente de novo em {resp.headers.get('retry-after', '?')} s)"
                raise NodeError(f"API respondeu HTTP {resp.status_code}{hint}: {body[:300]}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    raise NodeError(f"erro da API: {chunk['error']}")
                usage = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage") or usage
                for choice in chunk.get("choices") or []:
                    text = think.feed((choice.get("delta") or {}).get("content") or "")
                    if text:
                        if t_first is None:
                            t_first = time.perf_counter_ns()
                        pieces += 1
                        yield piece(text)
            rest = think.flush()
            if rest:
                t_first = t_first or time.perf_counter_ns()
                pieces += 1
                yield piece(rest)
    except httpx.HTTPError as exc:
        raise NodeError(f"falha de comunicação com {base_url}: {exc!r}") from exc
    t_end = time.perf_counter_ns()
    first = t_first or t_end
    yield {
        "done": True, "message": {"role": "assistant", "content": ""},
        "prompt_eval_count": usage.get("prompt_tokens"),
        "eval_count": usage.get("completion_tokens") or pieces,
        # rede + fila do provedor + prefill, até o 1º token visível
        "prompt_eval_duration": first - t0, "eval_duration": max(1, t_end - first),
        "load_duration": 0,
    }


async def list_models(client: httpx.AsyncClient, base_url: str, api_key_env: str,
                      timeout: float = 5.0) -> list[str]:
    try:
        resp = await client.get(f"{base_url}/models", headers=_headers(api_key_env), timeout=timeout)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise NodeError(f"API inacessível: {exc!r}") from exc
    return [m.get("id", "") for m in resp.json().get("data", [])]
