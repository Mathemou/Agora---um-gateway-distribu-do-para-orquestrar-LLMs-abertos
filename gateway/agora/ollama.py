"""Cliente mínimo da API HTTP do Ollama (/api/chat em streaming NDJSON)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx


class NodeError(Exception):
    """Falha ao falar com o nó de inferência."""


async def stream_chat(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    options: dict[str, Any],
    keep_alive: str | int,
) -> AsyncIterator[dict[str, Any]]:
    """Gera os pedaços (chunks) da resposta do Ollama, um por linha NDJSON.

    Cada pedaço intermediário traz `message.content` (≈ 1 token); o último tem
    `done: true` e as contagens do nó (eval_count, eval_duration em ns, ...).
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": options,
        "keep_alive": keep_alive,
    }
    try:
        async with client.stream("POST", f"{base_url}/api/chat", json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")
                raise NodeError(f"nó respondeu HTTP {resp.status_code}: {body[:300]}")
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise NodeError(f"erro do nó: {chunk['error']}")
                yield chunk
    except httpx.HTTPError as exc:
        raise NodeError(f"falha de comunicação com {base_url}: {exc!r}") from exc


async def list_models(client: httpx.AsyncClient, base_url: str, timeout: float = 3.0) -> list[str]:
    """Modelos disponíveis no nó (GET /api/tags). Levanta NodeError se fora do ar."""
    try:
        resp = await client.get(f"{base_url}/api/tags", timeout=timeout)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise NodeError(f"nó {base_url} inacessível: {exc!r}") from exc
    return [m.get("name", "") for m in resp.json().get("models", [])]
