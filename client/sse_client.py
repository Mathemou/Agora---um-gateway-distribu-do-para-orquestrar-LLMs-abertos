"""Cliente assíncrono do gateway: envia a pergunta e lê os eventos SSE."""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx


def auth_headers() -> dict[str, str]:
    """Token da API (gateway na nuvem): defina AGORA_API_TOKEN no terminal."""
    token = os.environ.get("AGORA_API_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


@dataclass
class Event:
    name: str
    data: dict[str, Any]
    t: float  # time.perf_counter() quando o evento chegou ao cliente


async def stream_events(
    client: httpx.AsyncClient, base_url: str, payload: dict[str, Any]
) -> AsyncIterator[Event]:
    """Faz POST /v1/chat e gera um Event por mensagem SSE recebida."""
    async with client.stream("POST", f"{base_url}/v1/chat", json=payload, headers=auth_headers()) as resp:
        if resp.status_code != 200:
            body = (await resp.aread()).decode(errors="replace")
            yield Event("http_error", {"status": resp.status_code, "body": body}, time.perf_counter())
            return
        name, data_lines = "message", []
        async for line in resp.aiter_lines():
            if line == "":
                if data_lines:
                    yield Event(name, json.loads("\n".join(data_lines)), time.perf_counter())
                name, data_lines = "message", []
            elif line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())


@dataclass
class ClientResult:
    """Uma requisição vista pelo cliente (inclui a rede até o gateway)."""

    domain: str
    status: str = "pending"
    start: float = 0.0
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    tokens: int = 0
    text: str = ""
    server: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    # Failover no gateway: o texto parcial é descartado e a resposta recomeça noutro nó.
    failovers: int = 0
    discarded_tokens: int = 0

    @property
    def tps(self) -> float | None:
        if self.ttft_ms is None or self.e2e_ms is None or self.tokens <= 1:
            return None
        gen_s = (self.e2e_ms - self.ttft_ms) / 1000
        return (self.tokens - 1) / gen_s if gen_s > 0 else None


async def ask(
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, Any],
    on_token=None,
) -> ClientResult:
    res = ClientResult(domain=payload.get("domain", ""), start=time.perf_counter())
    async for ev in stream_events(client, base_url, payload):
        if ev.name == "token":
            if res.ttft_ms is None:
                res.ttft_ms = (ev.t - res.start) * 1000
            res.tokens += 1
            res.text += ev.data["text"]
            if on_token:
                on_token(ev.data["text"])
        elif ev.name == "failover":
            res.failovers += 1
            res.discarded_tokens += ev.data.get("discard_tokens", 0)
            res.tokens = 0
            res.text = ""
        elif ev.name == "done":
            res.server = ev.data
            res.status = "ok"
        elif ev.name == "error":
            res.status, res.error = "error", ev.data.get("message")
        elif ev.name == "http_error":
            res.status, res.error = f"http_{ev.data['status']}", ev.data["body"]
    res.e2e_ms = (time.perf_counter() - res.start) * 1000
    if res.status == "pending":
        res.status, res.error = "error", "stream terminou sem evento 'done'"
    return res
