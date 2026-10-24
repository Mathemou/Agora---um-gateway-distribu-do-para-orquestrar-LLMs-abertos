"""Métricas do gateway: TTFT, tokens/s, latência fim a fim e vazão.

Definições (todas medidas no gateway, a partir da chegada da requisição):
- queue_wait_ms: tempo esperando vaga na fila do nó.
- ttft_ms: chegada da requisição -> primeiro token repassado ao cliente
  (inclui fila + carregamento do modelo + prefill do prompt).
- e2e_ms: chegada -> último token (latência fim a fim).
- gateway_tps: tokens / (e2e - ttft), a taxa de geração vista pelo cliente.
- node_tps: eval_count / eval_duration, a taxa informada pelo próprio Ollama.
- vazão: tokens/s agregados de todas as requisições numa janela de tempo.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RequestRecord:
    request_id: str
    domain: str
    node: str
    model: str
    started_at: float  # epoch (s)
    status: str = "pending"  # ok | error | cancelled
    queue_position: int = 0
    queue_wait_ms: float | None = None
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    output_tokens: int = 0
    prompt_tokens: int | None = None
    gateway_tps: float | None = None
    node_tps: float | None = None
    node_load_ms: float | None = None
    node_prompt_eval_ms: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 3)


@dataclass
class MetricsStore:
    log_path: str | None = None
    maxlen: int = 5000
    records: deque[RequestRecord] = field(init=False)
    _token_times: deque[float] = field(init=False)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.records = deque(maxlen=self.maxlen)
        self._token_times = deque()
        self.started = time.time()
        self.total_tokens = 0
        if self.log_path:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)

    def count_token(self) -> None:
        now = time.monotonic()
        self._token_times.append(now)
        self.total_tokens += 1
        self._prune(now, 300)

    def _prune(self, now: float, window_s: float) -> None:
        while self._token_times and now - self._token_times[0] > window_s:
            self._token_times.popleft()

    def record(self, rec: RequestRecord) -> None:
        self.records.append(rec)
        if self.log_path:
            line = json.dumps(rec.to_dict(), ensure_ascii=False)
            with self._lock, open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def throughput(self, window_s: float = 60.0) -> float:
        now = time.monotonic()
        self._prune(now, 300)
        n = sum(1 for t in self._token_times if now - t <= window_s)
        elapsed = min(window_s, time.time() - self.started) or 1.0
        return round(n / elapsed, 3)

    def summary(self, window_s: float = 60.0) -> dict[str, Any]:
        recs = list(self.records)
        ok = [r for r in recs if r.status == "ok"]

        def stats(attr: str) -> dict[str, float | None]:
            vals = [getattr(r, attr) for r in ok if getattr(r, attr) is not None]
            return {
                "mean": round(sum(vals) / len(vals), 3) if vals else None,
                "p50": _percentile(vals, 0.50),
                "p95": _percentile(vals, 0.95),
            }

        by_status: dict[str, int] = {}
        for r in recs:
            by_status[r.status] = by_status.get(r.status, 0) + 1
        return {
            "requests": len(recs),
            "by_status": by_status,
            "ttft_ms": stats("ttft_ms"),
            "e2e_ms": stats("e2e_ms"),
            "queue_wait_ms": stats("queue_wait_ms"),
            "gateway_tps": stats("gateway_tps"),
            "node_tps": stats("node_tps"),
            "throughput_tokens_per_s": {"window_s": window_s, "value": self.throughput(window_s)},
            "total_tokens": self.total_tokens,
        }
