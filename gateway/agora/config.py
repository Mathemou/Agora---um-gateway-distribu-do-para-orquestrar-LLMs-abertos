"""Carregamento e validação da configuração do gateway (YAML).

O arquivo aceita variáveis de ambiente no formato ${VAR} ou ${VAR:-padrão},
o que permite trocar modelos/URLs pelo .env sem editar o YAML.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name, default if default is not None else "")

    return _ENV_PATTERN.sub(repl, text)


class NodeConfig(BaseModel):
    """Um nó de inferência (uma instância do Ollama)."""

    url: str
    # Requisições atendidas ao mesmo tempo pelo nó. Deve ser igual ao
    # OLLAMA_NUM_PARALLEL do nó, para que a fila "de verdade" seja a do gateway.
    max_concurrency: int = Field(default=1, ge=1)
    # Requisições que podem esperar na fila; acima disso o gateway responde 503.
    max_queue: int = Field(default=32, ge=0)


class DomainConfig(BaseModel):
    """Um domínio: modelo, prompt de sistema e parâmetros próprios."""

    description: str = ""
    node: str
    model: str
    system_prompt: str = ""
    options: dict[str, Any] = Field(default_factory=dict)


class GatewayConfig(BaseModel):
    nodes: dict[str, NodeConfig]
    domains: dict[str, DomainConfig]
    # Arquivo JSONL com uma linha por requisição (insumo para os experimentos).
    metrics_log: str | None = "logs/requests.jsonl"
    # Por quanto tempo o Ollama mantém o modelo carregado após uma requisição:
    # duração com unidade ("30m", "1h") ou número de segundos (-1 = para sempre).
    keep_alive: str | int = "30m"
    # Timeout de leitura entre dois pedaços do stream (CPU pode ser lenta).
    read_timeout_s: float = 300.0

    @field_validator("keep_alive", mode="before")
    @classmethod
    def _keep_alive_number(cls, v: Any) -> Any:
        # O Ollama rejeita "-1" como texto (sem unidade); número ele aceita.
        if isinstance(v, str) and re.fullmatch(r"-?\d+", v.strip()):
            return int(v)
        return v

    @model_validator(mode="after")
    def _check_refs(self) -> "GatewayConfig":
        for name, domain in self.domains.items():
            if domain.node not in self.nodes:
                raise ValueError(f"domínio '{name}' aponta para nó inexistente '{domain.node}'")
        if not self.domains:
            raise ValueError("configure pelo menos um domínio")
        return self


def load_config(path: str | os.PathLike[str] | None = None) -> GatewayConfig:
    path = path or os.environ.get("GATEWAY_CONFIG", "config/gateway.yaml")
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(_expand_env(raw))
    return GatewayConfig.model_validate(data)
