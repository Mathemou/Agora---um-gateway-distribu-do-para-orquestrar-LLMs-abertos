"""Carregamento e validação da configuração do gateway (YAML).

O arquivo aceita variáveis de ambiente no formato ${VAR} ou ${VAR:-padrão},
o que permite trocar modelos/URLs pelo .env sem editar o YAML.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name, default if default is not None else "")

    return _ENV_PATTERN.sub(repl, text)


class NodeConfig(BaseModel):
    """Um nó de inferência: uma instância do Ollama ou a API do Claude."""

    url: str
    # "ollama" (local, pesos abertos) ou "anthropic" (API do Claude, na nuvem).
    # "ollama" (local, pesos abertos), "anthropic" (API do Claude) ou "openai" (qualquer API
    # compatível com a da OpenAI: Groq, OpenRouter, Cerebras, LM Studio...).
    kind: Literal["ollama", "anthropic", "openai"] = "ollama"
    # Para os nós remotos: variável de ambiente com a chave da API.
    api_key_env: str = "ANTHROPIC_API_KEY"
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


class GameConfig(BaseModel):
    """Regras do Jogo da Imitação (salas, fases e comportamento da IA)."""

    # Domínio que joga disfarçado de humano e domínio do Mestre (narrador).
    impostor_domain: str = "impostor"
    # Se o domínio da IA for remoto e falhar, a fala é refeita neste (local).
    impostor_fallback_domain: str | None = "impostor"
    master_domain: str | None = "mestre"
    # Participantes, contando a(s) IA(s). Mínimo 3 = 2 humanos + 1 IA.
    min_players: int = Field(default=3, ge=2)
    max_players: int = Field(default=10, ge=3)
    ai_count: int = Field(default=1, ge=1, le=3)
    max_rounds: int = Field(default=4, ge=1)
    # Duração das fases, em segundos.
    discussion_s: float = Field(default=150, gt=0)
    vote_s: float = Field(default=40, gt=0)
    reveal_s: float = Field(default=8, ge=0)
    # Tempo máximo esperando o Mestre gerar a pergunta da rodada (senão, usa uma pronta).
    master_timeout_s: float = Field(default=25, gt=0)
    # Antes da 1ª rodada, espera o nó da IA carregar o prompt (cache de prefixo), até este
    # limite. Sem isso, a 1ª fala da IA demora o prefill inteiro (~30 s no 3b em CPU) e a entrega.
    warmup_timeout_s: float = Field(default=90, ge=0)
    # "Humanização" da IA: atraso de leitura + velocidade de digitação simulada.
    read_delay_s: tuple[float, float] = (1.5, 4.0)
    typing_cps: tuple[float, float] = (5.0, 9.0)
    # Intervalo mínimo entre duas mensagens da IA e quantas linhas de conversa ela lê.
    ai_cooldown_s: float = Field(default=5, ge=0)
    # Personalidade da IA (chave de personas.ARCHETYPES); None = gira a roda a cada partida.
    ai_persona: str | None = None
    ai_context_messages: int = Field(default=30, ge=4)
    max_message_chars: int = Field(default=280, ge=20)
    # Log de partidas e de mensagens da IA (insumo para a análise de detecção).
    log_path: str | None = "logs/games.jsonl"


class GatewayConfig(BaseModel):
    nodes: dict[str, NodeConfig]
    domains: dict[str, DomainConfig]
    game: GameConfig = Field(default_factory=GameConfig)
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
        g = self.game
        if g.impostor_domain not in self.domains:
            raise ValueError(f"game.impostor_domain '{g.impostor_domain}' não é um domínio configurado")
        if g.master_domain and g.master_domain not in self.domains:
            raise ValueError(f"game.master_domain '{g.master_domain}' não é um domínio configurado")
        if g.min_players - g.ai_count < 1:
            raise ValueError("game.min_players precisa deixar pelo menos 1 humano")
        if g.max_players < g.min_players:
            raise ValueError("game.max_players < game.min_players")
        return self


def load_config(path: str | os.PathLike[str] | None = None) -> GatewayConfig:
    path = path or os.environ.get("GATEWAY_CONFIG", "config/gateway.yaml")
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(_expand_env(raw))
    return GatewayConfig.model_validate(data)
