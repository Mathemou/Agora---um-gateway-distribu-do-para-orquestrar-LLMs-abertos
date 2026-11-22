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


BalancePolicy = Literal["round_robin", "least_queue", "least_latency", "peak_ewma"]


class DomainConfig(BaseModel):
    """Um domínio: modelo, prompt de sistema e parâmetros próprios.

    Um domínio é servido por um ou mais nós (`nodes`, o "pool"); o gateway escolhe o
    nó de cada requisição pela política de balanceamento. `node` (um nó só) continua
    valendo e equivale a `nodes: [node]`.
    """

    description: str = ""
    node: str | None = None
    nodes: list[str] = Field(default_factory=list)
    # Política deste domínio; vazio = a política global (`balance.policy`).
    balance: BalancePolicy | None = None
    model: str
    system_prompt: str = ""
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("nodes", mode="before")
    @classmethod
    def _nodes_list(cls, v: Any) -> Any:
        # Aceita "a, b" (vindo de variável de ambiente) além de [a, b].
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        if isinstance(v, list):
            return [str(x).strip() for x in v if x is not None and str(x).strip()]
        return v

    @model_validator(mode="after")
    def _pool(self) -> "DomainConfig":
        if not self.nodes:
            if not self.node:
                raise ValueError("informe 'node' ou 'nodes'")
            self.nodes = [self.node]
        self.nodes = list(dict.fromkeys(self.nodes))  # sem repetidos, na ordem
        if self.node is None:
            self.node = self.nodes[0]  # nó "principal" (aquecimento, painel, /v1/domains)
        return self

    @property
    def pool(self) -> list[str]:
        return self.nodes


class BalanceConfig(BaseModel):
    """Trilha Desempenho: balanceamento, saúde dos nós e failover."""

    # round_robin: um de cada vez · least_queue: menos ocupado (ativos + fila) / vagas
    # least_latency: menor TTFT de serviço observado (média móvel exponencial)
    # peak_ewma: TTFT observado × carga do nó (estimativa do TTFT com a fila)
    policy: BalancePolicy = "least_queue"
    # Peso da amostra nova na média móvel do TTFT (0..1; maior = reage mais rápido).
    ewma_alpha: float = Field(default=0.3, gt=0, le=1)
    # Falhas seguidas (requisição ou health check) para tirar um nó do pool.
    fail_threshold: int = Field(default=2, ge=1)
    # Sem health check ativo, um nó retirado volta a ser tentado depois deste tempo.
    down_cooldown_s: float = Field(default=15, ge=0)
    # Health check ativo (GET /api/tags) a cada N s; 0 = desligado. Só nós locais
    # (Ollama), a menos de probe_remote (evita gastar o limite de taxa das APIs pagas).
    health_interval_s: float = Field(default=5, ge=0)
    health_timeout_s: float = Field(default=2, gt=0)
    probe_remote: bool = False
    # Failover: se o nó cair (antes ou NO MEIO da resposta), refaz em outro nó do pool.
    failover: bool = True
    max_failovers: int = Field(default=2, ge=0)
    # Nó "travado": sem nenhum pedaço do stream por este tempo depois do 1º token.
    stall_timeout_s: float | None = Field(default=30, gt=0)
    # Limite para o 1º token (inclui fila e prefill); vazio = só o read_timeout_s.
    ttft_timeout_s: float | None = Field(default=None, gt=0)


class QueueConfig(BaseModel):
    """Política da fila de cada nó.

    fifo: ordem de chegada. priority: a chave da fila é `chegada + atraso da classe`;
    quem tem atraso menor passa na frente, mas quem espera mais que a diferença volta a
    ganhar (envelhecimento), então ninguém morre de fome.
    """

    policy: Literal["fifo", "priority"] = "fifo"
    offsets_s: dict[str, float] = Field(default_factory=lambda: {
        "impostor": 0.0,          # fala da IA: curta e interativa (orçamento de digitação)
        "api": 2.0,               # cliente da API /v1/chat
        "mestre": 4.0,            # narração do Mestre (tem pergunta pronta de reserva)
        "impostor-warmup": 8.0,   # aquecimento do cache de prefixo
    })
    default_offset_s: float = 2.0


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
    ai_context_messages: int = Field(default=30, ge=4)
    max_message_chars: int = Field(default=280, ge=20)
    # Log de partidas e de mensagens da IA (insumo para a análise de detecção).
    log_path: str | None = "logs/games.jsonl"


class GatewayConfig(BaseModel):
    nodes: dict[str, NodeConfig]
    domains: dict[str, DomainConfig]
    game: GameConfig = Field(default_factory=GameConfig)
    balance: BalanceConfig = Field(default_factory=BalanceConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    # Arquivo JSONL com uma linha por requisição (insumo para os experimentos).
    metrics_log: str | None = "logs/requests.jsonl"
    # Por quanto tempo o Ollama mantém o modelo carregado após uma requisição:
    # duração com unidade ("30m", "1h") ou número de segundos (-1 = para sempre).
    keep_alive: str | int = "30m"
    # Timeout de leitura entre dois pedaços do stream (CPU pode ser lenta).
    read_timeout_s: float = 300.0
    # Variável de ambiente com o token que protege a API (POST /v1/chat e /v1/balancer).
    # Vazia ou não definida = API aberta (uso local). O jogo (WebSocket) continua aberto.
    api_token_env: str | None = None

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
            for node in domain.pool:
                if node not in self.nodes:
                    raise ValueError(f"domínio '{name}' aponta para nó inexistente '{node}'")
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
