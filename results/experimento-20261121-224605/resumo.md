# Resumo do experimento — experimento-20261121-224605

## Ambiente

```json
{
  "data": "2026-11-21T22:46:05",
  "commit": "6f7f209",
  "url": "http://localhost:8000",
  "domain": "impostor",
  "model": "qwen2.5:3b",
  "nodes": [
    "node-impostor"
  ],
  "c": 2,
  "node_snapshots": [
    {
      "node": "node-impostor",
      "active": 0,
      "waiting": 0,
      "max_concurrency": 2,
      "max_queue": 32,
      "total_admitted": 55,
      "total_rejected": 0,
      "total_abandoned": 0,
      "healthy": true,
      "ewma_ttft_ms": 356.8,
      "selected": 14,
      "failures": 0,
      "failovers_from": 0
    }
  ],
  "queue_policy_antes": "fifo",
  "balance": "peak_ewma",
  "maquina": {
    "sistema": "Windows-10-10.0.26200-SP0",
    "cpu": "Intel64 Family 6 Model 183 Stepping 1, GenuineIntel",
    "nucleos_logicos": 20,
    "python": "3.11.9"
  },
  "ollama": {
    "http://localhost:21434": "0.40.11"
  },
  "parametros": {
    "domain": "impostor",
    "fases": [
      "bench",
      "salas",
      "fila"
    ],
    "clients": [
      1,
      2,
      4,
      8,
      16
    ],
    "requests": 2,
    "repeat": 5,
    "max_tokens": 64,
    "rooms": [
      1,
      2,
      4,
      8
    ],
    "rooms_repeat": 2,
    "discussion": 90,
    "talk": 60,
    "fila_repeat": 3,
    "fila_duration": 60,
    "games_log": null,
    "rapido": false,
    "out": "results"
  },
  "games_log": "C:\\Users\\mathe\\Desktop\\T2 - Redes\\Agora---um-gateway-distribu-do-para-orquestrar-LLMs-abertos\\logs\\games.jsonl"
}
```

## 1. Clientes simultâneos × TTFT e vazão (experimento obrigatório)

Arquivo: `bench-20261121-224611-raw.csv` · 5 repetições · 2 pedidos por cliente · c = 2 vagas no nó.

Parâmetros medidos do modelo de fila:

- S com 1 cliente: **3.02 s**; TTFT de serviço: 0.90 s; 14.2 tokens por resposta
- S com 2 em paralelo: **4.42 s** (×1.47); TTFT de serviço: 1.27 s
- teto de vazão previsto: c·tokens/S = **6.4 tokens/s**

| N | TTFT médio (s) | previsto ingênuo | previsto calibrado | TTFT p95 (s) | espera na fila (s) | vazão (tok/s) | prevista | falhas |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.90 ± 0.05 | 0.90 | 0.90 | 1.00 ± 0.07 | 0.00 ± 0.00 | 4.6 ± 0.8 | 4.7 | 0 |
| 2 | 1.27 ± 0.16 | 0.90 | 1.27 | 1.47 ± 0.34 | 0.00 ± 0.00 | 6.9 ± 0.8 | 6.4 | 0 |
| 4 | 4.76 ± 0.39 | 3.17 | 4.59 | 7.29 ± 0.50 | 3.56 ± 0.25 | 7.0 ± 0.7 | 6.4 | 0 |
| 8 | 12.30 ± 0.92 | 7.69 | 11.22 | 17.82 ± 1.69 | 11.27 ± 0.78 | 6.9 ± 0.7 | 6.4 | 0 |
| 16 | 26.22 ± 3.39 | 16.75 | 24.49 | 36.47 ± 4.97 | 25.67 ± 3.43 | 7.2 ± 0.1 | 6.4 | 0 |

Erro médio da previsão para N > c: ingênua **36 %**, calibrada **6 %**. TTFT passa do dobro do valor com 1 cliente a partir de N = **4**.

Figuras: `fig-ttft.png`, `fig-vazao.png`.

## 2. Versão jogo: salas simultâneas × estouro do orçamento de digitação

Arquivo: `rooms-20261121-225902-raw.csv` · log das partidas: `C:\Users\mathe\Desktop\T2 - Redes\Agora---um-gateway-distribu-do-para-orquestrar-LLMs-abertos\logs\games.jsonl`.

Com 1 sala, a fala da IA leva **3.97 s** (mediana, do gatilho ao texto pronto) e o orçamento mediano é **8.07 s**. Previsão para a rajada do início da rodada: começa a estourar com k* ≈ c·B/S = **4.1 salas**.

| salas | falas da IA | estouram (todas) | estouram (1ª da rodada) | previsto (1ª) | fila (s) | prefill (s) | geração (s) | total p50 / p95 (s) | orçamento (s) | IA venceu |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 0 % [0–32] | 0 % (n=2) | 0 % | 0.00 | 2.08 | 1.89 | 3.97 / 5.37 | 8.07 | 2 |
| 2 | 17 | 35 % [17–59] | 25 % (n=4) | 0 % | 0.00 | 2.11 | 2.78 | 4.89 / 11.72 | 7.72 | 4 |
| 4 | 33 | 42 % [27–59] | 75 % (n=8) | 0 % | 0.00 | 3.44 | 2.94 | 6.38 / 18.34 | 7.65 | 8 |
| 8 | 28 | 86 % [69–94] | 100 % (n=16) | 50 % | 20.10 | 5.28 | 3.85 | 29.23 / 50.20 | 9.78 | 10 |

Figuras: `fig-salas.png`, `fig-tempo-fala.png`.

## 3. Política de fila: fifo × priority

Arquivo: `desempenho-fila-20261121-231508-summary.csv`.

| política | reps | falas da IA que estouram | fala da IA e2e p50 / p95 (s) | espera da fala (s) | lote e2e p50 (s) | vazão do lote (tok/s) |
|---|---|---|---|---|---|---|
| fifo | 3 | 93 ± 15 % | 12.63 ± 2.75 / 15.53 ± 3.74 | 9.94 ± 2.47 | 14.47 ± 3.81 | 8.9 ± 0.7 |
| priority | 3 | 65 ± 33 % | 10.80 ± 3.14 / 14.07 ± 1.98 | 8.00 ± 2.86 | 14.53 ± 0.90 | 8.8 ± 0.6 |

Figura: `fig-fila.png`.
