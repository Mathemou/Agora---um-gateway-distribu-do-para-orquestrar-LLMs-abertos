# Roteiro do experimento (M2)

Ágora — Jogo da Imitação · Projeto 1 (Gateway distribuído de LLMs abertos)
Matheus Santos de Moura e Yan Maia · roteiro e previsão escritos em 21/11/2026, **antes** das medições

Este roteiro fixa hipótese, variáveis, repetições e ambiente, e registra a previsão do
modelo de fila **antes** de medir, como pede a proposta. As medições saem de um comando só
(`python client/experimento.py`), que grava o ambiente, roda as três fases e gera os
gráficos e as tabelas (`resumo.md`) a partir dos CSVs.

## 1. Pergunta e hipóteses

**Pergunta.** Em que ponto a fila do nó de inferência faz o TTFT disparar, e a partir de
quantas partidas simultâneas a IA impostora começa a se denunciar pelo tempo de resposta?

- **H1 (experimento obrigatório).** Com `c` vagas no nó (`OLLAMA_NUM_PARALLEL = 2`), o TTFT
  fica perto do tempo de prefill enquanto N ≤ c e cresce **linearmente** com N a partir daí.
  Já a vazão agregada (tokens/s) para de crescer em ≈ c · tokens / S.
- **H2 (CPU).** As `c` gerações paralelas dividem os mesmos núcleos. Por isso o tempo de serviço
  S com 2 pedidos juntos é **maior** que com 1, e o ganho de vazão de N = 1 para N = 2 fica
  **abaixo de 2×**. A previsão "ingênua" (S constante) vai **subestimar** o TTFT para N > c.
- **H3 (versão jogo).** No início de cada rodada, as IAs de todas as salas pedem a fala ao
  mesmo tempo (rajada). Com k salas, a i-ésima IA recebe o texto em ≈ (⌊i/c⌋ + 1) · S. Ela
  estoura o orçamento de digitação B ("lendo" + caracteres ÷ velocidade) quando isso passa de
  B. O estouro começa em **k\* ≈ c · B / S** salas.
- **H4 (política de fila).** Com `queue.policy = priority`, as falas da IA passam na frente
  dos pedidos longos da API no mesmo nó. A fração de falas que estouram e a espera delas na
  fila **caem**. Os pedidos em lote ficam um pouco mais lentos e a vazão total do nó não muda,
  porque o trabalho é o mesmo, só reordenado.

## 2. Previsão (registrada antes de medir)

### Parâmetros tirados das medições que já existiam

| Fonte | Modelo | Medida |
| --- | --- | --- |
| `logs/requests.jsonl`, 17 falas da IA em partidas reais (28/10) | qwen2.5:3b | prefill ≈ 2,5 s para ≈ 680 tokens de contexto (≈ 275 tokens/s); 14 tokens por fala a ≈ 6,4 tokens/s; **S ≈ 4,5 s** por fala |
| `logs/games.jsonl`, 10 falas da IA em 2 partidas | qwen2.5:3b | do gatilho ao texto pronto: **4,65 s** (mediana); orçamento de digitação **B ≈ 8,6 s** (mediana); 2 de 10 falas já estouraram com 1 sala |
| `results/bench-20261027-*` (1 cliente, 3 rep.) | qwen2.5:0.5b | TTFT ≈ 0,1 s, ≈ 15 tokens/s |
| `docker-compose.yml` / `.env` | — | `NODE_PARALLEL = 2` → **c = 2**; nó da IA com 4 CPUs e 4 GB |

No experimento obrigatório o contexto é menor do que numa partida: o prompt de sistema do
`impostor` mais uma pergunta curta dá ≈ 300 tokens, e o prefixo fica em cache entre pedidos.
A estimativa é **S₁ ≈ 3,0 s** (1 cliente), TTFT de serviço ≈ 0,8 s, ≈ 14 tokens por resposta
(o domínio para na quebra de linha) e r = 2 pedidos por cliente.

### H1 e H2: TTFT e vazão × clientes

O modelo é um sistema fechado: N clientes, c = 2 vagas, serviço S, fila FIFO. Com N > c, cada
pedido espera ≈ (N/c − 1) · S. No primeiro lote a espera média é a metade disso, então em
média W(N) ≈ (N/c − 1) · S · (1 − 1/(2r)). A tabela usa a simulação determinística desse
sistema (`client/analise_experimento.py`, função `simulate_closed`).

| N | TTFT previsto, S constante | TTFT previsto, S × 1,5 com 2 em paralelo | vazão prevista (tokens/s) |
| --- | --- | --- | --- |
| 1 | 0,8 s | 0,8 s | 4,7 |
| 2 | 0,8 s | 1,2 s | 9,3 → 6,2 |
| 4 | 3,0 s | 4,6 s | 9,3 → 6,2 |
| 8 | 7,5 s | 11,3 s | 9,3 → 6,2 |
| 16 | 16,6 s | 24,8 s | 9,3 → 6,2 |

O joelho fica em N = 2 → 4. Daí em diante o TTFT cresce ≈ S/c por cliente a mais
(≈ 1,1 s a 1,7 s por cliente). A vazão fica parada entre 6 e 9 tokens/s. A análise desenha as
duas curvas: a **ingênua**, com S medido com 1 cliente, e a **calibrada**, com S medido com 2
em paralelo. A distância entre elas mede o efeito da CPU compartilhada (H2).

### H3: salas simultâneas × estouro do orçamento

Com S ≈ 4,65 s e B ≈ 8,6 s: **k\* ≈ 2 × 8,6 / 4,65 ≈ 3,7 salas**.

| salas | 1ª fala da rodada que estoura (previsto) | raciocínio |
| --- | --- | --- |
| 1 | 0 % (mas ≈ 20 % das falas já estouram por variação de S, como em 28/10) | sem fila |
| 2 | 0 % | as duas cabem nas 2 vagas |
| 4 | ≈ 50 % | a 2ª "onda" fica pronta em ≈ 9,3 s > 8,6 s |
| 8 | ≈ 75 % | ondas em 4,7 / 9,3 / 14 / 18,6 s: só a 1ª cabe no orçamento |

Fora da rajada (falas por menção ou silêncio), os pedidos chegam espalhados e a fração de
estouros fica menor. A rajada é o pior caso que o jogo produz sozinho.

### H4: fifo × priority

Previsão qualitativa: com `priority`, a espera das falas da IA cai para perto de zero, porque
elas passam à frente dos pedidos longos do domínio `assistente` no mesmo nó. A fração de
estouros cai na mesma proporção. Os pedidos em lote ganham alguns segundos no p50, e a vazão
total muda menos de 10 %.

## 3. Variáveis

| Tipo | Variável | Valores |
| --- | --- | --- |
| Independente (H1, H2) | clientes simultâneos N | 1, 2, 4, 8, 16 |
| Independente (H3) | salas simultâneas k (2 bots "humanos" por sala) | 1, 2, 4, 8 |
| Independente (H4) | política da fila | fifo, priority |
| Controlada | modelo / quantização | qwen2.5:3b (`MODEL_IMPOSTOR`), mesma imagem do Ollama (`OLLAMA_IMAGE` fixada) |
| Controlada | vagas e recursos do nó | `NODE_PARALLEL = 2`, 4 CPUs, 4 GB (`docker-compose.yml`) |
| Controlada | carga | prompts fixos do `bench.py`, `max_tokens = 64`, 2 pedidos seguidos por cliente; bots com frases e semente fixas |
| Controlada | gateway | um nó só (`IMPOSTOR_NODES = node-impostor`), fila em fifo nas fases 1 e 2 |
| Medida | TTFT, latência fim a fim, espera na fila, tokens/s agregados | por pedido (cliente SSE + registro do gateway) |
| Medida | fração de falas da IA que estouram o orçamento; decomposição fila / prefill / geração | por fala (`logs/games.jsonl`) |

## 4. Repetições e estatística

- Clientes: **5 repetições** por nível, cada uma com N clientes × 2 pedidos. A métrica de
  cada repetição é a média (e o p50/p95) dos pedidos. O relatório mostra a média entre as
  repetições com **IC 95 % (t de Student)**.
- Salas: **2 repetições** por nível. Estouro é proporção, então o IC 95 % é o de **Wilson**,
  que funciona com poucas amostras e com 0 % ou 100 %.
- Fila: **3 repetições** de 60 s por política, com IC 95 % (t).
- Os níveis rodam em ordem crescente e um pedido de aquecimento é descartado antes, para
  carregar o modelo na memória.

## 5. Ambiente

O `client/experimento.py` grava em `ambiente.json`: data, commit, sistema, CPU, número de
núcleos, versão do Python e do Ollama, modelo, nó e `c` lido do gateway, e os parâmetros
usados. Antes de rodar:

1. Notebook na tomada, modo de energia "melhor desempenho", outros programas fechados.
2. `.env`: `IMPOSTOR_NODES=node-impostor`, `NODE_PARALLEL=2`, `GAME_IMPOSTOR_DOMAIN=impostor`,
   `MODEL_IMPOSTOR=qwen2.5:3b` e `OLLAMA_IMAGE` com versão fixa (ex.: `ollama/ollama:0.12.6`).
3. `docker compose up -d` e conferir `http://localhost:8000/health` com tudo `up`.

## 6. Procedimento

```bash
python client/experimento.py --rapido     # ensaio de ~5 min, confere que tudo roda
python client/experimento.py              # experimento completo (~1 h com o 3b em CPU)
```

O resultado vai para `results/experimento-<data>/`: CSVs brutos e resumidos de cada fase,
`log-*.txt`, `fig-ttft.png`, `fig-vazao.png`, `fig-salas.png`, `fig-tempo-fala.png`,
`fig-fila.png` e `resumo.md` com as tabelas do relatório. Para refazer só a análise:
`python client/analise_experimento.py results/experimento-<data>`.

## 7. Validação do método (nó falso)

O pipeline inteiro (carga, coleta, simulação do modelo, IC e gráficos) foi conferido com o
nó falso (`agora.fake_node`), que tem tempo de serviço determinístico (c = 2, S ≈ 2,3 s). A
simulação acertou o TTFT medido com **erro médio de 1 %** para N > c (N = 4, 8, 16) e a vazão
saturou em 56,6 tokens/s contra 56,5 previstos. Com o nó falso calibrado nos tempos do 3b
(prefill 2,5 s), 6 salas fizeram **83 %** das primeiras falas estourarem (previsto: 67 %). Isso
mostra que o mecanismo de H3 aparece e que a previsão fica do lado otimista, porque ignora as
falas de menção que entram na mesma fila. Esses números validam o método e **não** entram como
resultado: os resultados são os do Ollama.

## 8. Ameaças à validade

- **Cache de prefixo do Ollama:** prompts repetidos têm prefill menor que numa partida real.
  Por isso a versão jogo usa partidas de verdade, com o contexto crescendo.
- **Aquecimento térmico da CPU:** a ordem crescente dos níveis pode somar lentidão térmica
  aos níveis altos. As repetições e o IC mostram se isso pesou.
- **Bots não são humanos:** eles falam a cada 4–10 s e votam ao acaso. Por isso "a IA venceu"
  não é métrica. A métrica da versão jogo é só o tempo (estouro do orçamento).
- **Um nó só:** o balanceamento entre duas máquinas (trilha Desempenho) é medido à parte, com
  `client/exp_desempenho.py balance`.
