# Roteiro da demonstração M1 (5 minutos)

**Objetivo:** mostrar o núcleo funcionando de ponta a ponta: o jogo rodando ao vivo em cima do gateway, a fila e as métricas medindo cada fala da IA e o streaming token a token.

**Quem faz o quê**
- **Matheus** apresenta. O notebook dele roda o Ágora e fica ligado no projetor.
- **Yan** joga pelo próprio notebook ou pelo celular.
- **2 ou 3 colegas convidados** entram pelo celular. Quanto mais humanos, mais difícil achar a IA, e a turma participa.

---

## Antes da aula (checklist, uns 15 min antes)

- [ ] `docker compose up -d` pelo menos 10 min antes, com o notebook na tomada.
- [ ] `http://localhost:8000/health` mostra tudo verde (`status: ok`, nenhum `missing`).
- [ ] Jogar **uma partida de teste** rápida, para carregar os modelos na memória. A primeira carga é a mais lenta.
- [ ] Fechar navegador pesado, IDE e jogos. A CPU é toda da IA.
- [ ] Descobrir o IP do notebook na rede da sala (`ipconfig`, linha "Endereço IPv4") e testar `http://<IP>:8000` pelo celular do Yan. Se não abrir, liberar a porta 8000 no Firewall do Windows (ou rotear do celular e ligar todos nele).
- [ ] Deixar abertas três abas:
  1. o jogo: `http://localhost:8000`
  2. o painel: `http://localhost:8000/painel` (linha do tempo **escondida**)
  3. um terminal com o venv ativado, na pasta do projeto
- [ ] Opções da sala que vamos usar: **discussão de 1min30**, **2 rodadas**, **1 IA**.

---

## Durante (5:00)

### 0:00 → 0:45 · A ideia (enquanto a sala enche)

> "O Ágora é o Projeto 1, um gateway de LLMs abertos rodando local, só em CPU. Para dar sentido ao que trafega, a gente transformou ele no **jogo da imitação** de Turing, no estilo Among Us: numa sala de bate-papo, um dos participantes é um LLM tentando se passar por gente. Vocês vão jogar agora."

- **Já crie a sala** e projete o código em letra grande no lobby. Os colegas entram pelo `http://<IP>:8000/?sala=CÓDIGO` (o botão "Copiar convite" gera o link).
- Mostre o desenho da arquitetura do README em 15 segundos: navegadores → **WebSocket** → servidor do jogo → **mesma fila e pipeline do gateway** → nós Ollama.
- Clique em **Começar partida** assim que tiver gente. A tela "Preparando a partida…" aparece enquanto o modelo da IA aquece, e você continua falando por cima.

### 0:45 → 3:00 · Partida ao vivo

- Leia em voz alta a pergunta do Mestre. Ela chega **token a token**, aponte isso.
- Deixe a turma conversar. Incentive provocações: "pergunta pra alguém o que comeu ontem", "acusa alguém".
- Durante a discussão, **não** mostre o painel com a linha do tempo. Ele entrega a IA pelo horário de cada requisição (e é por isso que ele começa escondido).
- Na **votação**, todo mundo vota. O resultado diz se quem foi exilado era humano ou a IA.
- Se a partida não acabar na 1ª rodada, tudo bem: termine a votação da rodada 2 ou siga para o painel.

### 3:00 → 4:10 · O lado de Redes (painel)

Vá para a aba do **painel** e **ligue a linha do tempo**, já que agora não estraga nada.

> "Cada fala da IA passou por aqui. A fila do nó tem limite de concorrência: quando ela está ocupada, a próxima requisição espera e o TTFT sobe."

- **Quadrados das vagas e barra da fila:** o limite de concorrência do nó, ao vivo.
- **Barras "onde vai o tempo":** fila, depois prefill até o 1º token, depois a geração dos tokens.
- **Frase-chave:** "Um humano leva de 4 a 12 segundos para ler e digitar. A IA precisa caber nesse orçamento. A gente começa a gerar enquanto ela 'lê' e mostra 'digitando…'. Se a fila cresce, ela estoura o orçamento e **se entrega pelo relógio**. Essa é a nossa pergunta-desafio, TTFT × tokens/s, virada mecânica de jogo."
- Volte à tela de fim de jogo: ela mostra a resposta média da IA, o TTFT e quantas falas estouraram o orçamento.

### 4:10 → 5:00 · Streaming cru + próximos passos

No terminal:

```bash
python client/chat.py -d mestre "Faça uma pergunta curta para os jogadores"
```

O texto aparece token a token, e no fim vem a linha de métricas (TTFT, tokens/s, fila). Se preferir mostrar o SSE puro:

```powershell
curl.exe -N -X POST http://localhost:8000/v1/chat -H "Content-Type: application/json" -d '{\"domain\":\"mestre\",\"prompt\":\"oi\"}'
```

Feche com:

> "Para o M2: **red team** (os próprios jogadores tentando prompt injection na IA, medindo quantos ataques passam antes e depois dos filtros) e **balanceamento entre dois nós**, um em cada notebook, com failover no meio de uma fala. O experimento final varia o número de salas simultâneas e mede onde o TTFT dispara."

---

## Plano B

| Problema na hora | O que fazer |
| --- | --- |
| IA lenta demais (primeira fala demora, estouros) | No `.env`, `MODEL_IMPOSTOR=qwen2.5:1.5b` e `docker compose up -d gateway` (leva uns 30 s se o modelo já estiver baixado) |
| Celular não entra | Joguem em duas abas no notebook (uma anônima) + o Yan no notebook dele pelo IP; ou rotear do celular |
| Ollama travou | `docker compose restart node-impostor node-mestre` |
| Tudo travou | Opção C do README (nó falso): o jogo e as métricas funcionam igual, só a IA fala frases prontas. Serve para mostrar a parte de Redes |

## Perguntas que podem aparecer

- **"Por que não usar o ChatGPT/Claude?"** O projeto é sobre modelos abertos sob controle local (soberania). O Claude está plugado como comparação opcional (`impostor-claude`), não como base.
- **"Por que WebSocket no jogo e SSE na API?"** O jogo é bidirecional (chat, digitando, votos). A API só manda do servidor para o cliente, e SSE passa por qualquer proxy e se inspeciona com `curl -N`.
- **"Onde fica a fila?"** No gateway, não no Ollama: o limite de concorrência de cada nó é igual ao `OLLAMA_NUM_PARALLEL` dele, então a espera é visível e medida.
- **"Como a IA não se entrega escrevendo certinho?"** Identidade criada pelo modelo + humanizador (abreviações, minúsculas, erros de digitação, corte no tamanho que a sala usa) + filtros de eco e de "como uma IA…".
