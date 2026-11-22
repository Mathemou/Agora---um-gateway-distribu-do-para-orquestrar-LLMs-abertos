# Ágora na nuvem: passo a passo (Oracle Cloud, Always Free)

A VM roda só o gateway e o Caddy (HTTPS). A IA impostora e o Mestre usam o Groq,
com o OpenCode Go de reserva. Os experimentos do relatório continuam no Ollama local.

## 0. Antes de começar

- Faça commit e push do repositório (a VM vai baixar o código do GitHub).
- Tenha em mãos: a chave do Groq, a chave do OpenCode Go (do Yan) e um cartão de
  crédito para a verificação de identidade da Oracle (a conta Always Free não cobra).

## 1. Criar a conta

1. Acesse <https://www.oracle.com/cloud/free/> e clique em **Start for free**.
2. Escolha a **região principal** com cuidado: ela não muda depois. Para jogadores no
   Brasil, use **Brazil East (São Paulo)** ou **Brazil Southeast (Vinhedo)**.
3. Conclua a verificação do cartão e espere o e-mail de conta pronta.

## 2. Criar a VM

1. No console: **☰ → Compute → Instances → Create instance**. Nome: `agora`.
2. **Image:** Canonical Ubuntu 24.04.
3. **Shape → Change shape → Ampere → VM.Standard.A1.Flex**, com **1 OCPU e 6 GB**.
   O gateway sobra nisso e fica dentro da cota gratuita (2 OCPU e 12 GB no total).
4. **Networking:** deixe criar uma VCN nova com sub-rede pública e marque
   **Assign a public IPv4 address**.
5. **SSH keys:** **Generate a key pair for me** e baixe a chave privada (`.key`).
6. **Create**. Se aparecer *Out of capacity*, troque o *availability domain* ou tente
   de novo mais tarde (é comum nas contas gratuitas).
7. Anote o **Public IP address** da instância.

## 3. Abrir as portas 80 e 443 na Oracle

1. Na página da instância: clique na **sub-rede** → **Security Lists** → a lista padrão.
2. **Add Ingress Rules** duas vezes: *Source CIDR* `0.0.0.0/0`, *IP Protocol* TCP,
   *Destination Port* `80`; e o mesmo com `443`.

## 4. Entrar na VM

No PowerShell (Windows) ou no terminal:

```bash
ssh -i C:\caminho\da\chave.key ubuntu@<IP-PUBLICO>
```

Se o Windows reclamar da permissão da chave, rode antes:
`icacls C:\caminho\da\chave.key /inheritance:r /grant:r "$($env:USERNAME):R"`.

## 5. Preparar o Ubuntu

```bash
# o Ubuntu da Oracle vem com o iptables fechado: libere 80 e 443 também aqui
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save

# Docker + Compose
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
exit          # saia e entre de novo pelo ssh para o grupo docker valer
```

## 6. Baixar o projeto e configurar

```bash
git clone https://github.com/<usuario>/<repositorio>.git agora
cd agora
cp .env.cloud.example .env
openssl rand -hex 24        # copie o resultado: é o AGORA_API_TOKEN
nano .env
```

No `.env`, preencha:

- `DOMAIN`: o IP com hífens + `.sslip.io` (IP `129.146.10.20` → `129-146-10-20.sslip.io`).
- `GROQ_API_KEY` e `OPENCODE_API_KEY`.
- `AGORA_API_TOKEN`: o valor gerado acima.

Repositório privado: o `git clone` pede usuário e um *personal access token* do GitHub
(Settings → Developer settings → Fine-grained tokens, só leitura do repositório).

## 7. Subir

```bash
docker compose -f docker-compose.cloud.yml up -d --build
docker compose -f docker-compose.cloud.yml logs -f      # Ctrl+C para sair dos logs
```

Na primeira subida, o Caddy tira o certificado HTTPS (leva alguns segundos).

## 8. Conferir

- `https://<DOMAIN>/health`: os dois nós com `"up": true`.
- `https://<DOMAIN>`: crie uma sala e mande o link para a dupla.
- `https://<DOMAIN>/painel`: filas, TTFT e saúde dos nós ao vivo.
- A API pede o token:
  `curl -N https://<DOMAIN>/v1/chat -H "Authorization: Bearer <TOKEN>" -H "Content-Type: application/json" -d '{"domain":"mestre-groq","prompt":"oi"}'`

## 9. Manutenção

| O quê | Comando (na pasta `agora` da VM) |
| --- | --- |
| Atualizar o código | `git pull && docker compose -f docker-compose.cloud.yml up -d --build` |
| Ver os logs | `docker compose -f docker-compose.cloud.yml logs -f gateway` |
| Reiniciar | `docker compose -f docker-compose.cloud.yml restart gateway` |
| Parar | `docker compose -f docker-compose.cloud.yml down` |
| Trazer os logs das partidas | no seu PC: `scp -i chave.key ubuntu@<IP>:agora/logs/games.jsonl .` |

## Problemas comuns

| Sintoma | Causa provável | Solução |
| --- | --- | --- |
| O site não abre (tempo esgotado) | Porta fechada na Security List ou no iptables | Refaça os passos 3 e 5 |
| O Caddy não consegue o certificado | `DOMAIN` errado ou porta 80 fechada | Confira o IP com hífens e a porta 80 |
| A IA responde "kkkk" e frases prontas | Chave errada ou limite de taxa do Groq | `logs -f gateway`; confira as chaves; a reserva é o OpenCode |
| `401 token ausente ou inválido` | Faltou o cabeçalho na chamada da API | `-H "Authorization: Bearer <TOKEN>"` ou `export AGORA_API_TOKEN=...` |
| A página `/chat` não responde | Ela chama a API sem o token | Na nuvem, use o `chat.py` com `AGORA_API_TOKEN` |
