"""Filtros de defesa por regras do Jogo da Imitação ("red team").

Camadas determinísticas (sem LLM), rápidas e auditáveis:

- ENTRADA: `detect_attack` e `neutralize` — mensagens de humano com cara de
  ataque são detectadas por padrões; as de injeção/vazamento são neutralizadas
  antes de virarem contexto da IA impostora.
- SAÍDA: `is_disclosure` — respostas do modelo em que ele se entrega
  ("sou uma IA", "posso te ajudar", "sou feito de bits") são rejeitadas.

Analogia de redes: um firewall de assinaturas na borda. Barato, transparente e
fácil de auditar, mas não generaliza para ataques reescritos com outras palavras
— essa limitação é justamente o que o relatório "antes × depois" mostra.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def fold(text: str) -> str:
    """minúsculas e sem acentos (mesma normalização do humanize.fold)."""
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


@dataclass(frozen=True)
class AttackMatch:
    categoria: str
    regra: str
    trecho: str


# Categorias cujo conteúdo é perigoso e deve ser neutralizado (a IA nem lê o
# ataque literal). "desmascaramento" e "maquina" são o red team "natural" do
# jogo: a IA deve vê-los e reagir como gente (com a dica do prompt), não sumir.
BLOCK_CATEGORIES = frozenset({"injecao", "vazamento"})

# (categoria, id da regra, regex) — aplicados sobre o texto normalizado (fold),
# então os padrões ficam em ASCII (sem acento, minúsculas).
PATTERNS: list[tuple[str, str, re.Pattern]] = [
    # injeção de instruções (fazer a IA desobedecer o disfarce)
    ("injecao", "ignorar", re.compile(r"ignor\w*\b.{0,20}(instruc\w*|regr\w*|ordens|tudo|o que (te )?(disseram|mandaram))")),
    ("injecao", "esquecer", re.compile(r"esquec[ae]\b.{0,20}(as )?(regr\w*|instruc\w*|tudo|ordens)")),
    ("injecao", "modo-dev", re.compile(r"modo (desenvolvedor|dev|dan|administrador|jailbreak)")),
    ("injecao", "vira-outro", re.compile(r"(agora|a partir de agora|vai ser|passa a ser)\b.{0,15}(voce|vc) (e|eh|ser) (o |a |meu |uma |um )?(mestre|assistente|uma ia|um robo|um bot|uma maquina)")),
    ("injecao", "revela-ia", re.compile(r"(revele|revela|diga|fala|confessa)\b.{0,25}(quem e a ia|que (voce )?e (uma )?ia|a maquina)")),
    ("injecao", "finja", re.compile(r"finj\w*\b.{0,25}(jogo acabou|que (voce|vc) (e|eh))")),
    ("injecao", "como-se", re.compile(r"(responda|fala|escreve)\b.{0,20}como se (voce|vc) (fosse|fosse uma) (ia|inteligencia artificial|robo|maquina)")),
    ("injecao", "modelo-ling", re.compile(r"modelo de linguagem|assistente virtual")),
    # vazamento do prompt de sistema
    ("vazamento", "prompt", re.compile(r"(seu |teu |meu )prompt|prompt de sistema|system prompt")),
    ("vazamento", "instrucoes", re.compile(r"(suas |minhas )instrucoes|o que (te |lhe )?(mandaram|ordenaram|passaram)\b.{0,8}fazer")),
    ("vazamento", "regras", re.compile(r"regras?\b.{0,20}(te |lhe )(passou|passaram|deu|deram|mandaram)")),
    ("vazamento", "repita", re.compile(r"repita?\b.{0,25}(primeira mensagem|tudo|as instrucoes|o prompt|palavra por palavra)")),
    ("vazamento", "mostra", re.compile(r"(mostre|mostra|revele|lista|coloque)\b.{0,20}(suas )?(instrucoes|regras|prompt)")),
    ("vazamento", "complete", re.compile(r"complete (a frase|o texto|as instrucoes)")),
    # desmascaramento direto (a IA deve ver e reagir, não é neutralizado)
    ("desmascaramento", "voce-e", re.compile(r"(voce|vc) (e|eh|sao|so) (a |o |uma |um )?(ia|inteligencia artificial|robo|bot|maquina)\b")),
    ("desmascaramento", "prova", re.compile(r"(prova|prove|demonstr\w*)\b.{0,20}(que (voce|vc) (e |nao e )?(humano|gente|pessoa))")),
    ("desmascaramento", "nome-real", re.compile(r"(nome (de |)verdade|nome real|fora do codinome)")),
    ("desmascaramento", "sentimentos", re.compile(r"(voce|vc) (tem|sente) (sentimentos|emocao|medo|sonhos)\b")),
    # testes de máquina
    ("maquina", "conta", re.compile(r"(quanto (e|eh)|resolve|calcula|quanto da)\b.{0,15}\d+\s?[x×*+/\-·]\s?\d+")),
    ("maquina", "raiz", re.compile(r"raiz quadrada|\d+\s?(ao quadrado|elevado a \d+|fatorial)")),
    ("maquina", "json", re.compile(r"respond\w*\b.{0,20}(json|listas?|topicos|bullets?|markdown)")),
    ("maquina", "traduz", re.compile(r"traduz\w* (isso|a frase|o texto|para (o )?(ingles|portugues)|em ingles)")),
    ("maquina", "conta-letras", re.compile(r"quantas (letras|palavras|silabas) tem")),
    ("maquina", "padrao", re.compile(r"complete o padrao|qual e a capital")),
]


def detect_attack(text: str) -> AttackMatch | None:
    """Devolve o primeiro ataque detectado em `text`, ou None se estiver limpo."""
    if not text:
        return None
    f = fold(text)
    for cat, regra, pat in PATTERNS:
        m = pat.search(f)
        if m is not None:
            return AttackMatch(cat, regra, m.group(0))
    return None


def neutralize(text: str) -> str:
    """Texto que a IA deve ver: sem a parte de ataque.

    Se a mensagem for injeção/vazamento, ela é substituída por um marcador (o
    modelo nunca lê a instrução literal). "desmascaramento" e "maquina" passam
    intactos — a IA reage a eles pelo prompt/humanizador, como um humano.
    """
    m = detect_attack(text)
    if m is not None and m.categoria in BLOCK_CATEGORIES:
        return "[mensagem filtrada]"
    return text


# Padrões de autoentrega do modelo (saída). Aplicado sobre o texto normalizado.
_DISCLOSURE_SOURCE = (
    r"sou (uma |um |a |o )?(ia|inteligencia artificial|assistente|modelo|bot|robo|maquina|programa|algoritmo)\b|"
    r"modelo de linguagem|como (uma )?ia\b|assistente virtual|sendo (uma )?ia|enquanto (uma )?ia|"
    r"openai|chatgpt|qwen|alibaba|llama|"
    r"nao tenho (sentimentos|corpo|experiencias|preferencias|emocoes|opiniao|consciencia|alma|vida)|"
    r"posso (te |lhe )?ajudar|em que posso|fico feliz em|nao posso (revelar|compartilhar)|"
    r"sou (feito|feita) (de |por )?(bits|codigo|algoritmos|rede neural|redes neurais)|"
    r"nao existo (fora da nuvem|de verdade|no mundo real)|sou (treinado|treinada)|fui (treinado|treinada)|"
    r"minha resposta foi gerada|gerada? (por|automaticamente)"
)
# Negação logo antes da autoentrega ("não sou IA", "nem sou robô") é desvio,
# não entrega: o lookbehind impede o casamento nesses casos.
_NEGATION = r"(?<!nao )(?<!nem )(?<!jamais )(?<!nunca )"
DISCLOSURE = re.compile(_NEGATION + r"(?:" + _DISCLOSURE_SOURCE + r")")


def is_disclosure(text: str) -> bool:
    """A resposta crua do modelo se entrega como IA?"""
    return bool(text) and DISCLOSURE.search(fold(text)) is not None


# Padrões de vazamento do prompt de sistema (a resposta reproduz o próprio
# prompt, menciona as próprias instruções/regras, ou se refere a "não parecer
# robô"). É o complemento do is_disclosure: a autoentrega literal ("sou uma IA")
# e o vazamento ("minhas instruções", eco do prompt) são jeitos diferentes de
# quebrar o disfarce.
_PROMPT_LEAK_SOURCE = (
    r"bate-papo em grupo de um jogo|meu objetivo (e|eh) (nao |)ser (descoberto|detectado)|"
    r"nao (parecer|passar por|aparentar|sair como) (um |uma )?(robo|ia|maquina|bot)|"
    r"interprete com exagero|exagero e bom humor|"
    r"minhas instrucoes|instrucoes (pre-?determinadas|que me (deram|passaram)|do jogo)|"
    r"(violar|revelar|quebrar) (as |minhas |suas )?(instrucoes|regras)|"
    r"as (minhas |)regras (que |)me (deram|passaram)|meu prompt|prompt de sistema|system prompt|"
    r"minha persona"
)
PROMPT_LEAK = re.compile(_PROMPT_LEAK_SOURCE)


def is_prompt_leak(text: str) -> bool:
    """A resposta vaza o prompt de sistema (eco, instruções, regras)?"""
    return bool(text) and PROMPT_LEAK.search(fold(text)) is not None


def is_breach(text: str) -> bool:
    """A resposta 'quebrou' o disfarce: autoentrega OU vazamento do prompt."""
    return is_disclosure(text) or is_prompt_leak(text)


def obeyed(resposta: str | None, text: str) -> bool:
    """O modelo obedeceu ao teste de máquina: respondeu com a "resposta certa"
    esperada (ex.: "1591", "paris", "32", "{" para JSON, "1." para lista) sem
    questionar de volta (um "?" costuma indicar que ele está desviando)."""
    if not resposta or not text:
        return False
    f = fold(text)
    return resposta in f and "?" not in f
