"""Instruções (prompts) enviadas ao LLM. Ficam juntas para ser fácil revisar e citar no README."""

STREETLIGHT_SYSTEM = (
    "Você verifica a iluminação pública de São Paulo a partir de fotos. "
    "Responda somente com um objeto JSON, sem texto fora dele. "
    "Rostos e placas na foto já foram borrados de propósito, por privacidade; ignore isso. "
    "Não invente: se a foto não mostra uma luminária de iluminação pública, ou não é noturna, diga isso."
)

STREETLIGHT_PROMPT = """Esta foto foi tirada à noite por um técnico, depois do conserto de um poste.
A lâmpada do poste está acesa?

Responda neste formato JSON:
{
  "luminaria_visivel": true ou false,
  "foto_noturna": true ou false,
  "lampada": "ACESA", "APAGADA" ou "INDETERMINADA",
  "confianca": número de 0 a 1,
  "descricao": "uma frase curta, em português, dizendo o que você vê"
}

Regras:
- luminaria_visivel: true se aparece a luminária de iluminação pública, mesmo em close-up e sem o poste inteiro.
- ACESA: a luminária emite luz (ponto claro na luminária, luz no chão abaixo dela).
- APAGADA: a luminária aparece, a foto é noturna e ela não emite luz.
- INDETERMINADA: a luminária não aparece, a foto não é noturna, ou não dá para saber.
- Luz de farol, janela ou letreiro não conta como luminária pública."""


# ---------- /analisar: alagamento e descrição das detecções (uma chamada só por frame) ----------

SCENE_SYSTEM = (
    "Você ajuda a zeladoria urbana de São Paulo a descrever problemas vistos por câmeras de rua. "
    "Responda somente com um objeto JSON, sem texto fora dele. "
    "Rostos e placas já foram borrados de propósito, por privacidade; ignore isso e nunca tente identificar pessoas. "
    "Não invente: descreva só o que aparece na imagem. Os problemas já detectados vêm marcados com "
    "retângulos numerados; não descarte nem crie detecções, só descreva as que recebeu."
)

FLOODING_PART = """
"alagamento": {
  "alagada": true ou false,
  "gravidade": "LOW", "MEDIUM", "HIGH" ou "CRITICAL",
  "confianca": número de 0 a 1,
  "descricao": "uma frase técnica para a equipe da prefeitura",
  "mensagem_cidadao": "uma frase simples e calma para o cidadão"
}"""

FLOODING_RULES = """
Alagamento:
- alagada: true só se há água acumulada sobre a pista (lâmina d'água cobrindo parte da via).
  Asfalto molhado, poça pequena, reflexo de luz ou chuva caindo NÃO é alagamento.
- gravidade: LOW = poças grandes, carros passam; MEDIUM = lâmina d'água em uma faixa;
  HIGH = água cobrindo a pista inteira; CRITICAL = água na altura das rodas ou carros parados."""


def scene_prompt(detections: list[dict], ask_flooding: bool) -> str:
    """Monta o pedido conforme o frame: só descrições, só alagamento, ou os dois."""
    parts = []
    if ask_flooding:
        parts.append(FLOODING_PART.strip())
    parts.append(
        '"deteccoes": [\n  {"id": número, "descricao": "frase técnica", "mensagem_cidadao": "frase simples"}\n]'
    )
    listing = "\n".join(
        f"- id {d['id']}: {d['categoria']} (gravidade {d['gravidade']}, confiança {d['confianca']:.0%})"
        for d in detections
    ) or "- nenhuma"
    return (
        "Imagem de uma câmera de rua, já com rostos e placas borrados.\n\n"
        f"Detecções feitas pelos nossos modelos (retângulos numerados na imagem):\n{listing}\n\n"
        "Responda neste formato JSON:\n{\n" + ",\n".join(parts) + "\n}\n"
        + (FLOODING_RULES if ask_flooding else "")
        + "\nDescrições: uma por detecção da lista, com o mesmo id. Lista vazia se não houver detecções."
        "\nNa descricao, diga o que se vê e onde (ex.: faixa da direita, perto da calçada)."
        "\nNa mensagem_cidadao, nada de termos técnicos, porcentagens ou nomes de modelos."
    )


# ---------- Verificador do buraco (segunda opinião sobre as caixas do YOLO) ----------

POTHOLE_CHECK_SYSTEM = (
    "Você inspeciona o pavimento de ruas de São Paulo. Responda somente com um objeto JSON. "
    "Seja cético: um detector automático marcou candidatos, e muitos são alarmes falsos."
)


def pothole_check_prompt(count: int) -> str:
    return f"""A imagem é um mosaico com {count} recorte(s) numerado(s) de câmeras de rua.
Em cada recorte, um detector automático marcou com um retângulo vermelho um possível buraco.
Para cada número, diga o que está DENTRO do retângulo vermelho (use o entorno só como contexto).

Responda neste formato JSON:
{{
  "candidatos": [
    {{"id": número, "objeto": "BURACO", "SOMBRA", "BUEIRO_OU_TAMPA", "REMENDO", "MANCHA_OU_POCA", "SINALIZACAO", "VEICULO_OU_PESSOA" ou "OUTRO",
      "confianca": número de 0 a 1, "motivo": "frase curta"}}
  ]
}}

Regras:
- BURACO: cavidade no asfalto, com bordas quebradas ou irregulares (pode ter água ou terra dentro).
- Sombra projetada, tampa de bueiro, grelha, remendo plano, mancha, poça rasa, faixa pintada,
  veículo ou pessoa NÃO são buraco.
- Um item por número, de 1 a {count}. Se não der para ver, use OUTRO com confiança baixa."""
