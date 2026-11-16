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
