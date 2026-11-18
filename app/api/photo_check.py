"""POST /verificar-foto: a foto que o cidadão mandou serve como prova do conserto (ou da falta dele)?

Quem chama: a API .NET, quando o cidadão confirma ou contesta pelo app o conserto de uma ocorrência
resolvida. O objetivo é barrar o que claramente não serve (selfie, pássaro, print de tela) antes de
a foto chegar à equipe; o que passa ainda é revisado por uma pessoa.

Caminho da requisição:
    foto → reduz para até 1920 px → borra cabeças, rostos e placas → LLM de visão (só a foto borrada)
         → 3 perguntas de sim ou não → a REGRA, aqui no código, decide se vale

Por que a regra fica no código e não no LLM: o LLM só descreve o que vê ("é uma via pública?",
"aparece uma calçada?", "dá para ver o buraco?"). Quem decide é uma regra fixa, testável e igual
para todo mundo. Mudar a regra é mudar uma linha, sem reescrever prompt.

Sem borrão ou sem LLM, responde 503: por privacidade e segurança, nada passa sem as duas etapas.
"""

import base64
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field, ValidationError

from app.api.analyze import read_upload
from app.api.dependencies import anonymizer_dependency, llm_dependency
from app.llm.client import LlmClient, LlmUnavailableError
from app.llm.prompts import PHOTO_CHECK_SYSTEM, PHOTO_CHECK_TEXTS, photo_check_prompt
from app.security import require_api_key
from app.vision.anonymizer import AnonymizationError, Anonymizer
from app.vision.frames import InvalidImageError, decode_image, encode_jpeg, limit_size

router = APIRouter(tags=["Foto do cidadão"], dependencies=[Depends(require_api_key)])

Category = Literal["BURACO", "FAIXA_APAGADA", "POSTE_APAGADO", "VIA_ALAGADA"]
Decision = Literal["CONFIRM", "CONTEST"]

# O LLM recebe a foto menor: ver se é rua ou pássaro não precisa de 1920 px, e a chamada fica mais rápida.
LLM_IMAGE_SIDE = 1024
ACCEPTED = "Foto conferida pela IA."


class LlmPhotoAnswer(BaseModel):
    """Formato que exigimos do LLM. Resposta diferente conta como falha (503)."""

    via_publica: bool
    local_do_tipo: bool
    mostra_o_problema: bool
    confianca: float = Field(ge=0, le=1)
    descricao: str = Field(min_length=1, max_length=500)


class LlmUsed(BaseModel):
    provedor: str = Field(description="principal (Groq) ou reserva (Gemini)")
    modelo: str
    latencia_ms: int


class PhotoCheckResponse(BaseModel):
    valida: bool = Field(description="true se a foto serve como prova (ainda passa pela revisão da equipe)")
    motivo: str = Field(description="Frase para o cidadão: por que foi recusada, ou 'Foto conferida pela IA.'")
    descricao: str = Field(description="O que o LLM viu na foto")
    confianca: float = Field(ge=0, le=1)
    via_publica: bool
    local_do_tipo: bool
    mostra_o_problema: bool
    rostos_borrados: int
    placas_borradas: int
    pessoas_borradas: int
    llm: LlmUsed
    foto_borrada_base64: str = Field(description="JPEG borrado em base64: é o que a .NET guarda no Blob")


def decide(answer: LlmPhotoAnswer, category: str, decision: str) -> tuple[bool, str]:
    """A regra. CONTEST precisa mostrar o problema; CONFIRM precisa mostrar o lugar do tipo certo."""
    problem, place = PHOTO_CHECK_TEXTS[category]
    seen = answer.descricao.strip().rstrip(".")
    if not answer.via_publica:
        return False, f"A foto não mostra uma via pública: {seen}."
    if decision == "CONTEST" and not answer.mostra_o_problema:
        return False, f"A foto não mostra {problem}: {seen}."
    if decision == "CONFIRM" and not answer.local_do_tipo:
        return False, f"A foto não mostra {place}: {seen}."
    return True, ACCEPTED


@router.post(
    "/verificar-foto",
    response_model=PhotoCheckResponse,
    summary="Confere se a foto do cidadão serve como prova do conserto (borra antes, LLM depois)",
    responses={
        400: {"description": "O arquivo não é uma imagem válida"},
        401: {"description": "Chave ausente ou inválida no header X-API-Key"},
        413: {"description": "Imagem maior que 10 MB"},
        415: {"description": "Tipo de arquivo não aceito"},
        422: {"description": "categoria ou decisao inválidas"},
        503: {"description": "Sem borrão, sem LLM ou resposta do LLM fora do formato: nada foi conferido"},
    },
)
def verify_photo(
    file: UploadFile = File(description="Foto do cidadão (JPEG ou PNG)"),
    categoria: Category = Form(description="Tipo da ocorrência"),
    decisao: Decision = Form(description="CONFIRM (diz que foi consertado) ou CONTEST (diz que continua)"),
    anonymizer: Anonymizer = Depends(anonymizer_dependency),
    llm: LlmClient | None = Depends(llm_dependency),
) -> PhotoCheckResponse:
    data = read_upload(file)
    try:
        image = limit_size(decode_image(data))
    except InvalidImageError as error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(error)) from error

    # 1. Privacidade: o LLM (um serviço de fora) só vê a foto já borrada.
    try:
        blurred = anonymizer.anonymize(image)
    except AnonymizationError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error

    # 2. As três perguntas ao LLM.
    if llm is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "LLM não configurado (variáveis LLM_*).")
    try:
        answer = llm.ask_json(PHOTO_CHECK_SYSTEM, photo_check_prompt(categoria, decisao),
                              encode_jpeg(limit_size(blurred.image, LLM_IMAGE_SIDE)))
    except LlmUnavailableError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"LLM indisponível ({error}).") from error
    try:
        parsed = LlmPhotoAnswer.model_validate(answer.data)
    except ValidationError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            f"O LLM ({answer.provider}) respondeu fora do formato combinado.") from error

    # 3. A regra, no código.
    valid, reason = decide(parsed, categoria, decisao)
    return PhotoCheckResponse(
        valida=valid,
        motivo=reason,
        descricao=parsed.descricao,
        confianca=round(parsed.confianca, 2),
        via_publica=parsed.via_publica,
        local_do_tipo=parsed.local_do_tipo,
        mostra_o_problema=parsed.mostra_o_problema,
        rostos_borrados=len(blurred.faces),
        placas_borradas=len(blurred.plates),
        pessoas_borradas=len(blurred.persons),
        llm=LlmUsed(provedor=answer.provider, modelo=answer.model, latencia_ms=answer.latency_ms),
        foto_borrada_base64=base64.b64encode(encode_jpeg(blurred.image)).decode("ascii"),
    )
