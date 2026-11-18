"""POST /verificar-poste: confere, por uma foto noturna, se o poste consertado acendeu.

Quem chama é a API Java (depois que o técnico envia a foto pelo app). O app nunca chama a IA
direto: app → Java (JWT) → IA (X-API-Key).

Caminho da requisição:
foto → borra cabeças, rostos e placas → reduz para 1024 px → LLM de visão → resposta
                                                        └→ LLM fora do ar: regra de brilho (plano B)
"""

import logging
import time
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field, ValidationError

from app.api.analyze import AnonymizationSummary, read_upload
from app.api.dependencies import anonymizer_dependency, llm_dependency
from app.llm.client import LlmClient, LlmUnavailableError
from app.llm.prompts import STREETLIGHT_PROMPT, STREETLIGHT_SYSTEM
from app.security import require_api_key
from app.vision.anonymizer import AnonymizationError, Anonymizer
from app.vision.frames import InvalidImageError, decode_image, encode_jpeg
from app.vision.streetlight import LLM_MIN_CONFIDENCE, StreetlightStatus, read_brightness, shrink

logger = logging.getLogger("uvicorn.error")

router = APIRouter(tags=["Poste"], dependencies=[Depends(require_api_key)])


class LlmStreetlightAnswer(BaseModel):
    """Formato que exigimos do LLM. Se ele responder diferente, contamos como falha."""

    luminaria_visivel: bool  # a luminária basta: o técnico costuma fotografar só ela, em close
    foto_noturna: bool
    lampada: Literal["ACESA", "APAGADA", "INDETERMINADA"]
    confianca: float = Field(ge=0, le=1)
    descricao: str = Field(min_length=1, max_length=500)


class LlmInfo(BaseModel):
    provider: str = Field(description="principal (Groq) ou reserva (Gemini)")
    model: str
    latency_ms: int


class StreetlightCheckResponse(BaseModel):
    status: StreetlightStatus = Field(description="LIT (acesa), OFF (apagada) ou UNCERTAIN (não dá para saber)")
    repaired: bool | None = Field(description="true se acesa, false se apagada, null se indeterminado")
    confidence: float = Field(ge=0, le=1)
    method: Literal["llm", "brightness"] = Field(description="llm, ou brightness (regra de brilho, plano B)")
    description: str = Field(description="Explicação curta, em português")
    motivo_sem_llm: str | None = Field(description="Por que o LLM não foi usado (null quando foi)")
    llm: LlmInfo | None
    anonymization: AnonymizationSummary
    processing_time_ms: int


_STATUS_FROM_LLM: dict[str, StreetlightStatus] = {"ACESA": "LIT", "APAGADA": "OFF", "INDETERMINADA": "UNCERTAIN"}
_REPAIRED: dict[str, bool | None] = {"LIT": True, "OFF": False, "UNCERTAIN": None}


def _from_llm(answer: LlmStreetlightAnswer) -> tuple[StreetlightStatus, float]:
    status_ = _STATUS_FROM_LLM[answer.lampada]
    # Sem luminária na foto, de dia, ou com pouca certeza: não confirmamos o conserto.
    if not answer.luminaria_visivel or not answer.foto_noturna or answer.confianca < LLM_MIN_CONFIDENCE:
        status_ = "UNCERTAIN"
    return status_, answer.confianca


@router.post(
    "/verificar-poste",
    response_model=StreetlightCheckResponse,
    summary="Confere, por uma foto noturna, se a lâmpada do poste consertado está acesa",
    responses={
        401: {"description": "Chave ausente ou inválida"},
        413: {"description": "Imagem maior que 10 MB"},
        415: {"description": "O arquivo não é uma imagem"},
        422: {"description": "Imagem corrompida ou ilegível"},
        503: {"description": "Anonimização indisponível: por privacidade, a foto não vai ao LLM"},
    },
)
def verify_streetlight(
    file: UploadFile = File(description="Foto noturna do poste, tirada depois do conserto (JPEG ou PNG)"),
    anonymizer: Anonymizer = Depends(anonymizer_dependency),
    llm: LlmClient | None = Depends(llm_dependency),
) -> StreetlightCheckResponse:
    started = time.perf_counter()
    data = read_upload(file)
    try:
        image = decode_image(data)
    except InvalidImageError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error

    try:
        anonymized = anonymizer.anonymize(image)
    except AnonymizationError as error:
        # Privacidade primeiro: sem borrão, a foto não sai do serviço (nem para o LLM).
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error

    llm_info = None
    motivo_sem_llm = None
    if llm is None:
        motivo_sem_llm = "LLM não configurado (variáveis LLM_*)"
    else:
        try:
            answer = llm.ask_json(STREETLIGHT_SYSTEM, STREETLIGHT_PROMPT, encode_jpeg(shrink(anonymized.image)))
            parsed = LlmStreetlightAnswer.model_validate(answer.data)
        except LlmUnavailableError as error:
            motivo_sem_llm = f"LLM indisponível ({error})"
        except ValidationError:
            motivo_sem_llm = f"LLM ({answer.provider}) respondeu fora do formato combinado"
        else:
            status_, confidence = _from_llm(parsed)
            description = parsed.descricao
            method = "llm"
            llm_info = LlmInfo(provider=answer.provider, model=answer.model, latency_ms=answer.latency_ms)

    if motivo_sem_llm is not None:
        # Plano B: a regra de brilho. Nunca devolvemos erro só porque o LLM caiu.
        logger.warning("verificar-poste usando a regra de brilho: %s", motivo_sem_llm)
        reading = read_brightness(anonymized.image)
        status_, confidence, description, method = reading.status, reading.confidence, reading.describe(), "brightness"

    return StreetlightCheckResponse(
        status=status_,
        repaired=_REPAIRED[status_],
        confidence=round(confidence, 2),
        method=method,
        description=description,
        motivo_sem_llm=motivo_sem_llm,
        llm=llm_info,
        anonymization=AnonymizationSummary(
            faces_blurred=len(anonymized.faces),
            heads_blurred=len(anonymized.persons),
            plates_blurred=len(anonymized.plates),
            blurred_area_percent=round(anonymized.blurred_area_ratio * 100, 1),
        ),
        processing_time_ms=int((time.perf_counter() - started) * 1000),
    )
