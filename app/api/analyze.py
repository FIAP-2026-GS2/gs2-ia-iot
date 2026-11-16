"""POST /analisar: recebe uma imagem, borra rostos e placas e devolve o resultado.

Nesta fase a rota já faz a parte de privacidade e devolve a imagem anotada.
Os detectores de problemas (buraco, faixa, poste e alagamento) entram nas próximas fases.
"""

import base64
import time
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile, status
from pydantic import BaseModel, Field

from app.api.dependencies import anonymizer_dependency
from app.security import require_api_key
from app.vision.anonymizer import AnonymizationError, Anonymizer
from app.vision.annotate import Annotation, annotate
from app.vision.frames import InvalidImageError, decode_image, encode_jpeg

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

router = APIRouter(tags=["Análise"], dependencies=[Depends(require_api_key)])


class AnonymizationSummary(BaseModel):
    faces_blurred: int = Field(description="Rostos borrados (pessoas próximas e médias)")
    heads_blurred: int = Field(description="Cabeças borradas (pessoas de longe, a partir de 60 px de altura)")
    plates_blurred: int = Field(description="Placas borradas")
    blurred_area_percent: float = Field(description="Quanto da imagem foi coberto por borrão, em %")


class DetectionOut(BaseModel):
    category: str = Field(description="BURACO, FAIXA_APAGADA, POSTE_APAGADO ou VIA_ALAGADA")
    confidence: float = Field(ge=0, le=1)
    bbox: list[int] = Field(description="[x1, y1, x2, y2] em pixels")


class AnalyzeImages(BaseModel):
    anonymized_jpeg_base64: str = Field(description="Frame só com o borrão (é o que sobe para o Blob)")
    annotated_jpeg_base64: str = Field(description="Frame borrado com os retângulos das detecções")


class AnalyzeResponse(BaseModel):
    anonymization: AnonymizationSummary
    detections: list[DetectionOut]
    processing_time_ms: int
    images: AnalyzeImages


def _read_upload(file: UploadFile) -> bytes:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Tipo de arquivo não aceito ({file.content_type}). Envie uma imagem JPEG ou PNG.",
        )
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Imagem maior que 10 MB.")
    return data


@router.post(
    "/analisar",
    response_model=AnalyzeResponse,
    summary="Analisa uma imagem (borra cabeças, rostos e placas antes de tudo)",
    responses={
        200: {"content": {"image/jpeg": {}}, "description": "JSON, ou a imagem anotada com formato=imagem"},
        401: {"description": "Chave ausente ou inválida"},
        413: {"description": "Imagem maior que 10 MB"},
        415: {"description": "O arquivo não é uma imagem"},
        422: {"description": "Imagem corrompida ou ilegível"},
        503: {"description": "Anonimização indisponível: por privacidade, nada é processado"},
    },
)
def analyze(
    file: UploadFile = File(description="Foto ou frame de câmera (JPEG ou PNG)"),
    formato: Literal["json", "imagem"] = Query(
        "json", description="json (padrão) ou imagem, que devolve só o JPEG anotado (bom para o Swagger)"
    ),
    anonymizer: Anonymizer = Depends(anonymizer_dependency),
) -> AnalyzeResponse | Response:
    # Função comum (def), e não async: o FastAPI roda numa thread separada, então o
    # processamento pesado da imagem não trava as outras requisições.
    started = time.perf_counter()
    data = _read_upload(file)
    try:
        image = decode_image(data)
    except InvalidImageError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error

    try:
        anonymized = anonymizer.anonymize(image)
    except AnonymizationError as error:
        # Privacidade primeiro: sem borrão, a imagem não segue adiante.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error

    # As detecções de problemas entram aqui nas próximas fases.
    annotations: list[Annotation] = []
    annotated = annotate(anonymized.image, annotations)
    annotated_jpeg = encode_jpeg(annotated)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if formato == "imagem":
        return Response(
            content=annotated_jpeg,
            media_type="image/jpeg",
            headers={
                "X-Faces-Blurred": str(len(anonymized.faces)),
                "X-Heads-Blurred": str(len(anonymized.persons)),
                "X-Plates-Blurred": str(len(anonymized.plates)),
                "X-Blurred-Area-Percent": f"{anonymized.blurred_area_ratio * 100:.1f}",
                "X-Processing-Time-Ms": str(elapsed_ms),
            },
        )

    return AnalyzeResponse(
        anonymization=AnonymizationSummary(
            faces_blurred=len(anonymized.faces),
            heads_blurred=len(anonymized.persons),
            plates_blurred=len(anonymized.plates),
            blurred_area_percent=round(anonymized.blurred_area_ratio * 100, 1),
        ),
        detections=[
            DetectionOut(category=a.label, confidence=a.confidence, bbox=[a.x1, a.y1, a.x2, a.y2])
            for a in annotations
        ],
        processing_time_ms=elapsed_ms,
        images=AnalyzeImages(
            anonymized_jpeg_base64=base64.b64encode(encode_jpeg(anonymized.image)).decode(),
            annotated_jpeg_base64=base64.b64encode(annotated_jpeg).decode(),
        ),
    )
