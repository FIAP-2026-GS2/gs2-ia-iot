"""POST /analisar: recebe um frame, borra rostos e placas e roda os cenários pedidos.

Caminho da requisição:
frame → borra cabeças, rostos e placas (sem borrão = 503) → cenários (app/analysis.py)
      → buraco (YOLO) · poste (ROI meio-dia × meia-noite) · alagamento (LLM)
      → 1 chamada ao LLM descreve tudo (sem LLM: textos fixos + motivo_sem_llm)
      → JSON no formato das detecções da .NET, ou a imagem anotada (formato=imagem)
"""

import base64
import time
from typing import Literal

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile, status
from pydantic import BaseModel, Field

from app.analysis import AnalysisInputError, AnalysisType, Period, run_analysis
from app.api.dependencies import (
    anonymizer_dependency,
    cameras_dependency,
    llm_dependency,
    pothole_dependency,
    reference_store_dependency,
)
from app.llm.client import LlmClient
from app.security import require_api_key
from app.vision.pothole import PotholeDetector
from app.vision.streetlight_roi import Camera, LocalReferenceStore
from app.vision.anonymizer import AnonymizationError, Anonymizer
from app.vision.annotate import Annotation, annotate
from app.vision.frames import InvalidImageError, decode_image, encode_jpeg

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

router = APIRouter(tags=["Análise"], dependencies=[Depends(require_api_key)])


class AnonymizationSummary(BaseModel):
    faces_blurred: int = Field(description="Rostos borrados (pessoas próximas e médias)")
    heads_blurred: int = Field(description="Cabeças borradas (pessoas de longe, a partir de 60 px de altura)")
    plates_blurred: int = Field(description="Placas borradas")
    faces_unconfirmed: int = Field(
        default=0, description="\"Rostos\" de nota média sem pessoa embaixo (ex.: desenho no chão): não borrados")
    blurred_area_percent: float = Field(description="Quanto da imagem foi coberto por borrão, em %")


class DetectionOut(BaseModel):
    category: str = Field(description="BURACO, FAIXA_APAGADA, POSTE_APAGADO ou VIA_ALAGADA")
    confidence: float = Field(ge=0, le=1)
    severity: str = Field(description="LOW, MEDIUM, HIGH ou CRITICAL (estimatedSeverity na .NET)")
    bbox: list[int] = Field(description="[x1, y1, x2, y2] em pixels (detectionCoordinates na .NET)")
    reportable: bool = Field(description="Confiança ≥ 0,5: a .NET abre ou vincula ocorrência; abaixo disso só registra")
    source: str = Field(description="Quem detectou: yolo (buraco), roi (poste) ou llm (alagamento)")
    description: str = Field(description="Texto técnico para a prefeitura (LLM, ou texto fixo sem LLM)")
    citizen_message: str = Field(description="Texto simples para o cidadão")
    model_confidence: float | None = Field(description="Buraco: confiança original do YOLO, antes da verificação")
    verified: bool | None = Field(description="Buraco: true se o LLM confirmou; null se não foi verificado (aí confidence ≤ 0,49)")
    check_reason: str = Field(description="Buraco: o que o verificador disse")


class DiscardedOut(BaseModel):
    category: str = "BURACO"
    model_confidence: float
    bbox: list[int]
    by: str = Field(description="zona (config/cameras.json) ou llm (verificador)")
    reason: str


class LampOut(BaseModel):
    lamp_id: str
    status: str = Field(description="LIT (acesa), OFF (apagada) ou UNCERTAIN")
    score: float = Field(description="Nota de 0 a 1: ≥ 0,5 acesa, ≤ 0,25 apagada")
    confidence: float
    has_reference: bool = Field(description="Comparado com a referência do meio-dia?")
    bbox: list[int]
    description: str


class LlmUsage(BaseModel):
    purpose: str = Field(description="verificar_buraco ou descrever")
    provider: str = Field(description="principal (Groq) ou reserva (Gemini)")
    model: str
    latency_ms: int


class AnalyzeImages(BaseModel):
    anonymized_jpeg_base64: str = Field(description="Frame só com o borrão (é o que sobe para o Blob)")
    annotated_jpeg_base64: str = Field(description="Frame borrado com os retângulos das detecções")


class AnalyzeResponse(BaseModel):
    analyses: list[str] = Field(description="Cenários rodados (mesmos nomes da agenda da .NET)")
    period: str | None
    camera: str | None
    anonymization: AnonymizationSummary
    detections: list[DetectionOut]
    discarded: list[DiscardedOut] = Field(description="Candidatos do YOLO descartados (alarme falso), com o motivo")
    lamps: list[LampOut] = Field(description="Resultado de cada luminária (só no STREETLIGHT da meia-noite)")
    notes: list[str] = Field(description="Avisos: o que não foi analisado e por quê")
    models: list[str] = Field(description="Modelos usados (aiModel na .NET)")
    llm_calls: list[LlmUsage] = Field(description="Chamadas ao LLM feitas neste frame (0, 1 ou 2)")
    motivo_sem_llm: str | None = Field(description="Por que o LLM não foi usado (null quando foi, ou quando não precisou)")
    processing_time_ms: int
    images: AnalyzeImages


def read_upload(file: UploadFile) -> bytes:
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
    summary="Analisa um frame: buraco, poste e alagamento (borra cabeças, rostos e placas antes de tudo)",
    responses={
        200: {"content": {"image/jpeg": {}}, "description": "JSON, ou a imagem anotada com formato=imagem"},
        401: {"description": "Chave ausente ou inválida"},
        413: {"description": "Imagem maior que 10 MB"},
        415: {"description": "O arquivo não é uma imagem"},
        422: {"description": "Imagem ilegível, câmera desconhecida, ou faltou câmera/período no STREETLIGHT"},
        503: {"description": "Anonimização indisponível: por privacidade, nada é processado"},
    },
)
def analyze(
    file: UploadFile = File(description="Foto ou frame de câmera (JPEG ou PNG)"),
    formato: Literal["json", "imagem"] = Query(
        "json", description="json (padrão) ou imagem, que devolve só o JPEG anotado (bom para o Swagger)"
    ),
    analises: list[AnalysisType] = Query(
        ["POTHOLE_CROSSWALK"],
        description="Cenários, como na agenda da .NET: POTHOLE_CROSSWALK (12:00), STREETLIGHT (12:00 e 00:00), FLOODING (alerta de chuva)",
    ),
    periodo: Period | None = Query(
        None,
        description="MEIO_DIA (o STREETLIGHT guarda a referência) ou MEIA_NOITE (o STREETLIGHT compara). Obrigatório com STREETLIGHT",
    ),
    camera: str | None = Query(
        None, description="cameraCode, como em config/cameras.json (ex.: CET-IGUATEMI). Obrigatório com STREETLIGHT"
    ),
    anonymizer: Anonymizer = Depends(anonymizer_dependency),
    pothole_detector: PotholeDetector | None = Depends(pothole_dependency),
    cameras: dict[str, Camera] = Depends(cameras_dependency),
    reference_store: LocalReferenceStore = Depends(reference_store_dependency),
    llm: LlmClient | None = Depends(llm_dependency),
) -> AnalyzeResponse | Response:
    # Função comum (def), e não async: o FastAPI roda numa thread separada, então o
    # processamento pesado da imagem não trava as outras requisições.
    started = time.perf_counter()
    if camera is not None and camera not in cameras:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Câmera {camera} não está em config/cameras.json. Disponíveis: {', '.join(sorted(cameras)) or 'nenhuma'}",
        )
    data = read_upload(file)
    try:
        image = decode_image(data)
    except InvalidImageError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error

    try:
        anonymized = anonymizer.anonymize(image)
    except AnonymizationError as error:
        # Privacidade primeiro: sem borrão, a imagem não segue adiante.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error

    try:
        result = run_analysis(
            anonymized.image,
            set(analises),
            period=periodo,
            camera=cameras.get(camera) if camera else None,
            pothole_detector=pothole_detector,
            reference_store=reference_store,
            llm=llm,
            captured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    except AnalysisInputError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error

    annotations = [Annotation(f.category, f.confidence, *f.bbox, tag=str(i)) for i, f in enumerate(result.findings, 1)]
    # Luminárias acesas ou incertas também aparecem (verde/cinza): mostra onde a regra olhou.
    annotations += [
        Annotation("POSTE_ACESO" if lamp.status == "LIT" else "POSTE_INCERTO", lamp.confidence, *lamp.roi)
        for lamp in result.lamps
        if lamp.status != "OFF"
    ]
    # Descartados em cinza, com o motivo: mostra o que o YOLO achou e por que não virou ocorrência.
    annotations += [
        Annotation("DESCARTADO", item.box.confidence, item.box.x1, item.box.y1, item.box.x2, item.box.y2,
                   tag=item.reason.split(":")[0] if item.by == "llm" else "ZONA")
        for item in result.discarded
    ]
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
                "X-Faces-Unconfirmed": str(len(anonymized.unconfirmed_faces)),
                "X-Blurred-Area-Percent": f"{anonymized.blurred_area_ratio * 100:.1f}",
                "X-Processing-Time-Ms": str(elapsed_ms),
                "X-Detections": ",".join(f.category for f in result.findings) or "nenhuma",
            },
        )

    return AnalyzeResponse(
        analyses=list(dict.fromkeys(analises)),
        period=periodo,
        camera=camera,
        anonymization=AnonymizationSummary(
            faces_blurred=len(anonymized.faces),
            heads_blurred=len(anonymized.persons),
            plates_blurred=len(anonymized.plates),
            faces_unconfirmed=len(anonymized.unconfirmed_faces),
            blurred_area_percent=round(anonymized.blurred_area_ratio * 100, 1),
        ),
        detections=[
            DetectionOut(
                category=f.category, confidence=f.confidence, severity=f.severity, bbox=list(f.bbox),
                reportable=f.reportable, source=f.source, description=f.description,
                citizen_message=f.citizen_message, model_confidence=f.model_confidence, verified=f.verified,
                check_reason=f.check_reason,
            )
            for f in result.findings
        ],
        discarded=[
            DiscardedOut(model_confidence=d.box.confidence, bbox=[d.box.x1, d.box.y1, d.box.x2, d.box.y2],
                         by=d.by, reason=d.reason)
            for d in result.discarded
        ],
        lamps=[
            LampOut(lamp_id=lamp.lamp_id, status=lamp.status, score=lamp.score, confidence=lamp.confidence,
                    has_reference=lamp.has_reference, bbox=list(lamp.roi), description=lamp.describe())
            for lamp in result.lamps
        ],
        notes=result.notes,
        models=result.models,
        llm_calls=[LlmUsage(**vars(call)) for call in result.llm_calls],
        motivo_sem_llm=result.motivo_sem_llm,
        processing_time_ms=elapsed_ms,
        images=AnalyzeImages(
            anonymized_jpeg_base64=base64.b64encode(encode_jpeg(anonymized.image)).decode(),
            annotated_jpeg_base64=base64.b64encode(annotated_jpeg).decode(),
        ),
    )
