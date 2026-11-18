"""O caminho completo de um frame agendado, da câmera até a ocorrência na .NET.

    frame da câmera (vídeo público)
      → borra cabeças, rostos e placas         (sem borrão: para aqui; nada sai do serviço)
      → roda os cenários pedidos pela agenda   (app/analysis.py)
      → sobe o JPEG borrado ao Blob            (nome determinístico: <câmera>/<slotAt>.jpg)
      → POST /api/captures                     (Idempotency-Key capture:<cameraId>:<slotAt>)
      → POST /api/analyses                     (Idempotency-Key analysis:<imageId>)
      → a .NET decide: abre ocorrência, vincula a uma existente, ou só registra (< 0,5)

O agendador (próxima fase) chama process_frame para cada item do GET /api/schedules/due.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app import __version__
from app.analysis import AnalysisResult, Finding, run_analysis
from app.config import Settings
from app.integrations.blob import BlobStorage, slot_text
from app.integrations.dotnet import DotnetApiError, DotnetClient, DueItem
from app.llm.client import LlmClient
from app.vision.anonymizer import AnonymizationError, Anonymizer
from app.vision.frames import encode_jpeg, limit_size
from app.vision.pothole import PotholeDetector
from app.vision.streetlight_roi import Camera

logger = logging.getLogger("uvicorn.error")

AI_MODEL_MAX = 100
DESCRIPTION_MAX = 1000


class PipelineConfigError(RuntimeError):
    """Falta configuração para enviar à .NET (ex.: id do tipo de problema)."""


def issue_type_ids(settings: Settings) -> dict[str, int]:
    ids = {
        "BURACO": settings.issue_type_id_buraco,
        "FAIXA_APAGADA": settings.issue_type_id_faixa_apagada,
        "POSTE_APAGADO": settings.issue_type_id_poste_apagado,
        "VIA_ALAGADA": settings.issue_type_id_via_alagada,
    }
    missing = [name for name, value in ids.items() if value is None]
    if missing:
        raise PipelineConfigError("Preencha no .env: " + ", ".join(f"ISSUE_TYPE_ID_{name}" for name in missing))
    return ids  # type: ignore[return-value]


# ---------- corpos das requisições ----------

def capture_key(item: DueItem) -> str:
    return f"capture:{item.camera_id}:{slot_text(item.slot_at)}"


def analysis_key(image_id: int) -> str:
    # Uma análise por imagem: a .NET só aceita analisar uma captura PENDING.
    return f"analysis:{image_id}"


def capture_payload(item: DueItem, file_name: str, url: str, sha256: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "captureDate": item.slot_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cameraId": item.camera_id,
        "image": {"fileName": file_name, "filePath": url, "fileHash": sha256, "isAnonymized": True},
    }
    if item.reason == "TRIGGER_EVENT":
        payload |= {"source": "EVENT", "triggerEventId": item.trigger_event_id}
    else:
        payload |= {"source": "RECORDING_FRAME", "captureScheduleId": item.capture_schedule_id}
    return payload


def coordinates(bbox: tuple[int, int, int, int]) -> str:
    """A .NET guarda "x,y,largura,altura"; nós trabalhamos com os dois cantos (x1,y1,x2,y2)."""
    x1, y1, x2, y2 = bbox
    return f"{max(0, x1)},{max(0, y1)},{max(1, x2 - x1)},{max(1, y2 - y1)}"


def detection_payload(finding: Finding, type_ids: dict[str, int]) -> dict[str, Any]:
    # A .NET tem um campo de texto só; a rota pública mostra esse texto ao cidadão,
    # então vão os dois: o técnico (prefeitura) e o simples (cidadão).
    text = finding.description
    if finding.citizen_message:
        text = f"{text} | Para o cidadão: {finding.citizen_message}"
    return {
        "issueTypeId": type_ids[finding.category],
        "confidenceScore": round(float(finding.confidence), 3),
        "estimatedSeverity": finding.severity,
        "detectionCoordinates": coordinates(finding.bbox),
        "description": text[:DESCRIPTION_MAX],
    }


def analysis_payload(image_id: int, result: AnalysisResult | None, processing_ms: int,
                     type_ids: dict[str, int]) -> dict[str, Any]:
    """result None = a análise falhou: status FAILED e nenhuma detecção (regra da .NET)."""
    models = "+".join(result.models) if result and result.models else "sampatech-ia"
    return {
        "imageId": image_id,
        "aiModel": models[:AI_MODEL_MAX],
        "modelVersion": __version__,
        "status": "COMPLETED" if result is not None else "FAILED",
        "analysisDate": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "processingTimeMs": processing_ms,
        "detections": [detection_payload(f, type_ids) for f in result.findings] if result else [],
    }


# ---------- o caminho de um frame ----------

@dataclass
class FrameOutcome:
    camera_code: str
    slot_at: datetime
    status: str                 # SENT, ALREADY_SENT, ANONYMIZATION_FAILED, ANALYSIS_FAILED
    image_id: int | None = None
    detections: list[dict] | None = None  # o que a .NET respondeu para cada detecção
    message: str = ""


def _period(item: DueItem) -> str | None:
    """Agenda das 12:00 → MEIO_DIA; das 00:00 → MEIA_NOITE (no horário de São Paulo, UTC−3)."""
    if item.reason != "FIXED_TIME":
        return None
    local_hour = (item.slot_at.astimezone(timezone.utc).hour - 3) % 24
    return "MEIO_DIA" if 6 <= local_hour < 18 else "MEIA_NOITE"


def process_frame(
    item: DueItem,
    frame: np.ndarray,
    *,
    anonymizer: Anonymizer,
    dotnet: DotnetClient,
    blob: BlobStorage,
    type_ids: dict[str, int],
    camera: Camera | None,
    pothole_detector: PotholeDetector | None,
    reference_store: Any,
    llm: LlmClient | None,
) -> FrameOutcome:
    started = time.perf_counter()
    frame = limit_size(frame)  # mesma regra das rotas: lado maior até 1920 px
    try:
        anonymized = anonymizer.anonymize(frame)
    except AnonymizationError as error:
        # Sem borrão, o frame não sobe ao Blob nem vira captura. A agenda tenta de novo depois.
        logger.error("%s %s: anonimização falhou, nada enviado (%s)", item.camera_code, slot_text(item.slot_at), error)
        return FrameOutcome(item.camera_code, item.slot_at, "ANONYMIZATION_FAILED", message=str(error))

    result: AnalysisResult | None
    try:
        result = run_analysis(anonymized.image, set(item.analyses), period=_period(item), camera=camera,
                              pothole_detector=pothole_detector, reference_store=reference_store, llm=llm,
                              captured_at=item.slot_at.isoformat())
        failure = ""
    except Exception as error:  # noqa: BLE001 — qualquer falha (inclusive AnalysisInputError) vira análise FAILED
        logger.exception("%s %s: análise falhou", item.camera_code, slot_text(item.slot_at))
        result, failure = None, f"{type(error).__name__}: {error}"

    file_name, url, sha256 = blob.upload_frame(item.camera_code, item.slot_at, encode_jpeg(anonymized.image))
    capture = dotnet.create_capture(capture_payload(item, file_name, url, sha256), capture_key(item))
    image_id = capture["images"][0]["id"]

    payload = analysis_payload(image_id, result, int((time.perf_counter() - started) * 1000), type_ids)
    try:
        answer = dotnet.create_analysis(payload, analysis_key(image_id))
    except DotnetApiError as error:
        # 400 "já foi analisada" ou 422 (mesma chave, corpo diferente: o LLM escreveu outro texto
        # num reprocessamento). Nos dois casos a análise desta imagem já está registrada.
        if error.status_code == 422 or (error.status_code == 400 and "analisada" in str(error)):
            return FrameOutcome(item.camera_code, item.slot_at, "ALREADY_SENT", image_id, message=str(error))
        raise
    status = "SENT" if result is not None else "ANALYSIS_FAILED"
    return FrameOutcome(item.camera_code, item.slot_at, status, image_id, answer.get("detections", []), failure)


def build_clients(settings: Settings) -> tuple[DotnetClient, BlobStorage]:
    """Cria os clientes a partir do .env. Falta algo? Erro claro na hora, e não no meio da madrugada."""
    if not settings.dotnet_configured:
        raise PipelineConfigError("Preencha DOTNET_API_URL, IA_USERNAME e IA_PASSWORD no .env.")
    if not settings.blob_configured:
        raise PipelineConfigError("Preencha BLOB_CONNECTION_STRING no .env.")
    dotnet = DotnetClient(str(settings.dotnet_api_url), settings.ia_username, settings.ia_password)
    blob = BlobStorage.from_connection_string(settings.blob_connection_string.get_secret_value(),
                                              settings.blob_container, settings.blob_prefix)
    return dotnet, blob
