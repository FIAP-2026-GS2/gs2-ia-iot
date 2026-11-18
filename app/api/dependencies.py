"""Objetos pesados criados uma vez só e reaproveitados entre as requisições."""

import logging
from functools import lru_cache

from fastapi import Depends, HTTPException, status

from app.config import Settings, get_settings
from app.integrations.blob import BlobReferenceStore, BlobStorage
from app.llm.client import LlmClient, providers_from_settings
from app.vision.anonymizer import AnonymizationError, Anonymizer, build_default_anonymizer
from app.vision.pothole import PotholeDetector, PotholeModelError
from app.vision.streetlight_roi import Camera, CameraConfigError, LocalReferenceStore, load_cameras

logger = logging.getLogger("uvicorn.error")


@lru_cache
def get_anonymizer() -> Anonymizer:
    # lru_cache não guarda exceções: se o modelo faltar, a próxima chamada tenta de novo.
    return build_default_anonymizer()


def anonymizer_dependency() -> Anonymizer:
    """Entrega o anonimizador às rotas. Sem ele, responde 503: por privacidade, nada é processado."""
    try:
        return get_anonymizer()
    except AnonymizationError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error


def llm_dependency(settings: Settings = Depends(get_settings)) -> LlmClient | None:
    """Entrega o cliente do LLM, ou None se nenhum provedor estiver no .env (aí entra o plano B)."""
    providers = providers_from_settings(settings)
    return LlmClient(providers, settings.llm_timeout_seconds) if providers else None


@lru_cache
def _pothole_detector(path: str) -> PotholeDetector:
    return PotholeDetector(path)


def pothole_dependency(settings: Settings = Depends(get_settings)) -> PotholeDetector | None:
    """Detector de buraco, ou None se o modelo faltar (a resposta avisa em notes)."""
    try:
        return _pothole_detector(settings.pothole_model_path)
    except PotholeModelError as error:
        logger.error("%s", error)
        return None


def cameras_dependency(settings: Settings = Depends(get_settings)) -> dict[str, Camera]:
    """Câmeras e retângulos dos postes. Lido a cada requisição: editar o JSON vale na hora."""
    try:
        return load_cameras(settings.cameras_config_path)
    except CameraConfigError as error:
        logger.error("%s", error)
        return {}


def reference_store_dependency(settings: Settings = Depends(get_settings)) -> LocalReferenceStore | BlobReferenceStore:
    """Referência do poste no Blob (sobrevive a um novo deploy) ou, sem Blob, em arquivo local."""
    if settings.blob_configured:
        return BlobReferenceStore(_blob_storage(settings.blob_connection_string.get_secret_value(),
                                                settings.blob_container, settings.blob_prefix))
    return LocalReferenceStore(settings.reference_dir)


@lru_cache
def _blob_storage(connection_string: str, container: str, prefix: str) -> BlobStorage:
    return BlobStorage.from_connection_string(connection_string, container, prefix)
