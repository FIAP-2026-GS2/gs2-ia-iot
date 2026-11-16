"""Objetos pesados criados uma vez só e reaproveitados entre as requisições."""

from functools import lru_cache

from fastapi import Depends, HTTPException, status

from app.config import Settings, get_settings
from app.llm.client import LlmClient, providers_from_settings
from app.vision.anonymizer import AnonymizationError, Anonymizer, build_default_anonymizer


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
