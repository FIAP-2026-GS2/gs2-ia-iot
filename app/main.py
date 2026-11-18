"""Ponto de entrada da API (FastAPI).

Rodar no local:  uvicorn app.main:app --reload --port 8000
Swagger:         http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI
from pydantic import BaseModel

import logging

from app import __version__
from app.api.analyze import router as analyze_router
from app.api.scheduler import router as scheduler_router
from app.api.streetlight import router as streetlight_router
from app.api.dependencies import get_anonymizer
from pydantic import ValidationError

from app.config import Settings, describe_settings_error, get_settings
from app.vision.anonymizer import AnonymizationError

# Usa o logger do próprio uvicorn: as mensagens saem no mesmo formato ("ERROR:    ...").
logger = logging.getLogger("uvicorn.error")


class ConfigurationError(RuntimeError):
    """Configuração inválida: o servidor não sobe."""


@asynccontextmanager
async def lifespan(application: FastAPI):
    # Fail fast: lê e valida a configuração na subida. Se faltar a IA_API_KEY ou houver
    # valor de exemplo, o servidor nem sobe, e o erro diz qual variável corrigir.
    try:
        settings = get_settings()
    except ValidationError as error:
        summary = describe_settings_error(error)
        logger.error(summary)
        # "from None" corta o erro original do pydantic do traceback: a última linha que
        # aparece no terminal é o resumo legível, sem nenhum valor do .env.
        raise ConfigurationError(summary) from None
    # Carrega os modelos de anonimização já na subida, para a 1ª requisição não esperar.
    # Se falharem, a API sobe mesmo assim (o /health mostra), mas o /analisar responde 503.
    try:
        get_anonymizer()
    except AnonymizationError as error:
        logger.error("Anonimização indisponível: %s", error)
    # Agendador: só com SCHEDULER_ENABLED=true (no local e nos testes fica desligado).
    application.state.scheduler = None
    if settings.scheduler_enabled:
        from app.scheduler import FrameScheduler

        application.state.scheduler = FrameScheduler(settings)
        application.state.scheduler.start()
    yield
    if application.state.scheduler is not None:
        application.state.scheduler.stop()


app = FastAPI(
    title="SampaTech · Serviço de IA",
    description=(
        "Analisa frames das câmeras (com rostos e placas borrados) para encontrar problemas "
        "de zeladoria: buraco, faixa apagada, poste apagado e via alagada."
    ),
    version=__version__,
    lifespan=lifespan,
)
app.include_router(analyze_router)
app.include_router(streetlight_router)
app.include_router(scheduler_router)

ComponentState = Literal["configured", "not_configured"]


class HealthComponents(BaseModel):
    anonymizer: Literal["ready", "unavailable"]
    dotnet_api: ComponentState
    blob_storage: ComponentState
    llm: ComponentState
    llm_fallback: ComponentState


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
    environment: str
    components: HealthComponents


def _state(configured: bool) -> ComponentState:
    return "configured" if configured else "not_configured"


def _anonymizer_state() -> Literal["ready", "unavailable"]:
    # Os modelos são locais (arquivos .onnx): conferir é instantâneo e não chama nada externo.
    try:
        get_anonymizer()
        return "ready"
    except AnonymizationError:
        return "unavailable"


@app.get("/health", response_model=HealthResponse, tags=["Saúde"])
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Diz se o processo está de pé. Chamado pelo health check da API .NET e pelo ACI.

    Não testa a conexão com a .NET, o Blob nem o LLM, de propósito:
    - a .NET consulta este /health com limite de 3 s, então ele precisa ser instantâneo;
    - se ele chamasse a .NET, e a .NET chama ele, um ficaria esperando o outro (dependência circular).
    Por isso mostra só se cada integração está configurada, sem nenhum segredo.
    """
    return HealthResponse(
        status="ok",
        service="sampatech-ia",
        version=__version__,
        environment=settings.app_env,
        components=HealthComponents(
            anonymizer=_anonymizer_state(),
            dotnet_api=_state(settings.dotnet_configured),
            blob_storage=_state(settings.blob_configured),
            llm=_state(settings.llm_configured),
            llm_fallback=_state(settings.llm_fallback_configured),
        ),
    )
