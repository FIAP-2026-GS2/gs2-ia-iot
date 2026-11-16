"""Ponto de entrada da API (FastAPI).

Rodar no local:  uvicorn app.main:app --reload --port 8000
Swagger:         http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from app import __version__
from app.config import Settings, get_settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Fail fast: lê e valida a configuração na subida. Se faltar a IA_API_KEY ou houver
    # valor de exemplo, o servidor nem sobe, e o erro diz qual variável corrigir.
    get_settings()
    yield


app = FastAPI(
    title="SampaTech · Serviço de IA",
    description=(
        "Analisa frames das câmeras (com rostos e placas borrados) para encontrar problemas "
        "de zeladoria: buraco, faixa apagada, poste apagado e via alagada."
    ),
    version=__version__,
    lifespan=lifespan,
)

ComponentState = Literal["configured", "not_configured"]


class HealthComponents(BaseModel):
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
            dotnet_api=_state(settings.dotnet_configured),
            blob_storage=_state(settings.blob_configured),
            llm=_state(settings.llm_configured),
            llm_fallback=_state(settings.llm_fallback_configured),
        ),
    )
