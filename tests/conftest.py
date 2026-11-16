import pytest
from fastapi.testclient import TestClient

from app.config import get_settings

# Chave fictícia, só para os testes (não é usada em nenhum ambiente real).
TEST_API_KEY = "chave-de-teste-0123456789abcdef"


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    """Cada teste começa com uma configuração limpa e controlada.

    - IA_ENV_FILE vazio: o teste nunca lê o seu .env de verdade.
    - Apaga do ambiente as variáveis do serviço, para o resultado não depender da sua máquina.
    - Limpa o cache do get_settings antes e depois.
    """
    monkeypatch.setenv("IA_ENV_FILE", "")
    for name in (
        "DOTNET_API_URL", "IA_USERNAME", "IA_PASSWORD", "BLOB_CONNECTION_STRING", "BLOB_PREFIX",
        "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY",
        "LLM_FALLBACK_BASE_URL", "LLM_FALLBACK_MODEL", "LLM_FALLBACK_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("IA_API_KEY", TEST_API_KEY)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client():
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
