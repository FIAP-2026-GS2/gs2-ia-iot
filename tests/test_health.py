def test_health_responde_200_sem_integracoes(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "sampatech-ia"
    assert body["environment"] == "test"
    assert body["components"] == {
        "anonymizer": "ready",
        "dotnet_api": "not_configured",
        "blob_storage": "not_configured",
        "llm": "not_configured",
        "llm_fallback": "not_configured",
    }


def test_health_mostra_integracoes_configuradas(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("DOTNET_API_URL", "http://localhost:5242")
    monkeypatch.setenv("IA_USERNAME", "ia.teste")
    monkeypatch.setenv("IA_PASSWORD", "senha-ficticia-123")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("LLM_MODEL", "modelo-qualquer")
    monkeypatch.setenv("LLM_API_KEY", "gsk_ficticia_123456")
    get_settings.cache_clear()

    body = client.get("/health").json()

    assert body["components"]["dotnet_api"] == "configured"
    assert body["components"]["llm"] == "configured"
    assert body["components"]["blob_storage"] == "not_configured"


def test_health_nunca_expoe_segredos(client, monkeypatch):
    from app.config import get_settings
    from tests.conftest import TEST_API_KEY

    monkeypatch.setenv("IA_PASSWORD", "senha-ficticia-123")
    get_settings.cache_clear()

    text = client.get("/health").text

    assert TEST_API_KEY not in text
    assert "senha-ficticia-123" not in text
