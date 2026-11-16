import json

import httpx
import pytest
from pydantic import SecretStr

from app.config import Settings
from app.llm.client import LlmClient, LlmProvider, LlmUnavailableError, parse_json_object, providers_from_settings

KEY_PRINCIPAL = "gsk_chave_ficticia_principal"
KEY_RESERVA = "chave_ficticia_reserva"
PRINCIPAL = LlmProvider("principal", "https://groq.teste/openai/v1", "modelo-a", SecretStr(KEY_PRINCIPAL))
RESERVA = LlmProvider("reserva", "https://gemini.teste/v1beta/openai/", "modelo-b", SecretStr(KEY_RESERVA))


def _chat(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def _client(handler, providers=(PRINCIPAL, RESERVA)) -> LlmClient:
    # MockTransport: o httpx não sai para a internet; quem responde é a função handler.
    return LlmClient(list(providers), timeout_seconds=5, transport=httpx.MockTransport(handler))


def test_extrai_json_mesmo_dentro_de_bloco_de_codigo():
    assert parse_json_object('Claro!\n```json\n{"lampada": "ACESA"}\n```') == {"lampada": "ACESA"}


def test_texto_sem_json_e_erro():
    with pytest.raises(ValueError):
        parse_json_object("Não sei dizer.")


def test_envia_imagem_e_chave_no_formato_openai():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return _chat('{"ok": true}')

    answer = _client(handler).ask_json("sistema", "pergunta", b"\xff\xd8jpeg")

    assert answer.data == {"ok": True}
    assert answer.provider == "principal"
    assert seen["url"] == "https://groq.teste/openai/v1/chat/completions"
    assert seen["auth"] == f"Bearer {KEY_PRINCIPAL}"
    image_part = seen["body"]["messages"][1]["content"][1]
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize(
    "falha_principal, motivo",
    [
        (lambda: httpx.Response(429, json={"error": "rate limit"}), "HTTP 429"),
        (lambda: httpx.Response(500), "HTTP 500"),
        (lambda: _chat("desculpe, não consigo"), "fora do formato JSON"),
    ],
    ids=["limite-429", "erro-500", "sem-json"],
)
def test_principal_falha_e_reserva_responde(falha_principal, motivo):
    def handler(request: httpx.Request):
        if "groq" in request.url.host:
            return falha_principal()
        return _chat('{"lampada": "ACESA"}')

    answer = _client(handler).ask_json("s", "p")

    assert answer.provider == "reserva"
    assert answer.model == "modelo-b"
    assert any(motivo in failure for failure in answer.previous_failures)


def test_timeout_conta_como_falha():
    def handler(request: httpx.Request):
        if "groq" in request.url.host:
            raise httpx.ReadTimeout("demorou", request=request)
        return _chat('{"ok": 1}')

    answer = _client(handler).ask_json("s", "p")

    assert answer.provider == "reserva"
    assert "sem resposta em 5 s" in answer.previous_failures[0]


def test_todos_falham_levanta_erro_com_os_motivos_e_sem_as_chaves():
    def handler(request: httpx.Request):
        raise httpx.ConnectError("sem rede", request=request)

    with pytest.raises(LlmUnavailableError) as caught:
        _client(handler).ask_json("s", "p")

    message = str(caught.value)
    assert "principal: falha de conexão" in message
    assert "reserva: falha de conexão" in message
    assert KEY_PRINCIPAL not in message and KEY_RESERVA not in message


def test_chave_nao_aparece_no_repr_do_provedor():
    assert KEY_PRINCIPAL not in repr(PRINCIPAL)


def test_monta_so_os_provedores_configurados():
    settings = Settings(
        _env_file=None,
        ia_api_key="chave-de-teste-0123456789abcdef",
        llm_fallback_base_url="https://gemini.teste/v1beta/openai/",
        llm_fallback_model="modelo-b",
        llm_fallback_api_key=KEY_RESERVA,
    )

    assert [p.name for p in providers_from_settings(settings)] == ["reserva"]
