import pytest
from pydantic import ValidationError

from app.config import Settings


def test_sem_ia_api_key_nao_sobe(monkeypatch):
    monkeypatch.delenv("IA_API_KEY", raising=False)

    with pytest.raises(ValidationError, match="ia_api_key"):
        Settings(_env_file=None)


def test_ia_api_key_curta_e_recusada(monkeypatch):
    monkeypatch.setenv("IA_API_KEY", "curta")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("valor", ["<sua-chave-aqui>", "troque-esta-chave-agora", "changeme-changeme-123"])
def test_valor_de_exemplo_e_recusado(monkeypatch, valor):
    monkeypatch.setenv("IA_API_KEY", valor)

    with pytest.raises(ValidationError, match="valor de exemplo"):
        Settings(_env_file=None)


def test_variavel_vazia_conta_como_nao_configurada(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("ISSUE_TYPE_ID_BURACO", "")

    settings = Settings(_env_file=None)

    assert settings.llm_api_key is None
    assert settings.issue_type_id_buraco is None


@pytest.mark.parametrize(("entrada", "esperado"), [("dev", "dev/"), ("/dev/", "dev/"), ("", "")])
def test_prefixo_do_blob_e_normalizado(monkeypatch, entrada, esperado):
    monkeypatch.setenv("BLOB_PREFIX", entrada)

    assert Settings(_env_file=None).blob_prefix == esperado
