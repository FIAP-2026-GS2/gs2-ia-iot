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


def test_erro_de_validacao_nao_mostra_o_segredo(monkeypatch):
    monkeypatch.setenv("IA_API_KEY", "segredo-curto")

    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None)

    assert "segredo-curto" not in str(error.value)


def test_chave_faltando_nao_mostra_os_outros_segredos(monkeypatch):
    """Sem a IA_API_KEY, o pydantic listava TODAS as outras variáveis lidas no erro."""
    monkeypatch.delenv("IA_API_KEY", raising=False)
    monkeypatch.setenv("IA_PASSWORD", "senha-ficticia-999")
    monkeypatch.setenv("LLM_API_KEY", "gsk_ficticia_do_llm_123")

    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None)

    assert "senha-ficticia-999" not in str(error.value)
    assert "gsk_ficticia_do_llm_123" not in str(error.value)


def test_resumo_do_erro_usa_o_nome_da_variavel_e_esconde_o_valor(monkeypatch):
    from app.config import describe_settings_error

    monkeypatch.setenv("IA_API_KEY", "abc")

    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None)
    summary = describe_settings_error(error.value)

    assert "IA_API_KEY" in summary
    assert "16" in summary
    assert "abc" not in summary


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
