"""Configuração do serviço, lida das variáveis de ambiente (ou do arquivo .env no local).

Regras:
- Segredo nunca tem valor padrão no código. Vem só do ambiente.
- Valor de exemplo (ex.: "<sua-chave>") é recusado na subida, para o erro aparecer na hora
  e não virar um "plano B" silencioso lá na frente.
- Integrações opcionais (.NET, Blob, LLM) ficam desligadas enquanto não forem configuradas.
"""

import os
from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Trechos que denunciam um valor copiado do .env.example sem trocar.
_EXAMPLE_MARKERS = ("<", "troque", "sua-chave", "your-key", "changeme", "example", "exemplo")

_SECRET_FIELDS = (
    "ia_api_key",
    "ia_password",
    "blob_connection_string",
    "llm_api_key",
    "llm_fallback_api_key",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Variável vazia no .env conta como "não informada", e não como texto vazio.
        env_ignore_empty=True,
        extra="ignore",
    )

    # Ambiente: muda só o que aparece no /health (local, cloud ou test).
    app_env: Literal["local", "cloud", "test"] = "local"

    # Chave que protege as rotas da IA (header X-API-Key). Obrigatória.
    ia_api_key: SecretStr = Field(min_length=16)

    # API .NET (login com o papel IA_SERVICE).
    dotnet_api_url: AnyHttpUrl | None = None
    ia_username: str | None = None
    ia_password: SecretStr | None = None

    # Azure Blob Storage: as imagens só sobem já borradas.
    blob_connection_string: SecretStr | None = None
    blob_container: str = "frames"
    # "dev/" no local, para não misturar com os frames da nuvem.
    blob_prefix: str = ""

    # IDs da tabela ISSUE_TYPE. Na nuvem são 1 a 4; no local podem ser outros.
    issue_type_id_buraco: int | None = Field(default=None, gt=0)
    issue_type_id_faixa_apagada: int | None = Field(default=None, gt=0)
    issue_type_id_poste_apagado: int | None = Field(default=None, gt=0)
    issue_type_id_via_alagada: int | None = Field(default=None, gt=0)

    # LLM principal (Groq) e reserva (Gemini), os dois no padrão da OpenAI.
    llm_base_url: AnyHttpUrl | None = None
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_fallback_base_url: AnyHttpUrl | None = None
    llm_fallback_model: str | None = None
    llm_fallback_api_key: SecretStr | None = None
    llm_timeout_seconds: float = Field(default=30, gt=0, le=120)

    @field_validator(*_SECRET_FIELDS)
    @classmethod
    def _reject_example_values(cls, value: SecretStr | None, info) -> SecretStr | None:
        if value is None:
            return value
        lowered = value.get_secret_value().strip().lower()
        if any(marker in lowered for marker in _EXAMPLE_MARKERS):
            raise ValueError(
                f"{info.field_name.upper()} ainda está com o valor de exemplo. Troque no .env."
            )
        return value

    @field_validator("blob_prefix")
    @classmethod
    def _normalize_prefix(cls, value: str) -> str:
        # "dev" e "dev/" viram "dev/"; vazio continua vazio (raiz do container).
        value = value.strip().strip("/")
        return f"{value}/" if value else ""

    # --- Quais integrações estão ligadas (usadas no /health e pelas próximas fases) ---

    @property
    def dotnet_configured(self) -> bool:
        return all((self.dotnet_api_url, self.ia_username, self.ia_password))

    @property
    def blob_configured(self) -> bool:
        return self.blob_connection_string is not None

    @property
    def llm_configured(self) -> bool:
        return all((self.llm_base_url, self.llm_model, self.llm_api_key))

    @property
    def llm_fallback_configured(self) -> bool:
        return all((self.llm_fallback_base_url, self.llm_fallback_model, self.llm_fallback_api_key))


@lru_cache
def get_settings() -> Settings:
    """Lê a configuração uma vez só e reaproveita (o cache evita reler o .env a cada requisição).

    IA_ENV_FILE permite trocar o arquivo; vazio desliga o .env (usado nos testes, para que
    eles nunca leiam os segredos reais da sua máquina).
    """
    env_file = os.getenv("IA_ENV_FILE", ".env") or None
    return Settings(_env_file=env_file)
