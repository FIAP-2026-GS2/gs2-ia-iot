"""Proteção das rotas da IA por uma chave no header X-API-Key.

Quem chama: a API Java (verificação do poste) e nós, pelo Swagger (demonstração).
O mobile nunca chama a IA direto: a chave ficaria dentro do app, e qualquer pessoa extrai um APK.
"""

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from app.config import Settings, get_settings

# auto_error=False: sem o header, nós mesmos respondemos 401 com uma mensagem clara.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="Chave do serviço de IA")


def require_api_key(
    provided: str | None = Depends(api_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    expected = settings.ia_api_key.get_secret_value()
    # compare_digest leva o mesmo tempo acertando ou errando: medir o tempo da resposta
    # não ajuda a adivinhar a chave caractere por caractere (timing attack).
    if not provided or not secrets.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Chave ausente ou inválida no header X-API-Key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
