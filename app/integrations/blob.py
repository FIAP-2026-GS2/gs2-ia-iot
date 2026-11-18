"""Azure Blob Storage: guarda o frame já borrado e a referência do meio-dia do poste.

Nomes determinísticos (o contrato com a .NET exige):
- frame:       <prefixo><cameraCode>/<slotAt em UTC, ex. 20261117T150000Z>.jpg
- referência:  <prefixo>refs/streetlight/<cameraCode>.json
Um reenvio grava o mesmo nome com os mesmos bytes: a URL e o SHA-256 não mudam, então o corpo
do POST /api/captures também não muda (com outra URL, a Idempotency-Key daria 422).

O prefixo ("dev/" no local) separa os testes da máquina dos frames da nuvem no mesmo container.
O container é privado: a URL guardada na .NET não abre sem credencial.
"""

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Protocol

from app.vision.streetlight_roi import LampMeasure


class BlobContainer(Protocol):
    """O pedaço do ContainerClient do Azure que usamos (nos testes, um falso)."""

    def upload_blob(self, name: str, data: bytes, overwrite: bool = ..., content_settings: Any = ...) -> Any: ...
    def download_blob(self, blob: str) -> Any: ...
    def get_blob_client(self, blob: str) -> Any: ...


class BlobError(RuntimeError):
    """Falha ao gravar ou ler no Blob."""


def slot_text(slot_at: datetime) -> str:
    return slot_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe(code: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in code)


class BlobStorage:
    def __init__(self, container: BlobContainer, prefix: str = ""):
        self.container = container
        self.prefix = prefix

    @classmethod
    def from_connection_string(cls, connection_string: str, container_name: str, prefix: str = "") -> "BlobStorage":
        from azure.storage.blob import BlobServiceClient  # import aqui: só carrega se o Blob estiver configurado

        service = BlobServiceClient.from_connection_string(connection_string)
        return cls(service.get_container_client(container_name), prefix)

    def frame_name(self, camera_code: str, slot_at: datetime) -> str:
        return f"{self.prefix}{_safe(camera_code)}/{slot_text(slot_at)}.jpg"

    def upload_frame(self, camera_code: str, slot_at: datetime, jpeg: bytes) -> tuple[str, str, str]:
        """Sobe o frame (JÁ BORRADO). Devolve (nome do arquivo, URL https, SHA-256)."""
        from azure.storage.blob import ContentSettings

        name = self.frame_name(camera_code, slot_at)
        try:
            self.container.upload_blob(name, jpeg, overwrite=True,
                                       content_settings=ContentSettings(content_type="image/jpeg"))
        except Exception as error:  # o SDK do Azure tem dezenas de exceções; todas viram BlobError
            raise BlobError(f"Falha ao enviar {name} ao Blob: {type(error).__name__}") from error
        url = self.container.get_blob_client(name).url
        return name.rsplit("/", 1)[-1], url, hashlib.sha256(jpeg).hexdigest()

    # ---------- arquivos com nome livre (sem o prefixo): as fontes de demonstração ----------

    def upload(self, name: str, data: bytes, content_type: str) -> None:
        from azure.storage.blob import ContentSettings

        try:
            self.container.upload_blob(name, data, overwrite=True,
                                       content_settings=ContentSettings(content_type=content_type))
        except Exception as error:
            raise BlobError(f"Falha ao enviar {name} ao Blob: {type(error).__name__}") from error

    def download(self, name: str) -> bytes | None:
        """Bytes do arquivo, ou None se ele não existir no container."""
        from azure.core.exceptions import ResourceNotFoundError

        try:
            return self.container.download_blob(name).readall()
        except ResourceNotFoundError:
            return None
        except Exception as error:
            raise BlobError(f"Falha ao ler {name} do Blob: {type(error).__name__}") from error


class BlobReferenceStore:
    """Mesma interface do LocalReferenceStore, mas no Blob: sobrevive a um novo deploy do ACI."""

    def __init__(self, storage: BlobStorage):
        self.storage = storage

    def _name(self, camera_code: str) -> str:
        return f"{self.storage.prefix}refs/streetlight/{_safe(camera_code)}.json"

    def save(self, camera_code: str, measures: dict[str, LampMeasure], captured_at: str) -> None:
        from azure.storage.blob import ContentSettings

        payload = {"camera": camera_code, "captured_at": captured_at,
                   "lamps": {lamp_id: asdict(m) for lamp_id, m in measures.items()}}
        try:
            self.storage.container.upload_blob(self._name(camera_code), json.dumps(payload).encode(), overwrite=True,
                                               content_settings=ContentSettings(content_type="application/json"))
        except Exception as error:
            raise BlobError(f"Falha ao gravar a referência de {camera_code}: {type(error).__name__}") from error

    def load(self, camera_code: str) -> tuple[dict[str, LampMeasure], str] | None:
        from azure.core.exceptions import ResourceNotFoundError

        try:
            raw = self.storage.container.download_blob(self._name(camera_code)).readall()
        except ResourceNotFoundError:
            return None
        except Exception as error:
            raise BlobError(f"Falha ao ler a referência de {camera_code}: {type(error).__name__}") from error
        payload = json.loads(raw)
        return {lamp_id: LampMeasure(**m) for lamp_id, m in payload["lamps"].items()}, payload["captured_at"]
