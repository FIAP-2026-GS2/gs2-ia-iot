"""Cliente da API .NET (SampaTech), do jeito que o contrato pede.

Rotas usadas (todas exigem o papel IA_SERVICE):
- POST /api/auth/login          → token JWT de 60 min (renovado 5 min antes de vencer, ou num 401)
- GET  /api/schedules/due       → quais frames extrair agora
- POST /api/captures            → registra o frame (já borrado e no Blob)        [Idempotency-Key]
- POST /api/analyses            → registra a análise e as detecções do frame       [Idempotency-Key]

Robustez:
- 429 (limite da .NET): espera o Retry-After e tenta de novo, até 3 vezes.
- 5xx ou rede fora: espera 2, 4, 8 s e tenta de novo. Como as rotas POST usam
  Idempotency-Key, reenviar é seguro: a .NET devolve a resposta original.
- A senha nunca aparece em log, mensagem de erro ou repr.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import httpx
from pydantic import SecretStr

logger = logging.getLogger("uvicorn.error")

TOKEN_MARGIN = timedelta(minutes=5)
MAX_ATTEMPTS = 4          # 1 tentativa + 3 novas
MAX_RETRY_AFTER = 60      # segundos: a .NET costuma pedir bem menos


class DotnetApiError(RuntimeError):
    """Resposta de erro da .NET (4xx que não adianta repetir, ou falha depois das tentativas)."""

    def __init__(self, status_code: int | None, message: str):
        super().__init__(f"API .NET: {message}" + (f" (HTTP {status_code})" if status_code else ""))
        self.status_code = status_code


@dataclass(frozen=True)
class DueItem:
    """Um frame a extrair (GET /api/schedules/due)."""

    camera_id: int
    camera_code: str
    slot_at: datetime          # UTC
    reason: str                # FIXED_TIME ou TRIGGER_EVENT
    capture_schedule_id: int | None
    trigger_event_id: int | None
    analyses: tuple[str, ...]  # POTHOLE_CROSSWALK, STREETLIGHT, FLOODING

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "DueItem":
        slot = datetime.fromisoformat(data["slotAt"].replace("Z", "+00:00"))
        if slot.tzinfo is None:  # a .NET manda UTC; sem fuso no texto, assumimos UTC
            slot = slot.replace(tzinfo=timezone.utc)
        return cls(
            camera_id=data["cameraId"], camera_code=data["cameraCode"], slot_at=slot.astimezone(timezone.utc),
            reason=data["reason"], capture_schedule_id=data.get("captureScheduleId"),
            trigger_event_id=data.get("triggerEventId"), analyses=tuple(data.get("analyses") or ()),
        )


def _problem_message(response: httpx.Response) -> str:
    """Resumo do erro no formato ProblemDetails da .NET (title, detail e errors de validação)."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200] or response.reason_phrase
    parts = [body.get("detail") or body.get("title") or response.reason_phrase]
    for name, messages in (body.get("errors") or {}).items():
        parts.append(f"{name}: {'; '.join(messages)}")
    return " | ".join(str(part) for part in parts if part)


@dataclass
class DotnetClient:
    base_url: str
    username: str
    password: SecretStr = field(repr=False)
    timeout_seconds: float = 30
    transport: httpx.BaseTransport | None = field(default=None, repr=False)
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)  # trocável nos testes
    _token: str | None = field(default=None, init=False, repr=False)
    _expires_at: datetime | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._http = httpx.Client(base_url=self.base_url.rstrip("/"), timeout=self.timeout_seconds,
                                  transport=self.transport)

    # ---------- autenticação ----------

    def _login(self) -> None:
        response = self._http.post("/api/auth/login",
                                   json={"username": self.username, "password": self.password.get_secret_value()})
        if response.status_code != 200:
            raise DotnetApiError(response.status_code, f"login recusado: {_problem_message(response)}")
        body = response.json()
        self._token = body["accessToken"]
        expires = datetime.fromisoformat(body["expiresAt"].replace("Z", "+00:00"))
        self._expires_at = expires if expires.tzinfo else expires.replace(tzinfo=timezone.utc)
        logger.info("Login na API .NET ok (papel %s, token até %s)", body.get("role"), self._expires_at.isoformat())

    def _auth_header(self) -> dict[str, str]:
        if not self._token or not self._expires_at or datetime.now(timezone.utc) >= self._expires_at - TOKEN_MARGIN:
            self._login()
        return {"Authorization": f"Bearer {self._token}"}

    # ---------- requisição com novas tentativas ----------

    def _request(self, method: str, path: str, *, json: Any = None, params: dict | None = None,
                 idempotency_key: str | None = None) -> httpx.Response:
        relogged = False
        for attempt in range(1, MAX_ATTEMPTS + 1):
            headers = self._auth_header()
            if idempotency_key:
                headers["Idempotency-Key"] = idempotency_key
            try:
                response = self._http.request(method, path, json=json, params=params, headers=headers)
            except httpx.TransportError as error:  # rede fora, timeout, DNS
                if attempt == MAX_ATTEMPTS:
                    raise DotnetApiError(None, f"{method} {path} sem resposta: {type(error).__name__}") from error
                self._wait(2 ** attempt, f"{method} {path}: {type(error).__name__}")
                continue

            if response.status_code == 401 and not relogged:
                self._token, relogged = None, True  # token revogado ou relógio adiantado: entra de novo
                continue
            if response.status_code == 429 and attempt < MAX_ATTEMPTS:
                self._wait(self._retry_after(response), f"{method} {path}: HTTP 429 (limite da .NET)")
                continue
            if response.status_code >= 500 and attempt < MAX_ATTEMPTS:
                self._wait(2 ** attempt, f"{method} {path}: HTTP {response.status_code}")
                continue
            if response.status_code >= 400:
                raise DotnetApiError(response.status_code, f"{method} {path}: {_problem_message(response)}")
            return response
        raise DotnetApiError(None, f"{method} {path}: sem sucesso depois de {MAX_ATTEMPTS} tentativas")

    @staticmethod
    def _retry_after(response: httpx.Response) -> float:
        try:
            return min(float(response.headers.get("Retry-After", "5")), MAX_RETRY_AFTER)
        except ValueError:
            return 5.0

    def _wait(self, seconds: float, why: str) -> None:
        logger.warning("%s; tentando de novo em %.0f s", why, seconds)
        self.sleep(seconds)

    # ---------- rotas ----------

    def due(self, at: datetime | None = None) -> list[DueItem]:
        """Frames a extrair agora. `at` simula outro horário (útil na demonstração)."""
        params = {"at": at.astimezone(timezone.utc).isoformat()} if at else None
        response = self._request("GET", "/api/schedules/due", params=params)
        return [DueItem.from_json(item) for item in response.json()]

    def create_capture(self, payload: dict[str, Any], idempotency_key: str) -> dict[str, Any]:
        response = self._request("POST", "/api/captures", json=payload, idempotency_key=idempotency_key)
        return response.json()

    def create_analysis(self, payload: dict[str, Any], idempotency_key: str) -> dict[str, Any]:
        response = self._request("POST", "/api/analyses", json=payload, idempotency_key=idempotency_key)
        return response.json()

    def close(self) -> None:
        self._http.close()
