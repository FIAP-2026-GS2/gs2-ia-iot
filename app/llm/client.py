"""Cliente do LLM de visão, no padrão de API da OpenAI (POST {base_url}/chat/completions).

O Groq e o Gemini aceitam esse mesmo formato. Por isso, trocar de provedor é só trocar
LLM_BASE_URL, LLM_MODEL e LLM_API_KEY no .env, sem mudar o código.

Ordem: o principal (Groq) primeiro; se ele falhar (fora do ar, limite 429, timeout ou resposta
que não é JSON), tenta o reserva (Gemini). Se os dois falharem, levanta LlmUnavailableError com
o motivo, e quem chamou usa o plano B (texto pronto ou regra de brilho) com motivo_sem_llm.
"""

import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import SecretStr

from app.config import Settings

logger = logging.getLogger("uvicorn.error")

# Resposta do modelo às vezes vem dentro de ```json ... ```; pegamos só o objeto.
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class LlmUnavailableError(RuntimeError):
    """Nenhum provedor respondeu um JSON válido. A mensagem vira o motivo_sem_llm."""


@dataclass(frozen=True)
class LlmProvider:
    name: str
    base_url: str
    model: str
    # repr=False: a chave nunca aparece num print ou log do objeto.
    api_key: SecretStr = field(repr=False)

    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


@dataclass(frozen=True)
class LlmAnswer:
    data: dict[str, Any]
    provider: str
    model: str
    latency_ms: int
    # Falhas dos provedores anteriores (ex.: o Groq caiu e o Gemini respondeu).
    previous_failures: list[str] = field(default_factory=list)


def parse_json_object(text: str) -> dict[str, Any]:
    """Extrai o primeiro objeto JSON do texto do modelo. Levanta ValueError se não houver."""
    match = _JSON_OBJECT.search(text or "")
    if not match:
        raise ValueError("a resposta não tem um objeto JSON")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("a resposta não é um objeto JSON")
    return data


class LlmClient:
    def __init__(
        self,
        providers: list[LlmProvider],
        timeout_seconds: float = 30,
        transport: httpx.BaseTransport | None = None,
    ):
        if not providers:
            raise ValueError("Informe ao menos um provedor de LLM.")
        self.providers = providers
        self._timeout = timeout_seconds
        # transport: nos testes, trocamos a internet por respostas falsas (httpx.MockTransport).
        self._transport = transport

    def ask_json(self, system: str, prompt: str, image_jpeg: bytes | None = None) -> LlmAnswer:
        """Manda a pergunta (e a imagem, já borrada) e devolve o JSON da resposta."""
        failures: list[str] = []
        for provider in self.providers:
            started = time.perf_counter()
            try:
                content = self._call(provider, system, prompt, image_jpeg)
                data = parse_json_object(content)
            except httpx.TimeoutException:
                failures.append(f"{provider.name}: sem resposta em {self._timeout:.0f} s")
                continue
            except httpx.HTTPStatusError as error:
                failures.append(f"{provider.name}: HTTP {error.response.status_code}")
                continue
            except httpx.HTTPError as error:
                failures.append(f"{provider.name}: falha de conexão ({type(error).__name__})")
                continue
            except (ValueError, KeyError, IndexError, TypeError):
                failures.append(f"{provider.name}: resposta fora do formato JSON")
                continue
            latency_ms = int((time.perf_counter() - started) * 1000)
            if failures:
                logger.warning("LLM %s respondeu depois de falha(s): %s", provider.name, "; ".join(failures))
            return LlmAnswer(data, provider.name, provider.model, latency_ms, failures)
        logger.warning("Nenhum LLM respondeu: %s", "; ".join(failures))
        raise LlmUnavailableError("; ".join(failures))

    def _call(self, provider: LlmProvider, system: str, prompt: str, image_jpeg: bytes | None) -> str:
        user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_jpeg is not None:
            # A imagem vai dentro do JSON, em base64 (texto), como uma "data URL".
            encoded = base64.b64encode(image_jpeg).decode()
            user_content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
            )
        body = {
            "model": provider.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            # Baixa temperatura: respostas mais estáveis (menos "criatividade").
            "temperature": 0.1,
            "max_tokens": 400,
        }
        headers = {"Authorization": f"Bearer {provider.api_key.get_secret_value()}"}
        with httpx.Client(timeout=self._timeout, transport=self._transport) as http:
            response = http.post(provider.chat_url, json=body, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]


def providers_from_settings(settings: Settings) -> list[LlmProvider]:
    """Monta a lista na ordem: principal, depois reserva. Só entra quem está configurado."""
    providers = []
    if settings.llm_configured:
        providers.append(
            LlmProvider("principal", str(settings.llm_base_url), settings.llm_model, settings.llm_api_key)
        )
    if settings.llm_fallback_configured:
        providers.append(
            LlmProvider(
                "reserva",
                str(settings.llm_fallback_base_url),
                settings.llm_fallback_model,
                settings.llm_fallback_api_key,
            )
        )
    return providers
