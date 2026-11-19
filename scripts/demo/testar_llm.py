"""Confere os provedores de LLM do .env: lista os modelos e mede a latência de uma pergunta com imagem.

Uso:
    python -m scripts.demo.testar_llm --listar                 # modelos que a sua chave enxerga
    python -m scripts.demo.testar_llm                          # pergunta do poste com uma foto sintética
    python -m scripts.demo.testar_llm --imagem foto.jpg        # pergunta do poste com uma foto sua

Atenção: a foto vai para fora (Groq/Gemini) SEM passar pelo borrão. Use só fotos sem pessoas
nem placas, ou passe pela rota /verificar-poste, que borra antes de enviar.
Nunca imprime as chaves.
"""

import argparse
import time

import cv2
import httpx
import numpy as np

from app.config import get_settings
from app.llm.client import LlmClient, LlmUnavailableError, providers_from_settings
from app.llm.prompts import STREETLIGHT_PROMPT, STREETLIGHT_SYSTEM
from app.vision.frames import decode_image, encode_jpeg
from app.vision.streetlight import shrink


def synthetic_night_photo() -> np.ndarray:
    """Cena noturna desenhada: céu escuro, poste cinza e uma luminária acesa com o cone de luz."""
    image = np.full((480, 640, 3), 18, dtype=np.uint8)
    cv2.rectangle(image, (0, 380), (640, 480), (40, 40, 40), -1)  # calçada
    cv2.rectangle(image, (300, 120), (312, 380), (90, 90, 90), -1)  # poste
    cv2.rectangle(image, (312, 120), (370, 130), (90, 90, 90), -1)  # braço
    cv2.ellipse(image, (365, 140), (60, 120), 0, 0, 180, (60, 110, 140), -1)  # cone de luz
    cv2.circle(image, (365, 135), 12, (255, 255, 255), -1)  # lâmpada
    return image


def list_models(settings) -> None:
    """Lista os modelos de cada provedor. Só precisa da URL e da chave (o modelo pode estar vazio)."""
    candidates = [
        ("principal", settings.llm_base_url, settings.llm_api_key, settings.llm_model),
        ("reserva", settings.llm_fallback_base_url, settings.llm_fallback_api_key, settings.llm_fallback_model),
    ]
    for name, base_url, api_key, model in candidates:
        if not (base_url and api_key):
            print(f"[{name}] sem URL ou chave no .env: pulei")
            continue
        url = str(base_url).rstrip("/") + "/models"
        headers = {"Authorization": f"Bearer {api_key.get_secret_value()}"}
        try:
            response = httpx.get(url, headers=headers, timeout=15)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            print(f"[{name}] não consegui listar: HTTP {error.response.status_code}")
            continue
        except httpx.HTTPError as error:
            print(f"[{name}] não consegui listar: {type(error).__name__}")
            continue
        names = sorted(item.get("id", "?") for item in response.json().get("data", []))
        print(f"[{name}] {base_url} · {len(names)} modelos (no .env: {model or 'VAZIO'})")
        for item in names:
            same = model and item.removeprefix("models/") == model.removeprefix("models/")
            print(f"   {item}{'  <- no .env' if same else ''}")


def ask(client: LlmClient, image: np.ndarray) -> None:
    jpeg = encode_jpeg(shrink(image))
    print(f"Imagem enviada: {len(jpeg) / 1024:.0f} KB")
    # Testa cada provedor sozinho, para ver a latência e a resposta de cada um.
    for provider in client.providers:
        single = LlmClient([provider], client._timeout)
        started = time.perf_counter()
        try:
            answer = single.ask_json(STREETLIGHT_SYSTEM, STREETLIGHT_PROMPT, jpeg)
        except LlmUnavailableError as error:
            print(f"[{provider.name}] FALHOU em {time.perf_counter() - started:.1f} s: {error}")
            continue
        print(f"[{provider.name}] {provider.model} · {answer.latency_ms} ms")
        print(f"   {answer.data}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--listar", action="store_true", help="lista os modelos de cada provedor")
    parser.add_argument("--imagem", help="foto para a pergunta do poste (sem pessoas nem placas)")
    args = parser.parse_args()

    settings = get_settings()
    if args.listar:
        list_models(settings)
        return

    providers = providers_from_settings(settings)
    if not providers:
        raise SystemExit("Nenhum LLM completo no .env: preencha a URL, o MODEL e a chave (rode --listar para ver os modelos).")
    client = LlmClient(providers, settings.llm_timeout_seconds)
    # np.fromfile + decode: o cv2.imread falha com acento no caminho do Windows.
    image = decode_image(np.fromfile(args.imagem, dtype=np.uint8).tobytes()) if args.imagem else synthetic_night_photo()
    ask(client, image)


if __name__ == "__main__":
    main()
