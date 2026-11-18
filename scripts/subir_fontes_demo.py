"""Sobe as fotos de teste (já borradas) ao Blob como fonte das câmeras, ou baixa de volta para ver.

    python -m scripts.subir_fontes_demo                          # borra e sobe todas as fotos da lista
    python -m scripts.subir_fontes_demo --camera DEMO-BURACO     # só uma câmera
    python -m scripts.subir_fontes_demo --baixar data/amostras/demo-blob   # baixa do Blob para ver

A lista de fotos fica em config/fontes_demo.json: cada câmera diz qual arquivo local é a foto
de "dia" (frame das 12:00) e qual é a de "noite" (00:00). O script não supõe nenhum padrão de
nome: trocar ou acrescentar uma câmera de teste é só editar o JSON.

Por que no Blob: as fotos de teste não vão para o GitHub. No Blob (privado), o agendador acha
as fotos num clone limpo e também no container da nuvem: basta o .env.

Nomes no Blob (sem o prefixo dev/, porque servem ao local e à nuvem):
    demo/<cameraCode>/dia.jpg   e   demo/<cameraCode>/noite.jpg
O config/cameras.json aponta para eles com "blob:demo/<cameraCode>/dia.jpg".

Privacidade: a foto é borrada ANTES de sair da máquina (mesmo anonimizador do /analisar).
Se o borrão falhar, a foto não sobe.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from app.api.dependencies import get_anonymizer
from app.config import get_settings
from app.integrations.blob import BlobStorage
from app.vision.anonymizer import AnonymizationError
from app.vision.frames import encode_jpeg

PHOTOS_CONFIG = Path("config/fontes_demo.json")
PERIODS = ("dia", "noite")


def load_photos(path: Path) -> dict[str, dict[str, Path]]:
    """{câmera: {período: arquivo local}} lido do JSON. Período desconhecido é erro (evita erro de digitação)."""
    config = json.loads(path.read_text(encoding="utf-8"))
    photos: dict[str, dict[str, Path]] = {}
    for code, periods in (config.get("cameras") or {}).items():
        unknown = set(periods) - set(PERIODS)
        if unknown:
            raise ValueError(f"{code}: período desconhecido {sorted(unknown)} em {path} (use 'dia' ou 'noite')")
        photos[code] = {period: Path(file) for period, file in periods.items()}
    return photos


def blob_name(camera_code: str, period: str) -> str:
    return f"demo/{camera_code}/{period}.jpg"


def upload(storage: BlobStorage, photos: dict[str, dict[str, Path]]) -> int:
    anonymizer = get_anonymizer()
    sent = 0
    for code, periods in photos.items():
        for period, path in periods.items():
            if not path.is_file():
                print(f"  - {code} {period}: arquivo não encontrado ({path}), pulando")
                continue
            image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                print(f"  - {code} {period}: não é uma imagem válida ({path}), pulando")
                continue
            try:
                blurred = anonymizer.anonymize(image)
            except AnonymizationError as error:
                print(f"  - {code} {period}: borrão falhou, NÃO enviada ({error})")
                continue
            storage.upload(blob_name(code, period), encode_jpeg(blurred.image), "image/jpeg")
            print(f"  - {code} {period}: borrada ({len(blurred.faces)} rosto(s), {len(blurred.plates)} placa(s), "
                  f"{len(blurred.persons)} cabeça(s); {len(blurred.unconfirmed_faces)} \"rosto(s)\" sem confirmação)"
                  f" e enviada → {blob_name(code, period)}", flush=True)
            sent += 1
    return sent


def download(storage: BlobStorage, photos: dict[str, dict[str, Path]], folder: Path) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    saved = 0
    for code, periods in photos.items():
        for period in periods:
            data = storage.download(blob_name(code, period))
            if data is None:
                print(f"  - {code} {period}: não está no Blob")
                continue
            target = folder / f"{code}_{period}.jpg"
            target.write_bytes(data)
            print(f"  - {code} {period} → {target}", flush=True)
            saved += 1
    return saved


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baixar", metavar="PASTA", help="baixa as fotos do Blob para esta pasta")
    parser.add_argument("--camera", help="só esta câmera (ex.: DEMO-BURACO)")
    parser.add_argument("--lista", default=str(PHOTOS_CONFIG), help=f"lista de fotos (padrão: {PHOTOS_CONFIG})")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.blob_configured:
        print("Preencha BLOB_CONNECTION_STRING no .env.")
        return 1
    try:
        photos = load_photos(Path(args.lista))
    except (OSError, ValueError) as error:
        print(f"Lista de fotos: {error}")
        return 1
    if args.camera:
        if args.camera not in photos:
            print(f"Câmera {args.camera} não está em {args.lista}.")
            return 1
        photos = {args.camera: photos[args.camera]}
    storage = BlobStorage.from_connection_string(settings.blob_connection_string.get_secret_value(),
                                                 settings.blob_container)  # sem prefixo: demo/ é comum

    if args.baixar:
        print(f"Baixando as fotos para {args.baixar}:")
        return 0 if download(storage, photos, Path(args.baixar)) else 1
    print("Borrando e enviando as fotos:")
    sent = upload(storage, photos)
    print(f"\n{sent} foto(s) no Blob.")
    return 0 if sent else 1


if __name__ == "__main__":
    sys.exit(main())
