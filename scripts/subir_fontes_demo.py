"""Sobe as fotos de demonstração (já borradas) ao Blob, ou baixa de volta para ver.

    python -m scripts.subir_fontes_demo                     # borra e sobe as fotos de data/amostras/cet
    python -m scripts.subir_fontes_demo --baixar fotos-demo # baixa as do Blob para a pasta fotos-demo

Por que no Blob: as fotos da CET são só para teste e não vão para o GitHub. No Blob (privado),
o agendador acha as fotos num clone limpo e também no container da nuvem: basta o .env.

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

LOCAL_DIR = Path("data/amostras/cet")
PERIODS = {"dia": "dia", "noite": "noite"}


def local_photo(camera_code: str, period: str) -> Path:
    """CET-CIDADE-JARDIM + dia → data/amostras/cet/cet_cidade_jardim_dia_1.png (o padrão dos seus prints)."""
    slug = camera_code.removeprefix("CET-").lower().replace("-", "_")
    return LOCAL_DIR / f"cet_{slug}_{period}_1.png"


def blob_name(camera_code: str, period: str) -> str:
    return f"demo/{camera_code}/{period}.jpg"


def upload(storage: BlobStorage, cameras: list[str]) -> int:
    anonymizer = get_anonymizer()
    sent = 0
    for code in cameras:
        for period in PERIODS:
            path = local_photo(code, period)
            if not path.is_file():
                print(f"  - {code} {period}: sem foto local ({path}), pulando")
                continue
            image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            try:
                blurred = anonymizer.anonymize(image)
            except AnonymizationError as error:
                print(f"  - {code} {period}: borrão falhou, NÃO enviada ({error})")
                continue
            storage.upload(blob_name(code, period), encode_jpeg(blurred.image), "image/jpeg")
            print(f"  - {code} {period}: borrada ({len(blurred.faces)} rosto(s), {len(blurred.plates)} placa(s), {len(blurred.persons)} pessoa(s); {len(blurred.unconfirmed_faces)} \"rosto(s)\" sem confirmação) e enviada → "
                  f"{blob_name(code, period)}", flush=True)
            sent += 1
    return sent


def download(storage: BlobStorage, cameras: list[str], folder: Path) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    saved = 0
    for code in cameras:
        for period in PERIODS:
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
    args = parser.parse_args()

    settings = get_settings()
    if not settings.blob_configured:
        print("Preencha BLOB_CONNECTION_STRING no .env.")
        return 1
    storage = BlobStorage.from_connection_string(settings.blob_connection_string.get_secret_value(),
                                                 settings.blob_container)  # sem prefixo: demo/ é comum
    config = json.loads(Path(settings.cameras_config_path).read_text(encoding="utf-8"))
    cameras = list(config.get("cameras") or {})

    if args.baixar:
        print(f"Baixando as fotos de demonstração para {args.baixar}:")
        return 0 if download(storage, cameras, Path(args.baixar)) else 1
    print("Borrando e enviando as fotos de demonstração:")
    sent = upload(storage, cameras)
    print(f"\n{sent} foto(s) no Blob.")
    return 0 if sent else 1


if __name__ == "__main__":
    sys.exit(main())
