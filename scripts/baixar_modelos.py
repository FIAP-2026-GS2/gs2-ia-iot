"""Baixa os modelos grandes demais para o GitHub e confere o SHA-256 de cada um.

    python -m scripts.baixar_modelos

O SHA-256 é como o lacre de um pacote: se o arquivo baixado mudar um único byte
(download cortado ou arquivo trocado na origem), o hash não bate e o script recusa.
"""

import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelFile:
    path: Path
    url: str
    sha256: str
    description: str


MODELS = [
    ModelFile(
        ROOT / "models" / "anonymizer" / "rf-detr-nano-384-coco.onnx",
        "https://github.com/ankandrew/open-image-models/releases/download/assets/rf-detr-nano-384-coco.onnx",
        "fdcdd1c566ed4ab4d3700ca1ce6f03edadfec2a0655432e40e4c2734623feaec",
        "Detector de pessoas (RF-DETR nano, COCO, Apache 2.0, 114 MB)",
    ),
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure(model: ModelFile) -> bool:
    if model.path.is_file() and sha256_of(model.path) == model.sha256:
        print(f"OK (já existe): {model.path.name}")
        return True

    print(f"Baixando {model.description}...")
    model.path.parent.mkdir(parents=True, exist_ok=True)
    partial = model.path.with_suffix(".part")
    urllib.request.urlretrieve(model.url, partial)  # noqa: S310 (URL fixa, HTTPS)

    actual = sha256_of(partial)
    if actual != model.sha256:
        partial.unlink()
        print(f"ERRO: SHA-256 não confere para {model.path.name} (esperado {model.sha256}, veio {actual}).")
        return False

    # Só troca o nome depois de conferido: nunca fica um modelo pela metade com o nome certo.
    partial.replace(model.path)
    print(f"OK (baixado e conferido): {model.path.name}")
    return True


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    results = [ensure(model) for model in MODELS]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
