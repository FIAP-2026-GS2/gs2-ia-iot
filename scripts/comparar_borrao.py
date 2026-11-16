"""Compara as estratégias de anonimização lado a lado, para decidir com imagens reais.

    python -m scripts.comparar_borrao data/amostras/
    python -m scripts.comparar_borrao foto1.jpg foto2.jpg

Para cada imagem, gera em data/comparacao/ um painel com 7 quadros:
    ORIGINAL | ATUAL (rosto + placa) | B (rosto 2x + placa) | A (pessoa inteira) | Cabeça (topo e centro da pessoa)
    | Combinada (rosto 2x + cabeça fatiada) | Regra 60px (combinada sem quem está longe demais)
e imprime uma tabela com o que cada uma borrou, a área apagada e o tempo.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from app.vision.anonymizer import (
    MIN_FACE_WIDTH_PX,
    MIN_PERSON_HEIGHT_PX,
    Anonymizer,
    RfDetrPersonDetector,
    YoloPlateDetector,
    YuNetFaceDetector,
)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".jfif", ".webp", ".bmp"}


def build_variants() -> dict[str, Anonymizer]:
    plates = YoloPlateDetector()
    faces = YuNetFaceDetector()
    return {
        "ATUAL": Anonymizer(faces, plates),
        "B rosto 2x": Anonymizer(YuNetFaceDetector(upscale=2.0), plates),
        "A pessoa": Anonymizer(faces, plates, RfDetrPersonDetector()),
        # Medido nas fotos de teste: de corpo inteiro, o rosto termina em ~15% da altura da pessoa
        # e fica entre 32% e 77% da largura. 20% da altura e 60% da largura (centro) cobrem a cabeça
        # com folga. De perto (pessoa cortada), quem acha o rosto é o YuNet.
        "Cabeça": Anonymizer(faces, plates, RfDetrPersonDetector(top_fraction=0.2, width_fraction=0.6)),
        # Combinação: rosto ampliado 2x (de perto/médio) + cabeça com fatiamento 2x2 (multidão).
        "Combinada": Anonymizer(
            YuNetFaceDetector(upscale=2.0), plates,
            RfDetrPersonDetector(top_fraction=0.2, width_fraction=0.6, tiles=2),
        ),
        # Combinada + regra "longe demais para reconhecer" (pessoa < 60 px, rosto < 8 px: não borra).
        "Regra 60px": Anonymizer(
            YuNetFaceDetector(upscale=2.0, min_width=MIN_FACE_WIDTH_PX), plates,
            RfDetrPersonDetector(top_fraction=0.2, width_fraction=0.6, tiles=2, min_height=MIN_PERSON_HEIGHT_PX),
        ),
    }


def label(image: np.ndarray, text: str) -> np.ndarray:
    """Escreve o nome da variante numa faixa preta em cima do quadro."""
    bar_height = max(28, image.shape[0] // 18)
    bar = np.zeros((bar_height, image.shape[1], 3), dtype=np.uint8)
    scale = bar_height / 40
    cv2.putText(bar, text, (8, int(bar_height * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                max(1, int(scale * 2)), cv2.LINE_AA)
    return np.vstack([bar, image])


def collect(paths: list[Path]) -> list[Path]:
    images: list[Path] = []
    for path in paths:
        if path.is_dir():
            images += sorted(p for p in path.iterdir() if p.suffix.lower() in EXTENSIONS)
        elif path.suffix.lower() in EXTENSIONS:
            images.append(path)
    return images


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Compara as estratégias de borrão lado a lado.")
    parser.add_argument("entradas", nargs="+", type=Path, help="imagens ou pastas com imagens")
    parser.add_argument("--saida", type=Path, default=Path("data/comparacao"))
    args = parser.parse_args()

    images = collect(args.entradas)
    if not images:
        print("Nenhuma imagem encontrada.")
        return 1

    variants = build_variants()
    args.saida.mkdir(parents=True, exist_ok=True)
    print(f"{'imagem':28} {'variante':12} {'rostos':>6} {'pessoas':>7} {'placas':>6} {'área borrada':>12} {'tempo':>8}")

    for path in images:
        # imdecode em vez de imread: aceita nomes com acento no Windows.
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"{path.name[:28]:28} não é uma imagem válida")
            continue

        panels = [label(image, "ORIGINAL")]
        for name, anonymizer in variants.items():
            started = time.perf_counter()
            result = anonymizer.anonymize(image)
            elapsed = (time.perf_counter() - started) * 1000
            print(f"{path.name[:28]:28} {name:12} {len(result.faces):>6} {len(result.persons):>7} "
                  f"{len(result.plates):>6} {result.blurred_area_ratio:>11.1%} {elapsed:>6.0f}ms")
            panels.append(label(result.image, f"{name}: {result.blurred_area_ratio:.0%} borrado"))

        output = args.saida / f"{path.stem}_comparacao.jpg"
        cv2.imencode(".jpg", np.hstack(panels))[1].tofile(str(output))
        print(f"{'':28} → {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
