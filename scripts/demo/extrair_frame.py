"""Extrai frames de um vídeo e, opcionalmente, borra rostos e placas. Útil para teste manual.

Exemplos (na raiz do projeto, com o venv ativado):
    python -m scripts.demo.extrair_frame data/videos/rua.mp4 --segundo 12
    python -m scripts.demo.extrair_frame data/videos/rua.mp4 --segundo 12 --quantidade 3 --borrar
"""

import argparse
import sys
from pathlib import Path

import cv2

from app.vision.anonymizer import AnonymizationError, build_default_anonymizer
from app.vision.frames import FrameError, extract_frames


def main() -> int:
    # O terminal do Windows nem sempre usa UTF-8: sem isso, "vídeo" aparece como "vÝdeo".
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Extrai frames de um vídeo (e borra rostos e placas).")
    parser.add_argument("video", type=Path, help="caminho do vídeo (mp4, avi...)")
    parser.add_argument("--segundo", type=float, default=0, help="instante do primeiro frame, em segundos")
    parser.add_argument("--quantidade", type=int, default=1, help="quantos frames seguidos (1 por segundo)")
    parser.add_argument("--borrar", action="store_true", help="borra rostos e placas antes de salvar")
    parser.add_argument("--saida", type=Path, default=Path("data/frames"), help="pasta de saída")
    args = parser.parse_args()

    try:
        frames = extract_frames(args.video, args.segundo, count=args.quantidade)
    except FrameError as error:
        print(f"Erro ao ler o vídeo: {error}")
        return 1

    anonymizer = None
    if args.borrar:
        try:
            anonymizer = build_default_anonymizer()
        except AnonymizationError as error:
            print(f"Anonimização indisponível, nada foi salvo: {error}")
            return 1

    args.saida.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        moment = args.segundo + index
        name = f"{args.video.stem}_{moment:07.1f}s"
        if anonymizer:
            result = anonymizer.anonymize(frame)
            frame = result.image
            name += "_borrado"
            print(f"{moment:.1f}s: {len(result.faces)} rosto(s) e {len(result.plates)} placa(s) borrados")
        path = args.saida / f"{name}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"Salvo: {path} ({frame.shape[1]}x{frame.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
