"""Leitura de frames de vídeo e conversão de imagens.

O vídeo público simula a gravação de uma câmera do Smart Sampa. Para a IA, um frame é só
uma matriz de números (altura × largura × 3 canais BGR, de 0 a 255), igual a uma foto.
"""

from pathlib import Path

import cv2
import numpy as np


class FrameError(Exception):
    """Não foi possível obter o frame pedido (vídeo ausente, corrompido ou tempo fora do vídeo)."""


class InvalidImageError(Exception):
    """O conteúdo recebido não é uma imagem que o OpenCV consiga ler."""


# Qualidade fixa: o mesmo frame gera sempre os mesmos bytes, e portanto o mesmo SHA-256.
# Isso mantém o corpo do POST /api/captures igual num reenvio (a Idempotency-Key não dá 422).
JPEG_QUALITY = 90


def extract_frame(video_path: str | Path, at_seconds: float) -> np.ndarray:
    """Devolve o frame do vídeo no instante pedido (em segundos desde o início)."""
    return extract_frames(video_path, at_seconds, count=1)[0]


def extract_frames(
    video_path: str | Path, at_seconds: float, count: int = 3, step_seconds: float = 1.0
) -> list[np.ndarray]:
    """Devolve `count` frames seguidos, a partir de `at_seconds`, separados por `step_seconds`.

    Usado no poste: 2 ou 3 frames seguidos evitam que o farol de um carro passando
    seja confundido com a lâmpada acesa.
    """
    if count < 1:
        raise ValueError("count precisa ser pelo menos 1.")
    if at_seconds < 0 or step_seconds <= 0:
        raise ValueError("at_seconds não pode ser negativo e step_seconds precisa ser positivo.")

    path = Path(video_path)
    if not path.is_file():
        raise FrameError(f"Vídeo não encontrado: {path}")

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise FrameError(f"Não foi possível abrir o vídeo: {path.name}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        total = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        if fps <= 0 or total <= 0:
            raise FrameError(f"Vídeo sem informação de FPS ou de duração: {path.name}")

        duration = total / fps
        frames: list[np.ndarray] = []
        for index in range(count):
            moment = at_seconds + index * step_seconds
            frame_number = int(round(moment * fps))
            if frame_number >= total:
                raise FrameError(
                    f"O instante {moment:.1f}s passa do fim do vídeo ({duration:.1f}s)."
                )
            # Posicionar pelo número do frame é mais preciso que por milissegundos.
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise FrameError(f"Falha ao ler o frame {frame_number} de {path.name}.")
            frames.append(frame)
        return frames
    finally:
        capture.release()


def decode_image(data: bytes) -> np.ndarray:
    """Converte bytes (JPEG, PNG...) numa imagem BGR. Recusa o que não for imagem."""
    if not data:
        raise InvalidImageError("Arquivo vazio.")
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise InvalidImageError("O arquivo não é uma imagem válida (JPEG ou PNG).")
    return image


def encode_jpeg(image: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise InvalidImageError("Falha ao gerar o JPEG.")
    return buffer.tobytes()
