"""Imagem anotada: o frame (já borrado) com um retângulo, o nome do problema e a confiança.

Serve para o vídeo, a apresentação e o README: mostra o que a IA viu e onde.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Uma cor por problema (BGR), fácil de distinguir na apresentação.
COLORS = {
    "BURACO": (0, 0, 255),           # vermelho
    "FAIXA_APAGADA": (0, 215, 255),  # amarelo
    "POSTE_APAGADO": (255, 128, 0),  # azul
    "VIA_ALAGADA": (255, 255, 0),    # ciano
}
DEFAULT_COLOR = (255, 255, 255)


@dataclass(frozen=True)
class Annotation:
    label: str  # categoria, ex.: "BURACO"
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


def annotate(image: np.ndarray, annotations: list[Annotation]) -> np.ndarray:
    result = image.copy()
    # Espessura e fonte proporcionais ao tamanho da imagem (legível em 720p e em 4K).
    thickness = max(2, image.shape[1] // 400)
    font_scale = max(0.5, image.shape[1] / 1600)
    for item in annotations:
        color = COLORS.get(item.label, DEFAULT_COLOR)
        cv2.rectangle(result, (item.x1, item.y1), (item.x2, item.y2), color, thickness)
        text = f"{item.label} {item.confidence:.0%}"
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        top = max(0, item.y1 - text_h - baseline - 4)
        cv2.rectangle(result, (item.x1, top), (item.x1 + text_w + 6, top + text_h + baseline + 4), color, -1)
        cv2.putText(result, text, (item.x1 + 3, top + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return result
