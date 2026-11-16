"""Verificação do conserto do poste a partir de uma foto noturna (rota /verificar-poste).

Caminho principal: o LLM de visão olha a foto (já borrada) e diz se a lâmpada está acesa.
Plano B (LLM fora do ar): uma regra de brilho simples, com confiança baixa (0,6).

A regra de brilho em uma frase: à noite, uma lâmpada acesa "estoura" o sensor da câmera,
então aparece um grupo de pixels quase brancos (≥ 245 de 255) numa imagem escura.
"""

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

StreetlightStatus = Literal["LIT", "OFF", "UNCERTAIN"]

# Lado maior da imagem enviada ao LLM e usada na regra de brilho. Foto de celular tem 4000 px;
# 1024 px bastam para ver um poste e deixam o envio ~15× menor (mais rápido e mais barato).
MAX_SIDE_PX = 1024

# Regra de brilho (plano B)
NIGHT_MAX_MEDIAN = 90  # mediana do cinza (0 a 255) acima disso: a foto não parece noturna
LAMP_PIXEL_MIN = 245  # pixel "estourado" pela luz da lâmpada
LIT_MIN_RATIO = 0.001  # ≥ 0,1% dos pixels estourados: há uma luz acesa na cena
OFF_MAX_RATIO = 0.0001  # ≤ 0,01%: nenhuma luz forte
# > 5%: área clara demais para ser uma lâmpada (céu de dia, fachada ao sol). Medido nas fotos de teste:
# lâmpadas acesas à noite ocupam de 0,3% a 1,1%; a rua de dia com céu claro, 18%.
LAMP_MAX_RATIO = 0.05
BRIGHTNESS_CONFIDENCE = 0.6  # regra simples: confiança limitada de propósito

# Abaixo disto, a resposta do LLM vira UNCERTAIN (na dúvida, não confirma o conserto).
LLM_MIN_CONFIDENCE = 0.6


@dataclass(frozen=True)
class BrightnessReading:
    status: StreetlightStatus
    confidence: float
    median_gray: float
    bright_ratio: float

    def describe(self) -> str:
        percent = self.bright_ratio * 100
        if self.status == "LIT":
            return f"Há um ponto de luz forte na foto ({percent:.2f}% dos pixels): a lâmpada parece acesa."
        if self.status == "OFF":
            return f"A foto é noturna e não tem ponto de luz forte ({percent:.2f}% dos pixels): a lâmpada parece apagada."
        if self.median_gray > NIGHT_MAX_MEDIAN:
            return "A foto não parece noturna; de dia, não dá para saber se a lâmpada acende."
        if self.bright_ratio > LAMP_MAX_RATIO:
            return f"Área clara grande demais para uma lâmpada ({percent:.1f}% dos pixels): parece céu de dia ou reflexo."
        return f"Há pouca luz forte na foto ({percent:.2f}% dos pixels): não dá para afirmar se a lâmpada acende."


def shrink(image: np.ndarray, max_side: int = MAX_SIDE_PX) -> np.ndarray:
    """Reduz a imagem para o lado maior ter no máximo max_side (nunca aumenta)."""
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1:
        return image
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def read_brightness(image: np.ndarray) -> BrightnessReading:
    gray = cv2.cvtColor(shrink(image), cv2.COLOR_BGR2GRAY)
    median = float(np.median(gray))
    ratio = float(np.count_nonzero(gray >= LAMP_PIXEL_MIN)) / gray.size

    if median > NIGHT_MAX_MEDIAN or ratio > LAMP_MAX_RATIO:
        return BrightnessReading("UNCERTAIN", 0.0, median, ratio)
    if ratio >= LIT_MIN_RATIO:
        return BrightnessReading("LIT", BRIGHTNESS_CONFIDENCE, median, ratio)
    if ratio <= OFF_MAX_RATIO:
        return BrightnessReading("OFF", BRIGHTNESS_CONFIDENCE, median, ratio)
    return BrightnessReading("UNCERTAIN", 0.0, median, ratio)
