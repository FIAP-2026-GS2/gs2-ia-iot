"""Anonimização: borra rostos e placas ANTES de qualquer imagem ser salva ou enviada.

Dois detectores pequenos, em ONNX, sem PyTorch:
- Rostos: YuNet (OpenCV Zoo, MIT, ~230 KB), que roda dentro do próprio OpenCV.
- Placas: YOLOv9-tiny 640 treinado em placas (open-image-models, MIT, ~7,5 MB), no ONNX Runtime.

Regra de privacidade: na dúvida, borra. Por isso os limiares de confiança são baixos:
um borrão a mais num pedaço de asfalto é aceitável; uma placa legível não é.

Falha segura: se um detector não carregar ou der erro, lança AnonymizationError, e quem
chamou NÃO sobe a imagem. Nunca devolvemos a imagem original "sem borrar por precaução".
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "anonymizer"
FACE_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
PLATE_MODEL = MODELS_DIR / "yolo-v9-t-640-license-plates-end2end.onnx"

# Limiares baixos de propósito (privacidade > precisão). O padrão do YuNet é 0,9 e o das placas, 0,25.
FACE_SCORE_THRESHOLD = 0.6
PLATE_SCORE_THRESHOLD = 0.2

# Quanto a caixa cresce antes do borrão (15% de cada lado), para cobrir cabelo, orelha e moldura da placa.
BOX_MARGIN = 0.15


class AnonymizationError(Exception):
    """A anonimização falhou: a imagem não pode ser salva nem enviada."""


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float

    def expanded(self, margin: float, width: int, height: int) -> "Box":
        dx = int((self.x2 - self.x1) * margin)
        dy = int((self.y2 - self.y1) * margin)
        return Box(
            max(0, self.x1 - dx), max(0, self.y1 - dy),
            min(width, self.x2 + dx), min(height, self.y2 + dy),
            self.score,
        )


class Detector(Protocol):
    """Qualquer coisa que receba uma imagem e devolva caixas. Permite trocar o modelo (ou usar um falso nos testes)."""

    def detect(self, image: np.ndarray) -> list[Box]: ...


class YuNetFaceDetector:
    def __init__(self, model_path: Path = FACE_MODEL, score_threshold: float = FACE_SCORE_THRESHOLD):
        if not model_path.is_file():
            raise AnonymizationError(f"Modelo de rosto não encontrado: {model_path.name}")
        # O tamanho de entrada é trocado a cada imagem (setInputSize).
        self._detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score_threshold, 0.3, 5000)

    def detect(self, image: np.ndarray) -> list[Box]:
        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(image)
        if faces is None:
            return []
        # Cada linha: x, y, largura, altura, 5 pontos do rosto (olhos, nariz, boca) e a confiança.
        return [
            Box(int(x), int(y), int(x + w), int(y + h), float(row[-1]))
            for row in faces
            for x, y, w, h in [row[:4]]
        ]


class YoloPlateDetector:
    def __init__(self, model_path: Path = PLATE_MODEL, score_threshold: float = PLATE_SCORE_THRESHOLD):
        if not model_path.is_file():
            raise AnonymizationError(f"Modelo de placa não encontrado: {model_path.name}")
        from open_image_models.detection.core.yolo_v9.inference import YoloV9Detector

        self._detector = YoloV9Detector(
            model_path, ("License Plate",), conf_thresh=score_threshold, providers=["CPUExecutionProvider"]
        )

    def detect(self, image: np.ndarray) -> list[Box]:
        results = self._detector.predict(image)
        return [
            Box(r.bounding_box.x1, r.bounding_box.y1, r.bounding_box.x2, r.bounding_box.y2, float(r.confidence))
            for r in results
        ]


def blur_regions(image: np.ndarray, boxes: list[Box], margin: float = BOX_MARGIN) -> np.ndarray:
    """Pixeliza e depois desfoca cada região. Devolve uma cópia; a imagem original não muda.

    Só o desfoque gaussiano pode ser parcialmente revertido em textos pequenos (placa).
    Pixelizar antes descarta a informação: cada bloco vira uma cor só.
    """
    result = image.copy()
    height, width = result.shape[:2]
    for box in boxes:
        b = box.expanded(margin, width, height)
        if b.x2 <= b.x1 or b.y2 <= b.y1:
            continue
        region = result[b.y1:b.y2, b.x1:b.x2]
        h, w = region.shape[:2]
        # Até 8 blocos no lado maior: rosto e placa ficam irreconhecíveis.
        scale = 8 / max(h, w)
        # INTER_AREA tira a MÉDIA dos pixels de cada bloco. O INTER_LINEAR só amostra alguns
        # pixels, e o detalhe fino (letras da placa) poderia sobreviver (o teste do xadrez pegou isso).
        small = cv2.resize(region, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        kernel = max(3, (min(h, w) // 4) | 1)  # ímpar, proporcional ao tamanho
        result[b.y1:b.y2, b.x1:b.x2] = cv2.GaussianBlur(pixelated, (kernel, kernel), 0)
    return result


@dataclass(frozen=True)
class AnonymizationResult:
    image: np.ndarray
    faces: list[Box]
    plates: list[Box]


class Anonymizer:
    def __init__(self, face_detector: Detector, plate_detector: Detector):
        self._face_detector = face_detector
        self._plate_detector = plate_detector

    def anonymize(self, image: np.ndarray) -> AnonymizationResult:
        try:
            faces = self._face_detector.detect(image)
            plates = self._plate_detector.detect(image)
            blurred = blur_regions(image, faces + plates)
        except AnonymizationError:
            raise
        except Exception as error:  # qualquer falha do detector bloqueia o envio
            raise AnonymizationError(f"Falha ao anonimizar a imagem: {error}") from error
        return AnonymizationResult(blurred, faces, plates)


def build_default_anonymizer() -> Anonymizer:
    try:
        return Anonymizer(YuNetFaceDetector(), YoloPlateDetector())
    except AnonymizationError:
        raise
    except Exception as error:
        raise AnonymizationError(f"Falha ao carregar os modelos de anonimização: {error}") from error
