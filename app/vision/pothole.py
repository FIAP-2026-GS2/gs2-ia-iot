"""Detecção de buraco com um YOLOv8 em ONNX (sem PyTorch: só o ONNX Runtime).

Caminho de uma imagem:
1. letterbox: reduz a imagem para caber em 640×640 sem distorcer, e completa com cinza
2. o modelo devolve 8400 caixas candidatas, cada uma com a nota de cada classe
3. ficam só as da classe "buraco" com nota ≥ 0,25; o NMS remove as caixas repetidas
4. as caixas voltam para o tamanho original da imagem
5. a gravidade sai da área: quanto da imagem o buraco ocupa

Faixas de confiança (combinado com a API .NET):
- < 0,25: descartada (ruído)
- 0,25 a 0,50: "abaixo do limiar": aparece na resposta, mas não abre ocorrência sozinha
- ≥ 0,50: detecção que abre ocorrência
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import onnxruntime as ort

INPUT_SIZE = 640
MIN_CONFIDENCE = 0.25
REPORT_CONFIDENCE = 0.50
NMS_IOU = 0.45

Severity = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class PotholeModelError(Exception):
    """Modelo de buraco ausente ou num formato inesperado."""


@dataclass(frozen=True)
class PotholeBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    area_ratio: float  # fração da imagem ocupada pela caixa (0 a 1)

    @property
    def severity(self) -> Severity:
        return severity_from_area(self.area_ratio)

    @property
    def reportable(self) -> bool:
        return self.confidence >= REPORT_CONFIDENCE


def severity_from_area(area_ratio: float) -> Severity:
    """< 3% da imagem: LOW; < 6%: MEDIUM; < 9%: HIGH; maior: CRITICAL."""
    if area_ratio < 0.03:
        return "LOW"
    if area_ratio < 0.06:
        return "MEDIUM"
    if area_ratio < 0.09:
        return "HIGH"
    return "CRITICAL"


def letterbox(image: np.ndarray, size: int = INPUT_SIZE) -> tuple[np.ndarray, float, int, int]:
    """Reduz sem distorcer e centraliza num quadrado cinza. Devolve a escala e o deslocamento."""
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def _class_names(session: ort.InferenceSession) -> dict[int, str]:
    # O ultralytics grava os nomes das classes no próprio ONNX, como texto de um dict Python.
    raw = session.get_modelmeta().custom_metadata_map.get("names", "{}")
    try:
        names = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}
    return {int(key): str(value) for key, value in names.items()}


class PotholeDetector:
    def __init__(self, model_path: str | Path, name: str | None = None):
        path = Path(model_path)
        if not path.is_file():
            raise PotholeModelError(f"Modelo de buraco não encontrado: {path} (rode python -m scripts.baixar_modelos)")
        self.name = name or path.stem
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name
        names = _class_names(self._session)
        # Modelos com várias classes (rachaduras etc.): fica só a classe de buraco.
        pothole_ids = [index for index, label in names.items() if "pothole" in label.lower()]
        if names and not pothole_ids:
            raise PotholeModelError(f"O modelo {path.name} não tem uma classe de buraco: {names}")
        self.class_id = pothole_ids[0] if pothole_ids else 0
        self.num_classes = max(len(names), 1)

    def detect(self, image: np.ndarray, min_confidence: float = MIN_CONFIDENCE) -> list[PotholeBox]:
        height, width = image.shape[:2]
        canvas, scale, pad_x, pad_y = letterbox(image)
        # BGR → RGB, 0-255 → 0-1, (altura, largura, canal) → (1, canal, altura, largura)
        blob = canvas[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        output = self._session.run(None, {self._input: blob})[0][0]  # (4 + classes, 8400)
        if output.shape[0] != 4 + self.num_classes:
            raise PotholeModelError(f"Saída inesperada do modelo {self.name}: {output.shape}")

        scores = output[4 + self.class_id]
        keep = scores >= min_confidence
        if not keep.any():
            return []
        cx, cy, w, h = output[:4, keep]
        scores = scores[keep]

        # Centro/tamanho no quadrado 640 → cantos na imagem original.
        x1 = (cx - w / 2 - pad_x) / scale
        y1 = (cy - h / 2 - pad_y) / scale
        x2 = (cx + w / 2 - pad_x) / scale
        y2 = (cy + h / 2 - pad_y) / scale
        rects = [[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in zip(x1, y1, x2, y2)]
        chosen = cv2.dnn.NMSBoxes(rects, scores.tolist(), min_confidence, NMS_IOU)

        boxes = []
        for index in np.array(chosen).flatten():
            bx1 = int(np.clip(x1[index], 0, width - 1))
            by1 = int(np.clip(y1[index], 0, height - 1))
            bx2 = int(np.clip(x2[index], 0, width - 1))
            by2 = int(np.clip(y2[index], 0, height - 1))
            if bx2 <= bx1 or by2 <= by1:
                continue
            area = (bx2 - bx1) * (by2 - by1) / (width * height)
            boxes.append(PotholeBox(bx1, by1, bx2, by2, round(float(scores[index]), 3), round(area, 4)))
        boxes.sort(key=lambda box: box.confidence, reverse=True)
        return boxes
