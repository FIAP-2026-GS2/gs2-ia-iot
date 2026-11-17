"""Poste apagado: compara a cabeça de cada poste à meia-noite com a mesma região ao meio-dia.

Por que ROI (região fixa) e não a foto inteira? Testado com câmeras reais da CET:
- à noite, a câmera aumenta o ganho sozinha: a mediana de brilho da foto inteira ficou entre
  86 e 110 (de dia, 110 a 118). A foto inteira não diz se é noite, nem se o poste acendeu;
- procurar "pontos brancos" automaticamente pega janelas de prédio e faróis.
Por isso cada câmera tem, em config/cameras.json, um retângulo marcado à mão em cada luminária.
"É noite?" vem do horário da agenda (00:00), não da imagem.

Dois sinais por luminária (os dois foram necessários nos testes):
- saturação: quanto do retângulo está "estourado" (≥ 245). Lâmpada forte estoura; o clarão
  pode vazar para o anel em volta e derrubar o contraste (poste 1 da Av. Brasil);
- contraste: o topo do retângulo (percentil 90) menos a mediana do anel em volta. Lâmpada
  pequena e longe não estoura, mas é bem mais clara que o redor (poste 2 da Iguatemi).
Com a referência do meio-dia, o contraste vira ganho: noite − dia. Isso desconta o que já é
claro de dia (céu ou parede branca atrás da luminária).

nota = maior dos dois sinais (0 a 1) → ≥ 0,50 acesa · ≤ 0,25 apagada · entre os dois: incerto
Sem referência do meio-dia, a confiança é multiplicada por 0,8.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

LAMP_PIXEL_MIN = 245
SATURATION_FULL = 0.02     # 2% do retângulo estourado já é lâmpada acesa (nota 1)
CONTRAST_ZERO = 30         # ganho de contraste até 30: nada (ruído, reflexo)
CONTRAST_FULL = 90         # ganho de 90 ou mais: nota 1
LIT_MIN_SCORE = 0.50
OFF_MAX_SCORE = 0.25
RING_PX = 20               # largura do anel em volta do retângulo
NO_REFERENCE_FACTOR = 0.8
MAX_CONFIDENCE = 0.95

LampStatus = Literal["LIT", "OFF", "UNCERTAIN"]


class CameraConfigError(Exception):
    """config/cameras.json ausente, inválido, ou câmera sem luminárias marcadas."""


@dataclass(frozen=True)
class Lamp:
    id: str
    roi: tuple[int, int, int, int]  # x1, y1, x2, y2 no tamanho de referência da câmera


@dataclass(frozen=True)
class Camera:
    code: str
    name: str
    frame_size: tuple[int, int]  # largura, altura em que os retângulos foram marcados
    lamps: tuple[Lamp, ...]
    # Buraco: objetos fixos que o YOLO confunde (tampa de bueiro, grelha) e, opcional, o
    # contorno da pista. Câmera fixa: o bueiro está sempre no mesmo lugar.
    pothole_ignore: tuple[tuple[int, int, int, int], ...] = ()
    road: tuple[tuple[int, int], ...] = ()

    def scale_box(self, box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
        sx, sy = width / self.frame_size[0], height / self.frame_size[1]
        return round(box[0] * sx), round(box[1] * sy), round(box[2] * sx), round(box[3] * sy)

    def scaled_road(self, width: int, height: int) -> list[tuple[int, int]]:
        sx, sy = width / self.frame_size[0], height / self.frame_size[1]
        return [(round(x * sx), round(y * sy)) for x, y in self.road]

    def scaled_roi(self, lamp: Lamp, width: int, height: int) -> tuple[int, int, int, int]:
        """O frame pode chegar em outra resolução: o retângulo acompanha a proporção."""
        return self.scale_box(lamp.roi, width, height)


def load_cameras(path: str | Path) -> dict[str, Camera]:
    file = Path(path)
    if not file.is_file():
        raise CameraConfigError(f"Arquivo de câmeras não encontrado: {file}")
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
        cameras = {}
        for code, item in raw["cameras"].items():
            lamps = tuple(Lamp(lamp["id"], tuple(lamp["roi"])) for lamp in item.get("streetlights", []))
            cameras[code] = Camera(
                code, item["name"], tuple(item["frame_size"]), lamps,
                pothole_ignore=tuple(tuple(zone["box"]) for zone in item.get("pothole_ignore", [])),
                road=tuple(tuple(point) for point in item.get("road", [])),
            )
        return cameras
    except (KeyError, TypeError, ValueError) as error:
        raise CameraConfigError(f"{file.name} em formato inválido: {error}") from error


@dataclass(frozen=True)
class LampMeasure:
    saturated_ratio: float
    top: float
    ring: float

    @property
    def contrast(self) -> float:
        return self.top - self.ring


def measure(gray: np.ndarray, roi: tuple[int, int, int, int]) -> LampMeasure:
    height, width = gray.shape
    x1, y1, x2, y2 = max(0, roi[0]), max(0, roi[1]), min(width, roi[2]), min(height, roi[3])
    if x2 <= x1 or y2 <= y1:
        raise CameraConfigError(f"Retângulo {roi} fora da imagem {width}×{height}")
    inner = gray[y1:y2, x1:x2]
    # Anel: um retângulo maior, sem o miolo.
    X1, Y1 = max(0, x1 - RING_PX), max(0, y1 - RING_PX)
    X2, Y2 = min(width, x2 + RING_PX), min(height, y2 + RING_PX)
    outer = gray[Y1:Y2, X1:X2].astype(np.float32)
    mask = np.ones(outer.shape, dtype=bool)
    mask[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1] = False
    ring = float(np.median(outer[mask])) if mask.any() else 0.0
    return LampMeasure(
        saturated_ratio=float((inner >= LAMP_PIXEL_MIN).mean()),
        top=float(np.percentile(inner, 90)),
        ring=ring,
    )


@dataclass(frozen=True)
class LampResult:
    lamp_id: str
    roi: tuple[int, int, int, int]
    status: LampStatus
    score: float
    confidence: float
    has_reference: bool
    night: LampMeasure
    day: LampMeasure | None

    def describe(self) -> str:
        ref = (f"contraste {self.night.contrast:+.0f} à noite × {self.day.contrast:+.0f} ao meio-dia"
               if self.day else f"contraste {self.night.contrast:+.0f} (sem referência do meio-dia)")
        state = {"LIT": "acesa", "OFF": "apagada", "UNCERTAIN": "incerta"}[self.status]
        return (f"Luminária {self.lamp_id} {state}: {self.night.saturated_ratio:.1%} do retângulo "
                f"estourado, {ref}.")


def score_lamp(night: LampMeasure, day: LampMeasure | None) -> float:
    saturation = min(1.0, night.saturated_ratio / SATURATION_FULL)
    gain = night.contrast - max(0.0, day.contrast) if day else night.contrast
    contrast = float(np.clip((gain - CONTRAST_ZERO) / (CONTRAST_FULL - CONTRAST_ZERO), 0, 1))
    return max(saturation, contrast)


def classify(score: float, has_reference: bool) -> tuple[LampStatus, float]:
    if score >= LIT_MIN_SCORE:
        status, confidence = "LIT", 0.5 + 0.5 * score
    elif score <= OFF_MAX_SCORE:
        status, confidence = "OFF", 0.6 + 0.4 * (1 - score / OFF_MAX_SCORE)
    else:
        status, confidence = "UNCERTAIN", 0.0
    if not has_reference:
        confidence *= NO_REFERENCE_FACTOR
    # Teto de 0,95: uma regra de brilho nunca tem certeza total (ex.: árvore cobrindo o poste).
    return status, round(min(confidence, MAX_CONFIDENCE), 2)


def measure_camera(image: np.ndarray, camera: Camera) -> dict[str, LampMeasure]:
    """Medidas de cada luminária. Ao meio-dia, viram a referência guardada."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    return {lamp.id: measure(gray, camera.scaled_roi(lamp, width, height)) for lamp in camera.lamps}


def check_night(image: np.ndarray, camera: Camera, reference: dict[str, LampMeasure] | None) -> list[LampResult]:
    """Meia-noite: decide, luminária por luminária, se está acesa."""
    height, width = image.shape[:2]
    results = []
    for lamp_id, night in measure_camera(image, camera).items():
        day = (reference or {}).get(lamp_id)
        score = score_lamp(night, day)
        status, confidence = classify(score, day is not None)
        lamp = next(item for item in camera.lamps if item.id == lamp_id)
        results.append(LampResult(lamp_id, camera.scaled_roi(lamp, width, height), status,
                                  round(score, 2), confidence, day is not None, night, day))
    return results


# ---------- Referência do meio-dia (guardada entre a captura das 12:00 e a das 00:00) ----------

class LocalReferenceStore:
    """Guarda a referência em arquivo JSON (uma por câmera).

    No local e no ACI fica em data/cache/refs/streetlight/<camera>.json. A versão com Blob
    (frames/<prefixo>refs/streetlight/<camera>.json) entra com a integração da .NET.
    """

    def __init__(self, folder: str | Path):
        self.folder = Path(folder) / "streetlight"

    def _path(self, camera_code: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in camera_code)
        return self.folder / f"{safe}.json"

    def save(self, camera_code: str, measures: dict[str, LampMeasure], captured_at: str) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        payload = {"camera": camera_code, "captured_at": captured_at,
                   "lamps": {lamp_id: asdict(m) for lamp_id, m in measures.items()}}
        self._path(camera_code).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, camera_code: str) -> tuple[dict[str, LampMeasure], str] | None:
        path = self._path(camera_code)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {lamp_id: LampMeasure(**m) for lamp_id, m in payload["lamps"].items()}, payload["captured_at"]
