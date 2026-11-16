"""Anonimização: borra cabeças, rostos e placas ANTES de qualquer imagem ser salva ou enviada.

Três detectores em ONNX, sem PyTorch (estratégia escolhida comparando 6 opções em fotos reais):
- Rostos de perto e médios: YuNet (OpenCV Zoo, MIT, ~230 KB) com a imagem ampliada 2×.
- Pessoas de longe: RF-DETR nano (COCO, Apache 2.0, 114 MB) com fatiamento 2×2. Borra só
  a CABEÇA (topo 20% e centro 60% da caixa), e não o corpo: o chão (buraco, faixa) fica visível.
- Placas: YOLOv9-tiny 640 treinado em placas (open-image-models, MIT, ~7,5 MB).

Regra "longe demais para reconhecer": pessoa com menos de 60 px de altura e rosto com menos
de 8 px não são borrados (norma de CFTV DORI: reconhecer alguém exige ~210 px).

Regra de privacidade: na dúvida, borra. Por isso os limiares de confiança são baixos:
um borrão a mais num pedaço de asfalto é aceitável; uma placa legível não é.

Falha segura: se um detector não carregar ou der erro, lança AnonymizationError, e quem
chamou NÃO sobe a imagem. Nunca devolvemos a imagem original "sem borrar por precaução".
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "anonymizer"
FACE_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
PLATE_MODEL = MODELS_DIR / "yolo-v9-t-640-license-plates-end2end.onnx"
# Detector de objetos COCO, do qual usamos só a classe "person".
# 114 MB: passa do limite de 100 MB do GitHub, então é baixado por script (scripts/baixar_modelos.py).
PERSON_MODEL = MODELS_DIR / "rf-detr-nano-384-coco.onnx"

# Limiares baixos de propósito (privacidade > precisão). O padrão do YuNet é 0,9 e o das placas, 0,25.
FACE_SCORE_THRESHOLD = 0.6
PLATE_SCORE_THRESHOLD = 0.2
# O padrão do RF-DETR é 0,5. Mais baixo pega pedestres pequenos e parcialmente escondidos.
PERSON_SCORE_THRESHOLD = 0.3

# Quanto a caixa cresce antes do borrão (15% de cada lado), para cobrir cabelo, orelha e moldura da placa.
BOX_MARGIN = 0.15

# Regra "longe demais para reconhecer" (norma de CFTV DORI, IEC 62676-4): reconhecer alguém exige
# ~210 px de altura. Abaixo de 60 px (3,5× menos), nem com zoom há rosto, então não é dado pessoal.
MIN_PERSON_HEIGHT_PX = 60
MIN_FACE_WIDTH_PX = 8
# Uma pessoa de pé tem ~2,5× mais altura que largura. Serve para estimar a altura de quem está
# cortado pela borda da foto: a caixa fica baixa, mas a largura continua a de uma pessoa perto.
PERSON_HEIGHT_TO_WIDTH = 2.5

# Estratégia final (medida nas fotos): de corpo inteiro, o rosto termina em ~15% da altura e fica
# entre 32% e 77% da largura da caixa. 20% da altura e 60% da largura cobrem a cabeça com folga.
HEAD_TOP_FRACTION = 0.2
HEAD_WIDTH_FRACTION = 0.6
FACE_UPSCALE = 2.0
PERSON_TILES = 2


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
    def __init__(
        self,
        model_path: Path = FACE_MODEL,
        score_threshold: float = FACE_SCORE_THRESHOLD,
        upscale: float = 1.0,
        min_width: int = 0,
    ):
        """upscale > 1 é a opção B: amplia a imagem antes de procurar rostos.

        O YuNet não acha rosto com menos de ~10 px. Ampliando 2×, um rosto de 6 px vira 12 px
        e entra no alcance. O custo é ~4× mais pixels para processar.

        min_width: rostos mais estreitos que isso não são borrados (longe demais para reconhecer).
        """
        if not model_path.is_file():
            raise AnonymizationError(f"Modelo de rosto não encontrado: {model_path.name}")
        if upscale < 1:
            raise ValueError("upscale precisa ser >= 1.")
        self._upscale = upscale
        self._min_width = min_width
        # O tamanho de entrada é trocado a cada imagem (setInputSize).
        self._detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score_threshold, 0.3, 5000)

    def detect(self, image: np.ndarray) -> list[Box]:
        if self._upscale > 1:
            image = cv2.resize(image, None, fx=self._upscale, fy=self._upscale, interpolation=cv2.INTER_CUBIC)
        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(image)
        if faces is None:
            return []
        # Cada linha: x, y, largura, altura, 5 pontos do rosto (olhos, nariz, boca) e a confiança.
        # Dividir por upscale devolve as coordenadas para o tamanho da imagem original.
        s = self._upscale
        boxes = [
            Box(int(x / s), int(y / s), int((x + w) / s), int((y + h) / s), float(row[-1]))
            for row in faces
            for x, y, w, h in [row[:4]]
        ]
        return [b for b in boxes if b.x2 - b.x1 >= self._min_width]


class RfDetrPersonDetector:
    """Acha PESSOAS (corpo inteiro), e não rostos, e devolve a região da cabeça para borrar.

    De longe, um rosto tem 4 a 8 px, mas o corpo tem 40 a 80 px: muito mais fácil de achar.
    O modelo detecta 80 classes do COCO; ficamos só com "person".
    """

    def __init__(
        self,
        model_path: Path = PERSON_MODEL,
        score_threshold: float = PERSON_SCORE_THRESHOLD,
        top_fraction: float = 1.0,
        width_fraction: float = 1.0,
        tiles: int = 1,
        tile_overlap: float = 0.25,
        min_height: int = 0,
    ):
        """Recorta a caixa da pessoa para borrar só a cabeça.

        - top_fraction: quanto da altura, a partir do topo (ex.: 0,2 = os 20% de cima).
        - width_fraction: quanto da largura, centralizado (ex.: 0,6 = do 20% ao 80% da largura).
        - tiles: fatiamento. 2 = além da imagem inteira, analisa 4 pedaços (2 × 2).

        Medido nas fotos de teste: de corpo inteiro, o rosto termina em ~15% da altura e fica
        entre 32% e 77% da largura da caixa (a caixa inclui ombros e braços). O rosto, que
        identifica, é borrado; o corpo e o chão (buraco, faixa) continuam visíveis.

        Por que fatiar: o modelo reduz qualquer imagem para 384 × 384 antes de olhar. Numa
        multidão, cada pessoa vira 3 ou 4 pixels e some. Cada pedaço também é reduzido para
        384, então a pessoa fica 2× maior para o modelo (técnica conhecida como SAHI, "slicing").
        Os pedaços se sobrepõem 25% para que ninguém fique cortado ao meio sem aparecer inteiro
        em outro pedaço.

        - min_height: pessoas com altura estimada menor que isso não são borradas (longe demais
          para reconhecer). A altura é ESTIMADA (ver estimated_height): quem está cortado pela
          borda da foto tem a caixa baixa, mas continua perto.
        """
        if not 0 < top_fraction <= 1 or not 0 < width_fraction <= 1:
            raise ValueError("top_fraction e width_fraction precisam estar entre 0 e 1.")
        if tiles < 1 or not 0 <= tile_overlap < 1:
            raise ValueError("tiles precisa ser >= 1 e tile_overlap entre 0 e 1.")
        self._top_fraction = top_fraction
        self._width_fraction = width_fraction
        self._tiles = tiles
        self._tile_overlap = tile_overlap
        self._min_height = min_height
        if not model_path.is_file():
            raise AnonymizationError(
                f"Modelo de pessoas não encontrado: {model_path.name}. Rode: python -m scripts.baixar_modelos"
            )
        from open_image_models.detection.core.coco import COCO_CLASSES
        from open_image_models.detection.core.rf_detr.inference import RFDETRDetector

        self._detector = RFDETRDetector(
            model_path, COCO_CLASSES, conf_thresh=score_threshold, providers=["CPUExecutionProvider"]
        )

    def detect(self, image: np.ndarray) -> list[Box]:
        people = self._people_in(image, 0, 0)
        for x, y, tile in tile_image(image, self._tiles, self._tile_overlap):
            people += self._people_in(tile, x, y)
        # A mesma pessoa aparece na imagem inteira e em 1 ou 2 pedaços: fica só a caixa mais confiante.
        people = non_max_suppression(people, iou_threshold=0.5)

        boxes = []
        for b in people:
            height = estimated_height(b)
            if height < self._min_height:
                continue  # longe demais para reconhecer: não é dado pessoal
            # A cabeça é medida pela altura ESTIMADA: numa pessoa cortada pela borda, 20% da caixa
            # visível seria pequeno demais e deixaria o queixo de fora. Nunca passa do fim da caixa.
            bottom = min(b.y2, b.y1 + max(1, int(height * self._top_fraction)))
            # Sobra de cada lado: com 0,6 da largura, 20% à esquerda e 20% à direita ficam de fora.
            side = int((b.x2 - b.x1) * (1 - self._width_fraction) / 2)
            boxes.append(Box(b.x1 + side, b.y1, b.x2 - side, bottom, b.score))
        return boxes

    def _people_in(self, image: np.ndarray, offset_x: int, offset_y: int) -> list[Box]:
        """Pessoas inteiras achadas num pedaço, com as coordenadas levadas para a imagem toda."""
        return [
            Box(r.bounding_box.x1 + offset_x, r.bounding_box.y1 + offset_y,
                r.bounding_box.x2 + offset_x, r.bounding_box.y2 + offset_y, float(r.confidence))
            for r in self._detector.predict(image)
            if r.label == "person"
        ]


def estimated_height(person: Box) -> int:
    """Altura que a pessoa teria se aparecesse inteira: a maior entre a vista e a largura × 2,5.

    Pessoa inteira: 84 px de altura e 27 de largura (27 × 2,5 = 68) → vale 84, a altura vista.
    Pessoa cortada na borda: 40 px de altura visível e 27 de largura → vale 68, pela largura.
    Na dúvida, a estimativa erra para MAIS, o que leva a borrar: o lado seguro para a privacidade.
    """
    height = person.y2 - person.y1
    width = person.x2 - person.x1
    return max(height, int(width * PERSON_HEIGHT_TO_WIDTH))


def tile_image(image: np.ndarray, tiles: int, overlap: float) -> list[tuple[int, int, np.ndarray]]:
    """Divide a imagem em tiles × tiles pedaços que se sobrepõem. Devolve (x, y, pedaço).

    Com tiles=1 não há pedaços (a imagem inteira já é analisada à parte).
    """
    if tiles <= 1:
        return []
    height, width = image.shape[:2]
    # Tamanho do pedaço para que `tiles` pedaços, com sobreposição, cubram a imagem toda.
    tile_w = int(width / (tiles - (tiles - 1) * overlap))
    tile_h = int(height / (tiles - (tiles - 1) * overlap))
    step_x = (width - tile_w) / (tiles - 1)
    step_y = (height - tile_h) / (tiles - 1)
    pieces = []
    for row in range(tiles):
        for col in range(tiles):
            x, y = int(round(col * step_x)), int(round(row * step_y))
            pieces.append((x, y, image[y:y + tile_h, x:x + tile_w]))
    return pieces


def non_max_suppression(boxes: list[Box], iou_threshold: float) -> list[Box]:
    """Remove caixas repetidas: se duas se sobrepõem mais que o limite, fica a de maior confiança.

    IoU (Intersection over Union) = área em comum ÷ área total das duas. 1 = idênticas, 0 = separadas.
    """
    if not boxes:
        return []
    rects = [[b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1] for b in boxes]
    keep = cv2.dnn.NMSBoxes(rects, [b.score for b in boxes], score_threshold=0.0, nms_threshold=iou_threshold)
    return [boxes[i] for i in np.array(keep).flatten()]


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
    persons: list[Box] = field(default_factory=list)

    @property
    def blurred_area_ratio(self) -> float:
        """Fração da imagem coberta por borrão (0 a 1). Mede o quanto a cena foi "apagada".

        Conta cada pixel uma vez só, mesmo quando um rosto está dentro de uma pessoa.
        """
        height, width = self.image.shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        for box in self.faces + self.plates + self.persons:
            b = box.expanded(BOX_MARGIN, width, height)
            mask[b.y1:b.y2, b.x1:b.x2] = True
        return float(mask.mean())


class Anonymizer:
    def __init__(self, face_detector: Detector, plate_detector: Detector, person_detector: Detector | None = None):
        self._face_detector = face_detector
        self._plate_detector = plate_detector
        # Opcional: só existe na opção A.
        self._person_detector = person_detector

    def anonymize(self, image: np.ndarray) -> AnonymizationResult:
        try:
            faces = self._face_detector.detect(image)
            plates = self._plate_detector.detect(image)
            persons = self._person_detector.detect(image) if self._person_detector else []
            blurred = blur_regions(image, faces + plates + persons)
        except AnonymizationError:
            raise
        except Exception as error:  # qualquer falha do detector bloqueia o envio
            raise AnonymizationError(f"Falha ao anonimizar a imagem: {error}") from error
        return AnonymizationResult(blurred, faces, plates, persons)


def build_default_anonymizer() -> Anonymizer:
    """A estratégia que a API usa: rosto 2×, cabeça das pessoas com fatiamento, placas e a regra de 60 px."""
    try:
        return Anonymizer(
            YuNetFaceDetector(upscale=FACE_UPSCALE, min_width=MIN_FACE_WIDTH_PX),
            YoloPlateDetector(),
            RfDetrPersonDetector(
                top_fraction=HEAD_TOP_FRACTION,
                width_fraction=HEAD_WIDTH_FRACTION,
                tiles=PERSON_TILES,
                min_height=MIN_PERSON_HEIGHT_PX,
            ),
        )
    except AnonymizationError:
        raise
    except Exception as error:
        raise AnonymizationError(f"Falha ao carregar os modelos de anonimização: {error}") from error
