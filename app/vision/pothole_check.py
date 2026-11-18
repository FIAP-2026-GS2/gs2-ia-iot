"""Segunda etapa do buraco: tirar os alarmes falsos que o YOLO dá em câmera de rua.

Testado com prints da CET: o YOLO marcou como buraco uma sombra de poste, tampas de bueiro,
uma faixa pintada e uma moto. Ele vê forma e cor, mas não entende o que a coisa é.

Duas camadas, da mais barata para a mais cara:
1. Zonas da câmera (config/cameras.json): a câmera é fixa, então o bueiro está sempre no mesmo
   lugar. Caixa em cima de uma zona ignorada, ou com o centro fora da pista, sai na hora.
2. Verificador (LLM): os recortes das caixas que sobraram vão numa imagem só (mosaico), e o
   LLM diz o que há em cada um. Só continua BURACO o que ele confirmar.
Analogia: triagem e especialista. O YOLO aponta tudo o que parece; o LLM olha só o apontado.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from app.vision.pothole import PotholeBox
from app.vision.streetlight_roi import Camera

IGNORE_OVERLAP = 0.3   # 30% da caixa em cima da zona ignorada: é o objeto fixo
CROP_PAD = 2.0         # o recorte leva 2× o tamanho da caixa de cada lado (contexto)
CROP_MIN_SIDE = 96     # caixa pequena demais: recorte mínimo, para o LLM ver o entorno
TILE = 256
MAX_TILES = 6          # mais que isso: as demais ficam sem verificação


@dataclass(frozen=True)
class Discarded:
    box: PotholeBox
    reason: str   # ex.: "SOMBRA: sombra projetada do poste"
    by: str       # zona ou llm


def _overlap(box: PotholeBox, zone: tuple[int, int, int, int]) -> float:
    ix = max(0, min(box.x2, zone[2]) - max(box.x1, zone[0]))
    iy = max(0, min(box.y2, zone[3]) - max(box.y1, zone[1]))
    area = max(1, (box.x2 - box.x1) * (box.y2 - box.y1))
    return ix * iy / area


def filter_by_zones(boxes: list[PotholeBox], camera: Camera | None, width: int, height: int
                    ) -> tuple[list[PotholeBox], list[Discarded]]:
    if camera is None:
        return boxes, []
    zones = [camera.scale_box(zone, width, height) for zone in camera.pothole_ignore]
    road = np.array(camera.scaled_road(width, height), dtype=np.int32) if camera.road else None
    kept, dropped = [], []
    for box in boxes:
        center = ((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2)
        if road is not None and cv2.pointPolygonTest(road, center, False) < 0:
            dropped.append(Discarded(box, "fora da pista marcada para esta câmera", "zona"))
        elif any(_overlap(box, zone) >= IGNORE_OVERLAP for zone in zones):
            dropped.append(Discarded(box, "objeto fixo conhecido (zona ignorada em config/cameras.json)", "zona"))
        else:
            kept.append(box)
    return kept, dropped


def crop_with_context(image: np.ndarray, box: PotholeBox) -> np.ndarray:
    """Recorte com o entorno e o retângulo vermelho, ampliado para TILE×TILE."""
    height, width = image.shape[:2]
    bw = max(box.x2 - box.x1, CROP_MIN_SIDE / (2 * CROP_PAD + 1))
    bh = max(box.y2 - box.y1, CROP_MIN_SIDE / (2 * CROP_PAD + 1))
    cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
    half = max(bw, bh) * (CROP_PAD + 0.5)  # quadrado, para não distorcer no mosaico
    x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
    x2, y2 = int(min(width, cx + half)), int(min(height, cy + half))
    crop = image[y1:y2, x1:x2].copy()
    cv2.rectangle(crop, (box.x1 - x1, box.y1 - y1), (box.x2 - x1, box.y2 - y1), (0, 0, 255), 2)
    # INTER_CUBIC amplia sem serrilhar; o quadrado cinza completa recortes na borda da imagem.
    scale = TILE / max(crop.shape[:2])
    resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    tile = np.full((TILE, TILE, 3), 114, dtype=np.uint8)
    tile[:resized.shape[0], :resized.shape[1]] = resized[:TILE, :TILE]
    return tile


def mosaic(image: np.ndarray, boxes: list[PotholeBox]) -> np.ndarray:
    """Junta os recortes numa imagem só, numerados de 1 em diante (1 chamada ao LLM, não N)."""
    tiles = []
    for number, box in enumerate(boxes, start=1):
        tile = crop_with_context(image, box)
        cv2.rectangle(tile, (0, 0), (34, 30), (0, 0, 0), -1)
        cv2.putText(tile, str(number), (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    columns = min(3, len(tiles))
    while len(tiles) % columns:
        tiles.append(np.full((TILE, TILE, 3), 255, dtype=np.uint8))
    gap = np.full((TILE, 6, 3), 255, dtype=np.uint8)
    rows = []
    for start in range(0, len(tiles), columns):
        row = [part for tile in tiles[start:start + columns] for part in (tile, gap)][:-1]
        rows.append(np.hstack(row))
    spacer = np.full((6, rows[0].shape[1], 3), 255, dtype=np.uint8)
    return np.vstack([part for row in rows for part in (row, spacer)][:-1])
