from dataclasses import dataclass

import numpy as np
import pytest

from app.vision.anonymizer import (
    PERSON_MODEL,
    AnonymizationError,
    AnonymizationResult,
    Anonymizer,
    Box,
    RfDetrPersonDetector,
    YoloPlateDetector,
    YuNetFaceDetector,
    blur_regions,
    build_default_anonymizer,
    estimated_height,
    non_max_suppression,
    tile_image,
)


class FakeDetector:
    def __init__(self, boxes=None, error=None):
        self.boxes = boxes or []
        self.error = error

    def detect(self, image):
        if self.error:
            raise self.error
        return self.boxes


def _checkerboard(size=200, square=10):
    """Imagem com detalhe fino (xadrez): depois do borrão, o detalhe some."""
    y, x = np.indices((size, size))
    board = (((x // square) + (y // square)) % 2 * 255).astype(np.uint8)
    return np.dstack([board] * 3)


def test_borrao_apaga_o_detalhe_dentro_da_caixa():
    image = _checkerboard()
    box = Box(50, 50, 150, 150, 0.9)

    result = blur_regions(image, [box], margin=0)

    inside_before = image[50:150, 50:150].std()
    inside_after = result[50:150, 50:150].std()
    assert inside_after < inside_before * 0.5


def test_borrao_nao_mexe_fora_da_caixa():
    image = _checkerboard()

    result = blur_regions(image, [Box(50, 50, 100, 100, 0.9)], margin=0)

    assert np.array_equal(result[150:, 150:], image[150:, 150:])


def test_borrao_nao_altera_a_imagem_original():
    image = _checkerboard()
    copy = image.copy()

    blur_regions(image, [Box(0, 0, 100, 100, 0.9)])

    assert np.array_equal(image, copy)


def test_margem_respeita_a_borda_da_imagem():
    box = Box(0, 0, 100, 100, 0.9).expanded(0.5, width=120, height=120)

    assert (box.x1, box.y1, box.x2, box.y2) == (0, 0, 120, 120)


def test_anonymizer_borra_rostos_e_placas():
    image = _checkerboard()
    anonymizer = Anonymizer(
        FakeDetector([Box(10, 10, 60, 60, 0.9)]),
        FakeDetector([Box(120, 150, 190, 180, 0.8)]),
    )

    result = anonymizer.anonymize(image)

    assert len(result.faces) == 1
    assert len(result.plates) == 1
    assert result.image[10:60, 10:60].std() < image[10:60, 10:60].std()


def test_falha_no_detector_bloqueia_a_imagem():
    anonymizer = Anonymizer(FakeDetector(error=RuntimeError("onnx quebrou")), FakeDetector())

    with pytest.raises(AnonymizationError, match="onnx quebrou"):
        anonymizer.anonymize(_checkerboard())


def test_modelo_ausente_gera_erro_de_anonimizacao(tmp_path):
    with pytest.raises(AnonymizationError, match="não encontrado"):
        YuNetFaceDetector(model_path=tmp_path / "nao-existe.onnx")
    with pytest.raises(AnonymizationError, match="não encontrado"):
        YoloPlateDetector(model_path=tmp_path / "nao-existe.onnx")


@pytest.mark.skipif(not PERSON_MODEL.is_file(), reason="rode python -m scripts.modelos.baixar_modelos")
def test_modelos_reais_carregam_e_rodam():
    """Integração: os dois .onnx do repositório carregam e rodam numa imagem sem rosto nem placa."""
    anonymizer = build_default_anonymizer()
    image = np.full((720, 1280, 3), 120, dtype=np.uint8)

    result = anonymizer.anonymize(image)

    assert result.faces == []
    assert result.plates == []
    assert result.image.shape == image.shape


# --- Opções em avaliação: A (pessoa), A2 (cabeça e ombros) e B (rosto ampliado) ---

needs_person_model = pytest.mark.skipif(
    not PERSON_MODEL.is_file(), reason="rode python -m scripts.modelos.baixar_modelos para baixar o detector de pessoas"
)


@dataclass
class _FakeBBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class _FakeResult:
    label: str
    confidence: float
    bounding_box: _FakeBBox


class _FakeRfDetr:
    def predict(self, image):
        return [
            _FakeResult("person", 0.9, _FakeBBox(10, 100, 50, 200)),
            _FakeResult("car", 0.9, _FakeBBox(60, 60, 120, 120)),
        ]


def _person_detector(top_fraction, width_fraction=1.0):
    """Cria o detector sem carregar o modelo de 114 MB e troca a rede por uma falsa."""
    detector = RfDetrPersonDetector.__new__(RfDetrPersonDetector)
    detector._top_fraction = top_fraction
    detector._width_fraction = width_fraction
    detector._tiles, detector._tile_overlap, detector._min_height = 1, 0.25, 0
    detector._detector = _FakeRfDetr()
    return detector


def test_detector_de_pessoas_ignora_outras_classes():
    boxes = _person_detector(1.0).detect(np.zeros((300, 300, 3), np.uint8))

    assert len(boxes) == 1  # o carro não entra
    assert (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2) == (10, 100, 50, 200)


def test_borra_so_a_cabeca_topo_e_centro_da_pessoa():
    boxes = _person_detector(top_fraction=0.2, width_fraction=0.6).detect(np.zeros((300, 300, 3), np.uint8))

    # Pessoa de x=10..50 (40 px) e y=100..200 (100 px):
    # altura → só os 20 px de cima; largura → tira 20% (8 px) de cada lado.
    assert (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2) == (18, 100, 42, 120)


@pytest.mark.parametrize("campo", ["top_fraction", "width_fraction"])
@pytest.mark.parametrize("valor", [0, -0.1, 1.5])
def test_fracoes_invalidas(campo, valor, tmp_path):
    with pytest.raises(ValueError):
        RfDetrPersonDetector(model_path=tmp_path / "x.onnx", **{campo: valor})


def test_modelo_de_pessoas_ausente_explica_como_baixar(tmp_path):
    with pytest.raises(AnonymizationError, match="baixar_modelos"):
        RfDetrPersonDetector(model_path=tmp_path / "nao-existe.onnx")


def test_area_borrada_conta_sobreposicao_uma_vez():
    image = np.zeros((100, 100, 3), np.uint8)
    person = Box(0, 0, 50, 100, 0.9)
    face_inside = Box(10, 10, 30, 30, 0.9)

    result = AnonymizationResult(image, faces=[face_inside], plates=[], persons=[person])

    # A pessoa (com margem de 15%) cobre ~57% da imagem; o rosto dentro dela não soma.
    assert result.blurred_area_ratio == pytest.approx(0.57, abs=0.01)


class _FakeYuNet:
    """Imita o cv2.FaceDetectorYN: guarda o tamanho que recebeu e acha um rosto fixo."""

    def setInputSize(self, size):
        self.size = size

    def detect(self, image):
        # Linha do YuNet: x, y, largura, altura, 10 números dos pontos do rosto e a confiança.
        return 1, np.array([[100, 60, 40, 50] + [0] * 10 + [0.95]], dtype=np.float32)


def test_rosto_ampliado_devolve_coordenadas_da_imagem_original():
    """Opção B: procura na imagem 2x maior, mas a caixa volta no tamanho original."""
    detector = YuNetFaceDetector(upscale=2.0)
    fake = _FakeYuNet()
    detector._detector = fake

    boxes = detector.detect(np.zeros((240, 320, 3), np.uint8))

    assert fake.size == (640, 480)  # a rede recebeu a imagem ampliada
    # Na imagem ampliada o rosto foi achado em x=100..140, y=60..110; na original, metade.
    assert (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2) == (50, 30, 70, 55)


def test_upscale_menor_que_1_e_recusado():
    with pytest.raises(ValueError):
        YuNetFaceDetector(upscale=0.5)


@needs_person_model
def test_modelo_real_de_pessoas_carrega_e_roda():
    detector = RfDetrPersonDetector()

    assert detector.detect(np.full((480, 640, 3), 120, np.uint8)) == []


# --- Fatiamento (tiling) e remoção de caixas repetidas (NMS) ---



def test_sem_fatiamento_nao_gera_pedacos():
    assert tile_image(np.zeros((100, 100, 3), np.uint8), tiles=1, overlap=0.25) == []


def test_fatiamento_2x2_cobre_a_imagem_toda_com_sobreposicao():
    image = np.zeros((480, 640, 3), np.uint8)

    pieces = tile_image(image, tiles=2, overlap=0.25)

    assert len(pieces) == 4
    # O 1º pedaço começa no canto e o último termina no canto oposto.
    assert (pieces[0][0], pieces[0][1]) == (0, 0)
    x, y, last = pieces[-1]
    assert x + last.shape[1] == 640 and y + last.shape[0] == 480
    # Cada pedaço é maior que a metade: é a sobreposição que evita cortar alguém ao meio sem salvação.
    assert last.shape[1] > 320 and last.shape[0] > 240


def test_nms_remove_a_mesma_pessoa_achada_duas_vezes():
    whole = Box(100, 100, 140, 200, 0.6)       # achada na imagem inteira
    in_tile = Box(102, 98, 141, 203, 0.9)      # a mesma, achada num pedaço (mais confiante)
    other = Box(300, 100, 340, 200, 0.7)       # outra pessoa

    kept = non_max_suppression([whole, in_tile, other], iou_threshold=0.5)

    assert len(kept) == 2
    assert in_tile in kept and other in kept


def test_nms_lista_vazia():
    assert non_max_suppression([], iou_threshold=0.5) == []


class _FakeRfDetrPorTamanho:
    """Só 'enxerga' a pessoa quando a imagem é pequena (um pedaço), como o modelo real numa multidão."""

    def predict(self, image):
        if image.shape[1] >= 600:  # imagem inteira: pessoa pequena demais, não acha
            return []
        return [_FakeResult("person", 0.8, _FakeBBox(10, 10, 30, 60))]


def test_fatiamento_acha_pessoa_que_a_imagem_inteira_perde():
    detector = _person_detector(top_fraction=1.0)
    detector._tiles, detector._tile_overlap = 2, 0.25
    detector._detector = _FakeRfDetrPorTamanho()

    boxes = detector.detect(np.zeros((480, 640, 3), np.uint8))

    # Os 4 pedaços acham a pessoa, cada um na sua posição, já convertida para a imagem inteira.
    assert len(boxes) == 4
    assert all(b.x1 >= 10 and b.y1 >= 10 for b in boxes)


# --- Regra "longe demais para reconhecer" (60 px de pessoa, 8 px de rosto) ---



class _FakeRfDetrVariasPessoas:
    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, image):
        return [_FakeResult("person", 0.9, _FakeBBox(*b)) for b in self.boxes]


def test_altura_estimada_de_pessoa_inteira_e_a_altura_vista():
    assert estimated_height(Box(0, 0, 27, 84, 0.9)) == 84


def test_altura_estimada_de_pessoa_cortada_vem_da_largura():
    # Só 40 px aparecem (cortada pela borda), mas 27 px de largura = pessoa de ~68 px.
    assert estimated_height(Box(0, 440, 27, 480, 0.9)) == 67


@pytest.mark.parametrize(("altura", "borra"), [(59, False), (60, True), (120, True)])
def test_regra_de_60px(altura, borra):
    detector = _person_detector(top_fraction=0.2)
    detector._min_height = 60
    # Pessoa estreita (10 px), para a largura não influenciar a estimativa.
    detector._detector = _FakeRfDetrVariasPessoas([(100, 100, 110, 100 + altura)])

    boxes = detector.detect(np.zeros((400, 400, 3), np.uint8))

    assert (len(boxes) == 1) is borra


def test_pessoa_cortada_na_borda_e_borrada_e_a_cabeca_usa_a_altura_estimada():
    detector = _person_detector(top_fraction=0.2)
    detector._min_height = 60
    # Caixa de 40 px de altura (cortada) e 30 de largura → altura estimada 75 px.
    detector._detector = _FakeRfDetrVariasPessoas([(100, 440, 130, 480)])

    boxes = detector.detect(np.zeros((480, 640, 3), np.uint8))

    assert len(boxes) == 1
    # Cabeça = 20% de 75 = 15 px (e não 20% de 40 = 8 px, que deixaria o queixo de fora).
    assert boxes[0].y2 - boxes[0].y1 == 15


def test_rosto_menor_que_o_minimo_nao_e_borrado():
    detector = YuNetFaceDetector(min_width=8)
    fake = _FakeYuNet()  # acha um rosto de 40 px de largura
    detector._detector = fake

    assert len(detector.detect(np.zeros((240, 320, 3), np.uint8))) == 1
    detector._min_width = 41
    assert detector.detect(np.zeros((240, 320, 3), np.uint8)) == []


# ---------- confirmação dupla do rosto ----------

def _anonymizer_com_pessoas(faces, heads):
    return Anonymizer(FakeDetector(faces), FakeDetector(), FakeDetector(heads))


def test_rosto_de_nota_alta_borra_mesmo_sem_pessoa():
    result = _anonymizer_com_pessoas([Box(10, 10, 60, 60, 0.9)], []).anonymize(_checkerboard())

    assert len(result.faces) == 1 and result.unconfirmed_faces == []


def test_rosto_de_nota_media_com_pessoa_embaixo_borra():
    face = Box(20, 20, 40, 40, 0.7)
    head = Box(15, 10, 45, 45, 0.8)  # cabeça da pessoa achada pelo RF-DETR, no mesmo lugar

    result = _anonymizer_com_pessoas([face], [head]).anonymize(_checkerboard())

    assert result.faces == [face] and result.unconfirmed_faces == []


def test_rosto_de_nota_media_sem_pessoa_nao_borra_e_fica_registrado():
    # O desenho na faixa do Ibirapuera: nota 0,69, nenhum corpo embaixo.
    drawing = Box(120, 120, 180, 190, 0.69)
    image = _checkerboard()

    result = _anonymizer_com_pessoas([drawing], [Box(10, 10, 40, 40, 0.8)]).anonymize(image)

    assert result.faces == [] and result.unconfirmed_faces == [drawing]
    assert np.array_equal(result.image[120:190, 120:180], image[120:190, 120:180])  # não borrou


def test_sem_detector_de_pessoas_borra_todo_rosto():
    anonymizer = Anonymizer(FakeDetector([Box(10, 10, 60, 60, 0.65)]), FakeDetector())

    result = anonymizer.anonymize(_checkerboard())

    assert len(result.faces) == 1 and result.unconfirmed_faces == []
