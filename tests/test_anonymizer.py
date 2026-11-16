import numpy as np
import pytest

from app.vision.anonymizer import (
    AnonymizationError,
    Anonymizer,
    Box,
    YoloPlateDetector,
    YuNetFaceDetector,
    blur_regions,
    build_default_anonymizer,
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


def test_modelos_reais_carregam_e_rodam():
    """Integração: os dois .onnx do repositório carregam e rodam numa imagem sem rosto nem placa."""
    anonymizer = build_default_anonymizer()
    image = np.full((720, 1280, 3), 120, dtype=np.uint8)

    result = anonymizer.anonymize(image)

    assert result.faces == []
    assert result.plates == []
    assert result.image.shape == image.shape
