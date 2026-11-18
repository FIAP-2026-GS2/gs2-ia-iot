"""POST /anonymize: a rota que a API .NET chama antes de guardar a foto do cidadão."""
import cv2
import numpy as np
import pytest

from app.api.dependencies import anonymizer_dependency
from app.main import app
from app.vision.anonymizer import AnonymizationError, Anonymizer, Box
from app.vision.frames import encode_jpeg
from tests.conftest import TEST_API_KEY
from tests.test_anonymizer import FakeDetector

HEADERS = {"X-API-Key": TEST_API_KEY}


def _foto(width=320, height=240):
    # Imagem com detalhe (xadrez), para o borrão mudar os pixels de verdade.
    tile = np.indices((height, width)).sum(axis=0) % 2 * 255
    return encode_jpeg(np.dstack([tile, tile, tile]).astype(np.uint8))


def _upload(data, name="foto.jpg", content_type="image/jpeg"):
    return {"file": (name, data, content_type)}


@pytest.fixture
def fake_anonymizer():
    """Anonimizador falso que 'acha' 1 rosto e 1 placa."""
    fake = Anonymizer(FakeDetector([Box(10, 10, 60, 60, 0.9)]), FakeDetector([Box(200, 180, 280, 210, 0.7)]))
    app.dependency_overrides[anonymizer_dependency] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


def test_sem_chave_responde_401(client, fake_anonymizer):
    assert client.post("/anonymize", files=_upload(_foto())).status_code == 401


def test_devolve_jpeg_borrado_e_conta_as_areas(client, fake_anonymizer):
    original = _foto()

    response = client.post("/anonymize", files=_upload(original), headers=HEADERS)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["x-faces-blurred"] == "1"
    assert response.headers["x-plates-blurred"] == "1"
    blurred = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    before = cv2.imdecode(np.frombuffer(original, np.uint8), cv2.IMREAD_COLOR)
    # A área do rosto mudou (borrada); um canto longe dos boxes ficou praticamente igual.
    assert np.abs(blurred[15:55, 15:55].astype(int) - before[15:55, 15:55].astype(int)).mean() > 30
    assert np.abs(blurred[100:140, 100:140].astype(int) - before[100:140, 100:140].astype(int)).mean() < 30


def test_aceita_png_e_devolve_jpeg(client, fake_anonymizer):
    ok, png = cv2.imencode(".png", np.full((120, 160, 3), 90, dtype=np.uint8))

    response = client.post("/anonymize", files=_upload(png.tobytes(), "foto.png", "image/png"), headers=HEADERS)

    assert response.status_code == 200
    assert response.content[:3] == b"\xff\xd8\xff"


def test_arquivo_que_nao_e_imagem_responde_400(client, fake_anonymizer):
    response = client.post("/anonymize", files=_upload(b"%PDF-1.7 nada", "doc.jpg"), headers=HEADERS)

    assert response.status_code == 400


def test_sem_anonimizador_responde_503_e_nao_devolve_foto(client):
    class Broken:
        def anonymize(self, image):
            raise AnonymizationError("modelo indisponível")

    app.dependency_overrides[anonymizer_dependency] = lambda: Broken()
    try:
        response = client.post("/anonymize", files=_upload(_foto()), headers=HEADERS)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/json")


def test_foto_grande_volta_reduzida_para_1920_mantendo_a_proporcao(client, fake_anonymizer):
    ok, big = cv2.imencode(".jpg", np.full((3000, 4000, 3), 120, dtype=np.uint8))  # celular comum

    response = client.post("/anonymize", files=_upload(big.tobytes()), headers=HEADERS)

    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert response.status_code == 200
    assert image.shape[:2] == (1440, 1920)  # 4000 × 3000 → 1920 × 1440 (mesma proporção 4:3)


def test_conta_os_rostos_sem_confirmacao(client):
    # Rosto de nota média e nenhuma pessoa achada: não borra, mas aparece no header.
    fake = Anonymizer(FakeDetector([Box(10, 10, 60, 60, 0.7)]), FakeDetector(), FakeDetector([]))
    app.dependency_overrides[anonymizer_dependency] = lambda: fake
    try:
        response = client.post("/anonymize", files=_upload(_foto()), headers=HEADERS)
    finally:
        app.dependency_overrides.clear()

    assert response.headers["x-faces-blurred"] == "0"
    assert response.headers["x-faces-unconfirmed"] == "1"
