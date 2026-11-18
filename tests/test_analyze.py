import base64

import numpy as np
import pytest
from fastapi import HTTPException

from app.api.dependencies import anonymizer_dependency
from app.main import app
from app.vision.anonymizer import AnonymizationError, Anonymizer, Box
from app.vision.frames import decode_image, encode_jpeg
from tests.conftest import TEST_API_KEY
from tests.test_anonymizer import FakeDetector, needs_person_model

HEADERS = {"X-API-Key": TEST_API_KEY}


def _jpeg(width=320, height=240):
    return encode_jpeg(np.full((height, width, 3), 90, dtype=np.uint8))


def _upload(data, name="frame.jpg", content_type="image/jpeg"):
    return {"file": (name, data, content_type)}


@pytest.fixture
def fake_anonymizer():
    """Troca o anonimizador real por um falso, que 'acha' 1 rosto e 1 placa."""
    fake = Anonymizer(FakeDetector([Box(10, 10, 50, 50, 0.9)]), FakeDetector([Box(200, 180, 280, 210, 0.7)]))
    app.dependency_overrides[anonymizer_dependency] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


def test_sem_chave_responde_401(client):
    response = client.post("/analisar", files=_upload(_jpeg()))

    assert response.status_code == 401


def test_chave_errada_responde_401(client):
    response = client.post("/analisar", files=_upload(_jpeg()), headers={"X-API-Key": "errada" * 5})

    assert response.status_code == 401


def test_analisa_imagem_e_devolve_json(client, fake_anonymizer):
    response = client.post("/analisar", files=_upload(_jpeg()), headers=HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["anonymization"]["faces_blurred"] == 1
    assert body["anonymization"]["heads_blurred"] == 0  # o anonimizador falso não tem detector de pessoas
    assert body["anonymization"]["plates_blurred"] == 1
    assert body["anonymization"]["blurred_area_percent"] > 0
    assert body["detections"] == []
    annotated = decode_image(base64.b64decode(body["images"]["annotated_jpeg_base64"]))
    assert annotated.shape == (240, 320, 3)


def test_formato_imagem_devolve_jpeg(client, fake_anonymizer):
    response = client.post("/analisar?formato=imagem", files=_upload(_jpeg()), headers=HEADERS)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["x-faces-blurred"] == "1"
    assert response.headers["x-heads-blurred"] == "0"
    assert decode_image(response.content).shape == (240, 320, 3)


def test_arquivo_que_nao_e_imagem_responde_415(client, fake_anonymizer):
    response = client.post("/analisar", files=_upload(b"oi", "nota.txt", "text/plain"), headers=HEADERS)

    assert response.status_code == 415


def test_imagem_corrompida_responde_422(client, fake_anonymizer):
    response = client.post("/analisar", files=_upload(b"\xff\xd8\xff nada"), headers=HEADERS)

    assert response.status_code == 422


def test_imagem_grande_demais_responde_413(client, fake_anonymizer):
    response = client.post("/analisar", files=_upload(b"0" * (10 * 1024 * 1024 + 1)), headers=HEADERS)

    assert response.status_code == 413


def test_falha_na_anonimizacao_responde_503(client):
    broken = Anonymizer(FakeDetector(error=RuntimeError("detector caiu")), FakeDetector())
    app.dependency_overrides[anonymizer_dependency] = lambda: broken
    try:
        response = client.post("/analisar", files=_upload(_jpeg()), headers=HEADERS)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "detector caiu" in response.json()["detail"]


def test_modelos_ausentes_respondem_503(client):
    def unavailable():
        raise HTTPException(503, "Modelo de rosto não encontrado")

    app.dependency_overrides[anonymizer_dependency] = unavailable
    try:
        response = client.post("/analisar", files=_upload(_jpeg()), headers=HEADERS)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503


@needs_person_model
def test_analisa_com_os_modelos_reais(client):
    response = client.post("/analisar", files=_upload(_jpeg(1280, 720)), headers=HEADERS)

    assert response.status_code == 200
    assert response.json()["anonymization"] == {
        "faces_blurred": 0, "heads_blurred": 0, "plates_blurred": 0, "faces_unconfirmed": 0,
            "blurred_area_percent": 0.0,
    }


def test_falha_de_anonimizacao_nunca_devolve_a_imagem(client):
    """Mesmo com erro, a resposta não carrega a imagem original (só a mensagem)."""
    broken = Anonymizer(FakeDetector(), FakeDetector(error=AnonymizationError("placa: erro")))
    app.dependency_overrides[anonymizer_dependency] = lambda: broken
    try:
        response = client.post("/analisar", files=_upload(_jpeg()), headers=HEADERS)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "images" not in response.text
