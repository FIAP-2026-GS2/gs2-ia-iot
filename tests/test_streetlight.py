import cv2
import numpy as np
import pytest

from app.api.dependencies import anonymizer_dependency, llm_dependency
from app.llm.client import LlmAnswer, LlmUnavailableError
from app.main import app
from app.vision.anonymizer import AnonymizationError, Anonymizer
from app.vision.frames import encode_jpeg
from app.vision.streetlight import read_brightness, shrink
from tests.conftest import TEST_API_KEY
from tests.test_anonymizer import FakeDetector

HEADERS = {"X-API-Key": TEST_API_KEY}


def night(lamp: bool) -> np.ndarray:
    """Cena noturna sintética: fundo escuro, um poste e (opcional) uma lâmpada branca."""
    image = np.full((480, 640, 3), 20, dtype=np.uint8)
    cv2.rectangle(image, (300, 100), (310, 400), (80, 80, 80), -1)
    if lamp:
        cv2.circle(image, (330, 110), 18, (255, 255, 255), -1)
    return image


def day() -> np.ndarray:
    return np.full((480, 640, 3), 170, dtype=np.uint8)


def _upload(image):
    return {"file": ("poste.jpg", encode_jpeg(image), "image/jpeg")}


class FakeLlm:
    def __init__(self, data=None, error=None):
        self.data, self.error, self.calls, self.image_jpeg = data, error, 0, None

    def ask_json(self, system, prompt, image_jpeg=None):
        self.calls += 1
        self.image_jpeg = image_jpeg
        if self.error:
            raise self.error
        return LlmAnswer(self.data, "principal", "modelo-fake", 850)


def _answer(lampada="ACESA", confianca=0.92, luminaria=True, noturna=True):
    return {
        "luminaria_visivel": luminaria, "foto_noturna": noturna, "lampada": lampada,
        "confianca": confianca, "descricao": "Luminária acesa iluminando a calçada.",
    }


@pytest.fixture
def use(client):
    """Troca o anonimizador e o LLM pelos falsos que o teste pedir."""

    def _use(llm=None, anonymizer=None):
        anonymizer = anonymizer or Anonymizer(FakeDetector(), FakeDetector())
        app.dependency_overrides[anonymizer_dependency] = lambda: anonymizer
        app.dependency_overrides[llm_dependency] = lambda: llm
        return llm

    yield _use
    app.dependency_overrides.clear()


# --- regra de brilho (plano B) ---

def test_brilho_lampada_acesa():
    assert read_brightness(night(lamp=True)).status == "LIT"


def test_brilho_lampada_apagada():
    assert read_brightness(night(lamp=False)).status == "OFF"


def test_brilho_foto_de_dia_e_indeterminada():
    reading = read_brightness(day())
    assert reading.status == "UNCERTAIN"
    assert "não parece noturna" in reading.describe()


def test_brilho_ceu_claro_com_rua_escura_e_indeterminado():
    # Caso real da calibração: rua escura (mediana baixa), mas céu de dia estourado (18% dos pixels).
    image = np.full((480, 640, 3), 30, dtype=np.uint8)
    image[:150] = 255
    reading = read_brightness(image)
    assert reading.status == "UNCERTAIN"
    assert "parece céu de dia" in reading.describe()


def test_reduz_foto_grande_e_nao_aumenta_a_pequena():
    assert shrink(np.zeros((3000, 4000, 3), np.uint8)).shape == (768, 1024, 3)
    assert shrink(np.zeros((100, 200, 3), np.uint8)).shape == (100, 200, 3)


# --- rota ---

def test_sem_chave_responde_401(client):
    assert client.post("/verificar-poste", files=_upload(night(True))).status_code == 401


def test_llm_confirma_lampada_acesa(client, use):
    llm = use(FakeLlm(_answer()))

    body = client.post("/verificar-poste", files=_upload(night(True)), headers=HEADERS).json()

    assert body["status"] == "LIT"
    assert body["repaired"] is True
    assert body["method"] == "llm"
    assert body["confidence"] == 0.92
    assert body["motivo_sem_llm"] is None
    assert body["llm"] == {"provider": "principal", "model": "modelo-fake", "latency_ms": 850}
    assert llm.calls == 1


def test_llm_diz_apagada(client, use):
    use(FakeLlm(_answer("APAGADA", 0.8)))

    body = client.post("/verificar-poste", files=_upload(night(False)), headers=HEADERS).json()

    assert (body["status"], body["repaired"]) == ("OFF", False)


@pytest.mark.parametrize(
    "answer",
    [_answer(confianca=0.4), _answer(luminaria=False), _answer(noturna=False), _answer("INDETERMINADA", 0.9)],
    ids=["pouca-confianca", "sem-luminaria", "foto-de-dia", "indeterminada"],
)
def test_na_duvida_nao_confirma_o_conserto(client, use, answer):
    use(FakeLlm(answer))

    body = client.post("/verificar-poste", files=_upload(night(True)), headers=HEADERS).json()

    assert (body["status"], body["repaired"]) == ("UNCERTAIN", None)


def test_llm_fora_do_ar_usa_regra_de_brilho(client, use):
    use(FakeLlm(error=LlmUnavailableError("principal: HTTP 429; reserva: sem resposta em 30 s")))

    body = client.post("/verificar-poste", files=_upload(night(True)), headers=HEADERS).json()

    assert body["method"] == "brightness"
    assert body["status"] == "LIT"
    assert body["confidence"] == 0.6
    assert "HTTP 429" in body["motivo_sem_llm"]
    assert body["llm"] is None


def test_llm_fora_do_formato_usa_regra_de_brilho(client, use):
    use(FakeLlm({"resposta": "acho que sim"}))

    body = client.post("/verificar-poste", files=_upload(night(False)), headers=HEADERS).json()

    assert body["method"] == "brightness"
    assert body["status"] == "OFF"
    assert "fora do formato" in body["motivo_sem_llm"]


def test_sem_llm_configurado_usa_regra_de_brilho(client, use):
    use(llm=None)

    body = client.post("/verificar-poste", files=_upload(night(True)), headers=HEADERS).json()

    assert body["method"] == "brightness"
    assert body["motivo_sem_llm"] == "LLM não configurado no .env"


def test_sem_borrao_a_foto_nao_vai_ao_llm(client, use):
    broken = Anonymizer(FakeDetector(error=AnonymizationError("rosto: erro")), FakeDetector())
    llm = use(FakeLlm(_answer()), broken)

    response = client.post("/verificar-poste", files=_upload(night(True)), headers=HEADERS)

    assert response.status_code == 503
    assert llm.calls == 0


def test_foto_enviada_ao_llm_e_reduzida(client, use):
    llm = use(FakeLlm(_answer()))
    big = np.full((2000, 3000, 3), 20, dtype=np.uint8)

    client.post("/verificar-poste", files=_upload(big), headers=HEADERS)

    sent = cv2.imdecode(np.frombuffer(llm.image_jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert max(sent.shape[:2]) == 1024
