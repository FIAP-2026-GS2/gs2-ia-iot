"""POST /verificar-foto: a .NET pergunta se a foto do cidadão serve como prova do conserto."""

import base64

import cv2
import numpy as np
import pytest

from app.api.dependencies import anonymizer_dependency, llm_dependency
from app.llm.client import LlmAnswer, LlmUnavailableError
from app.main import app
from app.vision.anonymizer import AnonymizationError, Anonymizer, Box
from app.vision.frames import encode_jpeg
from tests.conftest import TEST_API_KEY
from tests.test_anonymizer import FakeDetector

HEADERS = {"X-API-Key": TEST_API_KEY}
FACE = Box(10, 10, 60, 60, 0.9)  # o anonimizador falso "acha" um rosto aqui


def _photo() -> bytes:
    # Xadrez: o borrão muda os pixels de verdade, então dá para ver se o LLM recebeu a foto borrada.
    tile = (np.indices((240, 320)).sum(axis=0) % 2 * 255).astype(np.uint8)
    return encode_jpeg(np.dstack([tile, tile, tile]))


def _send(client, categoria="BURACO", decisao="CONTEST", data=None, headers=HEADERS,
          name="foto.jpg", content_type="image/jpeg"):
    return client.post("/verificar-foto", headers=headers,
                       files={"file": (name, data if data is not None else _photo(), content_type)},
                       data={"categoria": categoria, "decisao": decisao})


def _llm_answer(via=True, local=True, problema=True, descricao="Rua com um buraco grande no asfalto."):
    return {"via_publica": via, "local_do_tipo": local, "mostra_o_problema": problema,
            "confianca": 0.9, "descricao": descricao}


class FakeLlm:
    def __init__(self, data=None, error=None):
        self.data, self.error, self.calls, self.image_jpeg, self.prompt = data, error, 0, None, None

    def ask_json(self, system, prompt, image_jpeg=None):
        self.calls += 1
        self.prompt, self.image_jpeg = prompt, image_jpeg
        if self.error:
            raise self.error
        return LlmAnswer(self.data, "principal", "modelo-fake", 850)


@pytest.fixture
def use(client):
    def _use(llm=None, anonymizer=None):
        anonymizer = anonymizer or Anonymizer(FakeDetector([FACE]), FakeDetector())
        app.dependency_overrides[anonymizer_dependency] = lambda: anonymizer
        app.dependency_overrides[llm_dependency] = lambda: llm
        return llm

    yield _use
    app.dependency_overrides.clear()


# ---------- a regra ----------

def test_contestacao_que_mostra_o_problema_vale(client, use):
    use(FakeLlm(_llm_answer(problema=True)))

    body = _send(client, decisao="CONTEST").json()

    assert body["valida"] is True and body["motivo"] == "Foto conferida pela IA."
    assert body["descricao"] == "Rua com um buraco grande no asfalto."
    assert body["llm"] == {"provedor": "principal", "modelo": "modelo-fake", "latencia_ms": 850}


def test_contestacao_sem_o_problema_e_recusada(client, use):
    use(FakeLlm(_llm_answer(problema=False, descricao="Asfalto liso sem buracos.")))

    body = _send(client, decisao="CONTEST").json()

    assert body["valida"] is False
    assert body["motivo"] == "A foto não mostra buraco no asfalto: Asfalto liso sem buracos."


def test_confirmacao_no_lugar_certo_vale_mesmo_sem_o_problema(client, use):
    use(FakeLlm(_llm_answer(local=True, problema=False, descricao="Asfalto liso sem buracos")))

    body = _send(client, decisao="CONFIRM").json()

    assert body["valida"] is True


def test_confirmacao_fora_do_tipo_e_recusada(client, use):
    use(FakeLlm(_llm_answer(local=False, descricao="Uma calçada com uma árvore")))

    body = _send(client, categoria="POSTE_APAGADO", decisao="CONFIRM").json()

    assert body["valida"] is False
    assert body["motivo"] == "A foto não mostra poste ou luminária de iluminação pública: Uma calçada com uma árvore."


@pytest.mark.parametrize("decisao", ["CONFIRM", "CONTEST"])
def test_foto_que_nao_e_via_publica_e_recusada(client, use, decisao):
    # Mesmo que o LLM diga "sim" para o resto, a via pública vem primeiro.
    use(FakeLlm(_llm_answer(via=False, descricao="Um pássaro em um galho.")))

    body = _send(client, decisao=decisao).json()

    assert body["valida"] is False
    assert body["motivo"] == "A foto não mostra uma via pública: Um pássaro em um galho."


def test_o_pedido_ao_llm_muda_com_a_categoria_e_a_decisao(client, use):
    llm = use(FakeLlm(_llm_answer()))

    _send(client, categoria="VIA_ALAGADA", decisao="CONFIRM")

    assert "rua com água acumulada sobre a pista" in llm.prompt
    assert "o problema foi consertado" in llm.prompt


# ---------- privacidade ----------

def test_o_llm_recebe_a_foto_ja_borrada_e_a_net_tambem(client, use):
    llm = use(FakeLlm(_llm_answer()))
    original = cv2.imdecode(np.frombuffer(_photo(), np.uint8), cv2.IMREAD_COLOR)

    body = _send(client).json()

    sent = cv2.imdecode(np.frombuffer(llm.image_jpeg, np.uint8), cv2.IMREAD_COLOR)
    stored = cv2.imdecode(np.frombuffer(base64.b64decode(body["foto_borrada_base64"]), np.uint8), cv2.IMREAD_COLOR)
    for image in (sent, stored):
        # A área do rosto mudou (borrada); um canto longe dela ficou praticamente igual.
        assert np.abs(image[15:55, 15:55].astype(int) - original[15:55, 15:55].astype(int)).mean() > 30
        assert np.abs(image[150:200, 200:280].astype(int) - original[150:200, 200:280].astype(int)).mean() < 30
    assert body["rostos_borrados"] == 1 and body["placas_borradas"] == 0


def test_sem_borrao_responde_503_e_nao_chama_o_llm(client, use):
    llm = use(FakeLlm(_llm_answer()),
              Anonymizer(FakeDetector(error=AnonymizationError("modelo ausente")), FakeDetector()))

    response = _send(client)

    assert response.status_code == 503 and llm.calls == 0


# ---------- erros ----------

def test_sem_chave_responde_401(client, use):
    use(FakeLlm(_llm_answer()))
    assert _send(client, headers={}).status_code == 401


def test_categoria_invalida_responde_422(client, use):
    use(FakeLlm(_llm_answer()))
    assert _send(client, categoria="LIXO").status_code == 422
    assert _send(client, decisao="TALVEZ").status_code == 422


def test_arquivo_que_nao_e_imagem_responde_400(client, use):
    use(FakeLlm(_llm_answer()))
    assert _send(client, data=b"isto nao e uma foto").status_code == 400


def test_sem_llm_configurado_responde_503(client, use):
    use(None)

    response = _send(client)

    assert response.status_code == 503 and "não configurado" in response.json()["detail"]


def test_llm_indisponivel_responde_503(client, use):
    use(FakeLlm(error=LlmUnavailableError("principal: HTTP 429; reserva: timeout")))

    response = _send(client)

    assert response.status_code == 503 and "indisponível" in response.json()["detail"]


def test_resposta_do_llm_fora_do_formato_responde_503(client, use):
    use(FakeLlm({"via_publica": "talvez", "descricao": ""}))

    response = _send(client)

    assert response.status_code == 503 and "fora do formato" in response.json()["detail"]