"""Testes dos 3 cenários (buraco, poste, alagamento) e do /analisar que os junta.

Tudo com imagens sintéticas e falsos (modelo, LLM, anonimizador): roda sem internet e sem
os modelos grandes. O teste com fotos reais da CET fica para o teste manual.
"""

import json

import cv2
import numpy as np
import pytest

from app.analysis import AnalysisInputError, run_analysis
from app.api.dependencies import (
    anonymizer_dependency,
    cameras_dependency,
    llm_dependency,
    pothole_dependency,
    reference_store_dependency,
)
from app.llm.client import LlmUnavailableError
from app.main import app
from app.vision.anonymizer import Anonymizer
from app.vision.frames import encode_jpeg
from app.vision.pothole import PotholeBox, PotholeDetector, letterbox, severity_from_area
from app.vision.streetlight_roi import (
    Camera,
    CameraConfigError,
    Lamp,
    LocalReferenceStore,
    check_night,
    load_cameras,
    measure_camera,
)
from tests.conftest import TEST_API_KEY
from tests.test_anonymizer import FakeDetector
from app.llm.client import LlmAnswer
from app.vision.pothole_check import MAX_TILES, TILE, filter_by_zones, mosaic
from tests.test_streetlight import FakeLlm


class SeqLlm:
    """LLM falso que responde em sequência: 1ª chamada = verificador, 2ª = descrição."""

    def __init__(self, *answers):
        self.answers, self.prompts, self.images = list(answers), [], []

    def ask_json(self, system, prompt, image_jpeg=None):
        self.prompts.append(prompt)
        self.images.append(image_jpeg)
        return LlmAnswer(self.answers.pop(0), "principal", "modelo-fake", 900)


def _check(*verdicts):
    """Resposta do verificador: _check("BURACO", ("SOMBRA", 0.9)) → candidatos 1 e 2."""
    items = []
    for number, verdict in enumerate(verdicts, start=1):
        objeto, confianca = verdict if isinstance(verdict, tuple) else (verdict, 0.9)
        items.append({"id": number, "objeto": objeto, "confianca": confianca, "motivo": f"parece {objeto.lower()}"})
    return {"candidatos": items}


def _describe(*texts):
    return {"deteccoes": [{"id": i, "descricao": t, "mensagem_cidadao": "Buraco aqui."} for i, t in enumerate(texts, 1)]}

HEADERS = {"X-API-Key": TEST_API_KEY}
ROI = (300, 100, 320, 120)
CAMERA = Camera("CAM-TESTE", "Câmera de teste", (640, 480), (Lamp("p1", ROI),))


def street(lamp_on: bool, base: int = 60, sky: int | None = None) -> np.ndarray:
    """Rua sintética 640×480. A câmera de verdade aumenta o ganho à noite, então o fundo não é preto."""
    image = np.full((480, 640, 3), base, dtype=np.uint8)
    if sky is not None:
        image[:200] = sky  # céu claro atrás da luminária (caso do meio-dia)
    if lamp_on:
        cv2.circle(image, (310, 110), 7, (255, 255, 255), -1)
    return image


# ---------- buraco ----------

@pytest.mark.parametrize("area, expected", [(0.0, "LOW"), (0.029, "LOW"), (0.03, "MEDIUM"),
                                            (0.059, "MEDIUM"), (0.06, "HIGH"), (0.09, "CRITICAL")])
def test_gravidade_pela_area(area, expected):
    assert severity_from_area(area) == expected


def test_letterbox_nao_distorce():
    canvas, scale, pad_x, pad_y = letterbox(np.zeros((480, 1280, 3), dtype=np.uint8))
    assert canvas.shape == (640, 640, 3)
    assert scale == 0.5 and pad_x == 0 and pad_y == 200  # 1280×480 → 640×240, centrado: (640 − 240) / 2


class FakeSession:
    """Imita o ONNX Runtime: devolve a saída (4 + classes, 8400) que o YOLOv8 devolveria."""

    def __init__(self, output):
        self.output = output

    def run(self, _names, _feeds):
        return [self.output[None]]


def _detector_with(output, class_id=0, num_classes=1) -> PotholeDetector:
    detector = PotholeDetector.__new__(PotholeDetector)
    detector.name, detector._session, detector._input = "fake", FakeSession(output), "images"
    detector.class_id, detector.num_classes = class_id, num_classes
    return detector


def test_detector_volta_a_caixa_para_o_tamanho_original_e_aplica_nms():
    output = np.zeros((5, 8400), dtype=np.float32)
    # Duas caixas quase iguais (o NMS fica com uma) e uma fraca demais (< 0,25).
    output[:, 0] = [320, 320, 64, 64, 0.9]
    output[:, 1] = [322, 321, 64, 64, 0.8]
    output[:, 2] = [100, 400, 30, 30, 0.1]
    boxes = _detector_with(output).detect(np.zeros((480, 1280, 3), dtype=np.uint8))

    assert len(boxes) == 1
    box = boxes[0]
    # No quadrado 640 a caixa vai de 288 a 352; tira a borda (200 px em cima) e divide pela escala 0,5:
    assert (box.x1, box.y1, box.x2, box.y2) == (576, 176, 704, 304)
    assert box.confidence == pytest.approx(0.9)
    assert box.reportable


def test_detector_com_varias_classes_usa_so_a_de_buraco():
    output = np.zeros((9, 8400), dtype=np.float32)
    output[:4, 0] = [320, 320, 64, 64]
    output[4, 0] = 0.95          # classe 0 (rachadura): ignorada
    output[4 + 3, 1] = 0.7       # classe 3 (buraco)
    output[:4, 1] = [100, 100, 40, 40]
    boxes = _detector_with(output, class_id=3, num_classes=5).detect(np.zeros((640, 640, 3), dtype=np.uint8))

    assert [b.confidence for b in boxes] == [pytest.approx(0.7)]


# ---------- poste (ROI) ----------

def test_poste_aceso_com_referencia():
    reference = measure_camera(street(lamp_on=False, sky=200), CAMERA)
    [lamp] = check_night(street(lamp_on=True), CAMERA, reference)
    assert lamp.status == "LIT" and lamp.has_reference


def test_poste_apagado_com_referencia():
    reference = measure_camera(street(lamp_on=False, sky=200), CAMERA)
    [lamp] = check_night(street(lamp_on=False), CAMERA, reference)
    assert lamp.status == "OFF"
    assert 0.6 <= lamp.confidence <= 0.95  # teto: regra de brilho nunca tem certeza total


def test_sem_referencia_a_confianca_cai_20_por_cento():
    reference = measure_camera(street(lamp_on=False), CAMERA)
    with_ref = check_night(street(lamp_on=False), CAMERA, reference)[0]
    without = check_night(street(lamp_on=False), CAMERA, None)[0]
    assert without.status == "OFF" and not without.has_reference
    assert with_ref.confidence == 0.95       # teto
    assert without.confidence == 0.8         # 1,0 × 0,8 (o teto não chega a agir)


def test_referencia_desconta_o_que_ja_e_claro_de_dia():
    """Luminária na frente de uma parede clara: o contraste existe de dia e de noite, sem lâmpada.
    Sem referência pareceria acesa; com a referência do meio-dia, o ganho é zero."""
    scene = street(lamp_on=False)
    scene[100:120, 300:320] = 200  # mancha clara fixa (parede, placa)
    without = check_night(scene, CAMERA, None)[0]
    with_ref = check_night(scene, CAMERA, measure_camera(scene, CAMERA))[0]
    assert without.status == "LIT"
    assert with_ref.status == "OFF"


def test_retangulo_acompanha_a_resolucao_do_frame():
    assert CAMERA.scaled_roi(CAMERA.lamps[0], 1280, 960) == (600, 200, 640, 240)


def test_referencia_salva_e_lida(tmp_path):
    store = LocalReferenceStore(tmp_path)
    measures = measure_camera(street(lamp_on=False), CAMERA)
    store.save("CAM/../TESTE", measures, "2026-11-17T15:00:00+00:00")  # nome perigoso vira seguro
    loaded, at = store.load("CAM/../TESTE")
    assert loaded == measures and at.startswith("2026-11-17")
    assert all(path.parent == tmp_path / "streetlight" for path in tmp_path.rglob("*.json"))


def test_cameras_json_do_repositorio_e_valido():
    cameras = load_cameras("config/cameras.json")
    assert "CET-IGUATEMI" in cameras and cameras["CET-IGUATEMI"].lamps


def test_cameras_json_invalido(tmp_path):
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps({"cameras": {"X": {"name": "sem frame_size"}}}))
    with pytest.raises(CameraConfigError):
        load_cameras(path)


# ---------- orquestrador ----------

class FakePothole:
    name = "fake-yolo"

    def __init__(self, boxes=None):
        self.boxes = boxes or []

    def detect(self, image):
        return self.boxes


POTHOLE = PotholeBox(100, 300, 200, 360, 0.82, 0.02)


def _flood(alagada=True, confianca=0.8):
    return {"alagamento": {"alagada": alagada, "gravidade": "HIGH", "confianca": confianca,
                           "descricao": "Água cobrindo as duas faixas.", "mensagem_cidadao": "Evite a via."},
            "deteccoes": []}


def test_streetlight_sem_camera_ou_periodo_e_erro(tmp_path):
    store = LocalReferenceStore(tmp_path)
    with pytest.raises(AnalysisInputError, match="câmera"):
        run_analysis(street(True), {"STREETLIGHT"}, period="MEIA_NOITE", reference_store=store)
    with pytest.raises(AnalysisInputError, match="período"):
        run_analysis(street(True), {"STREETLIGHT"}, camera=CAMERA, reference_store=store)


def test_meio_dia_guarda_e_meia_noite_compara(tmp_path):
    store = LocalReferenceStore(tmp_path)
    noon = run_analysis(street(False, sky=200), {"STREETLIGHT"}, period="MEIO_DIA", camera=CAMERA,
                        reference_store=store, captured_at="2026-11-17T15:00:00+00:00")
    assert noon.findings == [] and "referência do meio-dia guardada" in noon.notes[0]

    night = run_analysis(street(False), {"STREETLIGHT"}, period="MEIA_NOITE", camera=CAMERA, reference_store=store)
    [finding] = night.findings
    assert finding.category == "POSTE_APAGADO" and finding.bbox == ROI and finding.source == "roi"
    assert finding.description.startswith("Luminária p1 apagada")  # sem LLM: texto da regra
    assert night.motivo_sem_llm == "LLM não configurado (variáveis LLM_*)"


def test_buraco_nao_roda_a_meia_noite(tmp_path):
    result = run_analysis(street(True), {"POTHOLE_CROSSWALK"}, period="MEIA_NOITE",
                          pothole_detector=FakePothole([POTHOLE]), reference_store=LocalReferenceStore(tmp_path))
    assert result.findings == [] and "meia-noite" in result.notes[0]


def test_llm_descreve_cada_deteccao_pelo_numero(tmp_path):
    llm = SeqLlm(_check("BURACO"), _describe("Buraco na faixa da direita."))
    result = run_analysis(street(True), {"POTHOLE_CROSSWALK"}, pothole_detector=FakePothole([POTHOLE]),
                          reference_store=LocalReferenceStore(tmp_path), llm=llm)
    [finding] = result.findings
    assert finding.description == "Buraco na faixa da direita."
    assert finding.verified is True and finding.confidence == POTHOLE.confidence
    assert [c.purpose for c in result.llm_calls] == ["verificar_buraco", "descrever"]
    assert result.motivo_sem_llm is None


# ---------- buraco: zonas e verificador ----------

def test_zona_ignorada_tira_o_bueiro_conhecido():
    camera = Camera("C", "c", (640, 480), (), pothole_ignore=((90, 290, 210, 370),))
    kept, dropped = filter_by_zones([POTHOLE], camera, 640, 480)
    assert kept == [] and dropped[0].by == "zona"


def test_zona_vale_na_proporcao_do_frame():
    camera = Camera("C", "c", (320, 240), (), pothole_ignore=((45, 145, 105, 185),))  # metade do tamanho
    kept, _ = filter_by_zones([POTHOLE], camera, 640, 480)
    assert kept == []


def test_caixa_fora_da_pista_e_descartada():
    camera = Camera("C", "c", (640, 480), (), road=((300, 200), (640, 200), (640, 480), (300, 480)))
    kept, dropped = filter_by_zones([POTHOLE], camera, 640, 480)  # centro em x=150: fora da pista
    assert kept == [] and "fora da pista" in dropped[0].reason


def test_sem_camera_nao_filtra_nada():
    assert filter_by_zones([POTHOLE], None, 640, 480) == ([POTHOLE], [])


def test_mosaico_junta_os_recortes_numa_imagem_so():
    boxes = [POTHOLE] * 4
    image = mosaic(street(True), boxes)
    assert image.shape[0] == 2 * TILE + 6 and image.shape[1] == 3 * TILE + 2 * 6  # 3 colunas, 2 linhas


def test_verificador_descarta_sombra_e_mantem_buraco(tmp_path):
    other = PotholeBox(400, 300, 460, 340, 0.62, 0.008)
    llm = SeqLlm(_check("BURACO", ("SOMBRA", 0.95)), _describe("Buraco."))
    result = run_analysis(street(True), {"POTHOLE_CROSSWALK"}, pothole_detector=FakePothole([POTHOLE, other]),
                          reference_store=LocalReferenceStore(tmp_path), llm=llm)
    assert [f.bbox for f in result.findings] == [(100, 300, 200, 360)]
    [gone] = result.discarded
    assert gone.box == other and gone.by == "llm" and gone.reason.startswith("SOMBRA")
    assert "2 recorte(s)" in llm.prompts[0]


def test_buraco_com_pouca_certeza_do_verificador_e_descartado(tmp_path):
    llm = SeqLlm(_check(("BURACO", 0.5)))
    result = run_analysis(street(True), {"POTHOLE_CROSSWALK"}, pothole_detector=FakePothole([POTHOLE]),
                          reference_store=LocalReferenceStore(tmp_path), llm=llm)
    assert result.findings == [] and len(result.discarded) == 1


def test_sem_llm_o_buraco_fica_abaixo_do_limiar(tmp_path):
    result = run_analysis(street(True), {"POTHOLE_CROSSWALK"}, pothole_detector=FakePothole([POTHOLE]),
                          reference_store=LocalReferenceStore(tmp_path), llm=None)
    [finding] = result.findings
    assert finding.confidence == 0.49 and finding.model_confidence == 0.82
    assert finding.verified is None and not finding.reportable
    assert "Não verificado" in finding.description
    assert any("segunda opinião" in note for note in result.notes)


def test_llm_esqueceu_uma_caixa_ela_fica_sem_verificacao(tmp_path):
    other = PotholeBox(400, 300, 460, 340, 0.7, 0.008)
    llm = SeqLlm(_check("BURACO"), _describe("Buraco.", "Outro."))  # respondeu só sobre o 1
    result = run_analysis(street(True), {"POTHOLE_CROSSWALK"}, pothole_detector=FakePothole([POTHOLE, other]),
                          reference_store=LocalReferenceStore(tmp_path), llm=llm)
    assert [f.verified for f in result.findings] == [True, None]
    assert result.findings[1].confidence == 0.49


def test_acima_do_limite_de_recortes_fica_sem_verificacao(tmp_path):
    boxes = [PotholeBox(10 * i, 300, 10 * i + 40, 340, 0.9 - i / 100, 0.005) for i in range(MAX_TILES + 1)]
    llm = SeqLlm(_check(*["BURACO"] * MAX_TILES), _describe(*["Buraco."] * (MAX_TILES + 1)))
    result = run_analysis(street(True), {"POTHOLE_CROSSWALK"}, pothole_detector=FakePothole(boxes),
                          reference_store=LocalReferenceStore(tmp_path), llm=llm)
    assert [f.verified for f in result.findings] == [True] * MAX_TILES + [None]


def test_llm_fora_do_ar_usa_texto_fixo_e_explica(tmp_path):
    llm = FakeLlm(error=LlmUnavailableError("principal: timeout; reserva: HTTP 429"))
    result = run_analysis(street(True), {"POTHOLE_CROSSWALK", "FLOODING"}, pothole_detector=FakePothole([POTHOLE]),
                          reference_store=LocalReferenceStore(tmp_path), llm=llm)
    [finding] = result.findings
    assert "detectado pelo modelo YOLO" in finding.description
    assert finding.citizen_message.startswith("Identificamos um possível buraco")
    assert result.motivo_sem_llm.startswith("LLM indisponível")
    assert "Alagamento: não avaliado" in result.notes[-1]
    assert not finding.reportable  # sem verificador, o buraco não abre ocorrência


def test_sem_deteccao_e_sem_alagamento_nao_chama_o_llm(tmp_path):
    llm = FakeLlm(_flood())
    result = run_analysis(street(True), {"POTHOLE_CROSSWALK"}, pothole_detector=FakePothole(),
                          reference_store=LocalReferenceStore(tmp_path), llm=llm)
    assert llm.calls == 0 and result.motivo_sem_llm is None


def test_alagamento_detectado(tmp_path):
    result = run_analysis(street(True), {"FLOODING"}, reference_store=LocalReferenceStore(tmp_path), llm=FakeLlm(_flood()))
    [finding] = result.findings
    assert (finding.category, finding.severity, finding.source) == ("VIA_ALAGADA", "HIGH", "llm")
    assert finding.bbox == (0, 0, 639, 479) and finding.reportable


def test_alagamento_com_pouca_confianca_fica_abaixo_do_limiar(tmp_path):
    result = run_analysis(street(True), {"FLOODING"}, reference_store=LocalReferenceStore(tmp_path),
                          llm=FakeLlm(_flood(confianca=0.4)))
    assert not result.findings[0].reportable  # a .NET registra como BELOW_CONFIDENCE_THRESHOLD


def test_via_seca_vira_aviso_e_nao_deteccao(tmp_path):
    result = run_analysis(street(True), {"FLOODING"}, reference_store=LocalReferenceStore(tmp_path),
                          llm=FakeLlm(_flood(alagada=False)))
    assert result.findings == [] and result.notes[-1].startswith("Alagamento: não")


def test_llm_sem_o_campo_de_alagamento_conta_como_falha(tmp_path):
    result = run_analysis(street(True), {"FLOODING"}, reference_store=LocalReferenceStore(tmp_path),
                          llm=FakeLlm({"deteccoes": []}))
    assert result.findings == [] and "não respondeu sobre o alagamento" in result.motivo_sem_llm


# ---------- rota /analisar ----------

@pytest.fixture
def scenario(client, tmp_path):
    def _use(llm=None, pothole=None):
        app.dependency_overrides[anonymizer_dependency] = lambda: Anonymizer(FakeDetector(), FakeDetector())
        app.dependency_overrides[llm_dependency] = lambda: llm
        app.dependency_overrides[pothole_dependency] = lambda: pothole
        app.dependency_overrides[cameras_dependency] = lambda: {CAMERA.code: CAMERA}
        app.dependency_overrides[reference_store_dependency] = lambda: LocalReferenceStore(tmp_path)

    yield _use
    app.dependency_overrides.clear()


def _post(client, image, **params):
    return client.post("/analisar", params=params, headers=HEADERS,
                       files={"file": ("frame.jpg", encode_jpeg(image), "image/jpeg")})


def test_rota_camera_desconhecida_da_422(client, scenario):
    scenario()
    response = _post(client, street(True), analises="STREETLIGHT", periodo="MEIA_NOITE", camera="NAO-EXISTE")
    assert response.status_code == 422 and "CAM-TESTE" in response.json()["detail"]


def test_rota_streetlight_sem_periodo_da_422(client, scenario):
    scenario()
    response = _post(client, street(True), analises="STREETLIGHT", camera=CAMERA.code)
    assert response.status_code == 422


def test_rota_junta_os_cenarios_no_formato_da_dotnet(client, scenario):
    other = PotholeBox(400, 300, 460, 340, 0.62, 0.008)
    scenario(llm=SeqLlm(_check("BURACO", "BUEIRO_OU_TAMPA"), {**_flood(), **_describe("Buraco grande.")}),
             pothole=FakePothole([POTHOLE, other]))
    response = client.post("/analisar", params=[("analises", "POTHOLE_CROSSWALK"), ("analises", "FLOODING")],
                           headers=HEADERS, files={"file": ("f.jpg", encode_jpeg(street(True)), "image/jpeg")})
    body = response.json()
    assert response.status_code == 200
    assert [d["category"] for d in body["detections"]] == ["BURACO", "VIA_ALAGADA"]
    first = body["detections"][0]
    assert (first["severity"], first["description"], first["reportable"]) == ("LOW", "Buraco grande.", True)
    assert [c["purpose"] for c in body["llm_calls"]] == ["verificar_buraco", "descrever"]
    assert body["motivo_sem_llm"] is None
    assert body["discarded"][0]["by"] == "llm" and body["discarded"][0]["reason"].startswith("BUEIRO")
    assert first["verified"] is True and first["model_confidence"] == 0.82
    assert body["models"] == ["pothole:fake-yolo", "flooding:llm:modelo-fake"]


def test_rota_poste_mostra_cada_luminaria(client, scenario):
    scenario()
    response = _post(client, street(False), analises="STREETLIGHT", periodo="MEIA_NOITE", camera=CAMERA.code)
    body = response.json()
    assert [d["category"] for d in body["detections"]] == ["POSTE_APAGADO"]
    assert body["lamps"][0]["status"] == "OFF" and body["lamps"][0]["has_reference"] is False


def test_rota_formato_imagem_lista_as_deteccoes_no_header(client, scenario):
    scenario(pothole=FakePothole([POTHOLE]))
    response = _post(client, street(True), formato="imagem")
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["X-Detections"] == "BURACO"
