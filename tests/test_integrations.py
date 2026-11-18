"""Cliente da .NET, Blob e o caminho completo de um frame (tudo com falsos: sem rede, sem Azure)."""

import hashlib
import json
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np
import pytest
from azure.core.exceptions import ResourceNotFoundError
from pydantic import SecretStr

from app.analysis import AnalysisResult, Finding
from app.config import Settings
from app.integrations.blob import BlobReferenceStore, BlobStorage
from app.integrations.dotnet import DotnetApiError, DotnetClient, DueItem
from app.pipeline import (
    PipelineConfigError,
    _period,
    analysis_payload,
    capture_key,
    capture_payload,
    coordinates,
    issue_type_ids,
    process_frame,
)
from app.vision.anonymizer import AnonymizationError, Anonymizer
from app.vision.streetlight_roi import LampMeasure, LocalReferenceStore
from tests.conftest import TEST_API_KEY
from tests.test_anonymizer import FakeDetector

PASSWORD = "senha-de-teste-123"
SLOT = datetime(2026, 11, 17, 15, 0, tzinfo=timezone.utc)  # 12:00 em São Paulo
TYPE_IDS = {"BURACO": 1, "FAIXA_APAGADA": 2, "POSTE_APAGADO": 3, "VIA_ALAGADA": 4}


# ---------- servidor .NET falso ----------

class FakeDotnet:
    """Responde como a .NET. `script` permite forçar respostas por rota (ex.: um 429 antes do 200)."""

    def __init__(self, token_minutes=60):
        self.calls: list[httpx.Request] = []
        self.logins = 0
        self.token_minutes = token_minutes
        self.script: dict[str, list[httpx.Response]] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        path = request.url.path
        if path == "/api/auth/login":
            self.logins += 1
            body = json.loads(request.content)
            if body["password"] != PASSWORD:
                return httpx.Response(401, json={"title": "Não autenticado"})
            expires = datetime.now(timezone.utc) + timedelta(minutes=self.token_minutes)
            return httpx.Response(200, json={"accessToken": f"token-{self.logins}", "tokenType": "Bearer",
                                             "expiresAt": expires.isoformat(), "role": "IA_SERVICE"})
        if self.script.get(path):
            return self.script[path].pop(0)
        if path == "/api/schedules/due":
            return httpx.Response(200, json=[{
                "cameraId": 7, "cameraCode": "CET-IGUATEMI", "slotAt": "2026-11-17T15:00:00Z", "reason": "FIXED_TIME",
                "captureScheduleId": 3, "triggerEventId": None, "analyses": ["POTHOLE_CROSSWALK", "STREETLIGHT"]}])
        if path == "/api/captures":
            return httpx.Response(201, json={"id": 11, "images": [{"id": 21}]})
        if path == "/api/analyses":
            return httpx.Response(201, json={"analysisId": 31, "detections": [{"outcome": "OPENED_NEW_ISSUE"}]})
        return httpx.Response(404)

    def requests_to(self, path):
        return [call for call in self.calls if call.url.path == path]


def _client(server: FakeDotnet, password=PASSWORD) -> tuple[DotnetClient, list[float]]:
    waits: list[float] = []
    client = DotnetClient("http://dotnet.test", "ia.sampatech", SecretStr(password),
                          transport=httpx.MockTransport(server), sleep=waits.append)
    return client, waits


def test_login_e_agenda():
    server = FakeDotnet()
    client, _ = _client(server)
    [item] = client.due()
    assert item == DueItem(7, "CET-IGUATEMI", SLOT, "FIXED_TIME", 3, None, ("POTHOLE_CROSSWALK", "STREETLIGHT"))
    assert server.requests_to("/api/schedules/due")[0].headers["Authorization"] == "Bearer token-1"


def test_reaproveita_o_token_e_renova_perto_de_vencer():
    server = FakeDotnet()
    client, _ = _client(server)
    client.due(); client.due()
    assert server.logins == 1
    server.token_minutes = 3  # menos que a margem de 5 min: cada chamada renova
    client._token = None
    client.due(); client.due()
    assert server.logins == 3


def test_401_entra_de_novo_uma_vez():
    server = FakeDotnet()
    server.script["/api/schedules/due"] = [httpx.Response(401)]
    client, _ = _client(server)
    client.due()
    assert server.logins == 2


def test_429_espera_o_retry_after():
    server = FakeDotnet()
    server.script["/api/captures"] = [httpx.Response(429, headers={"Retry-After": "7"})]
    client, waits = _client(server)
    client.create_capture({"x": 1}, "capture:7:20261117T150000Z")
    assert waits == [7.0]
    assert [r.headers["Idempotency-Key"] for r in server.requests_to("/api/captures")] == ["capture:7:20261117T150000Z"] * 2


def test_erro_500_tenta_de_novo_com_espera_crescente():
    server = FakeDotnet()
    server.script["/api/analyses"] = [httpx.Response(503), httpx.Response(500)]
    client, waits = _client(server)
    client.create_analysis({}, "analysis:21")
    assert waits == [2, 4]


def test_erro_400_mostra_a_mensagem_da_dotnet_sem_repetir():
    server = FakeDotnet()
    server.script["/api/analyses"] = [httpx.Response(400, json={
        "title": "Regra de negócio violada", "detail": "A captura 11 já foi analisada (status: ANALYZED)."})]
    client, waits = _client(server)
    with pytest.raises(DotnetApiError, match="já foi analisada") as error:
        client.create_analysis({}, "analysis:21")
    assert error.value.status_code == 400 and waits == []


def test_erros_de_validacao_aparecem_campo_a_campo():
    server = FakeDotnet()
    server.script["/api/captures"] = [httpx.Response(400, json={
        "title": "Dados inválidos", "errors": {"Image.FilePath": ["O caminho do arquivo deve ser uma URL https"]}})]
    client, _ = _client(server)
    with pytest.raises(DotnetApiError, match="Image.FilePath: O caminho"):
        client.create_capture({}, "k")


def test_senha_errada_nao_vaza_a_senha():
    client, _ = _client(FakeDotnet(), password="outra-senha-qualquer")
    with pytest.raises(DotnetApiError) as error:
        client.due()
    assert "outra-senha" not in str(error.value) and "outra-senha" not in repr(client)


# ---------- Blob falso ----------

class FakeBlob:
    def __init__(self, url):
        self.url = url


class FakeContainer:
    def __init__(self):
        self.files: dict[str, bytes] = {}

    def upload_blob(self, name, data, overwrite=False, content_settings=None):
        self.files[name] = data

    def download_blob(self, blob):
        if blob not in self.files:
            raise ResourceNotFoundError("não existe")
        data = self.files[blob]
        return type("Downloader", (), {"readall": lambda self: data})()

    def get_blob_client(self, blob):
        return FakeBlob(f"https://conta.blob.core.windows.net/frames/{blob}")


def test_frame_com_nome_deterministico_url_https_e_sha256():
    container = FakeContainer()
    name, url, sha = BlobStorage(container, "dev/").upload_frame("CET-IGUATEMI", SLOT, b"jpeg")
    assert name == "20261117T150000Z.jpg"
    assert url == "https://conta.blob.core.windows.net/frames/dev/CET-IGUATEMI/20261117T150000Z.jpg"
    assert sha == hashlib.sha256(b"jpeg").hexdigest()


def test_referencia_do_poste_no_blob():
    store = BlobReferenceStore(BlobStorage(FakeContainer(), "dev/"))
    assert store.load("CET-IGUATEMI") is None
    measures = {"p1": LampMeasure(0.0, 120.0, 100.0)}
    store.save("CET-IGUATEMI", measures, "2026-11-17T15:00:00+00:00")
    assert store.load("CET-IGUATEMI") == (measures, "2026-11-17T15:00:00+00:00")
    assert "dev/refs/streetlight/CET-IGUATEMI.json" in store.storage.container.files


# ---------- corpos das requisições ----------

ITEM = DueItem(7, "CET-IGUATEMI", SLOT, "FIXED_TIME", 3, None, ("POTHOLE_CROSSWALK",))


def test_coordenadas_no_formato_da_dotnet():
    assert coordinates((357, 310, 411, 344)) == "357,310,54,34"
    assert coordinates((5, 5, 5, 9)) == "5,5,1,4"  # largura nunca zero


def test_captura_agendada_e_por_alerta():
    payload = capture_payload(ITEM, "f.jpg", "https://x/f.jpg", "ab" * 32)
    assert payload["source"] == "RECORDING_FRAME" and payload["captureScheduleId"] == 3
    assert payload["captureDate"] == "2026-11-17T15:00:00Z" and payload["image"]["isAnonymized"] is True
    event = DueItem(7, "CET-IGUATEMI", SLOT, "TRIGGER_EVENT", None, 9, ("FLOODING",))
    assert capture_payload(event, "f", "https://x", "ab")["triggerEventId"] == 9
    assert capture_key(ITEM) == "capture:7:20261117T150000Z"


def test_analise_completa_e_com_falha():
    result = AnalysisResult(models=["pothole:yolo"])
    result.findings.append(Finding("BURACO", 0.82, "LOW", (100, 300, 200, 360), "yolo",
                                   "Buraco na faixa da direita.", "Há um buraco aqui."))
    body = analysis_payload(21, result, 950, TYPE_IDS)
    [detection] = body["detections"]
    assert body["status"] == "COMPLETED" and body["aiModel"] == "pothole:yolo"
    assert detection == {"issueTypeId": 1, "confidenceScore": 0.82, "estimatedSeverity": "LOW",
                         "detectionCoordinates": "100,300,100,60",
                         "description": "Buraco na faixa da direita. | Para o cidadão: Há um buraco aqui."}
    failed = analysis_payload(21, None, 950, TYPE_IDS)
    assert failed["status"] == "FAILED" and failed["detections"] == []


def test_ids_dos_tipos_precisam_estar_no_env():
    with pytest.raises(PipelineConfigError, match="ISSUE_TYPE_ID_BURACO"):
        issue_type_ids(Settings(_env_file=None, ia_api_key=TEST_API_KEY))


@pytest.mark.parametrize("hour_utc, expected", [(15, "MEIO_DIA"), (3, "MEIA_NOITE")])
def test_periodo_vem_do_horario_da_agenda(hour_utc, expected):
    item = DueItem(7, "C", SLOT.replace(hour=hour_utc), "FIXED_TIME", 3, None, ("STREETLIGHT",))
    assert _period(item) == expected


# ---------- caminho completo ----------

def _run(server, anonymizer=None, container=None, tmp_path=None):
    client, _ = _client(server)
    return process_frame(
        ITEM, np.full((480, 640, 3), 90, dtype=np.uint8),
        anonymizer=anonymizer or Anonymizer(FakeDetector(), FakeDetector()), dotnet=client,
        blob=BlobStorage(container or FakeContainer(), "dev/"), type_ids=TYPE_IDS, camera=None,
        pothole_detector=None, reference_store=LocalReferenceStore(tmp_path), llm=None)


def test_frame_vai_do_borrao_ate_a_analise(tmp_path):
    server, container = FakeDotnet(), FakeContainer()
    outcome = _run(server, container=container, tmp_path=tmp_path)
    assert outcome.status == "SENT" and outcome.image_id == 21
    assert list(container.files) == ["dev/CET-IGUATEMI/20261117T150000Z.jpg"]
    capture = json.loads(server.requests_to("/api/captures")[0].content)
    assert capture["image"]["filePath"].endswith("dev/CET-IGUATEMI/20261117T150000Z.jpg")
    analysis = server.requests_to("/api/analyses")[0]
    assert analysis.headers["Idempotency-Key"] == "analysis:21"
    assert json.loads(analysis.content)["status"] == "COMPLETED"


def test_sem_borrao_nada_sai_do_servico(tmp_path):
    server, container = FakeDotnet(), FakeContainer()
    broken = Anonymizer(FakeDetector(error=AnonymizationError("modelo ausente")), FakeDetector())
    outcome = _run(server, anonymizer=broken, container=container, tmp_path=tmp_path)
    assert outcome.status == "ANONYMIZATION_FAILED"
    assert container.files == {} and server.requests_to("/api/captures") == []


def test_reenvio_de_imagem_ja_analisada_nao_e_erro(tmp_path):
    server = FakeDotnet()
    server.script["/api/analyses"] = [httpx.Response(422, json={"title": "Idempotency-Key reutilizada com outro corpo"})]
    outcome = _run(server, tmp_path=tmp_path)
    assert outcome.status == "ALREADY_SENT"
