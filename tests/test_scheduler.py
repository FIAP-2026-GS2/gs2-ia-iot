"""Agendador: fonte do frame (vídeo em loop ou foto dia/noite) e a rodada (um item com erro não para os outros)."""

from dataclasses import dataclass
from datetime import datetime, timezone

import cv2
import numpy as np
import pytest

from app.integrations.dotnet import DueItem
from app.scheduler import FrameSource, SourceError, load_sources, run_once
from tests.conftest import TEST_API_KEY

NOON_SP = datetime(2026, 11, 18, 15, 0, 2, tzinfo=timezone.utc)      # 12:00:02 em São Paulo
MIDNIGHT_SP = datetime(2026, 11, 19, 3, 0, 0, tzinfo=timezone.utc)   # 00:00 em São Paulo


def _image(path, value):
    cv2.imwrite(str(path), np.full((60, 80, 3), value, dtype=np.uint8))
    return path


def _item(code="CAM-1", slot=NOON_SP):
    return DueItem(1, code, slot, "FIXED_TIME", 10, None, ("POTHOLE_CROSSWALK",))


@dataclass
class Outcome:
    status: str = "SENT"
    image_id: int | None = 7
    detections: list | None = None
    message: str = ""


# ---------- fonte do frame ----------

def test_foto_de_dia_ao_meio_dia_e_de_noite_a_meia_noite(tmp_path):
    source = FrameSource("CAM-1", day=_image(tmp_path / "dia.png", 200), night=_image(tmp_path / "noite.png", 20))

    assert source.frame_at(NOON_SP).mean() == pytest.approx(200, abs=2)
    assert source.frame_at(MIDNIGHT_SP).mean() == pytest.approx(20, abs=2)


def test_video_em_loop_usa_o_segundo_do_dia(tmp_path):
    path = tmp_path / "rua.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 60))
    for index in range(50):  # 5 s; o brilho indica o segundo (0, 50, 100, 150, 200)
        writer.write(np.full((60, 80, 3), (index // 10) * 50, dtype=np.uint8))
    writer.release()

    # 12:00:02 → segundo 43202 do dia; em loop de 4 s (5 s menos 1 de margem) → segundo 2 → brilho 100.
    frame = FrameSource("CAM-1", video=path).frame_at(NOON_SP)

    assert frame.mean() == pytest.approx(100, abs=8)


def test_sem_foto_do_periodo_ou_arquivo_faltando_da_erro_claro(tmp_path):
    with pytest.raises(SourceError, match="noite"):
        FrameSource("CAM-1", day=_image(tmp_path / "dia.png", 200)).frame_at(MIDNIGHT_SP)
    with pytest.raises(SourceError, match="não encontrado"):
        FrameSource("CAM-1", day=tmp_path / "sumiu.png").frame_at(NOON_SP)


def test_load_sources_le_video_e_dia_noite_e_ignora_camera_sem_source(tmp_path):
    config = {"cameras": {
        "A": {"source": "videos/a.mp4"},
        "B": {"source": {"dia": "b-dia.png", "noite": "b-noite.png"}},
        "C": {"name": "sem fonte"},
    }}

    sources = load_sources(config, tmp_path)

    assert set(sources) == {"A", "B"}
    assert sources["A"].video == tmp_path / "videos" / "a.mp4"
    assert sources["B"].night == tmp_path / "b-noite.png"


# ---------- a rodada ----------

def test_rodada_processa_cada_item_e_um_erro_nao_para_os_outros(tmp_path):
    sources = {code: FrameSource(code, day=_image(tmp_path / f"{code}.png", 100)) for code in ("A", "B")}
    seen = []

    def process(item, frame):
        seen.append(item.camera_code)
        if item.camera_code == "A":
            raise RuntimeError("Blob fora do ar")
        return Outcome(detections=[{"id": 1}])

    report = run_once(lambda at: [_item("A"), _item("B"), _item("SEM-FONTE")], sources, process)

    assert seen == ["A", "B"]
    assert [(i.camera_code, i.status) for i in report.items] == [("A", "ERROR"), ("B", "SENT"), ("SEM-FONTE", "NO_SOURCE")]
    assert "Blob fora do ar" in report.items[0].message
    assert report.items[1].image_id == 7 and report.items[1].detections == 1
    assert report.due == 3 and not report.error


def test_rodada_com_a_net_fora_do_ar_registra_o_erro(tmp_path):
    def due(at):
        raise ConnectionError("sem rede")

    report = run_once(due, {}, lambda item, frame: Outcome())

    assert report.error.startswith("ConnectionError") and report.items == []


def test_rodada_repassa_o_horario_simulado():
    asked = []
    run_once(lambda at: asked.append(at) or [], {}, lambda item, frame: Outcome(), at=NOON_SP)

    assert asked == [NOON_SP]


# ---------- rota ----------

def test_agendador_desligado_por_padrao(client):
    response = client.get("/agendador", headers={"X-API-Key": TEST_API_KEY})

    assert response.status_code == 200
    assert response.json() == {"enabled": False, "interval_minutes": 5, "last_run": None}


def test_agendador_exige_a_chave(client):
    assert client.get("/agendador").status_code == 401
