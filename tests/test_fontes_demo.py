"""Lista de fotos de teste (config/fontes_demo.json): o script não supõe padrão de nome."""

import json
from pathlib import Path

import pytest

from scripts.demo.subir_fontes_demo import blob_name, load_photos


def _write(tmp_path, cameras):
    path = tmp_path / "fontes.json"
    path.write_text(json.dumps({"cameras": cameras}), encoding="utf-8")
    return path


def test_le_cada_camera_com_os_arquivos_que_ela_declara(tmp_path):
    path = _write(tmp_path, {
        "CAM-A": {"dia": "fotos/qualquer-nome.png", "noite": "outra/pasta/n.jpg"},
        "CAM-B": {"dia": "b.jpg"},
    })

    photos = load_photos(path)

    assert photos == {
        "CAM-A": {"dia": Path("fotos/qualquer-nome.png"), "noite": Path("outra/pasta/n.jpg")},
        "CAM-B": {"dia": Path("b.jpg")},
    }


def test_periodo_com_erro_de_digitacao_e_recusado(tmp_path):
    with pytest.raises(ValueError, match="manha"):
        load_photos(_write(tmp_path, {"CAM-A": {"manha": "a.png"}}))


def test_nome_no_blob_e_o_que_o_cameras_json_espera():
    assert blob_name("DEMO-BURACO", "dia") == "demo/DEMO-BURACO/dia.jpg"


def test_lista_do_repositorio_cobre_as_cameras_com_fonte_no_blob():
    """Toda câmera que lê "blob:demo/..." no cameras.json tem a foto declarada na lista."""
    cameras = json.loads(Path("config/cameras.json").read_text(encoding="utf-8"))["cameras"]
    photos = load_photos(Path("config/fontes_demo.json"))
    for code, data in cameras.items():
        for period, location in (data.get("source") or {}).items():
            if location.startswith("blob:"):
                assert location == f"blob:{blob_name(code, period)}"
                assert period in photos.get(code, {}), f"{code} {period} sem foto em fontes_demo.json"
