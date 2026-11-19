"""Aquecimento dos modelos na subida: roda cada um uma vez e nunca derruba a API."""

import numpy as np

from app.warmup import WARM_UP_SHAPE, warm_up


def test_roda_cada_modelo_uma_vez_numa_imagem_preta():
    seen = []

    took = warm_up(("borrão", lambda image: seen.append(("borrão", image))),
                   ("buraco", lambda image: seen.append(("buraco", image))))

    assert [name for name, _ in seen] == ["borrão", "buraco"]
    assert all(image.shape == WARM_UP_SHAPE and not image.any() for _, image in seen)
    assert set(took) == {"borrão", "buraco"} and all(isinstance(s, float) for s in took.values())


def test_modelo_que_falha_nao_impede_os_outros():
    def broken(image):
        raise FileNotFoundError("smartathon-yolov8s.onnx")

    ran = []
    took = warm_up(("buraco", broken), ("borrão", lambda image: ran.append(True)))

    assert took["buraco"] is None and took["borrão"] is not None and ran == [True]


def test_nos_testes_a_subida_nao_aquece(client, monkeypatch):
    # O client sobe a API com APP_ENV=test; se aquecesse, chamaria o anonimizador real.
    import app.main as main

    calls = []
    monkeypatch.setattr(main, "warm_up", lambda *models: calls.append(models))
    from fastapi.testclient import TestClient

    with TestClient(main.app):
        pass

    assert calls == []


def test_fora_dos_testes_a_subida_aquece_o_borrao_e_o_buraco(monkeypatch):
    import app.main as main
    from app.config import get_settings
    from fastapi.testclient import TestClient

    monkeypatch.setenv("APP_ENV", "local")
    get_settings.cache_clear()
    calls = []
    monkeypatch.setattr(main, "warm_up", lambda *models: calls.append([name for name, _ in models]))

    with TestClient(main.app):
        pass

    assert calls == [["borrão", "buraco"]]
