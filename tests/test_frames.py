import cv2
import numpy as np
import pytest

from app.vision.frames import (
    FrameError,
    InvalidImageError,
    decode_image,
    encode_jpeg,
    extract_frame,
    extract_frames,
)

FPS = 10
SECONDS = 5


@pytest.fixture(scope="module")
def video_path(tmp_path_factory):
    """Vídeo sintético de 5 s: o brilho de cada frame indica o segundo (0, 50, 100, 150, 200)."""
    path = tmp_path_factory.mktemp("video") / "rua.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (160, 120))
    assert writer.isOpened(), "o OpenCV não conseguiu criar o vídeo de teste"
    for index in range(FPS * SECONDS):
        writer.write(np.full((120, 160, 3), (index // FPS) * 50, dtype=np.uint8))
    writer.release()
    return path


def test_extrai_o_frame_do_instante_pedido(video_path):
    frame = extract_frame(video_path, at_seconds=2.0)

    assert frame.shape == (120, 160, 3)
    # Segundo 2 → brilho 100 (com folga para a compressão MJPG).
    assert abs(float(frame.mean()) - 100) < 5


def test_extrai_frames_seguidos(video_path):
    frames = extract_frames(video_path, at_seconds=1.0, count=3, step_seconds=1.0)

    means = [round(float(f.mean()) / 50) * 50 for f in frames]
    assert means == [50, 100, 150]


def test_instante_depois_do_fim_do_video(video_path):
    with pytest.raises(FrameError, match="fim do vídeo"):
        extract_frame(video_path, at_seconds=SECONDS + 1)


def test_video_inexistente(tmp_path):
    with pytest.raises(FrameError, match="não encontrado"):
        extract_frame(tmp_path / "nao-existe.mp4", 0)


def test_arquivo_que_nao_e_video(tmp_path):
    fake = tmp_path / "falso.mp4"
    fake.write_text("isto não é um vídeo")

    with pytest.raises(FrameError):
        extract_frame(fake, 0)


def test_jpeg_e_deterministico():
    image = np.random.default_rng(42).integers(0, 255, (64, 64, 3), dtype=np.uint8)

    # Mesmos pixels → mesmos bytes → mesmo SHA-256 (a Idempotency-Key não dá 422 no reenvio).
    assert encode_jpeg(image) == encode_jpeg(image)


def test_decode_ida_e_volta():
    image = np.full((32, 48, 3), 77, dtype=np.uint8)

    decoded = decode_image(encode_jpeg(image))

    assert decoded.shape == (32, 48, 3)


@pytest.mark.parametrize("data", [b"", b"texto qualquer", b"\xff\xd8\xff\xe0 jpeg cortado"])
def test_decode_recusa_o_que_nao_e_imagem(data):
    with pytest.raises(InvalidImageError):
        decode_image(data)


def test_limit_size_reduz_so_o_que_passa_do_limite():
    from app.vision.frames import limit_size

    big = np.zeros((3000, 4000, 3), dtype=np.uint8)
    tall = np.zeros((4000, 1000, 3), dtype=np.uint8)
    small = np.zeros((480, 640, 3), dtype=np.uint8)

    assert limit_size(big).shape == (1440, 1920, 3)
    assert limit_size(tall).shape == (1920, 480, 3)      # foto em pé: o lado maior é a altura
    assert limit_size(small) is small                    # menor que o limite: a mesma imagem, sem cópia
