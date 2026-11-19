"""Baixa o modelo de buraco (.pt do PyTorch), confere o SHA-256 e converte para ONNX.

    python scripts/modelos/exportar_buraco.py models/pothole/smartathon-yolov8s.onnx

Por que converter e não baixar o ONNX pronto: o autor (Samdutse/pothole-yolov8, Hugging Face)
só publica o .pt. O serviço roda com ONNX Runtime, sem PyTorch (imagem ~1,5 GB menor).
Então a conversão acontece num estágio separado do Dockerfile, que tem PyTorch e Ultralytics
(requirements-export.txt), e só o .onnx pronto passa para a imagem final.

O lacre é o SHA-256 do .pt: se o arquivo na origem mudar, o build para. O .onnx gerado
não tem hash fixo (a conversão pode mudar bytes de metadados), por isso o script confere
o resultado de outro jeito: abre no ONNX Runtime e verifica a saída (1, 5, 8400).
"""

import hashlib
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/Samdutse/pothole-yolov8/resolve/main/best.pt"
SHA256 = "1067a09d37e4e9081164e6d6da337ad66d8ec7a884a6a194e58703beee804aa2"
EXPECTED_OUTPUT = [1, 5, 8400]  # 4 coordenadas + 1 classe (buraco), 8400 caixas candidatas


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    target = Path(sys.argv[1]).resolve()

    with tempfile.TemporaryDirectory() as work:
        weights = Path(work) / "best.pt"
        print(f"Baixando {URL}...")
        urllib.request.urlretrieve(URL, weights)  # noqa: S310 (URL fixa, HTTPS)
        actual = sha256_of(weights)
        if actual != SHA256:
            print(f"ERRO: SHA-256 do .pt não confere (esperado {SHA256}, veio {actual}).")
            return 1
        print("SHA-256 do .pt conferido.")

        from ultralytics import YOLO  # só existe no estágio de conversão

        # Mesmos parâmetros usados na avaliação (F1 78%): entrada fixa 640x640, opset 12.
        exported = Path(YOLO(str(weights)).export(format="onnx", imgsz=640, opset=12,
                                                  simplify=True, dynamic=False))

        import onnxruntime as ort

        shape = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"]).get_outputs()[0].shape
        if list(shape) != EXPECTED_OUTPUT:
            print(f"ERRO: saída do ONNX {shape}, esperado {EXPECTED_OUTPUT}.")
            return 1

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported), target)
    print(f"OK: {target.name} ({target.stat().st_size / 1e6:.0f} MB, saída {EXPECTED_OUTPUT})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
