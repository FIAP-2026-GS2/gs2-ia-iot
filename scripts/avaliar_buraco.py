"""Compara os modelos de buraco num conjunto de teste que nenhum deles viu no treino.

    python -m scripts.avaliar_buraco                       # avalia os modelos da lista MODELOS
    python -m scripts.avaliar_buraco --modelo outro.onnx   # avalia outro arquivo
    python -m scripts.avaliar_buraco --verificar           # com o verificador LLM (usa o .env; ~35 min, com pausa de 15 s)

Conjunto de teste (fora do Git, em data/eval/pothole/):
- images/ + labels/: 100 fotos com buracos marcados (formato YOLO), do dataset público
  "Potholes Detection" (Roboflow, CC BY 4.0), espelhado em huggingface.co/datasets/Ryukijano/Pothole-detection-Yolov8
- negativos: fotos de rua SEM buraco em data/eval/sem-buraco/ (qualquer nome), para medir alarme falso

Métricas:
- por caixa: uma detecção conta como acerto (TP) se cobrir um buraco marcado com IoU ≥ 0,5
- precisão = TP / (TP + FP): de tudo o que o modelo apontou, quanto era buraco
- recall = TP / (TP + FN): de todos os buracos, quantos ele achou
- F1 = média harmônica das duas (só fica alto se as duas forem altas)
- por imagem (matriz de confusão): "tem buraco?" sim/não, que é o que abre ou não uma ocorrência

Com --verificar, cada caixa do YOLO passa pelo LLM (o mesmo de app/analysis.py), e só contam as
confirmadas. As fotos vão ao LLM sem borrão: são do dataset público e dos prints de rua sem pessoas próximas.
"""

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.analysis import AnalysisResult, _verify
from app.config import get_settings
from app.llm.client import LlmClient, providers_from_settings
from app.vision.pothole import MIN_CONFIDENCE, REPORT_CONFIDENCE, PotholeDetector

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval" / "pothole"
# Fotos de rua SEM buraco (qualquer nome, .jpg ou .png): medem o alarme falso.
NEGATIVES_DIR = ROOT / "data" / "eval" / "sem-buraco"
NEGATIVES = sorted(p for p in NEGATIVES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
MODELOS = {
    "A · RDD (Japão/Índia, 5 classes)": ROOT / "models" / "pothole" / "rdd-yolov8s.onnx",
    "B · Smartathon (Arábia Saudita)": ROOT / "models" / "pothole" / "smartathon-yolov8s.onnx",
}
IOU_MATCH = 0.5
PAUSE_SECONDS = 15  # entre chamadas ao verificador
RETRY_SECONDS = 45  # sem resposta (429/timeout): espera e tenta uma vez de novo


def load(path: Path) -> np.ndarray:
    # np.fromfile + imdecode: funciona com acento no caminho do Windows (cv2.imread não).
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Não consegui ler {path}")
    return image


def ground_truth(label_path: Path, width: int, height: int) -> list[tuple[float, float, float, float]]:
    boxes = []
    if label_path.is_file():
        for line in label_path.read_text().split("\n"):
            parts = line.split()
            if len(parts) != 5:
                continue
            cx, cy, w, h = (float(value) for value in parts[1:])
            boxes.append(((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height))
    return boxes


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    # matriz por imagem: [real][previsto]
    image: dict = field(default_factory=lambda: {"sim": {"sim": 0, "nao": 0}, "nao": {"sim": 0, "nao": 0}})

    def prf(self) -> tuple[float, float, float]:
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1


def evaluate(detector: PotholeDetector, positives: list[Path], thresholds: list[float],
             llm: LlmClient | None = None) -> tuple[dict, float]:
    counts = {threshold: Counts() for threshold in thresholds}
    verdicts: Counter = Counter()
    timings = []
    samples = [(path, True) for path in positives] + [(path, False) for path in NEGATIVES if path.is_file()]
    for index, (path, has_label) in enumerate(samples, start=1):
        image = load(path)
        height, width = image.shape[:2]
        truth = ground_truth(EVAL_DIR / "labels" / f"{path.stem}.txt", width, height) if has_label else []
        started = time.perf_counter()
        found = detector.detect(image, min_confidence=min(thresholds))
        timings.append((time.perf_counter() - started) * 1000)
        if llm is not None and found:
            check = AnalysisResult()
            _verify(image, found, llm, check)
            if found and all(f.verified is None for f in check.findings) and not check.discarded:
                verdicts["nova tentativa"] += 1  # o LLM não respondeu: limite do plano grátis, não erro do modelo
                time.sleep(RETRY_SECONDS)
                check = AnalysisResult()
                _verify(image, found, llm, check)
            confirmed = {f.bbox for f in check.findings if f.verified}
            verdicts.update(d.reason.split(":")[0] for d in check.discarded)
            verdicts["sem resposta"] += sum(1 for f in check.findings if f.verified is None)
            verdicts["BURACO (confirmado)"] += len(confirmed)
            found = [b for b in found if (b.x1, b.y1, b.x2, b.y2) in confirmed]
            print(f"   [{index:3d}/{len(samples)}] {path.name[:40]:40s} {'com buraco' if has_label else 'sem buraco'}"
                  f" | YOLO {len(check.findings) + len(check.discarded)} caixa(s) → confirmadas {len(confirmed)}"
                  f" | descartadas {[d.reason.split(':')[0] for d in check.discarded]}", flush=True)
            time.sleep(PAUSE_SECONDS)  # o Groq gratuito limita chamadas por minuto (HTTP 429)
        elif llm is not None:
            print(f"   [{index:3d}/{len(samples)}] {path.name[:40]:40s} {'com buraco' if has_label else 'sem buraco'}"
                  " | YOLO 0 caixa(s), sem chamada ao LLM", flush=True)
        for threshold, count in counts.items():
            predicted = [(b.x1, b.y1, b.x2, b.y2) for b in found if b.confidence >= threshold]  # já em ordem de confiança
            matched = set()
            for box in predicted:
                best, best_iou = None, IOU_MATCH
                for index, real in enumerate(truth):
                    value = iou(box, real)
                    if index not in matched and value >= best_iou:
                        best, best_iou = index, value
                if best is None:
                    count.fp += 1
                else:
                    matched.add(best)
                    count.tp += 1
            count.fn += len(truth) - len(matched)
            count.image["sim" if truth else "nao"]["sim" if predicted else "nao"] += 1
    if verdicts:
        print("   Verificador:", dict(verdicts.most_common()))
    return counts, float(np.median(timings))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelo", action="append", help="caminho de um .onnx (pode repetir)")
    parser.add_argument("--verificar", action="store_true", help="passa cada caixa pelo verificador LLM")
    args = parser.parse_args()

    positives = sorted((EVAL_DIR / "images").glob("*.jpg"))
    if not positives:
        print(f"Conjunto de teste não encontrado em {EVAL_DIR}. Veja o docstring deste script.")
        return 1
    models = {Path(m).stem: Path(m) for m in args.modelo} if args.modelo else MODELOS
    negatives = [path for path in NEGATIVES if path.is_file()]
    print(f"Teste: {len(positives)} fotos com buraco + {len(negatives)} fotos de rua sem buraco\n")

    thresholds = [MIN_CONFIDENCE, REPORT_CONFIDENCE]
    llm = None
    if args.verificar:
        providers = providers_from_settings(get_settings())
        if not providers:
            print("--verificar precisa do LLM configurado no .env.")
            return 1
        llm = LlmClient(providers, 60)
    for label, path in models.items():
        if not path.is_file():
            print(f"{label}: arquivo não encontrado ({path.name}), pulando\n")
            continue
        detector = PotholeDetector(path)
        counts, median_ms = evaluate(detector, positives, thresholds, llm)
        print(f"== {label}  ({path.name}, mediana {median_ms:.0f} ms por imagem)")
        print("   limiar | TP  FP  FN | precisão recall  F1")
        for threshold, count in counts.items():
            p, r, f1 = count.prf()
            print(f"   {threshold:.2f}   | {count.tp:3d} {count.fp:3d} {count.fn:3d} |  {p:6.1%}  {r:6.1%} {f1:6.1%}")
        matrix = counts[REPORT_CONFIDENCE].image
        print(f"   Matriz por imagem (limiar {REPORT_CONFIDENCE:.2f}, o que abre ocorrência):")
        print("                     previsto: buraco | sem buraco")
        print(f"     real buraco      {matrix['sim']['sim']:15d} | {matrix['sim']['nao']:10d}")
        print(f"     real sem buraco  {matrix['nao']['sim']:15d} | {matrix['nao']['nao']:10d}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
