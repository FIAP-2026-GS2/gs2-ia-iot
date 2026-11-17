"""Confere a integração real com a API .NET e o Blob, usando o .env.

    python -m scripts.testar_dotnet                          # login + agenda (só leitura)
    python -m scripts.testar_dotnet --at 2026-11-18T12:00-03:00   # simula outro horário
    python -m scripts.testar_dotnet --enviar foto.png        # envia 1 frame do 1º item da agenda

Sem --enviar, nada é gravado: só login e GET /api/schedules/due.
Com --enviar, a foto é borrada, sobe ao Blob e vira captura + análise na .NET (grava no banco).
Nunca imprime senha, token nem connection string.
"""

import argparse
import sys
from datetime import datetime

import cv2
import numpy as np

from app.api.dependencies import _pothole_detector, get_anonymizer, llm_dependency
from app.config import get_settings
from app.integrations.blob import BlobReferenceStore
from app.integrations.dotnet import DotnetApiError
from app.pipeline import PipelineConfigError, build_clients, issue_type_ids, process_frame
from app.vision.pothole import PotholeModelError
from app.vision.streetlight_roi import load_cameras


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--at", help="horário simulado (ISO, ex.: 2026-11-18T12:00-03:00)")
    parser.add_argument("--enviar", metavar="IMAGEM", help="envia esta foto como o frame do 1º item da agenda")
    args = parser.parse_args()

    settings = get_settings()
    try:
        dotnet, blob = build_clients(settings)
        type_ids = issue_type_ids(settings) if args.enviar else {}
    except PipelineConfigError as error:
        print(f"Configuração: {error}")
        return 1

    at = datetime.fromisoformat(args.at) if args.at else None
    try:
        items = dotnet.due(at)
    except DotnetApiError as error:
        print(error)
        return 1
    print(f"Login ok. Agenda {'em ' + args.at if args.at else 'agora'}: {len(items)} frame(s)")
    for item in items:
        print(f"  - câmera {item.camera_code} (id {item.camera_id}) · {item.slot_at.isoformat()} · "
              f"{item.reason} · {', '.join(item.analyses)}")
    if not args.enviar:
        return 0
    if not items:
        print("Nada na agenda para enviar. Use --at com um horário de agenda (12:00 ou 00:00).")
        return 1

    item = items[0]
    frame = cv2.imdecode(np.fromfile(args.enviar, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        print(f"Não consegui ler {args.enviar}")
        return 1
    cameras = load_cameras(settings.cameras_config_path)
    try:
        detector = _pothole_detector(settings.pothole_model_path)
    except PotholeModelError:
        detector = None
    outcome = process_frame(
        item, frame, anonymizer=get_anonymizer(), dotnet=dotnet, blob=blob, type_ids=type_ids,
        camera=cameras.get(item.camera_code), pothole_detector=detector,
        reference_store=BlobReferenceStore(blob), llm=llm_dependency(settings),
    )
    print(f"\nResultado: {outcome.status} (imagem {outcome.image_id})")
    for detection in outcome.detections or []:
        print(f"  - tipo {detection.get('issueTypeId')} · confiança {detection.get('confidenceScore')} · "
              f"{detection.get('outcome')} · ocorrência {detection.get('issueReportId')}")
    if outcome.message:
        print(f"  {outcome.message}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
