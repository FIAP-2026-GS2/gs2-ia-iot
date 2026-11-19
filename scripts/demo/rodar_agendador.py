"""Roda o agendador no terminal, mostrando cada frame (o mesmo código que a API roda sozinha).

    python -m scripts.demo.rodar_agendador --at 2026-11-18T12:00-03:00   # uma rodada, simulando 12:00
    python -m scripts.demo.rodar_agendador --at 2026-11-19T00:00-03:00   # uma rodada, simulando 00:00
    python -m scripts.demo.rodar_agendador --loop                        # de 5 em 5 min, até Ctrl+C

Grava de verdade: frame borrado no Blob, captura e análise na .NET (pode abrir ocorrência).
Rodar de novo o mesmo horário não duplica nada: a .NET já não devolve o que foi capturado.
"""

import argparse
import sys
import time
from datetime import datetime

from app.config import get_settings
from app.pipeline import PipelineConfigError
from app.scheduler import ItemReport, build_runner, run_once


def show(line: ItemReport) -> None:
    extra = f" · imagem {line.image_id} · {line.detections} detecção(ões)" if line.image_id else ""
    print(f"  - {line.camera_code} · {line.slot_at} · {line.reason} → {line.status}{extra}", flush=True)
    for result in line.results:
        print(f"      {result}", flush=True)
    if line.message:
        print(f"      {line.message[:300]}", flush=True)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--at", help="horário simulado (ISO, ex.: 2026-11-18T12:00-03:00)")
    parser.add_argument("--loop", action="store_true", help="repete a cada SCHEDULER_INTERVAL_MINUTES")
    args = parser.parse_args()

    settings = get_settings()
    try:
        due, sources, process = build_runner(settings)
    except PipelineConfigError as error:
        print(f"Configuração: {error}")
        return 1
    print(f"Fontes de frame configuradas: {', '.join(sources) or 'nenhuma'}")
    at = datetime.fromisoformat(args.at) if args.at else None

    while True:
        print(f"\n[{datetime.now():%H:%M:%S}] Consultando a agenda {'em ' + args.at if args.at else 'agora'}...",
              flush=True)
        report = run_once(due, sources, process, at, on_item=show)
        if report.error:
            print(f"  Erro: {report.error}")
        else:
            print(f"  {report.due} frame(s) na agenda; "
                  f"{sum(1 for i in report.items if i.status == 'SENT')} enviado(s)")
        if not args.loop:
            return 1 if report.error else 0
        time.sleep(settings.scheduler_interval_minutes * 60)


if __name__ == "__main__":
    sys.exit(main())
