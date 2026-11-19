"""Agendador: de tempos em tempos pergunta à .NET quais frames extrair e processa cada um.

    a cada N minutos (APScheduler)
      → GET /api/schedules/due                   (a .NET diz: câmera X, 12:00, analisar buraco e poste)
      → pega o frame da câmera naquele instante   (FrameSource: vídeo público ou foto de teste)
      → process_frame                             (borra → analisa → Blob → captura → análise)

Por que perguntar à .NET e não decidir aqui: a agenda (12:00, 00:00, alerta de chuva) é regra de
negócio e mora no banco. A IA só executa. Se a IA cair e voltar às 12:20, o GET ainda devolve o
frame das 12:00 (tolerância de 30 min na .NET), e o que já foi enviado não volta (a .NET filtra).

Fonte do frame: não temos o stream do Smart Sampa. Cada câmera aponta, no config/cameras.json, para:
- "source": "caminho/do/video.mp4"  → um vídeo público em loop, simulando 24 h de gravação;
  o frame é o do segundo do dia (em São Paulo) do slot, contado em loop no vídeo;
- "source": {"dia": "foto.png", "noite": "foto.png"}  → uma imagem (ou vídeo) para cada período,
  usada com as fotos de teste da CET (dia para as 12:00, noite para as 00:00).

Cada caminho pode ser um arquivo local ou "blob:<nome no container>". As fotos de demonstração
ficam no Blob (já borradas, por scripts/demo/subir_fontes_demo.py): assim funcionam num clone limpo e
no container da nuvem, sem as fotos irem para o GitHub. Na 1ª leitura, o arquivo é baixado para
data/cache/fontes/ e reaproveitado nas próximas.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from app.integrations.dotnet import DueItem
from app.vision.frames import FrameError, extract_frame

logger = logging.getLogger("uvicorn.error")

SAO_PAULO = timezone(timedelta(hours=-3))  # sem horário de verão desde 2019
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class SourceError(RuntimeError):
    """A câmera não tem fonte configurada, ou o arquivo não abre."""


BLOB_SCHEME = "blob:"


@dataclass(frozen=True)
class BlobFile:
    """Um arquivo de fonte guardado no Blob, baixado para o disco só quando for usado."""

    storage: Any        # BlobStorage (Any para não importar o SDK do Azure aqui)
    name: str           # nome no container, ex.: demo/CET-CONSOLACAO/dia.jpg
    cache: Path         # onde a cópia local fica

    def local(self) -> Path:
        if not self.cache.is_file():
            data = self.storage.download(self.name)
            if data is None:
                raise SourceError(f"não está no Blob: {self.name} (rode python -m scripts.demo.subir_fontes_demo)")
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            partial = self.cache.with_suffix(self.cache.suffix + ".part")
            partial.write_bytes(data)
            partial.replace(self.cache)  # nunca fica um arquivo pela metade com o nome certo
        return self.cache

    @property
    def suffix(self) -> str:
        return Path(self.name).suffix


def is_daytime(moment: datetime) -> bool:
    return 6 <= moment.astimezone(SAO_PAULO).hour < 18


def _read(location: "Path | BlobFile", at_seconds: float) -> np.ndarray:
    path = location.local() if isinstance(location, BlobFile) else location
    if not path.is_file():
        raise SourceError(f"arquivo da fonte não encontrado: {path}")
    if path.suffix.lower() in VIDEO_SUFFIXES:
        capture = cv2.VideoCapture(str(path))
        try:
            fps, total = capture.get(cv2.CAP_PROP_FPS) or 0, capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        finally:
            capture.release()
        if fps <= 0 or total <= 0:
            raise SourceError(f"vídeo sem FPS ou duração: {path.name}")
        duration = total / fps
        try:
            # Loop: o segundo do dia "dá a volta" no vídeo. Margem de 1 s para não pedir além do fim.
            return extract_frame(path, at_seconds % max(duration - 1, 1))
        except FrameError as error:
            raise SourceError(str(error)) from error
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise SourceError(f"não é uma imagem válida: {path.name}")
    return image


@dataclass(frozen=True)
class FrameSource:
    """De onde sai o frame de uma câmera (no lugar do stream real do Smart Sampa)."""

    camera_code: str
    video: "Path | BlobFile | None" = None
    day: "Path | BlobFile | None" = None
    night: "Path | BlobFile | None" = None

    def frame_at(self, slot_at: datetime) -> np.ndarray:
        local = slot_at.astimezone(SAO_PAULO)
        second_of_day = local.hour * 3600 + local.minute * 60 + local.second
        if self.video is not None:
            return _read(self.video, second_of_day)
        path = self.day if is_daytime(slot_at) else self.night
        if path is None:
            raise SourceError(f"{self.camera_code} não tem fonte de {'dia' if is_daytime(slot_at) else 'noite'}")
        return _read(path, second_of_day)


def load_sources(config: dict[str, Any], base: Path, blob: Any = None,
                 cache_dir: Path = Path("data/cache/fontes")) -> dict[str, FrameSource]:
    """Lê o campo "source" de cada câmera do config/cameras.json (câmera sem "source" fica de fora)."""

    def location(value: str | None) -> "Path | BlobFile | None":
        if not value:
            return None
        if value.startswith(BLOB_SCHEME):
            name = value[len(BLOB_SCHEME):]
            if blob is None:
                raise SourceError(f"{value} pede o Blob, mas BLOB_CONNECTION_STRING não está configurada")
            return BlobFile(blob, name, cache_dir / name)
        return base / value

    sources = {}
    for code, data in (config.get("cameras") or {}).items():
        source = data.get("source")
        if isinstance(source, str):
            sources[code] = FrameSource(code, video=location(source))
        elif isinstance(source, dict):
            sources[code] = FrameSource(code, day=location(source.get("dia")), night=location(source.get("noite")))
    return sources


# ---------- uma rodada ----------

@dataclass
class ItemReport:
    camera_code: str
    slot_at: str
    reason: str
    status: str          # SENT, ALREADY_SENT, ANONYMIZATION_FAILED, ANALYSIS_FAILED, NO_SOURCE, ERROR
    image_id: int | None = None
    detections: int = 0
    message: str = ""
    # O que a .NET fez com cada detecção: OPENED_NEW_ISSUE, LINKED_TO_EXISTING_ISSUE ou BELOW_CONFIDENCE_THRESHOLD
    results: list[str] = field(default_factory=list)


@dataclass
class RunReport:
    started_at: str
    finished_at: str = ""
    at: str | None = None        # horário simulado, se houver
    due: int = 0
    items: list[ItemReport] = field(default_factory=list)
    error: str = ""              # falha antes dos itens (login, rede)


def run_once(
    due: Callable[[datetime | None], list[DueItem]],
    sources: dict[str, FrameSource],
    process: Callable[[DueItem, np.ndarray], Any],
    at: datetime | None = None,
    on_item: Callable[[ItemReport], None] | None = None,
) -> RunReport:
    """Uma rodada do agendador. Um item com erro não derruba os outros."""
    report = RunReport(started_at=datetime.now(timezone.utc).isoformat(), at=at.isoformat() if at else None)
    try:
        items = due(at)
    except Exception as error:  # noqa: BLE001 — .NET fora do ar: registra e tenta na próxima rodada
        report.error = f"{type(error).__name__}: {error}"
        report.finished_at = datetime.now(timezone.utc).isoformat()
        logger.error("Agendador: não consegui ler a agenda (%s)", report.error)
        return report
    report.due = len(items)
    for item in items:
        line = ItemReport(item.camera_code, item.slot_at.isoformat(), item.reason, "ERROR")
        source = sources.get(item.camera_code)
        try:
            if source is None:
                raise SourceError(f"câmera {item.camera_code} sem \"source\" no config/cameras.json")
            frame = source.frame_at(item.slot_at)
        except SourceError as error:
            line.status, line.message = "NO_SOURCE", str(error)
        else:
            try:
                outcome = process(item, frame)
                line.status, line.image_id, line.message = outcome.status, outcome.image_id, outcome.message
                line.detections = len(outcome.detections or [])
                line.results = [f"tipo {d.get('issueTypeId')} · confiança {d.get('confidenceScore')} · "
                                f"{d.get('outcome')} · ocorrência {d.get('issueReportId')}"
                                for d in outcome.detections or []]
            except Exception as error:  # noqa: BLE001 — Blob/.NET falhou neste item; os outros seguem
                logger.exception("Agendador: falha em %s %s", item.camera_code, item.slot_at.isoformat())
                line.message = f"{type(error).__name__}: {error}"
        report.items.append(line)
        logger.info("Agendador: %s %s → %s %s", line.camera_code, line.slot_at, line.status, line.message)
        if on_item:
            on_item(line)
    report.finished_at = datetime.now(timezone.utc).isoformat()
    return report


# ---------- montagem com as dependências reais ----------

def build_runner(settings) -> tuple[Callable, dict[str, FrameSource], Callable]:
    """Cria (due, sources, process) a partir do .env. Erro de configuração aparece aqui, na subida."""
    import json

    from app.api.dependencies import _pothole_detector, get_anonymizer, llm_dependency
    from app.integrations.blob import BlobReferenceStore
    from app.pipeline import build_clients, issue_type_ids, process_frame
    from app.vision.pothole import PotholeModelError
    from app.vision.streetlight_roi import load_cameras

    dotnet, blob = build_clients(settings)
    type_ids = issue_type_ids(settings)
    config_path = Path(settings.cameras_config_path)
    sources = load_sources(json.loads(config_path.read_text(encoding="utf-8")), Path.cwd(), blob)
    reference_store = BlobReferenceStore(blob)
    llm = llm_dependency(settings)

    def process(item: DueItem, frame: np.ndarray):
        try:
            detector = _pothole_detector(settings.pothole_model_path)
        except PotholeModelError:
            detector = None
        cameras = load_cameras(settings.cameras_config_path)  # relido: editar o JSON vale na próxima rodada
        return process_frame(item, frame, anonymizer=get_anonymizer(), dotnet=dotnet, blob=blob,
                             type_ids=type_ids, camera=cameras.get(item.camera_code),
                             pothole_detector=detector, reference_store=reference_store, llm=llm)

    return dotnet.due, sources, process


class FrameScheduler:
    """O APScheduler em volta do run_once, ligado na subida da API (SCHEDULER_ENABLED=true)."""

    def __init__(self, settings):
        self.settings = settings
        self.last: RunReport | None = None
        self._lock = threading.Lock()  # nunca duas rodadas ao mesmo tempo
        self._scheduler = None
        self._runner = None  # (due, sources, process): criado na 1ª rodada e reaproveitado (token, Blob)

    def tick(self) -> RunReport | None:
        if not self._lock.acquire(blocking=False):
            logger.warning("Agendador: rodada anterior ainda em andamento, pulando esta")
            return None
        try:
            if self._runner is None:
                self._runner = build_runner(self.settings)
            due, sources, process = self._runner
            self.last = run_once(due, sources, process)
            return self.last
        except Exception as error:  # noqa: BLE001 — configuração faltando: registra, não derruba a API
            logger.error("Agendador: %s", error)
            self.last = RunReport(started_at=datetime.now(timezone.utc).isoformat(),
                                  finished_at=datetime.now(timezone.utc).isoformat(), error=str(error))
            return self.last
        finally:
            self._lock.release()

    def start(self) -> None:
        from apscheduler.schedulers.background import BackgroundScheduler

        self._scheduler = BackgroundScheduler(timezone="America/Sao_Paulo")
        # max_instances=1 + coalesce: se uma rodada atrasar, as que acumularam viram uma só.
        self._scheduler.add_job(self.tick, "interval", minutes=self.settings.scheduler_interval_minutes,
                                next_run_time=datetime.now(timezone.utc), max_instances=1, coalesce=True,
                                id="frames")
        self._scheduler.start()
        logger.info("Agendador ligado: consulta a agenda a cada %d min", self.settings.scheduler_interval_minutes)

    def stop(self) -> None:
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
