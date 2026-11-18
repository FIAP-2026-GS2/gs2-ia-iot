"""GET /agendador: o que o agendador fez na última rodada (para acompanhar na demonstração)."""

from dataclasses import asdict

from fastapi import APIRouter, Depends, Request

from app.config import Settings, get_settings
from app.security import require_api_key

router = APIRouter(tags=["Agendador"], dependencies=[Depends(require_api_key)])


@router.get("/agendador")
def scheduler_status(request: Request, settings: Settings = Depends(get_settings)) -> dict:
    """Se o agendador está ligado, de quanto em quanto tempo roda e o resultado da última rodada."""
    scheduler = getattr(request.app.state, "scheduler", None)
    last = scheduler.last if scheduler else None
    return {
        "enabled": scheduler is not None,
        "interval_minutes": settings.scheduler_interval_minutes,
        "last_run": asdict(last) if last else None,
    }
