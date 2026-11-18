"""Orquestra a análise de um frame (já borrado), na ordem dos cenários pedidos pela agenda da .NET.

Os nomes das análises são os mesmos do GET /api/schedules/due:
- POTHOLE_CROSSWALK (agenda das 12:00): buraco (YOLO → zonas da câmera → verificador LLM)
  + faixa apagada (modelo próprio, próxima fase)
- STREETLIGHT (12:00 e 00:00): ao meio-dia guarda a referência; à meia-noite compara
- FLOODING (alerta de chuva): LLM de visão decide se a via está alagada

Depois dos modelos, UMA chamada ao LLM descreve todas as detecções (e julga o alagamento,
se pedido). Se o LLM falhar, as descrições saem de um modelo de texto fixo (template) e a
resposta diz o porquê em motivo_sem_llm. O alagamento não tem plano B: sem LLM, não há detecção.

Papéis do LLM: no buraco, é a segunda opinião (só descarta ou confirma o que o YOLO achou,
nunca cria caixa); no alagamento, decide (zero-shot, sem modelo treinado); em todas, descreve.
Buraco sem verificação (LLM fora do ar) fica com confiança no máximo 0,49: aparece, mas não abre
ocorrência na .NET. Na dúvida, não confirma.
"""

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from app.llm.client import LlmClient, LlmUnavailableError
from app.llm.prompts import POTHOLE_CHECK_SYSTEM, SCENE_SYSTEM, pothole_check_prompt, scene_prompt
from app.vision.annotate import Annotation, annotate
from app.vision.frames import encode_jpeg
from app.vision.pothole import PotholeBox, PotholeDetector
from app.vision.pothole_check import MAX_TILES, Discarded, filter_by_zones, mosaic
from app.vision.streetlight import shrink
from app.vision.streetlight_roi import Camera, LampResult, LocalReferenceStore, check_night, measure_camera

logger = logging.getLogger("uvicorn.error")

AnalysisType = Literal["POTHOLE_CROSSWALK", "STREETLIGHT", "FLOODING"]
Period = Literal["MEIO_DIA", "MEIA_NOITE"]
Category = Literal["BURACO", "FAIXA_APAGADA", "POSTE_APAGADO", "VIA_ALAGADA"]
Severity = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
REPORT_CONFIDENCE = 0.50  # mesmo mínimo da .NET (Detection__MinConfidence)
UNVERIFIED_MAX = 0.49     # buraco sem a segunda opinião do LLM não abre ocorrência
CHECK_MIN_CONFIDENCE = 0.6  # o LLM precisa dizer BURACO com pelo menos 0,6


class AnalysisInputError(ValueError):
    """Faltou um dado para a análise pedida (ex.: STREETLIGHT sem câmera ou sem período)."""


@dataclass
class Finding:
    category: Category
    confidence: float
    severity: Severity
    bbox: tuple[int, int, int, int]
    source: str  # quem detectou: yolo, roi ou llm
    description: str = ""
    citizen_message: str = ""
    model_confidence: float | None = None  # confiança original do YOLO (antes da verificação)
    verified: bool | None = None           # buraco: o LLM confirmou? (None = não se aplica / não verificado)
    check_reason: str = ""                 # o que o verificador disse

    @property
    def reportable(self) -> bool:
        return self.confidence >= REPORT_CONFIDENCE


@dataclass
class LlmUse:
    purpose: str  # verificar_buraco ou descrever
    provider: str
    model: str
    latency_ms: int


@dataclass
class AnalysisResult:
    findings: list[Finding] = field(default_factory=list)
    lamps: list[LampResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    discarded: list[Discarded] = field(default_factory=list)
    llm_calls: list[LlmUse] = field(default_factory=list)
    motivo_sem_llm: str | None = None

    def no_llm(self, reason: str) -> None:
        # Guarda só o primeiro motivo: se o LLM caiu no verificador, cai igual na descrição.
        self.motivo_sem_llm = self.motivo_sem_llm or reason


# ---------- Formato exigido do LLM ----------

class LlmFlooding(BaseModel):
    alagada: bool
    gravidade: Severity
    confianca: float = Field(ge=0, le=1)
    descricao: str = Field(min_length=1, max_length=500)
    mensagem_cidadao: str = Field(min_length=1, max_length=300)


class LlmDescription(BaseModel):
    id: int
    descricao: str = Field(min_length=1, max_length=500)
    mensagem_cidadao: str = Field(min_length=1, max_length=300)


class LlmCandidate(BaseModel):
    id: int
    objeto: Literal["BURACO", "SOMBRA", "BUEIRO_OU_TAMPA", "REMENDO_PLANO", "POCA_RASA", "MANCHA",
                    "SINALIZACAO", "VEICULO_OU_PESSOA", "OUTRO"]
    confianca: float = Field(ge=0, le=1)
    motivo: str = Field(min_length=1, max_length=300)


class LlmPotholeCheck(BaseModel):
    candidatos: list[LlmCandidate]


class LlmSceneAnswer(BaseModel):
    alagamento: LlmFlooding | None = None
    deteccoes: list[LlmDescription] = Field(default_factory=list)


# ---------- Textos de reserva (sem LLM) ----------

CITIZEN_TEMPLATES = {
    "BURACO": "Identificamos um possível buraco nesta via. A prefeitura foi avisada para avaliar o reparo.",
    "FAIXA_APAGADA": "Identificamos uma faixa de pedestres desgastada. A prefeitura foi avisada.",
    "POSTE_APAGADO": "Identificamos um poste apagado neste trecho. A prefeitura foi avisada.",
    "VIA_ALAGADA": "Há um ponto de alagamento nesta via. Evite passar por ela até a água baixar.",
}


def template_description(finding: Finding, image_area: int) -> str:
    if finding.category == "BURACO":
        x1, y1, x2, y2 = finding.bbox
        share = (x2 - x1) * (y2 - y1) / image_area
        return (f"Possível buraco ocupando {share:.1%} da imagem (gravidade {finding.severity}), "
                f"detectado pelo modelo YOLO com confiança {finding.model_confidence or finding.confidence:.0%}.")
    return f"{finding.category} detectado com confiança {finding.confidence:.0%}."


# ---------- Etapas ----------

def _ask(llm: LlmClient | None, purpose: str, system: str, prompt: str, picture: np.ndarray,
         schema: type[BaseModel], result: AnalysisResult) -> BaseModel | None:
    """Uma chamada ao LLM com resposta validada. Falhou? Registra o motivo e devolve None."""
    if llm is None:
        result.no_llm("LLM não configurado (variáveis LLM_*)")
        return None
    try:
        answer = llm.ask_json(system, prompt, encode_jpeg(shrink(picture)))
    except LlmUnavailableError as error:
        result.no_llm(f"LLM indisponível ({error})")
        return None
    try:
        parsed = schema.model_validate(answer.data)
    except ValidationError:
        result.no_llm(f"LLM ({answer.provider}) respondeu fora do formato combinado")
        return None
    result.llm_calls.append(LlmUse(purpose, answer.provider, answer.model, answer.latency_ms))
    return parsed


def _unverified(box: PotholeBox, why: str) -> Finding:
    return Finding("BURACO", min(box.confidence, UNVERIFIED_MAX), box.severity, (box.x1, box.y1, box.x2, box.y2),
                   "yolo", model_confidence=box.confidence, verified=None, check_reason=why)


def _verify(image: np.ndarray, boxes: list[PotholeBox], llm: LlmClient | None, result: AnalysisResult) -> None:
    checked, extra = boxes[:MAX_TILES], boxes[MAX_TILES:]
    answer = _ask(llm, "verificar_buraco", POTHOLE_CHECK_SYSTEM, pothole_check_prompt(len(checked)),
                  mosaic(image, checked), LlmPotholeCheck, result)
    verdicts = {item.id: item for item in answer.candidatos} if answer else {}
    for number, box in enumerate(checked, start=1):
        verdict = verdicts.get(number)
        if verdict is None:
            why = "sem verificação do LLM" if answer is None else "o LLM não respondeu sobre esta caixa"
            result.findings.append(_unverified(box, why))
        elif verdict.objeto == "BURACO" and verdict.confianca >= CHECK_MIN_CONFIDENCE:
            result.findings.append(Finding("BURACO", box.confidence, box.severity, (box.x1, box.y1, box.x2, box.y2),
                                           "yolo", model_confidence=box.confidence, verified=True,
                                           check_reason=verdict.motivo))
        else:
            result.discarded.append(Discarded(box, f"{verdict.objeto}: {verdict.motivo}", "llm"))
    for box in extra:
        result.findings.append(_unverified(box, f"passou do limite de {MAX_TILES} verificações por frame"))
    if answer is None and boxes:
        result.notes.append("Buraco: sem a segunda opinião do LLM; as caixas ficam abaixo do limiar (máx. 0,49).")


def _potholes(image: np.ndarray, period: Period | None, detector: PotholeDetector | None,
              camera: Camera | None, llm: LlmClient | None, result: AnalysisResult) -> None:
    if period == "MEIA_NOITE":
        result.notes.append("Buraco: não analisado à meia-noite (o modelo foi treinado com fotos de dia).")
        return
    if detector is None:
        result.notes.append("Buraco: modelo indisponível (rode python -m scripts.baixar_modelos).")
        return
    result.models.append(f"pothole:{detector.name}")
    height, width = image.shape[:2]
    kept, dropped = filter_by_zones(detector.detect(image), camera, width, height)
    result.discarded.extend(dropped)
    if kept:
        _verify(image, kept, llm, result)
    result.notes.append("Faixa apagada: o modelo próprio ainda está em treinamento (próxima fase).")


def _streetlight(image: np.ndarray, period: Period | None, camera: Camera | None,
                 store: LocalReferenceStore, captured_at: str, result: AnalysisResult) -> None:
    if camera is None:
        raise AnalysisInputError("STREETLIGHT precisa da câmera (camera), para saber onde estão os postes.")
    if period is None:
        raise AnalysisInputError("STREETLIGHT precisa do período: MEIO_DIA (guarda a referência) ou MEIA_NOITE (compara).")
    if not camera.lamps:
        result.notes.append(f"Poste: a câmera {camera.code} não tem luminárias marcadas em config/cameras.json.")
        return
    result.models.append("streetlight:roi-brilho")
    if period == "MEIO_DIA":
        store.save(camera.code, measure_camera(image, camera), captured_at)
        result.notes.append(f"Poste: referência do meio-dia guardada para {len(camera.lamps)} luminária(s) de {camera.code}.")
        return

    saved = store.load(camera.code)
    reference, reference_at = saved if saved else (None, None)
    if reference is None:
        result.notes.append("Poste: sem referência do meio-dia; comparado só com o entorno (confiança × 0,8).")
    else:
        result.notes.append(f"Poste: comparado com a referência do meio-dia de {reference_at}.")
    result.lamps = check_night(image, camera, reference)
    for lamp in result.lamps:
        if lamp.status == "OFF":
            result.findings.append(Finding("POSTE_APAGADO", lamp.confidence, "MEDIUM", lamp.roi, "roi",
                                           description=lamp.describe()))


def _describe(image: np.ndarray, ask_flooding: bool, llm: LlmClient | None,
              result: AnalysisResult) -> LlmSceneAnswer | None:
    listing = [{"id": i + 1, "categoria": f.category, "gravidade": f.severity, "confianca": f.confidence}
               for i, f in enumerate(result.findings)]
    # O LLM vê os retângulos numerados: assim "id 2" na resposta é o retângulo 2 da imagem.
    marked = annotate(image, [Annotation(f.category, f.confidence, *f.bbox, tag=str(i + 1))
                              for i, f in enumerate(result.findings)])
    parsed = _ask(llm, "descrever", SCENE_SYSTEM, scene_prompt(listing, ask_flooding), marked, LlmSceneAnswer, result)
    if parsed is not None and ask_flooding and parsed.alagamento is None:
        result.no_llm(f"LLM ({result.llm_calls[-1].provider}) não respondeu sobre o alagamento")
        return None
    return parsed


def run_analysis(
    image: np.ndarray,
    analyses: set[AnalysisType],
    *,
    period: Period | None = None,
    camera: Camera | None = None,
    pothole_detector: PotholeDetector | None = None,
    reference_store: LocalReferenceStore,
    llm: LlmClient | None = None,
    captured_at: str = "",
) -> AnalysisResult:
    """`image` precisa estar anonimizada: ela pode ir para o LLM."""
    result = AnalysisResult()
    if "POTHOLE_CROSSWALK" in analyses:
        _potholes(image, period, pothole_detector, camera, llm, result)
    if "STREETLIGHT" in analyses:
        _streetlight(image, period, camera, reference_store, captured_at, result)

    ask_flooding = "FLOODING" in analyses
    # Só chama o LLM se houver o que descrever ou um alagamento para julgar.
    answer = _describe(image, ask_flooding, llm, result) if (result.findings or ask_flooding) else None
    if result.motivo_sem_llm:
        logger.warning("analisar sem LLM: %s", result.motivo_sem_llm)

    height, width = image.shape[:2]
    descriptions = {item.id: item for item in answer.deteccoes} if answer else {}
    for index, finding in enumerate(result.findings, start=1):
        described = descriptions.get(index)
        if described:
            finding.description = described.descricao
            finding.citizen_message = described.mensagem_cidadao
        else:
            finding.description = finding.description or template_description(finding, width * height)
            if finding.category == "BURACO" and finding.verified is None:
                finding.description += f" Não verificado ({finding.check_reason})."
            finding.citizen_message = CITIZEN_TEMPLATES[finding.category]

    if ask_flooding:
        if answer and answer.alagamento:
            flood = answer.alagamento
            result.models.append(f"flooding:llm:{result.llm_calls[-1].model}")
            if flood.alagada:
                result.findings.append(Finding("VIA_ALAGADA", round(flood.confianca, 2), flood.gravidade,
                                               (0, 0, width - 1, height - 1), "llm",
                                               flood.descricao, flood.mensagem_cidadao))
            else:
                result.notes.append(f"Alagamento: não ({flood.descricao})")
        else:
            result.notes.append("Alagamento: não avaliado, porque depende do LLM.")
    return result
