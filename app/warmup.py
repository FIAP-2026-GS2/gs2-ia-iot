"""Aquecimento dos modelos na subida da API.

A primeira execução de um modelo ONNX é mais lenta que as seguintes: o ONNX Runtime prepara a
memória e escolhe como rodar cada operação na primeira vez. Medido no borrão: 11,0 s na 1ª
chamada e 5,6 s nas seguintes. Rodando uma vez numa imagem preta na subida, essa demora fica
no deploy, e não na primeira foto do cidadão (nem na frente da banca).

Falha no aquecimento nunca derruba a API: o modelo que faltar só fica sem aquecer, e a rota
que depende dele responde como já responderia (503 sem borrão; aviso sem o modelo de buraco).
"""

import logging
import time
from typing import Callable

import numpy as np

logger = logging.getLogger("uvicorn.error")

# Tamanho de uma foto de celular depois do limite de 1920 px (4:3).
WARM_UP_SHAPE = (1440, 1920, 3)


def warm_up(*models: tuple[str, Callable[[np.ndarray], object]]) -> dict[str, float | None]:
    """Roda cada modelo uma vez numa imagem preta. Devolve {nome: segundos} (None = falhou)."""
    image = np.zeros(WARM_UP_SHAPE, dtype=np.uint8)
    took: dict[str, float | None] = {}
    for name, run in models:
        started = time.perf_counter()
        try:
            run(image)
        except Exception as error:  # noqa: BLE001 — sem o modelo, a rota avisa; a subida continua
            logger.warning("Aquecimento de %s não rodou: %s", name, error)
            took[name] = None
            continue
        took[name] = round(time.perf_counter() - started, 1)
    logger.info("Aquecimento dos modelos: %s", ", ".join(f"{n} {s} s" if s is not None else f"{n} falhou"
                                                        for n, s in took.items()))
    return took
