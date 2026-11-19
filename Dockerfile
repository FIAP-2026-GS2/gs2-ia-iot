# Imagem do serviço de IA do SampaTech (FastAPI + ONNX Runtime, sem PyTorch).
# Build local:  docker build -t sampatech-ia .
# Rodar local:  docker run --rm -p 8000:8000 --env-file .env sampatech-ia
#
# Nenhum segredo entra na imagem: a chave da API e as do LLM chegam como variáveis de
# ambiente na hora de rodar (no ACI, como "secure environment variables").

# ---------- Estágio 1: converte o modelo de buraco (.pt → .onnx) ----------
# Tem PyTorch e Ultralytics só aqui. A imagem final recebe apenas o .onnx pronto.
FROM python:3.13-slim AS modelo-buraco

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 YOLO_OFFLINE=1
# O OpenCV que o Ultralytics instala precisa destas bibliotecas de sistema.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /export
COPY requirements-export.txt .
RUN pip install -r requirements-export.txt
COPY scripts/modelos/exportar_buraco.py .
# Confere o SHA-256 do .pt e a saída do .onnx; se algo não bater, o build para.
RUN python exportar_buraco.py /export/smartathon-yolov8s.onnx

# ---------- Estágio 2: a imagem que roda no ACI ----------
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_ENV=cloud \
    # Sem arquivo .env no container: a configuração vem só das variáveis de ambiente.
    IA_ENV_FILE=

WORKDIR /app

# Primeiro só as dependências: essa camada fica em cache enquanto o requirements.txt não muda.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ app/
COPY scripts/ scripts/
COPY models/ models/
COPY config/ config/

# O detector de pessoas (114 MB) não está no Git: baixa aqui, conferindo o SHA-256.
# Se o hash não bater, o build falha (melhor do que subir sem o borrão completo).
RUN python -m scripts.modelos.baixar_modelos

# O modelo de buraco já convertido no estágio 1.
COPY --from=modelo-buraco /export/smartathon-yolov8s.onnx models/pothole/smartathon-yolov8s.onnx

# Usuário sem privilégio de root.
# data/cache guarda a referência do meio-dia do poste: é a única pasta onde o app escreve.
RUN useradd --uid 10001 --no-create-home ia && mkdir -p data/cache && chown -R ia data
USER ia

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
