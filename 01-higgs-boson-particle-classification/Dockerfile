# Build multi-etapa: la etapa "builder" descarga los datos reales de CERN
# Open Data (sin autenticacion) y corre el pipeline real de principio a fin
# para producir el modelo entrenado -- la imagen final sirve ese modelo
# real, no un artefacto pre-entrenado copiado a mano (que ademas esta en
# .gitignore y no existe en un checkout limpio del repo).

FROM python:3.10-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

RUN mkdir -p data/raw && \
    python -c "import urllib.request; urllib.request.urlretrieve('https://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz', 'data/raw/atlas-higgs.csv.gz')" && \
    python -c "import gzip, shutil; shutil.copyfileobj(gzip.open('data/raw/atlas-higgs.csv.gz'), open('data/raw/atlas-higgs.csv', 'wb'))" && \
    rm data/raw/atlas-higgs.csv.gz

RUN python -m src.pipeline

FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY --from=builder /app/outputs/models/gbm_models.joblib outputs/models/gbm_models.joblib
COPY --from=builder /app/outputs/models/jet_medians.joblib outputs/models/jet_medians.joblib

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
