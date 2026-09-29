FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip && python -m pip install .

COPY app ./app
COPY evaluation ./evaluation
COPY docs ./docs
COPY data/raw ./data/raw

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "allianz_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
