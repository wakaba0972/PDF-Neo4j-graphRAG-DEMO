FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="PDF Neo4j GraphRAG Demo" \
      org.opencontainers.image.description="Local PDF GraphRAG UI powered by Gradio and Neo4j"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=8080 \
    GRADIO_INBROWSER=false

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --requirement requirements.txt

COPY src/ ./src/
COPY strategies/ ./strategies/
COPY config/ ./config/
RUN mkdir -p /app/data

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/', timeout=3).close()"

STOPSIGNAL SIGTERM

CMD ["python", "src/app.py"]
