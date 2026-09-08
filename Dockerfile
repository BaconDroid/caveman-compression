FROM python:3.11-slim

# Fixed cache location so models provisioned at build time are found at
# runtime regardless of which user the process runs as.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers

WORKDIR /app

# Build dependencies for wheels that must be compiled from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (pinned, CPU-only PyTorch).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code.
COPY caveman_compress_mlm.py .
COPY caveman_compress_nlp.py .
COPY server.py .
COPY mcp_server.py .
COPY download_models.py .
COPY entrypoint.sh .

# Models are downloaded at runtime on first start (see entrypoint.sh).
# This keeps the image ~1 GB smaller; models are cached in mounted volumes.
ARG LANGUAGES=en,fr
ARG CUSTOM_MLM_MODELS=""
ENV LANGUAGES=${LANGUAGES} CUSTOM_MLM_MODELS=${CUSTOM_MLM_MODELS}

# Run as a non-root user; model caches stay readable/writable by it.
RUN chmod +x /app/entrypoint.sh \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '3000') + '/health', timeout=5)"

ENTRYPOINT ["./entrypoint.sh"]
CMD ["gunicorn", "--workers", "2", "--timeout", "300", "server:app"]
