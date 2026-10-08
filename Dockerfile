# RAG Chatbot - production image (CPU)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DEVICE=cpu \
    OLLAMA_BASE=http://ollama:11434

# System libs: OpenCV/EasyOCR (libgl), PDF/OCR, clamav client not needed (socket only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 tesseract-ocr curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch CPU trước để tránh kéo wheel CUDA nặng
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY frontend/ frontend/
COPY tools/ tools/

# non-root + thư mục runtime
RUN useradd -m -u 10001 rag \
 && mkdir -p /app/chroma_db /app/data/processed /app/data/uploads /app/data/logs /app/backups \
 && chown -R rag:rag /app
USER rag

VOLUME ["/app/chroma_db", "/app/data"]

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=8).status==200 else 1)"

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
