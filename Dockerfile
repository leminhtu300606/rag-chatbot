# RAG Chatbot - production image (CPU)
# Stage 1: build frontend React + TypeScript + Vite
FROM node:20-slim AS frontend-build
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# Stage 2: Python runtime (FastAPI + RAG)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DEVICE=cpu \
    OLLAMA_BASE=http://ollama:11434

# System libs: OpenCV/EasyOCR (libgl), PDF/OCR, clamav client not needed (socket only)
# + libpq cho psycopg (pgvector) qua wheel binary thường không cần, giữ postgresql-client cho debug
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 tesseract-ocr curl postgresql-client \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch CPU trước để tránh kéo wheel CUDA nặng
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY tools/ tools/
# Frontend: chỉ lấy bản build (FastAPI serve frontend/dist)
COPY --from=frontend-build /web/dist frontend/dist

# non-root + thư mục runtime (vector lưu ở Postgres, không cần chroma_db local)
RUN useradd -m -u 10001 rag \
 && mkdir -p /app/data/processed /app/data/uploads /app/data/logs /app/backups \
 && chown -R rag:rag /app
USER rag

VOLUME ["/app/data"]

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=8).status==200 else 1)"

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
