"""
RAG Chatbot - Học viện Kỹ thuật Mật mã
====================================
Package chính được tổ chức theo 3 phần RAG chuẩn + các lớp hỗ trợ:

- backend.preprocessing  -> P1: Tiền xử lý (PDF/DOCX -> data/processed/*.parquet)
- backend.indexing        -> P2: Indexing (embedding -> ChromaDB) + retrieval
- backend.generation      -> P3: Generation (retrieve -> rerank -> prompt -> LLM)
                            + agent đa bước (generation/agent.py, LangGraph self-check)
- backend.api             -> Giao diện HTTP: FastAPI (app) + session + upload (api/app.py)
- backend.cli             -> Giao diện CLI: clean / index (cli/main.py)
- backend.config          -> Cấu hình + security dùng chung (config/settings.py)
- backend.common           -> Tiện ích dùng chung: dedup (common/utils.py)
- backend.security        -> Xác thực + phân quyền (security/auth.py)
- backend.infra           -> Hạ tầng dùng chung: cache (infra/cache.py)

Tương thích ngược (shim, không đổi logic):
- backend.core / backend.utils / backend.auth / backend.cache / backend.agent
  vẫn import được (re-export từ vị trí mới).
- `python -m backend.cli`, `python -m backend.api`, `uvicorn backend.api:app` giữ nguyên.

Ví dụ nhanh:
    from backend.preprocessing.pipeline import process_all
    from backend.cli.main import build
    from backend.generation.rag import answer

    process_all()          # P1: tiền xử lý
    build(rebuild=True)    # P2: indexing
    print(answer("Quy chế đào tạo là gì?"))  # P3: generation
"""

__version__ = "2.1.0"
__all__ = ["preprocessing", "indexing", "generation", "api", "cli", "config",
           "common", "security", "infra"]
