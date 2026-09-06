"""
backend/test.py - Entry point test RAG (chi goi generation)
========================================================
Khong xu ly logic truc tiep, chi goi den thu muc generation:
  - generation/test_service.py (show_chunking, show_embeddings, do_ask)
  - generation/rag.py, generator.py, reranker.py (duoc test_service goi)

CACH HOI BANG SUA CODE: sua DEFAULT_QUESTION trong backend/generation/test_service.py
Chay: python backend/test.py | python -m backend.test
"""
from pathlib import Path
import sys
if __package__ in (None, ""):
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Chi goi generation - khong xu ly logic tai day
from backend.generation.test_service import (
    DEFAULT_CATEGORY,
    DEFAULT_QUESTION,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_SHOW_CONTEXT,
    DEFAULT_SHOW_RAW,
    DEFAULT_TOP_K,
    DEFAULT_USE_RERANK,
    build_parser,
    do_ask,
    main,
    show_chunking,
    show_embeddings,
)

# Bí danh để tương thích với tên hàm cũ
show_chunking_results = show_chunking
show_embedding_results = show_embeddings

__all__ = [
    "DEFAULT_QUESTION", "DEFAULT_CATEGORY", "DEFAULT_TOP_K", "DEFAULT_RERANK_TOP_K",
    "show_chunking", "show_chunking_results", "show_embeddings", "show_embedding_results",
    "do_ask", "build_parser", "main",
]

if __name__ == "__main__":
    raise SystemExit(main())

