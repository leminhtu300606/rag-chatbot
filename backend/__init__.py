"""
RAG Chatbot - Học viện Kỹ thuật Mật mã
======================================
Package chính `src` được tổ chức theo 3 phần RAG chuẩn:

- Phần 1: backend.preprocessing  -> Tiền xử lý dữ liệu (PDF/DOCX/TXT -> data/processed)
- Phần 2: backend.indexing       -> Huấn luyện / Indexing (chunking, embedding, vector_store, reranking)
- Phần 3: backend.generation     -> Prompt & Sinh câu trả lời (retrieval -> rerank -> LLM)

Cấu trúc xem chi tiết ở README.md và backend/config.py

Ví dụ nhanh:
    from backend.preprocessing.pipeline import process_all
    from backend.index import build
    from backend.generation.rag import answer

    process_all()          # P1: tiền xử lý
    build(rebuild=True)    # P2: indexing
    print(answer("Quy chế đào tạo là gì?"))  # P3: generation
"""

__version__ = "2.0.0"
__all__ = ["preprocessing", "indexing", "generation", "config"]

