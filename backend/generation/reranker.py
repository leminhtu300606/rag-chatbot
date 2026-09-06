"""
backend/generation/reranker.py - Tinh chỉnh thứ tự context trước khi dựng prompt
===============================================================================
Nhiệm vụ:
- Nhận danh sách chunk đã retrieve và chấm điểm lại độ liên quan với câu hỏi
  bằng cross-encoder, giúp prompt chỉ chứa những chunk chất lượng cao nhất.
- Giảm hallucination và tăng độ chính xác câu trả lời.

Luồng RAG: retrieve (indexing) -> rerank (generation) -> build_messages (prompts) -> generate

Model mặc định: cross-encoder/ms-marco-MiniLM-L-6-v2 (đa năng, chạy CPU được).
Có thể đổi RERANK_MODEL sang model tiếng Việt nếu cần.
"""
from typing import Optional

from sentence_transformers import CrossEncoder

from backend.config import DEVICE

# Số kết quả tốt nhất giữ lại sau rerank
DEFAULT_RERANK_TOP_K = 3
RERANK_TOP_K = DEFAULT_RERANK_TOP_K  # bí danh tương thích

_reranker = None
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL, device=DEVICE)
    return _reranker


def rerank(query: str, documents: list[dict], top_k: Optional[int] = None) -> list[dict]:
    """Rerank documents dựa trên độ liên quan với query bằng cross-encoder.

    Args:
        query: câu hỏi người dùng
        documents: list retrieved docs mỗi phần tử {"text": str, "metadata": dict, "score": float}
        top_k: số lượng top results muốn giữ (None = DEFAULT_RERANK_TOP_K=3)

    Returns:
        list đã rerank, mỗi doc có cập nhật field "score" là điểm cross-encoder,
        đã sắp xếp giảm dần theo score, chỉ giữ top_k tốt nhất.
    """
    if not documents:
        return []

    if top_k is None:
        top_k = DEFAULT_RERANK_TOP_K

    pairs = [[query, doc["text"]] for doc in documents]
    scores = _get_reranker().predict(pairs)

    reranked = []
    for doc, score in zip(documents, scores):
        new_doc = doc.copy()
        new_doc["score"] = float(score)
        reranked.append(new_doc)

    reranked.sort(key=lambda x: x["score"], reverse=True)

    return reranked[:top_k]
