"""
backend/indexing/hybrid.py - Hybrid retrieval BM25 + Vector (CPU-friendly)
=========================================================================
Kết hợp vector search (Chroma + BGE-M3) và BM25 lexical search.
Trọng số lấy từ config RETRIEVAL: vector_weight + bm25_weight.
Chuẩn hóa scores về [0,1] bằng min-max trước khi weighted sum.

Fallback: nếu BM25 không có kết quả thì chỉ dùng vector, và ngược lại.
"""

from typing import List, Dict
import backend.config as cfg
from backend.indexing.retriever import retrieve as vector_retrieve
from backend.indexing.bm25 import bm25_search

try:
    RETR_CFG = getattr(cfg, "RETRIEVAL", {})
except Exception:
    RETR_CFG = {}

def _normalize_scores(docs: List[Dict], score_key: str = "score") -> List[float]:
    if not docs:
        return []
    scores = [float(d.get(score_key, 0) or 0) for d in docs]
    # Với vector distance (cosine): nhỏ = gần, cần đảo
    # Chroma trả distance, không phải similarity. Ta cần chuyển distance -> similarity
    # Nhưng vector_retrieve trả score là distance, còn BM25 là higher better.
    # Để thống nhất, ta sẽ normalize theo rank: cao hơn = tốt hơn.
    # Với vector, ta sẽ đảo: similarity = 1 - distance (hoặc 1/(1+distance))
    # Ở đây ta giả sử vector score đã là distance, chuyển thành 1 - score clipped
    # Nếu score là distance thì thường 0-2, ta làm 1 - min(score,1)
    # Kiểm tra: nếu scores đều >0 và nhỏ (0.2, 0.5) thì đảo sẽ cho high is better.
    # Heuristic: nếu top score < 2 và sorted ascending (vector distance thường tăng dần)
    # thì đảo.
    # Nhưng vector_retrieve đã trả raw distance, ta sẽ convert.
    # Đơn giản: normalize min-max sau khi convert.

    # Detect vector distance: if max score < 2 and scores are roughly increasing with rank (đã sort)
    # thì coi là distance và đảo
    is_distance = False
    if docs and "bm25_score" not in docs[0]:
        # vector docs have no bm25_score
        # Check if first doc has smallest score (distance) vs BM25 where first has largest
        # vector distance thường 0.2-0.6, BM25 thường 1-10
        # Ta cũng có thể check average
        if scores and max(scores) < 5 and min(scores) < 1.5:
            is_distance = True
    if is_distance:
        # Convert distance -> similarity (0-1)
        # similarity = 1 - distance clipped [0,1]
        conv = [max(0.0, 1.0 - min(s, 1.0)) for s in scores]
        scores = conv
    # Min-max normalize to 0-1
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]

def hybrid_retrieve(query: str, top_k: int = 3, category: str | None = None, vector_k: int | None = None, bm25_k: int | None = None) -> List[Dict]:
    """
    Hybrid search: vector_weight * norm_vector + bm25_weight * norm_bm25

    Args:
        query: câu hỏi
        top_k: số kết quả cuối
        category: filter category
        vector_k, bm25_k: số fetch mỗi nhánh (None = top_k*2)

    Returns:
        list dict {text, metadata, score} đã sort giảm dần theo hybrid_score
    """
    hybrid_enabled = True
    try:
        hybrid_enabled = bool(RETR_CFG.get("hybrid", True)) if isinstance(RETR_CFG, dict) else True
    except Exception:
        hybrid_enabled = True

    if not hybrid_enabled:
        # Fallback pure vector
        return vector_retrieve(query, top_k=top_k, category=category, dedup=True)

    vk = vector_k or (top_k * 2)
    bk = bm25_k or (top_k * 2)

    vec_docs = []
    bm25_docs = []
    try:
        vec_docs = vector_retrieve(query, top_k=vk, category=category, dedup=True)
    except Exception as e:
        print(f"[hybrid] vector fail: {e}")
        vec_docs = []
    try:
        bm25_docs = bm25_search(query, top_k=bk, category=category)
    except Exception as e:
        print(f"[hybrid] bm25 fail: {e}")
        bm25_docs = []

    if not vec_docs and not bm25_docs:
        return []
    if not vec_docs:
        return bm25_docs[:top_k]
    if not bm25_docs:
        return vec_docs[:top_k]

    # Normalize mỗi nhánh
    vec_norm = _normalize_scores(vec_docs, score_key="score")
    bm25_norm = _normalize_scores(bm25_docs, score_key="score")

    vw = RETR_CFG.get("vector_weight", 0.7) if isinstance(RETR_CFG, dict) else 0.7
    bw = RETR_CFG.get("bm25_weight", 0.3) if isinstance(RETR_CFG, dict) else 0.3
    # Chuẩn hóa tổng =1
    total = vw + bw
    if total:
        vw /= total
        bw /= total

    # Gộp theo text hash để tránh trùng, weighted sum nếu cùng doc xuất hiện 2 nhánh
    from backend.utils.dedup import normalize_text
    import hashlib

    merged: Dict[str, Dict] = {}

    for doc, norm in zip(vec_docs, vec_norm):
        key = hashlib.md5(normalize_text(doc.get("text","")).encode("utf-8")).hexdigest()
        if key not in merged:
            merged[key] = {
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": norm * vw,
                "vector_score": norm,
                "bm25_score": 0.0,
                "sources": ["vector"],
            }
        else:
            # đã có từ bm25 trước? cộng
            merged[key]["score"] += norm * vw
            merged[key]["vector_score"] = norm
            merged[key]["sources"].append("vector")

    for doc, norm in zip(bm25_docs, bm25_norm):
        key = hashlib.md5(normalize_text(doc.get("text","")).encode("utf-8")).hexdigest()
        if key not in merged:
            merged[key] = {
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": norm * bw,
                "vector_score": 0.0,
                "bm25_score": norm,
                "sources": ["bm25"],
            }
        else:
            merged[key]["score"] += norm * bw
            merged[key]["bm25_score"] = norm
            merged[key]["sources"].append("bm25")

    # Sắp xếp theo hybrid score
    sorted_docs = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

    # Chuyển về format retriever chuẩn
    result = []
    for d in sorted_docs[:top_k*2]:
        result.append({
            "text": d["text"],
            "metadata": d["metadata"],
            "score": float(d["score"]),
            "vector_score": d.get("vector_score", 0),
            "bm25_score": d.get("bm25_score", 0),
        })

    # Dedup lần cuối
    try:
        from backend.utils.dedup import deduplicate_docs
        result = deduplicate_docs(result, threshold=0.92)
    except Exception:
        pass

    return result[:top_k]
