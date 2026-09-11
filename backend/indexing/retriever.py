"""Truy xuất các chunk liên quan nhất với câu hỏi."""
from backend.config import RETRIEVE_TOP_K
from backend.indexing.embedder import embed
from backend.indexing.vectorstore import get_collection
import backend.config as cfg


def retrieve(query: str, top_k: int = RETRIEVE_TOP_K, category: str = None, dedup: bool = True) -> list[dict]:
    q_emb = embed([query])[0]
    where = {"category": category} if category else None
    # Fetch dư để bù sau khi dedup (nếu dedup bật)
    fetch_k = top_k * 2 if dedup else top_k
    # Chroma n_results không được vượt quá count
    try:
        from backend.indexing.vectorstore import count as _count
        total = _count()
        if total and fetch_k > total:
            fetch_k = total
    except Exception:
        pass
    res = get_collection().query(
        query_embeddings=[q_emb],
        n_results=fetch_k,
        where=where,
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    raw = [
        {"text": d, "metadata": m, "score": float(s)}
        for d, m, s in zip(docs, metas, dists)
    ]
    if dedup and raw:
        try:
            from backend.utils.dedup import deduplicate_docs
            deduped = deduplicate_docs(
                raw,
                threshold=cfg.CHUNK.get("dedup_threshold", 0.92) if hasattr(cfg, "CHUNK") else 0.92,
                exact_only=False,
            )
            # Trả về đủ top_k sau dedup (nếu dedup làm giảm nhiều thì lấy deduped)
            return deduped[:top_k]
        except Exception:
            pass
    return raw[:top_k]

