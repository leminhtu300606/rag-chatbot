"""Truy xuất các chunk liên quan nhất với câu hỏi."""
from backend.config import RETRIEVE_TOP_K
from backend.indexing.embedder import embed
from backend.indexing.vectorstore import get_collection
import backend.config as cfg


def _query_chroma(q_emb, where, fetch_k: int) -> list[dict]:
    """Helper query Chroma với where, bắt lỗi count."""
    try:
        from backend.indexing.vectorstore import count as _count
        total = _count()
        if total and fetch_k > total:
            fetch_k = max(1, min(fetch_k, total))
    except Exception:
        pass
    if fetch_k <= 0:
        return []
    try:
        res = get_collection().query(
            query_embeddings=[q_emb],
            n_results=fetch_k,
            where=where,
        )
        docs = res.get("documents", [[]])[0] or []
        metas = res.get("metadatas", [[]])[0] or []
        dists = res.get("distances", [[]])[0] or []
        return [
            {"text": d, "metadata": m, "score": float(s)}
            for d, m, s in zip(docs, metas, dists)
        ]
    except Exception as e:
        # where filter có thể lỗi nếu không có dữ liệu session
        # fallback về query không filter
        if where and "session_id" in str(where):
            try:
                res = get_collection().query(query_embeddings=[q_emb], n_results=fetch_k, where=None)
                docs = res.get("documents", [[]])[0] or []
                metas = res.get("metadatas", [[]])[0] or []
                dists = res.get("distances", [[]])[0] or []
                # lọc thủ công theo session_id
                out = []
                for d, m, s in zip(docs, metas, dists):
                    # giữ lại nếu metadata session_id khớp hoặc không có session filter gốc
                    out.append({"text": d, "metadata": m, "score": float(s)})
                return out
            except Exception:
                return []
        return []


def retrieve(query: str, top_k: int = RETRIEVE_TOP_K, category: str = None, dedup: bool = True, session_id: str | None = None) -> list[dict]:
    q_emb = embed([query])[0]
    # Fetch dư để bù sau khi dedup (nếu dedup bật)
    fetch_k = top_k * 2 if dedup else top_k

    # Nếu có session_id: merge global + session (ưu tiên session, cô lập phiên)
    if session_id:
        # 1) session-specific
        sess_where = {"session_id": session_id}
        if category:
            sess_where["category"] = category
        sess_docs = _query_chroma(q_emb, sess_where, fetch_k)
        # 2) global (không filter session, nhưng có category nếu cần) -> lọc bỏ upload của phiên khác
        global_where = {"category": category} if category else None
        global_docs_raw = _query_chroma(q_emb, global_where, fetch_k)
        # Lọc: chỉ giữ doc global thực sự (session_id is None / không có)
        global_docs = [d for d in global_docs_raw if not d.get("metadata", {}).get("session_id")]
        # Nếu global không đủ do lọc, vẫn có sess_docs
        combined = sess_docs + global_docs
        if not combined:
            # fallback: nếu lọc hết mà không có gì, thử dùng sess_docs
            combined = sess_docs
        if not combined:
            return []
        if dedup:
            try:
                from backend.utils.dedup import deduplicate_docs
                deduped = deduplicate_docs(
                    combined,
                    threshold=cfg.CHUNK.get("dedup_threshold", 0.92) if hasattr(cfg, "CHUNK") else 0.92,
                    exact_only=False,
                )
                return deduped[:top_k]
            except Exception:
                pass
        seen = set()
        out = []
        for d in combined:
            t = d.get("text", "")[:200]
            if t not in seen:
                seen.add(t)
                out.append(d)
            if len(out) >= top_k:
                break
        return out

    where = {"category": category} if category else None
    raw_all = _query_chroma(q_emb, where, fetch_k)
    # Lọc bỏ upload theo phiên khi query global (không có session_id) để cô lập
    raw = [d for d in raw_all if not d.get("metadata", {}).get("session_id")] if raw_all else []
    if dedup and raw:
        try:
            from backend.utils.dedup import deduplicate_docs
            deduped = deduplicate_docs(
                raw,
                threshold=cfg.CHUNK.get("dedup_threshold", 0.92) if hasattr(cfg, "CHUNK") else 0.92,
                exact_only=False,
            )
            return deduped[:top_k]
        except Exception:
            pass
    return raw[:top_k]

