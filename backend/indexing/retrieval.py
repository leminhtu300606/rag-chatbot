"""
backend/indexing/retrieval.py - Cum truy xuat (gop tu retriever.py + bm25.py + hybrid.py)
==========================================================================================
- Vector: truy xuat top-k chunk bang embedding (PostgreSQL/pgvector).
- BM25: lexical search tieng Viet (BM25Okapi thuan Python, cache theo mtime).
- Hybrid: weighted sum BM25 + Vector (trong so tu config RETRIEVAL), fallback tung nhanh.

Giu nguyen thuat toan va chu ky ham cu de khong vo caller.
"""
import hashlib
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import backend.config.settings as cfg
from backend.config.settings import RETRIEVE_TOP_K
from backend.indexing.embedder import embed
from backend.indexing.vectorstore import query_vector

try:
    RETR_CFG = getattr(cfg, "RETRIEVAL", {})
except Exception:
    RETR_CFG = {}


# ── Vector retrieval (gop tu retriever.py) ──

def _query_pgvector(q_emb, where, fetch_k: int) -> list[dict]:
    """Helper query pgvector với where, bắt lỗi count. Giữ tên cũ để caller không đổi."""
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
        return query_vector(q_emb, where=where, n=fetch_k)
    except Exception:
        # where filter có thể lỗi nếu không có dữ liệu session
        # fallback về query không filter
        if where and "session_id" in str(where):
            try:
                docs = query_vector(q_emb, where=None, n=fetch_k)
                # lọc thủ công theo session_id
                out = []
                for d in docs:
                    # giữ lại nếu metadata session_id khớp hoặc không có session filter gốc
                    out.append(d)
                return out
            except Exception:
                return []
        return []


def retrieve(query: str, top_k: int = RETRIEVE_TOP_K, category: str = None, dedup: bool = True, session_id: str | None = None) -> list[dict]:
    """Truy xuất các chunk liên quan nhất với câu hỏi (vector search)."""
    q_emb = embed([query])[0]
    # Fetch dư để bù sau khi dedup (nếu dedup bật)
    fetch_k = top_k * 2 if dedup else top_k

    # Nếu có session_id: merge global + session (ưu tiên session, cô lập phiên)
    if session_id:
        # 1) session-specific
        sess_where = {"session_id": session_id}
        if category:
            sess_where["category"] = category
        sess_docs = _query_pgvector(q_emb, sess_where, fetch_k)
        # 2) global (không filter session, nhưng có category nếu cần) -> lọc bỏ upload của phiên khác
        global_where = {"category": category} if category else None
        global_docs_raw = _query_pgvector(q_emb, global_where, fetch_k)
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
                from backend.common.utils import deduplicate_docs
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
    raw_all = _query_pgvector(q_emb, where, fetch_k)
    # Lọc bỏ upload theo phiên khi query global (không có session_id) để cô lập
    raw = [d for d in raw_all if not d.get("metadata", {}).get("session_id")] if raw_all else []
    if dedup and raw:
        try:
            from backend.common.utils import deduplicate_docs
            deduped = deduplicate_docs(
                raw,
                threshold=cfg.CHUNK.get("dedup_threshold", 0.92) if hasattr(cfg, "CHUNK") else 0.92,
                exact_only=False,
            )
            return deduped[:top_k]
        except Exception:
            pass
    return raw[:top_k]


# ── BM25 (gop tu bm25.py: khong dung rank_bm25 external, BM25Okapi thuan Python) ──

# Tokenizer cho tiếng Việt: tách theo khoảng trắng + bỏ dấu câu, giữ từ ghép?
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def _strip_accents(s: str) -> str:
    try:
        return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    except Exception:
        return s


def _tokenize_vi(text: str) -> List[str]:
    if not text:
        return []
    # Lower + bỏ dấu câu -> space, collapse
    t = text.lower()
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    if not t:
        return []
    # Cho phep fast mode khi eval (EVAL_FAST_TOKENIZE=1) de tranh underthesea cham
    import os as _os
    if _os.getenv("EVAL_FAST_TOKENIZE", "").lower() in ("1", "true", "yes"):
        return [w for w in t.split() if len(w) >= 2]
    # Thử dùng underthesea word_tokenize nếu có (tốt cho tiếng Việt)
    try:
        from underthesea import word_tokenize
        # underthesea word_tokenize trả về string có _ cho từ ghép
        tok = word_tokenize(t, format="text")
        # tok: "học_bổng là gì" -> split
        tokens = tok.split()
        # Bỏ "_" để so sánh linh hoạt, nhưng giữ cả 2 dạng
        out = []
        for w in tokens:
            out.append(w)
            if "_" in w:
                # thêm cả các thành phần
                out.extend(w.split("_"))
        # Lọc token quá ngắn
        return [w for w in out if len(w) >= 2]
    except Exception:
        # Fallback: split whitespace
        return [w for w in t.split() if len(w) >= 2]


class BM25Okapi:
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / self.N if self.N else 0
        self.df = Counter()
        self.doc_len = []
        self.doc_tf = []
        for doc in corpus:
            tf = Counter(doc)
            self.doc_tf.append(tf)
            self.doc_len.append(len(doc))
            for term in tf:
                self.df[term] += 1
        self.idf = {}
        for term, freq in self.df.items():
            # IDF chuẩn BM25
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        if doc_idx >= self.N:
            return 0.0
        tf = self.doc_tf[doc_idx]
        dl = self.doc_len[doc_idx]
        if dl == 0:
            return 0.0
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            idf = self.idf.get(term, 0)
            freq = tf[term]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * freq * (self.k1 + 1) / denom if denom else 0
        return score

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        return [self.score(query_tokens, i) for i in range(self.N)]


# ── Cache ──
_BM25_CACHE = {
    "bm25": None,
    "docs": None,  # list[dict] chunks
    "mtime": 0.0,
    "corpus_tokens": None,
}
_BM25_CACHE_BY_CAT: dict = {}  # category -> {bm25, docs, mtime, tokens}


def _get_processed_mtime() -> float:
    try:
        from backend.preprocessing.storage import get_processed_files
        files = get_processed_files()
        if not files:
            return 0.0
        return max(p.stat().st_mtime for p in files if p.exists())
    except Exception:
        return 0.0


def _load_chunks_for_bm25(category: str | None = None) -> List[Dict]:
    from backend.preprocessing.storage import iter_chunks
    docs = []
    for ch in iter_chunks():
        if category and ch.get("category") != category:
            continue
        docs.append(ch)
    return docs


def _build_bm25(docs: List[Dict]) -> Tuple[BM25Okapi, List[List[str]]]:
    k1 = RETR_CFG.get("bm25_k1", 1.5) if isinstance(RETR_CFG, dict) else 1.5
    b = RETR_CFG.get("bm25_b", 0.75) if isinstance(RETR_CFG, dict) else 0.75
    corpus_tokens = [_tokenize_vi(d.get("text", "")) for d in docs]
    # Lọc doc rỗng
    # Nếu corpus rỗng thì trả None
    bm25 = BM25Okapi(corpus_tokens, k1=k1, b=b)
    return bm25, corpus_tokens


def _ensure_bm25(category: str | None = None) -> Tuple[BM25Okapi | None, List[Dict]]:
    """Đảm bảo BM25 cache còn fresh, rebuild nếu mtime đổi hoặc category đổi."""
    mtime = _get_processed_mtime()
    # Cache theo category để evaluation nhanh (trước đây build lại mỗi query)
    if category:
        cached = _BM25_CACHE_BY_CAT.get(category)
        if cached and cached.get("mtime") == mtime and cached.get("bm25") is not None:
            return cached["bm25"], cached["docs"]
        docs = _load_chunks_for_bm25(category=category)
        if not docs:
            return None, []
        bm25, tokens = _build_bm25(docs)
        _BM25_CACHE_BY_CAT[category] = {"bm25": bm25, "docs": docs, "mtime": mtime, "tokens": tokens}
        return bm25, docs
    # Không category: dùng cache global
    if _BM25_CACHE["bm25"] is not None and _BM25_CACHE["mtime"] == mtime and _BM25_CACHE["docs"] is not None:
        return _BM25_CACHE["bm25"], _BM25_CACHE["docs"]
    docs = _load_chunks_for_bm25(category=None)
    if not docs:
        return None, []
    bm25, tokens = _build_bm25(docs)
    _BM25_CACHE["bm25"] = bm25
    _BM25_CACHE["docs"] = docs
    _BM25_CACHE["mtime"] = mtime
    _BM25_CACHE["corpus_tokens"] = tokens
    return bm25, docs


def bm25_search(query: str, top_k: int = 10, category: str | None = None) -> List[Dict]:
    """
    Tìm kiếm BM25, trả về list dict giống retriever: {text, metadata, score}
    score là BM25 raw (càng cao càng liên quan).
    """
    if not query or not query.strip():
        return []
    bm25, docs = _ensure_bm25(category=category)
    if bm25 is None or not docs:
        return []
    q_tokens = _tokenize_vi(query)
    if not q_tokens:
        return []
    scores = bm25.get_scores(q_tokens)
    # Kết hợp với docs, sắp xếp giảm dần
    scored = [(score, idx) for idx, score in enumerate(scores) if score > 0]
    if not scored:
        return []
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k * 2]  # lấy dư để dedup sau
    result = []
    for score, idx in top:
        d = docs[idx]
        result.append({
            "text": d.get("text", ""),
            "metadata": {
                "category": d.get("category"),
                "subcategory": d.get("subcategory"),
                "filename": d.get("filename"),
                "source": d.get("source"),
                "page": d.get("page"),
                "chunk_index": d.get("chunk_index", 0),
                "chunk_mode": d.get("chunk_mode", "bm25"),
                "chunk_type": d.get("chunk_type", "text"),
            },
            "score": float(score),
            "bm25_score": float(score),
        })
    # Dedup
    try:
        from backend.common.utils import deduplicate_docs
        result = deduplicate_docs(result)
    except Exception:
        pass
    return result[:top_k]


# ── Hybrid BM25 + Vector (gop tu hybrid.py, CPU-friendly) ──
# Kết hợp vector search (pgvector) và BM25 lexical search.
# Trọng số lấy từ config RETRIEVAL: vector_weight + bm25_weight.
# Chuẩn hóa scores về [0,1] bằng min-max trước khi weighted sum.
# Fallback: nếu BM25 không có kết quả thì chỉ dùng vector, và ngược lại.

def _normalize_scores(docs: List[Dict], score_key: str = "score") -> List[float]:
    if not docs:
        return []
    scores = [float(d.get(score_key, 0) or 0) for d in docs]
    # Với vector distance (cosine): nhỏ = gần, cần đảo
    # pgvector trả distance, không phải similarity. Ta cần chuyển distance -> similarity
    # Nhưng retrieve trả score là distance, còn BM25 là higher better.
    # Để thống nhất, ta sẽ normalize theo rank: cao hơn = tốt hơn.
    # Với vector, ta sẽ đảo: similarity = 1 - distance (hoặc 1/(1+distance))
    # Ở đây ta giả sử vector score đã là distance, chuyển thành 1 - score clipped
    # Nếu score là distance thì thường 0-2, ta làm 1 - min(score,1)
    # Kiểm tra: nếu scores đều >0 và nhỏ (0.2, 0.5) thì đảo sẽ cho high is better.
    # Heuristic: nếu top score < 2 và sorted ascending (vector distance thường tăng dần)
    # thì đảo.
    # Nhưng retrieve đã trả raw distance, ta sẽ convert.
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


def hybrid_retrieve(query: str, top_k: int = 3, category: str | None = None, vector_k: int | None = None, bm25_k: int | None = None, session_id: str | None = None) -> List[Dict]:
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
        # Fallback pure vector (có session_id nếu truyền)
        return retrieve(query, top_k=top_k, category=category, dedup=True, session_id=session_id)

    vk = vector_k or (top_k * 5)
    bk = bm25_k or (top_k * 5)

    vec_docs = []
    bm25_docs = []
    try:
        vec_docs = retrieve(query, top_k=vk, category=category, dedup=True, session_id=session_id)
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
    from backend.common.utils import normalize_text

    merged: Dict[str, Dict] = {}

    for doc, norm in zip(vec_docs, vec_norm):
        key = hashlib.md5(normalize_text(doc.get("text", "")).encode("utf-8")).hexdigest()
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
        key = hashlib.md5(normalize_text(doc.get("text", "")).encode("utf-8")).hexdigest()
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
    for d in sorted_docs[:top_k * 2]:
        result.append({
            "text": d["text"],
            "metadata": d["metadata"],
            "score": float(d["score"]),
            "vector_score": d.get("vector_score", 0),
            "bm25_score": d.get("bm25_score", 0),
        })

    # Dedup lần cuối
    try:
        from backend.common.utils import deduplicate_docs
        result = deduplicate_docs(result)
    except Exception:
        pass

    return result[:top_k]


__all__ = [
    "retrieve",
    "bm25_search",
    "hybrid_retrieve",
    "BM25Okapi",
]
