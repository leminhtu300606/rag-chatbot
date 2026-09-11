"""
backend/indexing/bm25.py - BM25 retrieval cho hybrid search (CPU-friendly)
=========================================================================
Không dùng rank_bm25 external để giảm dependency, implement BM25Okapi thuần Python.
Hỗ trợ tiếng Việt: tách từ đơn giản (lower, bỏ dấu câu), có thể dùng underthesea nếu có.
Cache theo mtime của processed files để không build lại mỗi query.
"""

import re
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple

import backend.config as cfg

try:
    RETR_CFG = getattr(cfg, "RETRIEVAL", {})
except Exception:
    RETR_CFG = {}

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
    # Với category filter, không cache theo category mà build riêng (vì ít chunks)
    # Nhưng vẫn cache theo mtime
    mtime = _get_processed_mtime()
    # Nếu có category, build riêng không dùng cache global (đơn giản)
    if category:
        docs = _load_chunks_for_bm25(category=category)
        if not docs:
            return None, []
        bm25, _ = _build_bm25(docs)
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
    top = scored[:top_k*2]  # lấy dư để dedup sau
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
        from backend.utils.dedup import deduplicate_docs
        result = deduplicate_docs(result, threshold=0.92)
    except Exception:
        pass
    return result[:top_k]

def clear_cache():
    _BM25_CACHE["bm25"] = None
    _BM25_CACHE["docs"] = None
    _BM25_CACHE["mtime"] = 0.0
    _BM25_CACHE["corpus_tokens"] = None
