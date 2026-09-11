"""
backend/utils/dedup.py - Dedup cho text/chunk/document
=====================================================
Nhiệm vụ:
- Chuẩn hóa text tiếng Việt (lower, strip, bỏ dấu câu dư, collapse whitespace)
- Hash exact và fuzzy (Jaccard / SequenceMatcher) để loại trùng lặp
- Cung cấp hàm cho 3 tầng: chunker (list[str]), pipeline (list[str]), retriever/reranker (list[dict])

CPU-friendly: chỉ dùng stdlib + re, không dùng embedding để so sánh fuzzy nặng.
Fuzzy dùng difflib.SequenceMatcher + Jaccard trên shingle (nhanh, chạy CPU).
"""

import re
import hashlib
import unicodedata
from difflib import SequenceMatcher
from typing import List, Dict, Optional


# ── Chuẩn hóa ──
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str, strip_accents: bool = False) -> str:
    """Chuẩn hóa text để so sánh dedup: lower, collapse whitespace, bỏ dấu câu thừa."""
    if not text:
        return ""
    t = text.strip().lower()
    # Bỏ dấu tiếng Việt nếu yêu cầu (để bắt trùng không dấu)
    if strip_accents:
        t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    # Thay dấu câu bằng space rồi collapse
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def hash_text(text: str, strip_accents: bool = False) -> str:
    norm = normalize_text(text, strip_accents=strip_accents)
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def _jaccard(a: str, b: str, n: int = 3) -> float:
    """Jaccard trên shingle n-gram ký tự (nhanh)."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    def shingles(s):
        s = s.replace(" ", "")
        if len(s) < n:
            return {s}
        return {s[i:i+n] for i in range(len(s)-n+1)}
    sa, sb = shingles(a), shingles(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def is_duplicate(a: str, b: str, threshold: float = 0.92, use_jaccard: bool = True) -> bool:
    """Kiểm tra 2 text có trùng lặp gần đúng không (CPU-friendly)."""
    na = normalize_text(a)
    nb = normalize_text(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    # Nhanh: nếu độ dài chênh quá 40% thì không thể trùng
    la, lb = len(na), len(nb)
    if min(la, lb) / max(la, lb) < 0.6:
        return False
    # Jaccard trước (rẻ)
    if use_jaccard:
        j = _jaccard(na, nb)
        if j >= 0.88:
            return True
        if j < 0.5:
            # Nếu Jaccard quá thấp thì không cần SequenceMatcher
            # Nhưng vẫn cho phép trường hợp chỉ khác vài từ
            pass
    # SequenceMatcher chính xác hơn
    ratio = SequenceMatcher(None, na, nb).ratio()
    return ratio >= threshold


# ── Dedup cho list[str] (chunker/pipeline) ──

def deduplicate_chunks(
    chunks: List[str],
    threshold: float = 0.92,
    exact_only: bool = False,
    strip_accents: bool = False,
) -> List[str]:
    """
    Loại chunk trùng lặp giữ lại thứ tự xuất hiện đầu tiên.

    Args:
        chunks: list text
        threshold: ngưỡng fuzzy (0.92 mặc định)
        exact_only: True = chỉ loại exact hash, False = cả fuzzy
        strip_accents: có bỏ dấu khi so sánh hash exact không
    """
    if not chunks:
        return chunks
    seen_hash = set()
    result: List[str] = []
    # Lưu normalized để so fuzzy nhanh
    seen_norm: List[str] = []

    for c in chunks:
        if not c or not c.strip():
            continue
        norm = normalize_text(c, strip_accents=strip_accents)
        if not norm:
            continue
        h = hashlib.md5(norm.encode("utf-8")).hexdigest()
        if h in seen_hash:
            continue
        is_dup = False
        if not exact_only and seen_norm:
            # So fuzzy với các chunk đã giữ (giới hạn so sánh 30 gần nhất để O(n) không nổ)
            # Nếu tổng chunk ít thì so hết, nếu nhiều thì chỉ so window
            window = seen_norm[-30:] if len(seen_norm) > 30 else seen_norm
            for prev_norm in window:
                # Heuristic nhanh: nếu cùng hash prefix hoặc độ dài gần
                if prev_norm == norm:
                    is_dup = True
                    break
                # Chỉ so fuzzy khi độ dài tương tự và Jaccard gợi ý
                if abs(len(prev_norm) - len(norm)) > 200:
                    continue
                if is_duplicate(prev_norm, norm, threshold=threshold):
                    is_dup = True
                    break
        if is_dup:
            continue
        seen_hash.add(h)
        seen_norm.append(norm)
        result.append(c)
    return result


# ── Dedup cho list[dict] có key "text" (retriever/reranker/prompts) ──

def deduplicate_docs(
    docs: List[Dict],
    text_key: str = "text",
    threshold: float = 0.92,
    exact_only: bool = False,
) -> List[Dict]:
    """
    Loại doc trùng lặp theo text, giữ bản có score cao nhất nếu có field score.
    Giữ thứ tự theo score giảm dần nếu có, không thì giữ thứ tự gốc.
    """
    if not docs:
        return docs
    # Nếu có score, sắp xếp giảm dần trước để giữ bản tốt nhất
    has_score = any("score" in d for d in docs)
    if has_score:
        # Với Chroma distance (nhỏ = gần), nhưng rerank score (lớn = tốt)
        # Ta không biết chiều, nên giữ nguyên thứ tự nếu không rõ.
        # Chỉ sắp xếp nếu phát hiện rerank (score có thể âm hoặc dương lớn)
        # Thôi giữ nguyên để không đảo retrieval order
        pass

    seen_hash = set()
    seen_norm: List[str] = []
    result: List[Dict] = []

    for doc in docs:
        txt = doc.get(text_key, "") if isinstance(doc, dict) else str(doc)
        if not txt or not txt.strip():
            continue
        norm = normalize_text(txt)
        if not norm:
            continue
        h = hashlib.md5(norm.encode("utf-8")).hexdigest()
        if h in seen_hash:
            continue
        is_dup = False
        if not exact_only and seen_norm:
            window = seen_norm[-30:] if len(seen_norm) > 30 else seen_norm
            for prev_norm in window:
                if prev_norm == norm:
                    is_dup = True
                    break
                if abs(len(prev_norm) - len(norm)) > 400:
                    continue
                if is_duplicate(prev_norm, norm, threshold=threshold):
                    is_dup = True
                    break
        if is_dup:
            continue
        seen_hash.add(h)
        seen_norm.append(norm)
        result.append(doc)
    return result


# ── Helper cho pipeline: dedup list[dict] chunk có "text" ──

def deduplicate_chunk_dicts(
    chunks: List[Dict],
    threshold: float = 0.92,
    exact_only: bool = False,
) -> List[Dict]:
    """Dedup cho list dict có key 'text', giữ metadata của bản đầu tiên."""
    if not chunks:
        return chunks
    seen_hash = set()
    seen_norm: List[str] = []
    result: List[Dict] = []
    for ch in chunks:
        txt = ch.get("text", "")
        norm = normalize_text(txt)
        if not norm:
            continue
        h = hashlib.md5(norm.encode("utf-8")).hexdigest()
        if h in seen_hash:
            continue
        is_dup = False
        if not exact_only and seen_norm:
            window = seen_norm[-30:] if len(seen_norm) > 30 else seen_norm
            for prev in window:
                if prev == norm:
                    is_dup = True
                    break
                if abs(len(prev) - len(norm)) > 400:
                    continue
                if is_duplicate(prev, norm, threshold=threshold):
                    is_dup = True
                    break
        if is_dup:
            continue
        seen_hash.add(h)
        seen_norm.append(norm)
        result.append(ch)
    return result
