"""
backend/cache.py - Cache cau hoi lap lai (GĐ4)
=============================================
LRU + TTL thuần stdlib cho câu hỏi đơn lượt giống hệt nhau
(question + category + top_k + style), khi không có session/history.
Giảm tải Ollama với câu hỏi phổ biến ("Quy chế đào tạo là gì?").

Tắt bằng CACHE_ENABLED=false. Không cache khi auth bật? Vẫn cache được vì
key không chứa nội dung user; nhưng category đã nằm trong key nên RBAC an toàn.
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

import backend.config.settings as cfg

_lock = threading.Lock()
_store: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()


def _enabled() -> bool:
    try:
        return bool(cfg.CACHE_ENABLED)
    except Exception:
        return True


def make_key(question: str, category: Optional[str], top_k: Optional[int], style: Optional[str]) -> str:
    raw = f"{(question or '').strip()}|{category or ''}|{top_k or ''}|{style or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get(key: str) -> Optional[Any]:
    if not _enabled():
        return None
    now = time.time()
    with _lock:
        hit = _store.get(key)
        if not hit:
            return None
        exp, val = hit
        if exp < now:
            _store.pop(key, None)
            return None
        _store.move_to_end(key)
        return val


def put(key: str, value: Any) -> None:
    if not _enabled():
        return
    try:
        ttl = int(cfg.CACHE_TTL_SECONDS)
        maxn = int(cfg.CACHE_MAX_ITEMS)
    except Exception:
        ttl, maxn = 600, 1000
    with _lock:
        _store[key] = (time.time() + max(1, ttl), value)
        _store.move_to_end(key)
        while len(_store) > max(1, maxn):
            _store.popitem(last=False)


def stats() -> dict:
    with _lock:
        return {"items": len(_store)}


def clear() -> None:
    with _lock:
        _store.clear()
