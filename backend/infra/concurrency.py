"""backend/infra/concurrency.py - Giới hạn đồng thời cho tài nguyên dùng chung.

Áp dụng cho Ollama/LLM và suy luận torch (embedding/rerank) để tránh quá tải
CPU/RAM khi nhiều request đến cùng lúc. Mỗi (name, limit) có một BoundedSemaphore
dùng chung trong tiến trình.
"""
from __future__ import annotations

from contextlib import contextmanager
from threading import BoundedSemaphore, Lock

_SEMAPHORES: dict = {}
_LOCK = Lock()


def get_semaphore(name: str, limit: int | None):
    """Semaphore theo tên; limit <= 0/None -> không giới hạn (trả None)."""
    try:
        lim = int(limit)
    except Exception:
        return None
    if lim <= 0:
        return None
    key = f"{name}:{lim}"
    with _LOCK:
        sem = _SEMAPHORES.get(key)
        if sem is None:
            sem = BoundedSemaphore(lim)
            _SEMAPHORES[key] = sem
        return sem


@contextmanager
def limit(name: str, limit_value: int | None, timeout: float | None = None):
    """Ngữ cảnh giới hạn đồng thời. Nếu chờ quá `timeout` -> raise TimeoutError."""
    sem = get_semaphore(name, limit_value)
    if sem is None:
        yield
        return
    acquired = sem.acquire(timeout=timeout) if timeout else sem.acquire()
    if not acquired:
        raise TimeoutError(f"Quá tải: vượt giới hạn đồng thời cho '{name}' ({limit_value})")
    try:
        yield
    finally:
        sem.release()
