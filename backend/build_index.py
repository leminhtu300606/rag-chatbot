"""
backend/build_index.py - Bí danh cho backend.index
==================================================
Nhiệm vụ:
- Re-export toàn bộ hàm từ backend.index để lệnh `python -m backend.build_index --rebuild`
  vẫn hoạt động.
- Khuyến nghị dùng `python -m backend.index --rebuild` cho thống nhất.
"""
from pathlib import Path
import sys

# Fix import khi chạy trực tiếp: py build_index.py trong E:\test\backend
if __package__ in (None, ""):
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from backend.index import *  # noqa: F401,F403
from backend.index import build, main, build_parser  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(main())
