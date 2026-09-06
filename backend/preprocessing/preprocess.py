"""
backend/preprocessing/preprocess.py - Bí danh cho pipeline
==========================================================
Cung cấp alias `preprocess` cho `pipeline` để import cũ vẫn hoạt động.
Khuyến nghị dùng `pipeline` cho code mới.
"""
from .pipeline import process_file, process_all, _out_path  # noqa: F401

__all__ = ["process_file", "process_all"]
