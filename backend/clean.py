"""
backend/clean.py - Entry point lam sach (chi goi preprocessing)
=============================================================
Khong xu ly logic truc tiep, chi goi den thu muc preprocessing:
  - preprocessing/clean_service.py (clean_single_file, clean_all, show_stats)
  - preprocessing/loader.py, cleaner.py, chunker.py, pipeline.py

Chay: python backend/clean.py | python -m backend.clean
"""
import os
import warnings
import logging
# Tat warning truoc khi import bat ky thu vien nao - chi hien thi tien trinh
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from pathlib import Path
import sys
if __package__ in (None, ""):
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Chi goi preprocessing - khong xu ly logic tai day
from backend.preprocessing.clean_service import (
    build_parser,
    clean_all,
    clean_single_file,
    main,
    show_stats,
)

# Xuất lại các hàm để có thể import qua backend.clean
__all__ = ["clean_single_file", "clean_all", "show_stats", "build_parser", "main"]

if __name__ == "__main__":
    raise SystemExit(main())

