"""
PHẦN 1: TIỀN XỬ LÝ DỮ LIỆU
=========================
Xử lý dữ liệu thô (PDF, DOCX, TXT) thành dữ liệu sạch, đã chunk, sẵn sàng cho huấn luyện/indexing.

Luồng:  loader (PDF/DOCX + OCR) -> cleaner (làm sạch) -> chunker (semantic chunk) -> pipeline (artifact .parquet)

Artifact: data/processed/<category>/<sub>/<file>.parquet  (mỗi dòng là 1 chunk + metadata, luu dang parquet)
"""

from .loader import load_file, load_all, load_docx, load_pdf
from .cleaner import clean_text
from .chunker import (
    semantic_chunk,
    contextual_chunk,
    split_sentences,
    split_by_sections,
    chunk_text,
)
from .pipeline import process_file, process_all

__all__ = [
    "load_file",
    "load_all",
    "load_docx",
    "load_pdf",
    "clean_text",
    "semantic_chunk",
    "contextual_chunk",
    "split_sentences",
    "split_by_sections",
    "chunk_text",
    "process_file",
    "process_all",
]
