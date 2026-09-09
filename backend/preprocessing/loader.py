"""Tải tài liệu từ thư mục data/classified.

Hỗ trợ:
- DOCX: đọc đoạn văn và bảng.
- PDF: trích text bằng pymupdf; nếu trang có quá ít text (nhiều khả năng là
  bản quét) thì tự động chạy OCR bằng easyocr (tiếng Việt).
Mỗi phần trả về một dict chứa text cùng metadata nguồn.
"""
import os
from pathlib import Path

import pymupdf as fitz
from docx import Document

from backend.config import DATA_DIR

PDF_TEXT_MIN = 50
OCR_READER = None


def _detect_ocr_gpu() -> bool:
    try:
        import backend.config as cfg
        if hasattr(cfg, "OCR_USE_GPU"):
            return bool(cfg.OCR_USE_GPU)
    except Exception:
        pass
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def _get_ocr():
    global OCR_READER
    if OCR_READER is None:
        import easyocr
        import warnings
        use_gpu = _detect_ocr_gpu()
        # An thong bao "Using CPU. Note: This module is much faster with a GPU."
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # easyocr verbose=False de khong in Using CPU
            try:
                OCR_READER = easyocr.Reader(["vi"], gpu=use_gpu, verbose=False)
            except TypeError:
                # fallback cho version cu khong co verbose
                import io, contextlib
                f = io.StringIO()
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    OCR_READER = easyocr.Reader(["vi"], gpu=use_gpu)
    return OCR_READER


def load_docx(path: Path) -> list[dict]:
    doc = Document(str(path))
    parts = []
    for p in doc.paragraphs:
        if p.text.strip():
            parts.append(p.text.strip())
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            line = " | ".join(c for c in cells if c)
            if line:
                parts.append(line)
    text = "\n".join(parts)
    return [{"text": text, "page": 0, "kind": "docx"}]


def _pdf_page_text(page: fitz.Page) -> str:
    return page.get_text("text").strip()


def _pdf_page_ocr(page: fitz.Page) -> str:
    import numpy as np
    pix = page.get_pixmap(dpi=200)
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, pix.n)[:, :, :3]
    reader = _get_ocr()
    results = reader.readtext(img, detail=0, paragraph=True)
    return "\n".join(results).strip()


def load_pdf(path: Path) -> list[dict]:
    doc = fitz.open(str(path))
    pages = []
    for i, page in enumerate(doc):
        text = _pdf_page_text(page)
        if len(text) < PDF_TEXT_MIN:
            try:
                text = _pdf_page_ocr(page)
            except Exception as e:
                print(f"[loader] OCR lỗi trang {i+1} của {path.name}: {e}")
        if text:
            pages.append({"text": text, "page": i + 1, "kind": "pdf"})
    doc.close()
    return pages


def load_file(path: Path) -> list[dict]:
    ext = Path(path).suffix.lower()
    if ext == ".docx":
        return load_docx(path)
    if ext == ".pdf":
        return load_pdf(path)
    return []


def load_all(data_dir: Path = DATA_DIR) -> list[dict]:
    records = []
    for root, _, files in os.walk(data_dir):
        rel = Path(root).relative_to(data_dir)
        category = rel.parts[0] if rel.parts else "unknown"
        subcategory = "/".join(rel.parts[1:]) if len(rel.parts) > 1 else ""
        for fn in files:
            if fn.startswith("~$"):
                continue
            path = Path(root) / fn
            try:
                for rec in load_file(path):
                    rec.update(
                        category=category,
                        subcategory=subcategory,
                        filename=fn,
                        source=str(path),
                    )
                    records.append(rec)
            except Exception as e:
                print(f"[loader] LỖI {path}: {e}")
    return records
