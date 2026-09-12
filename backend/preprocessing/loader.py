"""Tải tài liệu từ thư mục data/classified - Enhanced Multimodal Aware.

Hỗ trợ:
- DOCX: đọc đoạn văn và bảng có cấu trúc (headers/rows) + đọc thứ tự.
- PDF: trích text bằng pymupdf với layout blocks (reading order) + bảng + hình ảnh metadata;
        nếu trang có quá ít text (scan) thì OCR bằng easyocr (tiếng Việt) có bbox để giữ layout.
Mỗi phần trả về một dict chứa text cùng metadata nguồn + layout info (bbox, block_type, table_data, image_info).

Tương thích ngược: các field cũ (text, page, kind) vẫn giữ, thêm field mới tùy chọn.
CPU-friendly: chỉ dùng pymupdf + easyocr + stdlib, không yêu cầu GPU.
"""
import os
import re
import json
from pathlib import Path

import pymupdf as fitz
from docx import Document

from backend.config import DATA_DIR

try:
    import backend.config as cfg
    LAYOUT_CFG = getattr(cfg, "LAYOUT", {})
    CHUNK_CFG = getattr(cfg, "CHUNK", {})
except Exception:
    LAYOUT_CFG = {}
    CHUNK_CFG = {}

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                OCR_READER = easyocr.Reader(["vi"], gpu=use_gpu, verbose=False)
            except TypeError:
                import io, contextlib
                f = io.StringIO()
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    OCR_READER = easyocr.Reader(["vi"], gpu=use_gpu)
    return OCR_READER


def _should_use_blocks() -> bool:
    try:
        return bool(LAYOUT_CFG.get("use_blocks", True))
    except Exception:
        return True


# ── DOCX enhanced ──
def load_docx(path: Path) -> list[dict]:
    doc = Document(str(path))
    parts = []
    # Đoạn văn theo thứ tự xuất hiện (paragraphs đã giữ order)
    for p in doc.paragraphs:
        if p.text.strip():
            parts.append(p.text.strip())
    # Bảng có cấu trúc: lưu cả dạng pipe và structured
    tables_data = []
    for ti, table in enumerate(doc.tables):
        headers = []
        rows = []
        # Lấy dòng đầu làm headers nếu có style header, nếu không thì dòng đầu
        for ri, row in enumerate(table.rows):
            cells = [c.text.strip() for c in row.cells]
            if not any(cells):
                continue
            if ri == 0:
                headers = cells
                # Lưu dòng header cũng như row
                rows.append(cells)
                parts.append(" | ".join(c for c in cells if c))
            else:
                rows.append(cells)
                parts.append(" | ".join(c for c in cells if c))
        if headers or rows:
            tables_data.append({
                "table_index": ti,
                "headers": headers,
                "rows": rows,
                "bbox": None,
                "page": 0,
            })
    text = "\n".join(parts)
    rec = {"text": text, "page": 0, "kind": "docx", "block_type": "text"}
    if tables_data:
        rec["tables"] = tables_data
        rec["has_table"] = True
    return [rec]


# ── PDF helpers: blocks + tables + images ──

def _pdf_page_blocks_sorted(page: fitz.Page) -> str:
    """Lấy text theo blocks sorted theo reading order (y rồi x), hỗ trợ 2 cột đơn giản."""
    try:
        blocks = page.get_text("blocks")  # (x0,y0,x1,y1, text, block_no, block_type)
    except Exception:
        return page.get_text("text").strip()
    if not blocks:
        return ""
    # Lọc block_type 0 = text (1 = image)
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
    if not text_blocks:
        return ""
    # Sắp xếp theo y rồi x (reading order)
    # Thử phát hiện 2 cột: nếu có gap lớn theo x
    use_columns = LAYOUT_CFG.get("detect_columns", True) if isinstance(LAYOUT_CFG, dict) else True
    if use_columns and len(text_blocks) > 4:
        # Kiểm tra bimodal theo x0
        xs = sorted([b[0] for b in text_blocks])
        # Median split
        mid = xs[len(xs)//2]
        left = [b for b in text_blocks if b[0] < mid]
        right = [b for b in text_blocks if b[0] >= mid]
        # Nếu cả 2 bên đều có block và gap > 80px và y overlap lớn thì là 2 cột
        if left and right:
            gap = min(b[0] for b in right) - max(b[2] for b in left) if left and right else 0
            # Tính y overlap
            page_w = page.rect.width
            if gap > page_w * 0.15 and len(left) >= 2 and len(right) >= 2:
                # Sắp xếp từng cột riêng rồi ghép: cột trái trước (A1,A2,A3), rồi cột phải (B1,B2,B3)
                # Nhưng thứ tự đúng cho 2 cột là đọc hết cột trái rồi sang cột phải (như giấy A4 2 cột)
                # Nên sort left và right theo y riêng
                left_sorted = sorted(left, key=lambda b: (round(b[1]/10), b[0]))
                right_sorted = sorted(right, key=lambda b: (round(b[1]/10), b[0]))
                ordered = left_sorted + right_sorted
                return "\n".join(b[4].strip() for b in ordered).strip()
    # Mặc định: sort theo y (làm tròn 10px để tránh lệch 1-2px) rồi x
    text_blocks_sorted = sorted(text_blocks, key=lambda b: (round(b[1]/10), b[0]))
    return "\n".join(b[4].strip() for b in text_blocks_sorted).strip()


def _pdf_extract_tables(page: fitz.Page) -> list[dict]:
    """Trích bảng bằng pymupdf find_tables, trả về list dict {headers, rows, bbox, text}."""
    tables = []
    # LAYOUT table_mode
    mode = LAYOUT_CFG.get("table_mode", "auto") if isinstance(LAYOUT_CFG, dict) else "auto"
    if mode == "off":
        return tables
    try:
        # find_tables có từ pymupdf 1.23+
        tabs = page.find_tables()
        if tabs is None:
            return tables
        for ti, tab in enumerate(tabs):
            try:
                # tab.extract() trả về list rows (list[str])
                data = tab.extract()
                if not data or len(data) < 1:
                    continue
                headers = [str(c).strip() if c is not None else "" for c in data[0]]
                rows = []
                for r in data:
                    rows.append([str(c).strip() if c is not None else "" for c in r])
                # bbox
                bbox = list(tab.bbox) if hasattr(tab, "bbox") else None
                # Tạo text dạng pipe cho embedding
                lines = []
                for r in rows:
                    line = " | ".join(c for c in r if c)
                    if line:
                        lines.append(line)
                table_text = "\n".join(lines)
                if table_text.strip():
                    tables.append({
                        "table_index": ti,
                        "headers": headers,
                        "rows": rows,
                        "bbox": bbox,
                        "text": table_text,
                        "page": page.number + 1,
                    })
            except Exception:
                continue
    except Exception:
        # fallback: không có find_tables hoặc lỗi
        pass
    return tables


def _pdf_extract_images(page: fitz.Page, doc) -> list[dict]:
    """Trích metadata hình ảnh: bbox, width, height. Không lưu file nếu image_mode=metadata."""
    images = []
    mode = LAYOUT_CFG.get("image_mode", "metadata") if isinstance(LAYOUT_CFG, dict) else "metadata"
    if mode == "off":
        return images
    try:
        # get_images trả về list tuples, mỗi entry có xref
        img_list = page.get_images(full=True)
        for idx, img in enumerate(img_list):
            try:
                xref = img[0]
                # Lấy bbox của image trên page
                # Có thể có nhiều bbox nếu image lặp
                try:
                    # Tìm bbox qua get_image_info (pymupdf 1.24+) hoặc get_drawings
                    bboxes = []
                    # Thử dùng page.get_image_bbox
                    if hasattr(page, "get_image_bbox"):
                        bbox = page.get_image_bbox(img)
                        if bbox:
                            bboxes = [list(bbox)]
                    if not bboxes:
                        # Fallback: tìm trong drawings/images
                        bboxes = [None]
                    bbox = bboxes[0] if bboxes else None
                except Exception:
                    bbox = None
                # Lấy size gốc
                try:
                    pix = fitz.Pixmap(doc, xref)
                    w, h = pix.width, pix.height
                    # Không save nếu metadata only
                    pix = None
                except Exception:
                    w, h = 0, 0
                images.append({
                    "image_index": idx,
                    "xref": xref,
                    "bbox": bbox,
                    "width": w,
                    "height": h,
                    "page": page.number + 1,
                })
            except Exception:
                continue
    except Exception:
        pass
    return images


def _pdf_page_text(page: fitz.Page) -> str:
    if _should_use_blocks():
        txt = _pdf_page_blocks_sorted(page)
        if txt and len(txt.strip()) >= 10:
            return txt.strip()
    return page.get_text("text").strip()


def _pdf_page_ocr(page: fitz.Page) -> str:
    import numpy as np
    # Dùng detail=1 nếu LAYOUT ocr_detail=1 để giữ bbox, nhưng detail=1 chậm hơn -> mặc định dùng 0 cho tốc độ
    # Chỉ dùng detail=1 khi cần giữ layout chính xác, còn lại dùng paragraph nhanh
    detail_cfg = LAYOUT_CFG.get("ocr_detail", 0) if isinstance(LAYOUT_CFG, dict) else 0
    detail = 1 if detail_cfg == 1 else 0
    # Giảm dpi xuống 150 cho scan để tăng tốc (200 -> 150 giảm ~40% pixels)
    dpi = 150 if detail == 0 else 180
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, pix.n)[:, :, :3]
    reader = _get_ocr()
    if detail == 1:
        # easyocr detail=1 trả về [bbox, text, confidence]
        try:
            results = reader.readtext(img, detail=1, paragraph=False)
            # Sắp xếp theo y rồi x giống blocks
            # results: list of (bbox, text, prob)
            # bbox là 4 điểm [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            items = []
            for bbox, text, prob in results:
                if not text or not text.strip():
                    continue
                # Lấy y trung tâm
                try:
                    ys = [p[1] for p in bbox]
                    xs = [p[0] for p in bbox]
                    y0 = min(ys)
                    x0 = min(xs)
                except Exception:
                    y0, x0 = 0, 0
                items.append((y0, x0, text.strip()))
            items.sort(key=lambda x: (round(x[0]/10), x[1]))
            return "\n".join(t for _, _, t in items).strip()
        except Exception:
            # fallback paragraph
            results = reader.readtext(img, detail=0, paragraph=True)
            return "\n".join(results).strip()
    else:
        results = reader.readtext(img, detail=0, paragraph=True)
        return "\n".join(results).strip()


def load_pdf(path: Path) -> list[dict]:
    doc = fitz.open(str(path))
    pages: list[dict] = []
    enable_table = CHUNK_CFG.get("enable_table_struct", True) if isinstance(CHUNK_CFG, dict) else True
    for i, page in enumerate(doc):
        text = _pdf_page_text(page)
        is_scanned = False
        if len(text) < PDF_TEXT_MIN:
            try:
                text = _pdf_page_ocr(page)
                is_scanned = True
            except Exception as e:
                print(f"[loader] OCR lỗi trang {i+1} của {path.name}: {e}")
        if not text:
            continue
        # Trích bảng và hình - chỉ cho trang không phải scan để tiết kiệm CPU (scan đã là ảnh)
        tables = []
        images = []
        if not is_scanned:
            if enable_table:
                tables = _pdf_extract_tables(page)
            try:
                images = _pdf_extract_images(page, doc)
            except Exception:
                images = []
        # Tạo record chính
        main_rec = {
            "text": text,
            "page": i + 1,
            "kind": "pdf_scan" if is_scanned else "pdf",
            "block_type": "text",
            "bbox": None,
        }
        if tables:
            # Thêm text bảng vào main text để retriever vẫn tìm được nếu không có structured chunk
            table_texts = "\n".join(t["text"] for t in tables if t.get("text"))
            if table_texts and table_texts not in text:
                main_rec["text"] = text + "\n\n[BẢNG]\n" + table_texts
            main_rec["tables"] = tables
            main_rec["has_table"] = True
        if images:
            main_rec["images"] = images
            main_rec["has_image"] = True
            # Thêm placeholder caption nếu chưa có text mô tả
            # Không làm phình text, chỉ thêm marker để chunker biết có figure
            if len(images) <= 3:  # tránh spam khi scan full page image
                main_rec["text"] += f"\n[Có {len(images)} hình ảnh minh họa trang {i+1}]"
        pages.append(main_rec)
        # Nếu có bảng và enable_table_struct, tạo thêm record riêng cho mỗi bảng để chunker tách riêng
        if enable_table and tables:
            for t in tables:
                # Chỉ tạo separate chunk nếu bảng đủ lớn (>=2 hàng)
                if len(t.get("rows", [])) >= 2 and len(t.get("text","")) >= 30:
                    pages.append({
                        "text": f"Bảng trang {i+1}:\n{t['text']}",
                        "page": i + 1,
                        "kind": "pdf_table",
                        "block_type": "table",
                        "bbox": t.get("bbox"),
                        "table_data": t,
                        "images": None,
                    })
        # Hình riêng: nếu có ít hình và không phải scan full page, tạo record figure
        # Cho phép tới 5 hình để bắt cả trường hợp thong_bao có 3 hình (đồng hồ + logo)
        if images and not is_scanned and len(images) <= 5:
            for im in images[:2]:  # lấy tối đa 2 hình tiêu biểu để tránh spam
                # Bỏ qua hình quá nhỏ (logo header) hoặc quá lớn (full page scan)
                if im.get("width", 0) < 100 or im.get("height", 0) < 100:
                    continue
                if im.get("width", 0) > 1600 and im.get("height", 0) > 2300:
                    continue  # full page scan
                pages.append({
                    "text": f"Hình minh họa trang {i+1} (bbox {im.get('bbox')})",
                    "page": i + 1,
                    "kind": "pdf_figure",
                    "block_type": "figure",
                    "bbox": im.get("bbox"),
                    "image_info": im,
                })
    doc.close()
    return pages


def load_txt(path: Path) -> list[dict]:
    """Đọc file TXT/MD với fallback encoding."""
    text = ""
    for enc in ("utf-8", "utf-8-sig", "cp1258", "windows-1258", "iso-8859-1"):
        try:
            text = Path(path).read_text(encoding=enc)
            if text and text.strip():
                break
        except Exception:
            continue
    if not text or not text.strip():
        try:
            # fallback binary detect
            raw = Path(path).read_bytes()
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            text = ""
    if not text.strip():
        return []
    # Tách theo trang logic: mỗi ~3000 chars coi như 1 page để giữ metadata
    # Giữ nguyên text gốc để chunker xử lý semantic
    return [{"text": text.strip(), "page": 0, "kind": "txt", "block_type": "text"}]


def load_file(path: Path) -> list[dict]:
    ext = Path(path).suffix.lower()
    if ext == ".docx":
        return load_docx(path)
    if ext == ".pdf":
        return load_pdf(path)
    if ext in (".txt", ".md", ".csv", ".log"):
        return load_txt(path)
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
