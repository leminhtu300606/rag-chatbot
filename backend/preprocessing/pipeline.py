"""Tiền xử lý: load -> clean -> semantic chunk, lưu artifact ra data/processed.

Mỗi file nguồn sinh một file Parquet tại data/processed/<category>/<sub>/<ten>.parquet,
mỗi dòng là một chunk đã làm sạch kèm metadata. Đây là "data tốt nhất" dùng cho index.
"""
import os
import sys
import warnings
import logging
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
from pathlib import Path

import backend.config as cfg

# Đảm bảo stdout hỗ trợ UTF-8 (tránh lỗi cp1252 trên Windows)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from backend.preprocessing.loader import load_file
from backend.preprocessing.cleaner import clean_text
from backend.preprocessing.chunker import contextual_chunk, semantic_chunk, _effective_min_chars
from backend.indexing.embedder import embed


def _out_path(source: Path) -> Path:
    try:
        rel = source.resolve().relative_to(cfg.DATA_DIR.resolve())
    except ValueError:
        rel = Path("_external") / source.name
    ext = getattr(cfg, "PROCESSED_EXT", ".parquet")
    return cfg.PROCESSED_DIR / rel.with_suffix(ext)


def process_file(path) -> int:
    path = Path(path)
    records = load_file(path)
    # Bổ sung metadata dựa trên đường dẫn tương đối so với DATA_DIR (như loader.load_all)
    try:
        rel = path.resolve().relative_to(cfg.DATA_DIR.resolve())
        if len(rel.parts) > 1:
            category = rel.parts[0]
            if len(rel.parts) > 2:
                subcategory = "/".join(rel.parts[1:-1])
            else:
                subcategory = ""
        else:
            category = "unknown"
            subcategory = ""
        filename = path.name
        source = str(path)
    except ValueError:
        category = "unknown"
        subcategory = ""
        filename = path.name
        source = str(path)

    for rec in records:
        # Gắn metadata nếu loader chưa gắn (load_file trả về chỉ text/page/kind)
        rec.setdefault("category", category)
        rec.setdefault("subcategory", subcategory)
        rec.setdefault("filename", filename)
        rec.setdefault("source", source)

    # Chọn chunker theo config (contextual vs semantic)
    use_contextual = cfg.CHUNK.get("contextual", True)
    chunk_fn = contextual_chunk if use_contextual else semantic_chunk

    out_chunks = []
    # Parent context tracking cho multimodal chunking
    parent_text_by_page: dict[int, str] = {}
    # Lưu parent đầu tiên để dùng cho table/figure
    for rec in records:
        if rec.get("block_type") == "text" or rec.get("kind") in ("pdf", "pdf_scan", "docx"):
            # Chỉ lưu parent cho text chính, clean nhẹ để làm context
            pt = clean_text(rec.get("text", ""))[: cfg.CHUNK.get("parent_chars", 600) if isinstance(cfg.CHUNK, dict) else 600]
            if rec.get("page") not in parent_text_by_page and pt:
                parent_text_by_page[rec["page"]] = pt
    # Fallback nếu không có parent
    global_parent = ""
    if parent_text_by_page:
        try:
            global_parent = next(iter(parent_text_by_page.values()))
        except Exception:
            global_parent = ""

    for rec in records:
        block_type = rec.get("block_type", "text")
        kind = rec.get("kind", "pdf")
        # Xác định chunk_type cho metadata
        if kind == "pdf_table" or block_type == "table":
            chunk_type = "table"
        elif kind == "pdf_figure" or block_type == "figure":
            chunk_type = "figure"
        else:
            chunk_type = "text"

        # ── Xử lý bảng có cấu trúc ──
        if chunk_type == "table" and cfg.CHUNK.get("enable_table_struct", True):
            table_data = rec.get("table_data") or rec.get("tables", [None])[0] if isinstance(rec.get("tables"), list) else None
            # Nếu không có table_data structured thì fallback dùng text
            raw_text = rec.get("text", "")
            if table_data and isinstance(table_data, dict) and table_data.get("rows"):
                # Xây markdown table có header
                headers = table_data.get("headers") or []
                rows = table_data.get("rows") or []
                # Tạo text dạng structured markdown
                md_lines = []
                if headers:
                    md_lines.append(" | ".join(headers))
                    md_lines.append(" | ".join(["---"] * len(headers)))
                for r in rows[1 if headers and rows and rows[0]==headers else 0:]:
                    md_lines.append(" | ".join(str(c) for c in r))
                table_md = "\n".join(md_lines)
                # Parent context nếu bật
                parent_ctx = ""
                if cfg.CHUNK.get("enable_parent_child", True):
                    parent_ctx = parent_text_by_page.get(rec.get("page"), global_parent)[: cfg.CHUNK.get("parent_chars", 600)]
                    if parent_ctx:
                        raw_text = f"[Bối cảnh trang {rec.get('page')}]\n{parent_ctx}\n\n[BẢNG - {rec.get('filename')} trang {rec.get('page')}]\n{table_md}"
                    else:
                        raw_text = f"[BẢNG trang {rec.get('page')}]\n{table_md}"
                else:
                    raw_text = f"[BẢNG]\n{table_md}"
            # Bảng thường là 1 chunk duy nhất, không semantic split (giữ nguyên cấu trúc)
            text = clean_text(raw_text) if block_type != "table" else raw_text.strip()
            if not text:
                continue
            # Nếu bảng quá dài (> max_chars) thì tách theo dòng, không dùng semantic
            max_c = cfg.CHUNK.get("max_chars", 1200)
            if len(text) > max_c:
                # Tách theo dòng bảng
                lines = text.splitlines()
                cur = ""
                pieces = []
                for line in lines:
                    if len(cur) + len(line) + 1 <= max_c:
                        cur = (cur + "\n" + line).strip() if cur else line
                    else:
                        if cur:
                            pieces.append(cur)
                        cur = line
                if cur:
                    pieces.append(cur)
            else:
                pieces = [text]
            # Lưu metadata bảng
            for i, piece in enumerate(pieces):
                _eff_min = _effective_min_chars(None)
                if len(piece.strip()) < _eff_min:
                    # Với bảng, nới lỏng ngưỡng xuống 20 để không mất
                    if len(piece.strip()) < 20:
                        continue
                out_chunks.append(
                    {
                        "id": f"{rec['filename']}-{rec['page']}-table-{i}",
                        "text": piece,
                        "category": rec["category"],
                        "subcategory": rec["subcategory"],
                        "filename": rec["filename"],
                        "source": rec["source"],
                        "page": rec["page"],
                        "chunk_index": len(out_chunks),
                        "chunk_mode": "table_struct",
                        "chunk_type": "table",
                        "block_type": block_type,
                        "parent_context": parent_text_by_page.get(rec.get("page"), "")[:200],
                        "bbox": rec.get("bbox"),
                        "has_table": True,
                    }
                )
            continue

        # ── Xử lý hình ảnh/figure ──
        if chunk_type == "figure":
            raw_text = rec.get("text", "")
            # Parent context cho figure để LLM hiểu ngữ cảnh xung quanh
            if cfg.CHUNK.get("enable_parent_child", True):
                parent_ctx = parent_text_by_page.get(rec.get("page"), global_parent)[: cfg.CHUNK.get("parent_chars", 600)]
                if parent_ctx:
                    raw_text = f"[Bối cảnh trang {rec.get('page')}]\n{parent_ctx}\n\n[HÌNH ẢNH - {rec.get('filename')} trang {rec.get('page')} bbox {rec.get('bbox')}]\n{raw_text}"
            # Nếu bật vision description sau này sẽ enrich thêm, hiện chỉ là placeholder
            # Với figure, giữ nguyên 1 chunk
            text = raw_text.strip()
            if not text or len(text) < 20:
                continue
            out_chunks.append(
                {
                    "id": f"{rec['filename']}-{rec['page']}-figure-{len(out_chunks)}",
                    "text": text,
                    "category": rec["category"],
                    "subcategory": rec["subcategory"],
                    "filename": rec["filename"],
                    "source": rec["source"],
                    "page": rec["page"],
                    "chunk_index": len(out_chunks),
                    "chunk_mode": "figure",
                    "chunk_type": "figure",
                    "block_type": block_type,
                    "parent_context": parent_text_by_page.get(rec.get("page"), "")[:200],
                    "bbox": rec.get("bbox"),
                    "has_image": True,
                }
            )
            continue

        # ── Xử lý text thường (paragraph, section) ──
        text = clean_text(rec["text"])
        if not text:
            continue
        # Truyền metadata để chunk giàu ngữ cảnh (heading, filename, category)
        meta_for_chunk = {
            "filename": rec["filename"],
            "category": rec["category"],
            "subcategory": rec["subcategory"],
            "source": rec["source"],
            "page": rec["page"],
        }
        if use_contextual:
            # contextual_chunk đã tự xử lý heading/overlap/llm
            pieces = chunk_fn(text, embed, metadata=meta_for_chunk)
        else:
            pieces = chunk_fn(text, embed)

        for i, piece in enumerate(pieces):
            # Đồng bộ với ngưỡng hiệu dụng của chunker (max(min_chars, 20), kẹp theo max_chars)
            _eff_min = _effective_min_chars(None)
            if len(piece.strip()) < _eff_min:
                continue
            out_chunks.append(
                {
                    "id": f"{rec['filename']}-{rec['page']}-{i}",
                    "text": piece,
                    "category": rec["category"],
                    "subcategory": rec["subcategory"],
                    "filename": rec["filename"],
                    "source": rec["source"],
                    "page": rec["page"],
                    "chunk_index": len(out_chunks),
                    "chunk_mode": "contextual" if use_contextual else "semantic",
                    "chunk_type": "text",
                    "block_type": block_type,
                    "parent_context": parent_text_by_page.get(rec["page"], "")[:100] if cfg.CHUNK.get("enable_parent_child") else None,
                    "bbox": rec.get("bbox"),
                }
            )
    # Dedup toàn file để loại trùng do nhiều page có câu lặp (học kỳ hè...)
    if out_chunks:
        try:
            from backend.utils.dedup import deduplicate_chunk_dicts
            before = len(out_chunks)
            out_chunks = deduplicate_chunk_dicts(
                out_chunks,
                threshold=cfg.CHUNK.get("dedup_threshold", 0.92),
                exact_only=cfg.CHUNK.get("dedup_exact_only", False),
            )
            if len(out_chunks) != before:
                print(f"[dedup] {path.name}: {before} -> {len(out_chunks)} chunks (loại {before-len(out_chunks)} trùng)")
            # Re-index chunk_index sau dedup để id không trùng
            for idx, ch in enumerate(out_chunks):
                ch["chunk_index"] = idx
                ch["id"] = f"{ch['filename']}-{ch['page']}-{idx}"
        except Exception as e:
            print(f"[dedup] cảnh báo {path.name}: {e}")
    out = _out_path(path)
    # Luu dang parquet thong qua storage helper (fallback jsonl neu thieu pyarrow)
    from backend.preprocessing.storage import save_chunks
    save_chunks(out, out_chunks)
    return len(out_chunks)


def process_all(data_dir: Path | None = None) -> int:
    if data_dir is None:
        data_dir = cfg.DATA_DIR
    else:
        data_dir = Path(data_dir)
    total = 0
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if fn.startswith("~$"):
                continue
            p = Path(root) / fn
            try:
                n = process_file(p)
                total += n
                print(f"[preprocess] {p.name}: {n} chunks")
            except Exception as e:
                print(f"[preprocess] LỖI {p}: {e}")
    return total
