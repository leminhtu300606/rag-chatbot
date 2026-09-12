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


def _embed_for_chunk(texts):
    """Nạp embedding model chỉ khi semantic chunk thực sự cần dùng."""
    from backend.indexing.embedder import embed
    return embed(texts)


def _out_path(source: Path) -> Path:
    try:
        rel = source.resolve().relative_to(cfg.DATA_DIR.resolve())
    except ValueError:
        rel = Path("_external") / source.name
    ext = getattr(cfg, "PROCESSED_EXT", ".parquet")
    return cfg.PROCESSED_DIR / rel.with_suffix(ext)


def _build_chunks_from_records(records: list[dict], filename: str, category: str, subcategory: str, source: str, session_id: str | None = None) -> list[dict]:
    """Tách chunk từ records đã load - dùng chung cho global và session upload."""
    from backend.preprocessing.cleaner import clean_text as _clean
    from backend.preprocessing.chunker import _effective_min_chars as _eff

    use_contextual = cfg.CHUNK.get("contextual", True)
    chunk_fn = contextual_chunk if use_contextual else semantic_chunk
    out_chunks: list[dict] = []
    parent_text_by_page: dict[int, str] = {}
    for rec in records:
        if rec.get("block_type") == "text" or rec.get("kind") in ("pdf", "pdf_scan", "docx", "txt"):
            pt = _clean(rec.get("text", ""))[: cfg.CHUNK.get("parent_chars", 600) if isinstance(cfg.CHUNK, dict) else 600]
            if rec.get("page") not in parent_text_by_page and pt:
                parent_text_by_page[rec["page"]] = pt
    global_parent = next(iter(parent_text_by_page.values())) if parent_text_by_page else ""
    for rec in records:
        block_type = rec.get("block_type", "text")
        kind = rec.get("kind", "pdf")
        if kind == "pdf_table" or block_type == "table":
            chunk_type = "table"
        elif kind == "pdf_figure" or block_type == "figure":
            chunk_type = "figure"
        else:
            chunk_type = "text"
        if chunk_type == "table" and cfg.CHUNK.get("enable_table_struct", True):
            table_data = rec.get("table_data")
            if not table_data and isinstance(rec.get("tables"), list) and rec.get("tables"):
                table_data = rec["tables"][0]
            raw_text = rec.get("text", "")
            if table_data and isinstance(table_data, dict) and table_data.get("rows"):
                headers = table_data.get("headers") or []
                rows = table_data.get("rows") or []
                md_lines = []
                if headers:
                    md_lines.append(" | ".join(headers))
                    md_lines.append(" | ".join(["---"] * len(headers)))
                for r in rows[1 if headers and rows and rows[0]==headers else 0:]:
                    md_lines.append(" | ".join(str(c) for c in r))
                table_md = "\n".join(md_lines)
                parent_ctx = ""
                if cfg.CHUNK.get("enable_parent_child", True):
                    parent_ctx = parent_text_by_page.get(rec.get("page"), global_parent)[: cfg.CHUNK.get("parent_chars", 600)]
                    if parent_ctx:
                        raw_text = f"[Bối cảnh trang {rec.get('page')}]\n{parent_ctx}\n\n[BẢNG - {rec.get('filename')} trang {rec.get('page')}]\n{table_md}"
                    else:
                        raw_text = f"[BẢNG trang {rec.get('page')}]\n{table_md}"
                else:
                    raw_text = f"[BẢNG]\n{table_md}"
            text = _clean(raw_text) if block_type != "table" else raw_text.strip()
            if not text:
                continue
            max_c = cfg.CHUNK.get("table_max_chars", cfg.CHUNK.get("max_chars", 1200))
            if len(text) > max_c:
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
            for i, piece in enumerate(pieces):
                _eff_min = _eff(None)
                if len(piece.strip()) < cfg.CHUNK.get("table_min_chars", 30):
                    continue
                chunk_id_base = f"{session_id}-{filename}" if session_id else filename
                out_chunks.append({
                    "id": f"{chunk_id_base}-{rec['page']}-table-{i}-{len(out_chunks)}",
                    "text": piece,
                    "category": rec.get("category", category),
                    "subcategory": rec.get("subcategory", subcategory),
                    "filename": rec.get("filename", filename),
                    "source": rec.get("source", source),
                    "page": rec["page"],
                    "chunk_index": len(out_chunks),
                    "chunk_mode": "table_struct",
                    "chunk_type": "table",
                    "block_type": block_type,
                    "parent_context": parent_text_by_page.get(rec.get("page"), "")[:200],
                    "bbox": rec.get("bbox"),
                    "table_data": table_data,
                    "has_table": True,
                    "session_id": session_id,
                })
            continue
        if chunk_type == "figure":
            raw_text = rec.get("text", "")
            if cfg.CHUNK.get("enable_parent_child", True):
                parent_ctx = parent_text_by_page.get(rec.get("page"), global_parent)[: cfg.CHUNK.get("parent_chars", 600)]
                if parent_ctx:
                    raw_text = f"[Bối cảnh trang {rec.get('page')}]\n{parent_ctx}\n\n[HÌNH ẢNH - {rec.get('filename')} trang {rec.get('page')} bbox {rec.get('bbox')}]\n{raw_text}"
            text = raw_text.strip()
            if not text or len(text) < cfg.CHUNK.get("figure_min_chars", 30):
                continue
            figure_max = cfg.CHUNK.get("figure_max_chars", 900)
            if len(text) > figure_max:
                text = text[:figure_max].rstrip()
            chunk_id_base = f"{session_id}-{filename}" if session_id else filename
            out_chunks.append({
                "id": f"{chunk_id_base}-{rec['page']}-figure-{len(out_chunks)}",
                "text": text,
                "category": rec.get("category", category),
                "subcategory": rec.get("subcategory", subcategory),
                "filename": rec.get("filename", filename),
                "source": rec.get("source", source),
                "page": rec["page"],
                "chunk_index": len(out_chunks),
                "chunk_mode": "figure",
                "chunk_type": "figure",
                "block_type": block_type,
                "parent_context": parent_text_by_page.get(rec.get("page"), "")[:200],
                "bbox": rec.get("bbox"),
                "has_image": True,
                "image_info": rec.get("image_info"),
                "session_id": session_id,
            })
            continue
        text = _clean(rec.get("text", ""))
        if not text:
            continue
        meta_for_chunk = {
            "filename": rec.get("filename", filename),
            "category": rec.get("category", category),
            "subcategory": rec.get("subcategory", subcategory),
            "source": rec.get("source", source),
            "page": rec["page"],
        }
        if use_contextual:
            pieces = chunk_fn(text, _embed_for_chunk, metadata=meta_for_chunk)
        else:
            pieces = chunk_fn(text, _embed_for_chunk)
        for i, piece in enumerate(pieces):
            _eff_min = _eff(None)
            if len(piece.strip()) < _eff_min:
                continue
            chunk_id_base = f"{session_id}-{filename}" if session_id else filename
            out_chunks.append({
                "id": f"{chunk_id_base}-{rec['page']}-{i}-{len(out_chunks)}",
                "text": piece,
                "category": rec.get("category", category),
                "subcategory": rec.get("subcategory", subcategory),
                "filename": rec.get("filename", filename),
                "source": rec.get("source", source),
                "page": rec["page"],
                "chunk_index": len(out_chunks),
                "chunk_mode": "contextual" if use_contextual else "semantic",
                "chunk_type": "text",
                "block_type": block_type,
                "parent_context": parent_text_by_page.get(rec.get("page"), "")[:100] if cfg.CHUNK.get("enable_parent_child") else None,
                "bbox": rec.get("bbox"),
                "session_id": session_id,
            })
    if out_chunks:
        try:
            from backend.utils.dedup import deduplicate_chunk_dicts
            before = len(out_chunks)
            out_chunks = deduplicate_chunk_dicts(out_chunks, threshold=cfg.CHUNK.get("dedup_threshold", 0.92), exact_only=cfg.CHUNK.get("dedup_exact_only", False))
            if len(out_chunks) != before:
                print(f"[dedup] {filename}: {before} -> {len(out_chunks)} chunks")
            for idx, ch in enumerate(out_chunks):
                ch["chunk_index"] = idx
                chunk_id_base = f"{session_id}-{filename}" if session_id else filename
                # giữ id ổn định sau dedup
                ch["id"] = f"{chunk_id_base}-{ch['page']}-{idx}"
        except Exception as e:
            print(f"[dedup] cảnh báo {filename}: {e}")
    # Fallback cho upload: nếu file ngắn bị lọc hết thì giữ lại 1 chunk duy nhất
    if not out_chunks and session_id and records:
        try:
            from backend.preprocessing.cleaner import clean_text as _ct2
            raw = "\n".join(r.get("text","") for r in records).strip()
            cleaned = _ct2(raw) if raw else raw
            if cleaned and len(cleaned.strip()) >= 10:
                # cắt 1200 nếu quá dài
                txt = cleaned[:1200] if len(cleaned) > 1200 else cleaned
                out_chunks = [{
                    "id": f"{session_id}-{filename}-0-0",
                    "text": txt,
                    "category": "uploaded",
                    "subcategory": "",
                    "filename": filename,
                    "source": source,
                    "page": 0,
                    "chunk_index": 0,
                    "chunk_mode": "uploaded_fallback",
                    "chunk_type": "text",
                    "block_type": "text",
                    "session_id": session_id,
                }]
        except Exception as e:
            print(f"[fallback] lỗi {filename}: {e}")
    return out_chunks


def process_file(path, session_id: str | None = None) -> int:
    """Xử lý 1 file -> parquet (global) hoặc trả chunk với session_id."""
    path = Path(path)
    records = load_file(path)
    if not records:
        return 0
    try:
        rel = path.resolve().relative_to(cfg.DATA_DIR.resolve())
        if len(rel.parts) > 1:
            category = rel.parts[0]
            subcategory = "/".join(rel.parts[1:-1]) if len(rel.parts) > 2 else ""
        else:
            category = "unknown"
            subcategory = ""
        filename = path.name
        source = str(path)
    except ValueError:
        category = "uploaded" if session_id else "unknown"
        subcategory = ""
        filename = path.name
        source = str(path)
    for rec in records:
        rec.setdefault("category", category)
        rec.setdefault("subcategory", subcategory)
        rec.setdefault("filename", filename)
        rec.setdefault("source", source)
    out_chunks = _build_chunks_from_records(records, filename, category, subcategory, source, session_id=session_id)
    if session_id:
        # Không ghi parquet chung, trả số chunks để caller tự embed/add
        return out_chunks  # type: ignore[return-value]
    out = _out_path(path)
    from backend.preprocessing.storage import save_chunks
    save_chunks(out, out_chunks)
    return len(out_chunks)


def process_file_to_chunks(path: Path, session_id: str, filename_override: str | None = None) -> list[dict]:
    """Dành cho upload theo phiên: trả về chunks với session_id, không ghi parquet."""
    path = Path(path)
    records = load_file(path)
    if not records:
        return []
    # Giới hạn số trang để hỗ trợ dung lượng lớn mà không OOM
    max_pages = getattr(cfg, "MAX_UPLOAD_PAGES", 500)
    if len(records) > max_pages:
        records = records[:max_pages]
    filename = filename_override or path.name
    source = str(path)
    # Đặt category uploaded để phân biệt
    for rec in records:
        rec.setdefault("category", "uploaded")
        rec.setdefault("subcategory", "")
        rec.setdefault("filename", filename)
        rec.setdefault("source", source)
    chunks = _build_chunks_from_records(records, filename, "uploaded", "", source, session_id=session_id)
    return chunks


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
