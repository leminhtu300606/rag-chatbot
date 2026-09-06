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
from backend.preprocessing.chunker import contextual_chunk, semantic_chunk
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
    for rec in records:
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
            if len(piece) < 50:
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
                    "chunk_index": i,
                    "chunk_mode": "contextual" if use_contextual else "semantic",
                }
            )
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
