"""
backend/cli.py - Entry points CLI (gop tu cli/clean.py + cli/index.py)
=======================================================================
- clean: tien xu ly data tho -> data sach (logic trong preprocessing.pipeline)
- index: build vector store tu data/processed (parquet -> PostgreSQL/pgvector)

Chay:
  python -m backend.cli clean                 # lam sach toan bo data/classified
  python -m backend.cli clean --stats         # chi xem thong ke
  python -m backend.cli index                 # build incremental
  python -m backend.cli index --rebuild       # build lai toan bo
"""
import argparse
import json
import os
import sys
import warnings
import logging
from collections import Counter
from pathlib import Path

# Tat warning truoc khi import bat ky thu vien nao - chi hien thi tien trinh
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Fix import khi chay truc tiep
if __package__ in (None, ""):
    _ROOT = Path(__file__).resolve().parent.parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

# Dam bao stdout UTF-8 tren Windows
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import backend.config.settings as cfg

# ── clean: chi goi preprocessing.pipeline (khong xu ly logic tai day) ──
from backend.preprocessing.pipeline import (
    clean_all,
    clean_single_file,
    main as clean_main,
    show_stats,
)

# ── index: embedder / vectorstore / retrieval ──
from backend.indexing.embedder import embed
from backend.indexing.vectorstore import (
    add_chunks,
    count,
    reset_collection,
    get_stats,
)
from backend.indexing.retrieval import retrieve


# Logic build vector store
def _mkey(src: str) -> str:
    """Khoa manifest chuan posix (tuong thich manifest cu Windows backslash)."""
    try:
        return Path(str(src)).as_posix()
    except Exception:
        return str(src).replace("\\", "/")


def _msrc(src: str) -> Path:
    """Resolve source chunk thanh Path tuyet doi (ho tro ca relative nhu manifest cu)."""
    p = Path(str(src))
    if p.is_absolute():
        return p
    try:
        return cfg.BASE_DIR / p
    except Exception:
        return p


def _load_manifest() -> dict:
    manifest = cfg.INDEXED_MANIFEST
    if manifest.exists():
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            return {_mkey(k): v for k, v in raw.items()}
        except Exception:
            return {}
    return {}


def _save_manifest(m: dict) -> None:
    cfg.INDEXED_MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def _iter_chunks():
    # Đọc từ parquet (fallback jsonl cũ nếu chưa migrate)
    from backend.preprocessing.storage import iter_chunks as storage_iter
    yield from storage_iter()


def build(rebuild: bool = False) -> None:
    """Xây dựng/cập nhật vector store từ data/processed.

    Args:
        rebuild: True = xóa collection cũ và embed lại toàn bộ.
                 False = chỉ embed chunk của file mới/sửa (so mtime với manifest).
    """
    manifest = {} if rebuild else _load_manifest()

    seen = {}
    to_embed = []
    for ch in _iter_chunks():
        src = ch["source"]
        key = _mkey(src)
        try:
            mtime = _msrc(src).stat().st_mtime if _msrc(src).exists() else 0.0
        except Exception:
            mtime = 0.0
        seen[key] = mtime
        if not rebuild and key in manifest and manifest[key] == mtime:
            continue
        to_embed.append(ch)

    if not to_embed:
        _save_manifest(seen)
        print("Không có dữ liệu mới. Số bản ghi hiện tại:", count())
        try:
            stats = get_stats()
            print(f"  Stats vector store: {stats}")
        except Exception:
            pass
        return

    # Thống kê trước khi embed
    mode_cnt = Counter(c.get("chunk_mode", "unknown") for c in to_embed)
    cat_cnt = Counter(c.get("category", "unknown") for c in to_embed)
    print(f"Cần embed {len(to_embed)} chunks mới")
    print(f"  Phân bố chunk_mode: {dict(mode_cnt)}")
    print(f"  Phân bố category: {dict(cat_cnt)}")
    if to_embed:
        sample = to_embed[0]
        print(f"  Mẫu chunk[0] ({len(sample['text'])} chars): {sample['text'][:220]}...")
        print(f"  Mẫu metadata[0]: id={sample.get('id')} | file={sample.get('filename')} | cat={sample.get('category')} | page={sample.get('page')} | idx={sample.get('chunk_index')} | mode={sample.get('chunk_mode')}")

    texts = [c["text"] for c in to_embed]
    # Dùng batch size tối ưu theo phần cứng, tự chia nhỏ để tránh OOM
    try:
        batch = int(getattr(cfg, "EMBED_BATCH_SIZE", 32))
    except Exception:
        batch = 32
    embeddings = embed(texts, batch_size=batch)

    # Chỉ thay collection sau khi embed thành công, tránh --rebuild làm mất index cũ
    # khi model hoặc phần cứng embedding gặp lỗi giữa chừng.
    if rebuild:
        print("[build] --rebuild: xóa bảng vector cũ...")
        reset_collection(dim=len(embeddings[0]) if len(embeddings) > 0 else None)

    emb_dim = len(embeddings[0]) if len(embeddings) > 0 else 0
    print(f"  Embedding model: {cfg.EMBED_MODEL} | dim={emb_dim} | total={len(embeddings)}")
    if len(embeddings) > 0:
        try:
            preview = ", ".join(f"{x:.4f}" for x in embeddings[0][:5])
            print(f"  Embedding preview [0][:5]: [{preview} ...]")
        except Exception:
            pass

    items = []
    for c, e in zip(to_embed, embeddings):
        # Hỗ trợ metadata multimodal mới (chunk_type, has_table, bbox...)
        meta_extra = {}
        for k in ["chunk_type", "block_type", "has_table", "has_image", "parent_context", "bbox", "table_data", "image_info"]:
            if k in c and c[k] is not None:
                v = c[k]
                # Nếu là dict/list thì json dump (JSONB metadata chỉ nhận scalar, giới hạn độ dài)
                if isinstance(v, (dict, list)):
                    import json as _json
                    try:
                        v = _json.dumps(v, ensure_ascii=False)[:3000]
                    except Exception:
                        v = str(v)[:1000]
                meta_extra[k] = v
        items.append(
            {
                "id": c["id"],
                "text": c["text"],
                "embedding": e.tolist() if hasattr(e, "tolist") else list(e),
                "metadata": {
                    "category": c["category"],
                    "subcategory": c["subcategory"],
                    "filename": c["filename"],
                    "source": c["source"],
                    "page": c["page"],
                    "chunk_index": int(c.get("chunk_index", 0)),
                    "chunk_mode": c.get("chunk_mode", "unknown"),
                    "text_length": len(c["text"]),
                    "embedding_model": cfg.EMBED_MODEL,
                    "embedding_dim": int(emb_dim),
                    **meta_extra,
                },
            }
        )
    add_chunks(items)
    _save_manifest(seen)
    print("Đã cập nhật vector store. Tổng bản ghi:", count())
    print(f"  Đã lưu {len(items)} chunks + embeddings (dim={emb_dim}) vào pgvector bảng '{cfg.PGVECTOR_TABLE}'")
    try:
        print(f"  Stats sau cập nhật: {get_stats()}")
    except Exception:
        pass


def build_index_parser():
    p = argparse.ArgumentParser(
        prog="backend.cli index",
        description="Indexing - build vector store từ data/processed",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Ví dụ:\n"
            "  python -m backend.cli index --rebuild\n"
            "  python -m backend.cli index\n"
        ),
    )
    p.add_argument("--rebuild", action="store_true", help="Xóa collection cũ và embed lại toàn bộ")
    return p


def index_main(argv=None):
    parser = build_index_parser()
    args = parser.parse_args(argv)
    build(rebuild=args.rebuild)
    return 0


# ── Dispatcher: python -m backend.cli <clean|index> ──
# Subcommand clean co parser rieng phuc tap (positional path + options) nen dispatcher
# chi tach ten lenh, phan options do clean_main/index_main tu parse (giu 100% behavior cu).

def build_parser():
    p = argparse.ArgumentParser(
        prog="backend.cli",
        description="CLI RAG: clean (tien xu ly) | index (build vector store)",
    )
    p.add_argument("command", choices=["clean", "index"], help="clean: lam sach | index: build vector")
    return p


def main(argv=None):
    # Subcommand clean co parser rieng phuc tap (positional path + options) nen dispatch thu cong
    # de giu nguyen 100% behavior cu (khong mat option nao).
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print("Dung: python -m backend.cli <clean|index> [options]")
        print("  clean  : python -m backend.cli clean [--stats|--dry-run|--show-sample] [path]")
        print("  index  : python -m backend.cli index [--rebuild]")
        return 0
    cmd, rest = args[0], args[1:]
    if cmd == "clean":
        return clean_main(rest)
    if cmd == "index":
        return index_main(rest)
    print(f"[cli] lenh khong ro: {cmd} (dung clean|index)", file=sys.stderr)
    return 2


__all__ = [
    # clean (re-export tu pipeline)
    "clean_single_file", "clean_all", "show_stats", "clean_main",
    # embedder
    "embed",
    # vectorstore
    "add_chunks", "count", "reset_collection", "get_stats",
    # retrieval
    "retrieve",
    # build
    "build", "index_main", "build_index_parser", "main", "build_parser",
]


if __name__ == "__main__":
    raise SystemExit(main())
