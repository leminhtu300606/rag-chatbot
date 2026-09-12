"""
backend/index.py - Entry point cho indexing
============================================
Nhiệm vụ:
- Xây dựng và cập nhật vector store từ data/processed (parquet).
- Điều phối các module con:
  * indexing/embedder.py    -> tạo embedding
  * indexing/vectorstore.py -> quản lý collection ChromaDB
  * indexing/retriever.py   -> truy xuất
- Hỗ trợ build incremental dựa trên manifest + mtime và chế độ --rebuild để embed lại toàn bộ.

Chạy:
  python -m backend.index              # build incremental
  python -m backend.index --rebuild    # build lại toàn bộ
  python backend/index.py --rebuild
"""
import json
import os
import sys
import argparse
from pathlib import Path
from collections import Counter

# Fix import khi chạy trực tiếp
if __package__ in (None, ""):
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

import backend.config as cfg

# Xuất lại các hàm từ các module indexing để dùng qua backend.index
# embedder
from backend.indexing.embedder import embed, get_model_info, get_dimension
# vectorstore
from backend.indexing.vectorstore import (
    get_client,
    get_collection,
    add_chunks,
    count,
    reset_collection,
    peek,
    get_stats,
    get_by_ids,
    query_by_category,
)
# retriever
from backend.indexing.retriever import retrieve

# Đảm bảo stdout UTF-8 trên Windows
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# Logic build vector store
def _load_manifest() -> dict:
    manifest = cfg.INDEXED_MANIFEST
    if manifest.exists():
        try:
            return json.loads(manifest.read_text(encoding="utf-8"))
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
        mtime = Path(src).stat().st_mtime if Path(src).exists() else 0.0
        seen[src] = mtime
        if not rebuild and src in manifest and manifest[src] == mtime:
            continue
        to_embed.append(ch)

    if not to_embed:
        _save_manifest(seen)
        print("Không có dữ liệu mới. Số bản ghi hiện tại:", count())
        try:
            stats = get_stats()
            print(f"  Stats ChromaDB: {stats}")
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
        print("[build] --rebuild: xóa collection cũ...")
        reset_collection()

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
                # Nếu là dict/list thì json dump để Chroma metadata chỉ nhận str/int/float/bool
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
    print(f"  Đã lưu {len(items)} chunks + embeddings (dim={emb_dim}) vào ChromaDB collection '{cfg.COLLECTION_NAME}' tại {cfg.CHROMA_DIR}")
    try:
        print(f"  Stats sau cập nhật: {get_stats()}")
    except Exception:
        pass


def build_parser():
    p = argparse.ArgumentParser(
        prog="backend.index",
        description="Indexing - build vector store từ data/processed (thay thế indexing/build_index.py)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Ví dụ:\n"
            "  python -m backend.index --rebuild\n"
            "  python -m backend.index\n"
            "  python backend/index.py --rebuild\n"
        ),
    )
    p.add_argument("--rebuild", action="store_true", help="Xóa collection cũ và embed lại toàn bộ")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    build(rebuild=args.rebuild)
    return 0


__all__ = [
    # embedder
    "embed", "get_model_info", "get_dimension",
    # vectorstore
    "get_client", "get_collection", "add_chunks", "count", "reset_collection",
    "peek", "get_stats", "get_by_ids", "query_by_category",
    # retriever
    "retrieve",
    # build
    "build", "main", "build_parser",
]

if __name__ == "__main__":
    raise SystemExit(main())
