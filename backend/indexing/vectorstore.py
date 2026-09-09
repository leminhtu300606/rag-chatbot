"""
backend/indexing/vectorstore.py - Lưu và truy vấn embedding bằng ChromaDB
=========================================================================
Nhiệm vụ:
- Quản lý ChromaDB persistent trên đĩa (tạo collection, upsert, query).
- Cung cấp hàm kiểm tra kết quả chunking và embedding đã lưu
  (peek, get_stats, get_by_ids, query_by_category) phục vụ chẩn đoán.
- Lưu metadata đầy đủ cho mỗi chunk: chunk_index, chunk_mode, text_length, embedding_model, ...
"""
import chromadb

import backend.config as cfg

_client = None
_client_path = None


def get_client():
    global _client, _client_path
    # Tu dong reset client neu CHROMA_DIR thay doi (huu ich cho test voi tmp dir)
    cur_path = str(cfg.CHROMA_DIR)
    if _client is None or _client_path != cur_path:
        _client = chromadb.PersistentClient(path=cur_path)
        _client_path = cur_path
    return _client


def get_collection():
    # Tuning HNSW cho tiếng Việt: tăng M và ef để cân bằng tốc độ/chính xác
    # Giữ tương thích: nếu collection đã tồn tại thì metadata cũ được giữ
    hnsw_meta = {"hnsw:space": "cosine", "hnsw:M": 16, "hnsw:construction_ef": 200, "hnsw:search_ef": 64}
    try:
        return get_client().get_or_create_collection(
            name=cfg.COLLECTION_NAME, metadata=hnsw_meta
        )
    except Exception:
        return get_client().get_or_create_collection(
            name=cfg.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )


def reset_collection():
    try:
        get_client().delete_collection(cfg.COLLECTION_NAME)
    except Exception:
        pass
    return get_collection()


def add_chunks(chunks: list[dict], batch_size: int = 500) -> None:
    """Upsert danh sach chunks + embeddings vao ChromaDB.

    Moi chunk dict can co:
      id: str
      text: str (document)
      embedding: list[float]
      metadata: dict (da bao gom chunking results: chunk_index, chunk_mode, text_length...)
    Tự chia nhỏ theo batch để tránh OOM khi nhiều chunk.
    """
    if not chunks:
        return
    col = get_collection()
    # Chia nhỏ để tránh quá tải RAM / Chroma
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        col.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )


def count() -> int:
    return get_collection().count()


# Tiện ích kiểm tra kết quả chunking và embedding

def peek(limit: int = 5) -> dict:
    """Lay mau `limit` ban ghi trong ChromaDB kem ca embedding.

    Tra ve dict co keys: ids, documents, metadatas, embeddings
    Dung de kiem tra chunking (text, metadata) va embedding (dim, gia tri)
    da luu dung chua.
    """
    col = get_collection()
    # include phai chi dinh ro de lay embeddings
    try:
        # chromadb >=0.4
        return col.get(limit=limit, include=["documents", "metadatas", "embeddings"])
    except TypeError:
        # fallback cho ban cu
        return col.get(limit=limit)


def get_stats() -> dict:
    """Thong ke nhanh ve ChromaDB hien tai."""
    col = get_collection()
    c = col.count()
    out = {"total": c, "collection": cfg.COLLECTION_NAME, "chroma_dir": str(cfg.CHROMA_DIR)}
    if c == 0:
        return out
    try:
        sample = col.get(limit=min(c, 5), include=["metadatas", "embeddings"])
        metas = sample.get("metadatas", [])
        embeds = sample.get("embeddings", [])
        # Thong ke theo category / chunk_mode
        from collections import Counter
        cat_cnt = Counter(m.get("category", "unknown") for m in metas if m)
        mode_cnt = Counter(m.get("chunk_mode", "unknown") for m in metas if m)
        out["category_count"] = dict(cat_cnt)
        out["chunk_mode_count"] = dict(mode_cnt)
        if embeds is not None and len(embeds) > 0 and embeds[0] is not None:
            out["embedding_dim"] = len(embeds[0])
            # preview 3 gia tri dau
            try:
                out["embedding_preview"] = [float(f"{x:.4f}") for x in embeds[0][:3]]
            except Exception:
                pass
            # model tu metadata
            models = set(m.get("embedding_model", "") for m in metas if m and m.get("embedding_model"))
            if models:
                out["embedding_model"] = list(models)[0]
        # Lay 1 metadata mau
        if metas:
            out["sample_metadata"] = metas[0]
    except Exception as e:
        out["error"] = str(e)
    return out


def get_by_ids(ids: list[str]) -> dict:
    """Lay cac chunk theo danh sach ids (de kiem tra sau khi add)."""
    col = get_collection()
    return col.get(ids=ids, include=["documents", "metadatas", "embeddings"])


def query_by_category(category: str, limit: int = 5) -> dict:
    """Lay mau theo category (de kiem tra phan bo du lieu)."""
    col = get_collection()
    return col.get(where={"category": category}, limit=limit, include=["documents", "metadatas"])

