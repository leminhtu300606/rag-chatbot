"""
backend/indexing/vectorstore.py - Lưu và truy vấn embedding bằng PostgreSQL/pgvector
====================================================================================
Nhiệm vụ:
- Quản lý bảng vector (tạo bảng, upsert, query cosine) trên Postgres có extension pgvector.
- Public API ổn định cho caller:
  add_chunks / count / reset_collection / get_stats /
  delete_by_session / delete_chunks / count_by_session / list_uploads_by_session
  + query_vector(embedding, where, n) cho retrieval (score = cosine distance 0~2).
- Metadata đầy đủ cho mỗi chunk: chunk_index, chunk_mode, text_length, embedding_model, ...
- Embedding đã L2-normalize ở embedder -> cosine distance <=> inner product.

Bảng rag_chunks (mặc định, xem settings.PGVECTOR_TABLE):
  id TEXT PK | text TEXT | embedding vector(dim) | metadata JSONB |
  category TEXT | session_id TEXT NULL | filename TEXT | created_at TIMESTAMPTZ
"""
import json as _json
import re as _re
from contextlib import contextmanager as _ctx

import backend.config.settings as cfg

try:
    import psycopg as _pg
except Exception as _pg_err:  # pragma: no cover
    _pg = None
    _pg_import_error = _pg_err
else:
    _pg_import_error = None

_TABLE_RE = _re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _table() -> str:
    name = getattr(cfg, "PGVECTOR_TABLE", "rag_chunks") or "rag_chunks"
    if not _TABLE_RE.match(name):
        raise ValueError(f"Tên bảng vector không hợp lệ: {name!r}")
    return name


def _dsn() -> str:
    dsn = getattr(cfg, "DATABASE_URL", "") or ""
    if not dsn:
        raise RuntimeError("Chưa cấu hình DATABASE_URL cho pgvector (xem .env.example)")
    return dsn


def _need_pg():
    if _pg is None:
        raise RuntimeError(
            f"Thiếu package psycopg (cần cho pgvector): {_pg_import_error}. "
            "Cài: pip install 'psycopg[binary]' pgvector"
        )


@_ctx
def _conn():
    """Connection mới mỗi lần gọi (tải <20 user, đơn giản, không giữ pool)."""
    _need_pg()
    conn = _pg.connect(_dsn(), connect_timeout=5, autocommit=True)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_client():
    """Tương thích tên cũ: trả về 1 connection psycopg mới (nhớ .close())."""
    _need_pg()
    return _pg.connect(_dsn(), connect_timeout=5, autocommit=True)


def _ensure_extension(conn) -> None:
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except Exception as e:
        raise RuntimeError(
            "Postgres chưa có extension pgvector. Dùng image pgvector/pgvector "
            f"(xem docker-compose.yml). Chi tiết: {e}"
        )


def _table_exists(conn, table: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_schema='public' AND table_name=%s",
        (table,),
    )
    return cur.fetchone() is not None


def _table_dim(conn, table: str):
    """Trả về dim của cột embedding, None nếu bảng chưa có."""
    if not _table_exists(conn, table):
        return None
    # Ưu tiên đo trên dòng thật
    try:
        cur = conn.execute(f"SELECT vector_dims(embedding) FROM {table} LIMIT 1")
        row = cur.fetchone()
        if row and row[0]:
            return int(row[0])
    except Exception:
        pass
    # Bảng rỗng: đọc kiểu cột (format_type trả "vector(768)")
    try:
        cur = conn.execute(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid=%s::regclass AND attname='embedding'",
            (table,),
        )
        row = cur.fetchone()
        if row and row[0]:
            m = _re.search(r"\((\d+)\)", str(row[0]))
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return None


def _create_table(conn, table: str, dim: int) -> None:
    _ensure_extension(conn)
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {table} (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            embedding vector({int(dim)}),
            metadata JSONB NOT NULL DEFAULT '{{}}',
            category TEXT,
            session_id TEXT,
            filename TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )"""
    )
    conn.execute(
        f"""CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw
            ON {table} USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 200)"""
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {table}_session_idx ON {table} (session_id)"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {table}_category_idx ON {table} (category)"
    )


def _ensure_table(conn, dim: int) -> None:
    table = _table()
    if not _table_exists(conn, table):
        _create_table(conn, table, dim)
        return
    cur_dim = _table_dim(conn, table)
    if cur_dim is not None and cur_dim != int(dim):
        raise RuntimeError(
            f"Lệch dimension embedding: bảng '{table}' đang dim={cur_dim}, "
            f"dữ liệu mới dim={dim} (đổi EMBED_MODEL?). Chạy: python -m backend.cli index --rebuild"
        )


# ── Dịch where filter sang SQL ──
def _where_sql(where) -> tuple[str, list]:
    """Hỗ trợ: None, {category}, {session_id}, {filename}, {"$and": [...]} lồng 1 cấp."""
    if not where:
        return "", []
    if isinstance(where, dict) and len(where) == 1 and "$and" in where:
        parts, params = [], []
        for sub in where["$and"] or []:
            sql, ps = _where_sql(sub)
            if sql:
                parts.append(f"({sql})")
                params.extend(ps)
        return (" AND ".join(parts), params) if parts else ("", [])
    if isinstance(where, dict):
        parts, params = [], []
        for k, v in where.items():
            if k == "category":
                parts.append("category = %s")
                params.append(v)
            elif k == "session_id":
                parts.append("session_id = %s")
                params.append(v)
            elif k == "filename":
                parts.append("filename = %s")
                params.append(v)
            else:
                parts.append("(metadata ->> %s) = %s")
                params.extend([k, str(v)])
        return (" AND ".join(parts), params) if parts else ("", [])
    raise ValueError(f"where không hỗ trợ: {where!r}")


def _safe_json_value(v):
    """Chuyển metadata thành JSON-safe: NaN/Inf -> None, kiểu lạ -> str."""
    if v is None or isinstance(v, (str, int, bool)):
        return v
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return v
    if isinstance(v, dict):
        return {str(k): _safe_json_value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe_json_value(x) for x in v]
    try:
        return str(v)[:2000]
    except Exception:
        return None


def _meta_json(meta: dict) -> str:
    try:
        clean = _safe_json_value(dict(meta or {}))
        return _json.dumps(clean, ensure_ascii=False, allow_nan=False)
    except Exception:
        return _json.dumps({"_raw": str(meta)[:2000]}, ensure_ascii=False)


def _parse_vec(raw) -> list:
    """Parse '[0.1,0.2,...]' (embedding::text) thành list[float]."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        try:
            return [float(x) for x in raw]
        except Exception:
            return []
    try:
        s = str(raw).strip().strip("[]")
        if not s:
            return []
        return [float(x) for x in s.split(",")]
    except Exception:
        return []


def add_chunks(chunks: list[dict], batch_size: int = 500) -> None:
    """Upsert danh sach chunks + embeddings vào pgvector.

    Moi chunk dict can co:
      id: str
      text: str (document)
      embedding: list[float]
      metadata: dict (da bao gom chunking results: chunk_index, chunk_mode, text_length...)
    Tự chia nhỏ theo batch để tránh OOM khi nhiều chunk.
    """
    if not chunks:
        return
    table = _table()
    dim = len(chunks[0]["embedding"])
    with _conn() as conn:
        _ensure_table(conn, dim)
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            rows = []
            for c in batch:
                meta = _safe_json_value(c.get("metadata") or {})
                if not isinstance(meta, dict):
                    meta = {}
                emb = c["embedding"]
                if hasattr(emb, "tolist"):
                    emb = emb.tolist()
                rows.append((
                    c["id"],
                    c.get("text", ""),
                    list(emb),
                    _meta_json(meta),
                    meta.get("category"),
                    meta.get("session_id"),
                    meta.get("filename"),
                ))
            with conn.cursor() as cur:
                cur.executemany(
                    f"""INSERT INTO {table} (id, text, embedding, metadata, category, session_id, filename)
                        VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                          text = EXCLUDED.text,
                          embedding = EXCLUDED.embedding,
                          metadata = EXCLUDED.metadata,
                          category = EXCLUDED.category,
                          session_id = EXCLUDED.session_id,
                          filename = EXCLUDED.filename""",
                    rows,
                )


def query_vector(embedding, where=None, n: int = 5) -> list[dict]:
    """Truy vấn top-n gần nhất (cosine distance 0~2).

    Trả về [{text, metadata, score}] với score = distance (nhỏ = gần).
    """
    if n <= 0:
        return []
    table = _table()
    emb = list(embedding() if callable(embedding) else embedding)
    if hasattr(emb, "tolist"):
        emb = emb.tolist()
    cond, params = _where_sql(where)
    sql = (
        f"SELECT text, metadata, (embedding <=> %s::vector) AS dist "
        f"FROM {table} "
        + (f"WHERE {cond} " if cond else "")
        + "ORDER BY dist ASC LIMIT %s"
    )
    try:
        with _conn() as conn:
            if not _table_exists(conn, table):
                return []
            cur = conn.execute(sql, [emb, *params, int(n)])
            out = []
            for text, meta, dist in cur.fetchall():
                out.append({"text": text or "", "metadata": dict(meta or {}), "score": float(dist)})
            return out
    except Exception:
        return []


def count() -> int:
    try:
        with _conn() as conn:
            if not _table_exists(conn, _table()):
                return 0
            cur = conn.execute(f"SELECT count(*) FROM {_table()}")
            row = cur.fetchone()
            return int(row[0]) if row else 0
    except Exception:
        return 0


def reset_collection(dim: int | None = None):
    """Xóa bảng vector (DROP). Nếu cho dim thì tạo lại bảng rỗng + index."""
    table = _table()
    with _conn() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        if dim:
            _create_table(conn, table, int(dim))
    return True


# Tiện ích kiểm tra kết quả chunking và embedding

def get_stats() -> dict:
    """Thong ke nhanh ve vector store hien tai."""
    table = _table()
    out = {"total": 0, "collection": table, "table": table, "store": "pgvector"}
    try:
        with _conn() as conn:
            if not _table_exists(conn, table):
                return out
            cur = conn.execute(f"SELECT count(*) FROM {table}")
            c = int((cur.fetchone() or [0])[0])
            out["total"] = c
            if c == 0:
                return out
            dim = _table_dim(conn, table)
            if dim:
                out["embedding_dim"] = dim
            cur = conn.execute(f"SELECT metadata, embedding::text FROM {table} LIMIT 5")
            rows = cur.fetchall() or []
            from collections import Counter
            metas = [dict(r[0] or {}) for r in rows]
            out["category_count"] = dict(Counter(m.get("category", "unknown") for m in metas if m))
            out["chunk_mode_count"] = dict(Counter(m.get("chunk_mode", "unknown") for m in metas if m))
            out["chunk_type_count"] = dict(Counter(m.get("chunk_type", "text") for m in metas if m))
            if rows and rows[0][1]:
                try:
                    out["embedding_preview"] = [float(f"{x:.4f}") for x in _parse_vec(rows[0][1])[:3]]
                except Exception:
                    pass
            models = {m.get("embedding_model", "") for m in metas if m and m.get("embedding_model")}
            if models:
                out["embedding_model"] = sorted(models)[0]
            if metas:
                out["sample_metadata"] = metas[0]
    except Exception as e:
        out["error"] = str(e)
    return out


def delete_by_session(session_id: str) -> int:
    """Xóa tất cả chunk có session_id = sid (ephemeral upload)."""
    if not session_id:
        return 0
    try:
        with _conn() as conn:
            if not _table_exists(conn, _table()):
                return 0
            cur = conn.execute(f"DELETE FROM {_table()} WHERE session_id = %s", (session_id,))
            return int(cur.rowcount or 0)
    except Exception as e:
        print(f"[vectorstore] delete_by_session lỗi {session_id}: {e}")
        return 0


def delete_chunks(session_id: str, filename: str) -> int:
    """Xóa chunk của 1 file trong 1 phiên (trang quản lý upload)."""
    if not session_id or not filename:
        return 0
    try:
        with _conn() as conn:
            if not _table_exists(conn, _table()):
                return 0
            cur = conn.execute(
                f"DELETE FROM {_table()} WHERE session_id = %s AND filename = %s",
                (session_id, filename),
            )
            return int(cur.rowcount or 0)
    except Exception as e:
        print(f"[vectorstore] delete_chunks lỗi: {e}")
        return 0


def count_by_session(session_id: str) -> int:
    try:
        with _conn() as conn:
            if not _table_exists(conn, _table()):
                return 0
            cur = conn.execute(
                f"SELECT count(*) FROM {_table()} WHERE session_id = %s", (session_id,)
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0
    except Exception:
        return 0


def list_uploads_by_session(session_id: str) -> list[dict]:
    """Liệt kê file đã upload theo phiên (group by filename)."""
    try:
        with _conn() as conn:
            if not _table_exists(conn, _table()):
                return []
            cur = conn.execute(
                f"""SELECT filename, count(*),
                           max((metadata ->> 'page')),
                           max(category),
                           max(metadata ->> 'source')
                    FROM {_table()} WHERE session_id = %s GROUP BY filename""",
                (session_id,),
            )
            out = []
            for fn, c, page, cat, src in cur.fetchall() or []:
                try:
                    page_v = int(page) if page is not None else None
                except Exception:
                    page_v = None
                out.append({
                    "filename": fn or "unknown",
                    "chunks": int(c),
                    "page": page_v,
                    "category": cat,
                    "source": src,
                })
            return out
    except Exception as e:
        print(f"[vectorstore] list_uploads lỗi: {e}")
        return []


__all__ = [
    "get_client",
    "query_vector",
    "add_chunks",
    "count",
    "reset_collection",
    "get_stats",
    "delete_by_session",
    "delete_chunks",
    "count_by_session",
    "list_uploads_by_session",
]
