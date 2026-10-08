"""
PHAN 2: HUAN LUYEN / XAY DUNG MO HINH (INDEXING)
===============================================
Bien cac chunk da tien xu ly thanh vector va luu vao Vector Store, ho tro truy xuat.

Cau truc:
- embedder.py      : tao embedding tieng Viet (dangvantuan/vietnamese-embedding)
- vectorstore.py   : luu/truy van PostgreSQL/pgvector + helpers peek/get_stats
- retrieval.py     : cum truy xuat top-k (vector + BM25 + hybrid trong 1 file)
- (logic build)    : nam o backend/cli/main.py build() - embed + upsert parquet -> pgvector (incremental + --rebuild)

Reranker da chuyen han sang backend/generation/reranker.py (generation lo cham diem cho prompt).
"""

from .embedder import embed
from .vectorstore import (
    get_client,
    query_vector,
    add_chunks,
    count,
    reset_collection,
    get_stats,
)

__all__ = [
    "embed",
    "get_client",
    "query_vector",
    "add_chunks",
    "count",
    "reset_collection",
    "get_stats",
    "retrieve", "rerank", "RERANK_MODEL",
]


def __getattr__(name):
    if name == "retrieve":
        from .retrieval import retrieve
        return retrieve
    if name in ("rerank", "RERANK_MODEL"):
        from backend.generation.reranker import RERANK_MODEL, rerank
        return rerank if name == "rerank" else RERANK_MODEL
    raise AttributeError(name)

