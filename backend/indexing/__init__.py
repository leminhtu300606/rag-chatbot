"""
PHAN 2: HUAN LUYEN / XAY DUNG MO HINH (INDEXING)
===============================================
Bien cac chunk da tien xu ly thanh vector va luu vao Vector Store, ho tro truy xuat.

Cau truc:
- embedder.py      : tao embedding tieng Viet (dangvantuan/vietnamese-embedding)
- vectorstore.py   : luu/truy van ChromaDB + helpers peek/get_stats
- retriever.py     : truy xuat top-k chunk lien quan
- (logic build)    : da chuyen ra backend/index.py (truoc la build_index.py) - embed + upsert parquet -> Chroma (incremental + --rebuild)

Reranker da chuyen han sang backend/generation/reranker.py (generation lo cham diem cho prompt).
"""

from .embedder import embed
from .vectorstore import get_client, get_collection, add_chunks, count, reset_collection, peek, get_stats

__all__ = [
    "embed",
    "get_client",
    "get_collection",
    "add_chunks",
    "count",
    "reset_collection",
    "peek",
    "get_stats",
    "retrieve", "rerank", "RERANK_MODEL",
]


def __getattr__(name):
    if name == "retrieve":
        from .retriever import retrieve
        return retrieve
    if name in ("rerank", "RERANK_MODEL"):
        from backend.generation.reranker import RERANK_MODEL, rerank
        return rerank if name == "rerank" else RERANK_MODEL
    raise AttributeError(name)

