"""Truy xuất các chunk liên quan nhất với câu hỏi."""
from backend.config import RETRIEVE_TOP_K
from backend.indexing.embedder import embed
from backend.indexing.vectorstore import get_collection


def retrieve(query: str, top_k: int = RETRIEVE_TOP_K, category: str = None) -> list[dict]:
    q_emb = embed([query])[0]
    where = {"category": category} if category else None
    res = get_collection().query(
        query_embeddings=[q_emb],
        n_results=top_k,
        where=where,
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    return [
        {"text": d, "metadata": m, "score": float(s)}
        for d, m, s in zip(docs, metas, dists)
    ]

