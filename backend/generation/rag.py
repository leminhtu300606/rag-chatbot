"""
backend/generation/rag.py - Điều phối pipeline RAG
================================================
Nhiệm vụ:
- Kết nối tầng truy xuất (retrieval) với tầng sinh câu trả lời (generation).
- Thực hiện luồng: truy xuất → rerank → sinh câu trả lời.
- Cung cấp hai chế độ:
  * answer(): trả lời đơn lượt (single-turn)
  * answer_with_history(): trả lời có ngữ cảnh hội thoại (viết lại câu hỏi, quản lý cửa sổ lịch sử và tóm tắt)
"""

from backend.config import (
    RETRIEVE_TOP_K,
    CONVERSATION_WINDOW,
    CONVERSATION_SUMMARY_THRESHOLD,
    CONVERSATION_MAX_HISTORY_CHARS,
)
from backend.generation.generator import generate, generate_with_history, rewrite_query, summarize_history
from backend.indexing.retriever import retrieve

from backend.generation.reranker import DEFAULT_RERANK_TOP_K, rerank


def _prepare_history(
    history: list[dict] | None,
) -> tuple[list[dict] | None, str | None]:
    """Chuẩn hóa lịch sử hội thoại: cắt cửa sổ, cắt độ dài và tóm tắt khi vượt ngưỡng.

    Returns:
        (history_window, summary)
        - history_window: danh sách message gần nhất sau khi cắt (tối đa CONVERSATION_WINDOW*2)
        - summary: tóm tắt các turn cũ nếu có, None nếu không cần
    """
    if not history:
        return None, None

    # Chuẩn hóa và lọc message rỗng
    cleaned = []
    for m in history:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        content = m.get("content", "")
        if not content or not str(content).strip():
            continue
        if role not in ("user", "assistant", "system"):
            role = "user"
        txt = str(content).strip()
        if len(txt) > CONVERSATION_MAX_HISTORY_CHARS:
            txt = txt[:CONVERSATION_MAX_HISTORY_CHARS] + "..."
        cleaned.append({"role": role, "content": txt})

    if not cleaned:
        return None, None

    # Nếu vượt ngưỡng, tóm tắt phần cũ và chỉ giữ cửa sổ gần nhất
    summary = None
    if len(cleaned) > CONVERSATION_SUMMARY_THRESHOLD:
        window_size = CONVERSATION_WINDOW * 2  # mỗi turn ~2 messages
        old_part = cleaned[:-window_size] if len(cleaned) > window_size else cleaned[: len(cleaned) // 2]
        if old_part:
            try:
                summary = summarize_history(old_part)
            except Exception:
                summary = None
        cleaned = cleaned[-window_size:]

    elif len(cleaned) > CONVERSATION_WINDOW * 2:
        cleaned = cleaned[-(CONVERSATION_WINDOW * 2) :]

    return (cleaned if cleaned else None), summary


def answer(
    question: str,
    category: str = None,
    top_k: int = None,
    use_rerank: bool = False,
    rerank_k: int = None,
) -> dict:
    """Trả lời câu hỏi ở chế độ đơn lượt (single-turn).

    Args:
        question: câu hỏi người dùng
        category: lọc theo category nếu có
        top_k: số chunk cuối cùng đưa vào LLM (mặc định RETRIEVE_TOP_K / DEFAULT_RERANK_TOP_K)
        use_rerank: có dùng cross-encoder rerank không
        rerank_k: số chunk sau rerank

    Returns:
        {"answer": str, "sources": list[dict], "context": list[dict]}
    """
    if top_k is None:
        top_k = DEFAULT_RERANK_TOP_K if (use_rerank and rerank) else RETRIEVE_TOP_K
    fetch_k = top_k * 3 if use_rerank and rerank else top_k

    context = retrieve(question, category=category, top_k=fetch_k)

    if use_rerank and rerank and context:
        context = rerank(question, context, top_k=rerank_k if rerank_k is not None else top_k)

    text = generate(context, question)
    return {
        "answer": text,
        "sources": [c["metadata"] for c in context],
        "context": context,
    }


def answer_with_history(
    question: str,
    history: list[dict] | None = None,
    category: str = None,
    top_k: int = None,
    use_rerank: bool = False,
    rerank_k: int = None,
) -> dict:
    """Trả lời có ngữ cảnh hội thoại (conversational).

    Luồng xử lý:
      1. Chuẩn hóa lịch sử: cắt cửa sổ và tóm tắt nếu dài (_prepare_history)
      2. Viết lại câu hỏi thành dạng độc lập để retrieval chính xác (rewrite_query)
      3. Truy xuất và rerank bằng câu hỏi đã viết lại
      4. Sinh câu trả lời bằng câu hỏi gốc + lịch sử + tóm tắt + context (generate_with_history)

    Returns:
        {"answer": str, "sources": list[dict], "context": list[dict],
         "standalone_question": str, "summary": str|None, "history_used": list|None}
    """
    # Chuẩn bị cửa sổ lịch sử và tóm tắt
    history_window, summary = _prepare_history(history)

    # Viết lại câu hỏi để retrieval chính xác hơn
    standalone_q = question
    if history_window:
        try:
            standalone_q = rewrite_query(question, history_window)
        except Exception:
            standalone_q = question

    if top_k is None:
        top_k = DEFAULT_RERANK_TOP_K if (use_rerank and rerank) else RETRIEVE_TOP_K
    fetch_k = top_k * 3 if use_rerank and rerank else top_k

    query_for_retrieval = standalone_q if standalone_q and standalone_q.strip() else question
    context = retrieve(query_for_retrieval, category=category, top_k=fetch_k)

    if use_rerank and rerank and context:
        context = rerank(query_for_retrieval, context, top_k=rerank_k if rerank_k is not None else top_k)

    text = generate_with_history(context, question, history_window, summary)

    return {
        "answer": text,
        "sources": [c["metadata"] for c in context],
        "context": context,
        "standalone_question": standalone_q,
        "summary": summary,
        "history_used": history_window,
    }
