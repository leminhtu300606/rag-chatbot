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
    DEFAULT_STYLE,
)
from backend.generation.generator import generate, generate_with_history, generate_social, is_social_question, rewrite_query, summarize_history, detect_style_change, strip_style_instruction
from backend.indexing.retriever import retrieve

from backend.generation.reranker import DEFAULT_RERANK_TOP_K, rerank


def _prepare_history(
    history: list[dict] | None,
) -> tuple[list[dict] | None, str | None]:
    """Chuẩn hóa lịch sử hội thoại: nhớ từ đầu phiên, cân bằng tốc độ + chính xác.

    - Giữ 8 turns gần nhất nguyên văn (600 chars) để suy luận chính xác.
    - Phần cũ từ đầu phiên được nén thành tóm tắt để vẫn nhớ.
    Returns:
        (history_window, summary)
        - history_window: tối đa CONVERSATION_WINDOW*2 messages gần nhất
        - summary: tóm tắt phần cũ từ đầu phiên nếu vượt ngưỡng, None nếu không cần
    """
    if not history:
        return None, None

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

    # Tính ngưỡng theo turns (2 messages = 1 turn)
    threshold_msgs = CONVERSATION_SUMMARY_THRESHOLD * 2
    window_size = CONVERSATION_WINDOW * 2

    summary = None
    if len(cleaned) > threshold_msgs:
        # Nhớ từ đầu phiên: tóm tắt toàn bộ phần cũ từ đầu đến trước cửa sổ gần nhất
        old_part = cleaned[:-window_size] if len(cleaned) > window_size else cleaned[: len(cleaned) // 2]
        if old_part:
            try:
                summary = summarize_history(old_part)
            except Exception:
                summary = None
        cleaned = cleaned[-window_size:]
    elif len(cleaned) > window_size:
        cleaned = cleaned[-window_size:]

    return (cleaned if cleaned else None), summary


def answer(
    question: str,
    category: str = None,
    top_k: int = None,
    use_rerank: bool = False,
    rerank_k: int = None,
    style: str | None = None,
) -> dict:
    """Trả lời câu hỏi ở chế độ đơn lượt (single-turn) - hỗ trợ xã giao."""
    # Phát hiện đổi phong cách
    detected_style = detect_style_change(question)
    effective_style = detected_style or style or DEFAULT_STYLE
    stripped = strip_style_instruction(question) if detected_style else question
    if detected_style and not stripped:
        return {
            "answer": f"Đã đổi sang phong cách {effective_style}. Từ giờ mình sẽ trả lời theo phong cách này nhé.",
            "sources": [],
            "context": [],
            "style": effective_style,
        }
    effective_question = stripped if stripped and stripped.strip() else question

    # Nhánh xã giao: không cần nguồn, dùng kiến thức huấn luyện, mặc định casual
    if is_social_question(effective_question):
        # Nếu chưa có style riêng, dùng casual cho xã giao
        social_style = effective_style if effective_style in ["casual", "friendly", "humorous", "empathetic"] else "casual"
        # Nếu phong cách hiện tại là formal nhưng câu xã giao thì vẫn dùng casual cho tự nhiên
        if style is None and detected_style is None:
            social_style = "casual"
        else:
            social_style = effective_style
        text = generate_social(effective_question, history=None, summary=None, style=social_style)
        return {
            "answer": text,
            "sources": [],
            "context": [],
            "style": social_style,
        }

    if top_k is None:
        top_k = DEFAULT_RERANK_TOP_K if (use_rerank and rerank) else RETRIEVE_TOP_K
    fetch_k = top_k * 3 if use_rerank and rerank else top_k

    query_for_retrieval = effective_question
    context = retrieve(query_for_retrieval, category=category, top_k=fetch_k)

    if use_rerank and rerank and context:
        context = rerank(query_for_retrieval, context, top_k=rerank_k if rerank_k is not None else top_k)

    text = generate(context, query_for_retrieval, style=effective_style)
    return {
        "answer": text,
        "sources": [c["metadata"] for c in context],
        "context": context,
        "style": effective_style,
    }


def answer_with_history(
    question: str,
    history: list[dict] | None = None,
    category: str = None,
    top_k: int = None,
    use_rerank: bool = False,
    rerank_k: int = None,
    style: str | None = None,
) -> dict:
    """Trả lời có ngữ cảnh hội thoại (conversational) - hỗ trợ đổi phong cách qua câu tự nhiên."""
    # Phát hiện đổi phong cách
    detected_style = detect_style_change(question)
    effective_style = detected_style or style or DEFAULT_STYLE
    stripped = strip_style_instruction(question) if detected_style else question
    # Nếu chỉ đổi phong cách không kèm câu hỏi thực -> trả lời xác nhận, vẫn nhớ từ đầu phiên
    if detected_style and not stripped:
        # Vẫn chuẩn bị history để nhớ, nhưng không cần retrieve
        history_window, summary = _prepare_history(history)
        # Tạo câu trả lời xác nhận theo phong cách mới (không cần nguồn)
        confirm_context = []
        # Dùng generate_with_history với context rỗng để LLM nói theo style mới
        text = generate_with_history(confirm_context, f"Xác nhận đã đổi sang phong cách {effective_style}", history_window, summary, style=effective_style)
        # Nếu LLM không trả lời đúng, fallback
        if not text or "phong cách" not in text.lower():
            text = f"Đã đổi sang phong cách {effective_style}. Từ giờ mình sẽ trả lời theo phong cách này nhé. Bạn cần hỏi gì tiếp?"
        return {
            "answer": text,
            "sources": [],
            "context": [],
            "standalone_question": question,
            "summary": summary,
            "history_used": history_window,
            "style": effective_style,
        }

    # Dùng câu đã tách chỉ thị để xử lý tiếp
    effective_question = stripped if stripped and stripped.strip() else question

    # Chuẩn bị cửa sổ lịch sử và tóm tắt (nhớ từ đầu phiên) - luôn cần dù là xã giao
    history_window, summary = _prepare_history(history)

    # Nhánh xã giao mở rộng: nếu câu là xã giao đời thường thì không cần nguồn, trả lời tự nhiên
    # Chỉ cung cấp ngoài lề khi có nguồn chính xác hoặc kiến thức huấn luyện -> cho phép dùng generate_social
    if is_social_question(effective_question, history):
        # Chọn phong cách xã giao: mặc định casual, nếu người dùng đã chọn friendly/humorous thì giữ
        social_style = effective_style
        if effective_style not in ["casual", "friendly", "humorous", "empathetic", "plain", "simple"]:
            # nếu đang là formal mà câu xã giao thì dùng casual cho tự nhiên
            social_style = "casual"
        text = generate_social(effective_question, history_window, summary, style=social_style)
        return {
            "answer": text,
            "sources": [],
            "context": [],
            "standalone_question": effective_question,
            "summary": summary,
            "history_used": history_window,
            "style": social_style,
        }

    # Viết lại câu hỏi để retrieval chính xác hơn (dùng câu đã tách)
    standalone_q = effective_question
    if history_window:
        try:
            standalone_q = rewrite_query(effective_question, history_window)
        except Exception:
            standalone_q = effective_question

    if top_k is None:
        top_k = DEFAULT_RERANK_TOP_K if (use_rerank and rerank) else RETRIEVE_TOP_K
    fetch_k = top_k * 3 if use_rerank and rerank else top_k

    query_for_retrieval = standalone_q if standalone_q and standalone_q.strip() else effective_question
    context = retrieve(query_for_retrieval, category=category, top_k=fetch_k)

    if use_rerank and rerank and context:
        context = rerank(query_for_retrieval, context, top_k=rerank_k if rerank_k is not None else top_k)

    text = generate_with_history(context, effective_question, history_window, summary, style=effective_style)

    return {
        "answer": text,
        "sources": [c["metadata"] for c in context],
        "context": context,
        "standalone_question": standalone_q,
        "summary": summary,
        "history_used": history_window,
        "style": effective_style,
    }
