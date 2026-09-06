"""
backend/generation/prompts.py - Quản lý prompt cho hệ thống RAG
==============================================================
Nhiệm vụ:
- Định nghĩa prompt hệ thống và các template dựng ngữ cảnh cho LLM.
- Cung cấp hàm xây dựng messages cho nhiều chế độ:
  * Hội thoại đơn lượt (single-turn)
  * Hội thoại có lịch sử và tóm tắt (conversational)
  * Viết lại câu hỏi thành dạng độc lập và tóm tắt lịch sử
Tách riêng khỏi generator để dễ quản lý và thử nghiệm prompt.
"""

SYSTEM_PROMPT = (
    "Bạn là trợ lý ảo của Học viện Kỹ thuật Mật mã. "
    "Chỉ trả lời dựa trên ngữ cảnh được cung cấp, bằng tiếng Việt, "
    "ngắn gọn, rõ ràng và ghi rõ nguồn."
)

# Prompt dùng để viết lại câu hỏi thành dạng độc lập, đầy đủ ngữ nghĩa
REWRITE_SYSTEM_PROMPT = (
    "Bạn là chuyên gia viết lại câu hỏi cho hệ thống RAG. "
    "Nhiệm vụ: dựa trên lịch sử hội thoại, viết lại câu hỏi cuối cùng thành một câu hỏi độc lập, đầy đủ, "
    "không chứa đại từ mơ hồ (nó, cái đó, ở trên, trong đó, v.v.). "
    "Nếu câu hỏi đã đầy đủ thì giữ nguyên. "
    "Chỉ trả về câu hỏi đã viết lại, không thêm giải thích, không thêm tiền tố."
)

# Prompt dùng để tóm tắt lịch sử hội thoại dài
SUMMARY_SYSTEM_PROMPT = (
    "Bạn là trợ lý tóm tắt hội thoại. "
    "Hãy tóm tắt lịch sử hội thoại sau thành 3-5 câu ngắn gọn bằng tiếng Việt, "
    "giữ lại các thực thể chính (tên quy chế, điều khoản, đối tượng hỏi), "
    "để dùng làm ngữ cảnh cho câu hỏi tiếp theo. Không bịa thêm."
)


def build_messages(context: list[dict], question: str) -> list[dict]:
    """Dựng messages cho LLM từ context đã retrieve (chế độ đơn lượt, không kèm lịch sử)."""
    ctx = "\n\n".join(
        f"[{i+1}] {c['text']} (Nguồn: {c['metadata'].get('filename', '')})"
        for i, c in enumerate(context)
    )
    user_text = f"Ngữ cảnh:\n{ctx}\n\nCâu hỏi: {question}\nTrả lời:"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def build_messages_with_history(
    context: list[dict],
    question: str,
    history: list[dict] | None = None,
    summary: str | None = None,
) -> list[dict]:
    """Dựng messages có kèm lịch sử hội thoại và tóm tắt (chế độ hội thoại).

    Args:
        context: danh sách chunk đã retrieve
        question: câu hỏi hiện tại
        history: danh sách message lịch sử [{role, content}] đã được cắt cửa sổ
        summary: tóm tắt các turn cũ đã bị cắt (nếu hội thoại dài)

    Thứ tự messages:
    - SYSTEM_PROMPT đầu tiên
    - System note chứa tóm tắt (nếu có)
    - Lịch sử hội thoại theo thời gian
    - User message cuối chứa ngữ cảnh + câu hỏi
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Thêm tóm tắt hội thoại cũ nếu có
    if summary:
        messages.append(
            {
                "role": "system",
                "content": f"Tóm tắt hội thoại trước đó (để hiểu ngữ cảnh): {summary}",
            }
        )

    # Đưa lịch sử hội thoại gần nhất vào prompt
    if history:
        for msg in history:
            role = msg.get("role", "user")
            if role not in ("user", "assistant", "system"):
                role = "user"
            content = msg.get("content", "")
            if content and content.strip():
                messages.append({"role": role, "content": content.strip()})

    # Ngữ cảnh retrieve + câu hỏi hiện tại
    ctx = "\n\n".join(
        f"[{i+1}] {c['text']} (Nguồn: {c['metadata'].get('filename', '')})"
        for i, c in enumerate(context)
    )
    hint = ""
    if history or summary:
        hint = "\n(Lưu ý: Nếu lịch sử mâu thuẫn với ngữ cảnh mới, hãy ưu tiên ngữ cảnh.)"
    user_text = f"Ngữ cảnh:\n{ctx}\n\nCâu hỏi: {question}{hint}\nTrả lời:"
    messages.append({"role": "user", "content": user_text})
    return messages


def build_rewrite_messages(history: list[dict], question: str) -> list[dict]:
    """Dựng messages để viết lại câu hỏi cuối thành dạng độc lập, đầy đủ ngữ nghĩa."""
    hist_text = ""
    if history:
        parts = []
        for m in history[-8:]:  # chỉ lấy 8 messages gần nhất để đủ ngữ cảnh
            r = "Người dùng" if m.get("role") == "user" else "Trợ lý"
            parts.append(f"{r}: {m.get('content','')}")
        hist_text = "\n".join(parts)

    user_content = (
        f"Lịch sử hội thoại:\n{hist_text if hist_text else '(không có)'}\n\n"
        f"Câu hỏi cuối: {question}\n\n"
        f"Hãy viết lại câu hỏi cuối thành câu độc lập:"
    )
    return [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_summary_messages(history: list[dict]) -> list[dict]:
    """Dựng messages để tóm tắt lịch sử hội thoại dài."""
    parts = []
    for m in history:
        r = "Người dùng" if m.get("role") == "user" else "Trợ lý"
        parts.append(f"{r}: {m.get('content','')}")
    hist_text = "\n".join(parts)
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": f"Hãy tóm tắt hội thoại sau:\n{hist_text}"},
    ]


def build_prompt_text(context: list[dict], question: str) -> str:
    """Dựng prompt dạng text đơn giản cho transformers pipeline (đơn lượt)."""
    messages = build_messages(context, question)
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


def build_prompt_text_with_history(
    context: list[dict], question: str, history: list[dict] | None = None, summary: str | None = None
) -> str:
    """Dựng prompt dạng text đơn giản có kèm lịch sử (cho transformers pipeline)."""
    messages = build_messages_with_history(context, question, history, summary)
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
