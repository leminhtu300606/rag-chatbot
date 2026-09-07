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
    "Trả lời bằng tiếng Việt, ngắn gọn, rõ ràng và ghi rõ nguồn.\n"
    "Quy tắc suy luận và nhớ trong 1 phiên (cân bằng tốc độ + chính xác):\n"
    "- Luôn nhớ toàn bộ hội thoại từ đầu phiên: ưu tiên dùng lịch sử và tóm tắt đã cung cấp để hiểu ngữ cảnh.\n"
    "- Suy luận có bước: xác định câu hỏi thực sự là gì -> đối chiếu các mảnh ngữ cảnh đã cho -> tổng hợp rồi kết luận.\n"
    "- Chỉ trả lời dựa trên ngữ cảnh được cung cấp. Nếu ngữ cảnh không chứa đủ thông tin để trả lời chắc chắn, "
    "hãy nói thẳng: \"Không đủ nguồn trong tài liệu để trả lời chắc chắn.\" và gợi ý người dùng cung cấp thêm hoặc hỏi lại cụ thể hơn. "
    "Không bịa đặt, không suy đoán ngoài nguồn.\n"
    "- Khi lịch sử mâu thuẫn với ngữ cảnh mới, ưu tiên ngữ cảnh mới nhưng vẫn nhắc lại điểm khác biệt nếu cần."
)

# ── Bộ phong cách đổi được trong hội thoại bằng câu tự nhiên ──
# Người dùng có thể nói "đổi sang giọng thân thiện", "giải thích đơn giản thôi", "không dùng kiểu gọi hàm", v.v.
STYLE_PROMPTS = {
    "formal": "Phong cách: hành chính, trang trọng. Dùng câu đầy đủ, từ ngữ chuẩn mực, ghi rõ nguồn chi tiết.",
    "friendly": "Phong cách: thân thiện, gần gũi. Xưng hô ấm áp (bạn/mình), giọng vui vẻ, dễ gần.",
    "concise": "Phong cách: ngắn gọn, súc tích. Chỉ 3-5 câu, đi thẳng vào ý chính, không lan man.",
    "detailed": "Phong cách: chi tiết, đầy đủ. Giải thích cặn kẽ, có ví dụ, liệt kê rõ ràng.",
    "simple": "Phong cách: đơn giản dễ hiểu. Tránh thuật ngữ khó, dùng từ phổ thông, ví dụ đời thường.",
    "academic": "Phong cách: học thuật. Cấu trúc chặt chẽ, lập luận logic, có trích dẫn.",
    "casual": "Phong cách: tự nhiên, thoải mái như trò chuyện đời thường.",
    "humorous": "Phong cách: hài hước nhẹ, vui vẻ nhưng vẫn chính xác, không đùa quá trớn.",
    "empathetic": "Phong cách: đồng cảm, chia sẻ, an ủi, thấu hiểu cảm xúc người hỏi.",
    "creative": "Phong cách: sáng tạo, gợi mở, dùng ẩn dụ, góc nhìn mới.",
    "bullet": "Phong cách: gạch đầu dòng. Trình bày dạng liệt kê bullet rõ ràng, mỗi ý một dòng.",
    "step_by_step": "Phong cách: từng bước. Trình bày theo bước 1, 2, 3... có thứ tự.",
    "plain": "Phong cách: thuần túy, không dùng kiểu gọi hàm, không dùng thuật ngữ kỹ thuật, không nhắc tên hàm hay code, chỉ giải thích bằng lời tự nhiên, dễ đọc.",
}

def get_style_prompt(style: str | None) -> str:
    if not style:
        return ""
    style = str(style).strip().lower()
    return STYLE_PROMPTS.get(style, "")

def apply_style_to_system(base: str, style: str | None) -> str:
    extra = get_style_prompt(style)
    if extra:
        return base + "\n" + extra
    return base

# ── Prompt xã giao mở rộng: mặc định casual, chỉ cung cấp ngoài lề khi có nguồn chính xác hoặc kiến thức huấn luyện ──
SOCIAL_SYSTEM_PROMPT = (
    "Bạn là trợ lý thân thiện của Học viện Kỹ thuật Mật mã. "
    "Trả lời bằng giọng đời thường, tự nhiên, ấm áp, ngắn gọn bằng tiếng Việt.\n"
    "Quy tắc xã giao mở rộng:\n"
    "- Với câu chào hỏi, cảm ơn, hỏi thăm, trò chuyện đời thường thì trả lời tự nhiên, không cần nguồn tài liệu.\n"
    "- Chỉ cung cấp thông tin ngoài lề (không có trong tài liệu) khi bạn chắc chắn đó là kiến thức chung đã được huấn luyện trước hoặc khi có nguồn chính xác trong ngữ cảnh.\n"
    "- Nếu người dùng hỏi kiến thức ngoài lề mà bạn không chắc chắn, hãy nói thẳng là chưa đủ thông tin chắc chắn và gợi ý hỏi lại.\n"
    "- Luôn nhớ toàn bộ lịch sử chuyên môn từ đầu phiên để khi quay lại hỏi chuyên môn vẫn giữ mạch.\n"
    "- Mặc định dùng giọng đời thường (casual) trừ khi người dùng yêu cầu đổi phong cách."
)

# Prompt dùng để viết lại câu hỏi thành dạng độc lập, đầy đủ ngữ nghĩa (cân bằng)
REWRITE_SYSTEM_PROMPT = (
    "Bạn là chuyên gia viết lại câu hỏi cho hệ thống RAG. "
    "Nhiệm vụ: dựa trên lịch sử hội thoại từ đầu phiên, viết lại câu hỏi cuối cùng thành một câu hỏi độc lập, đầy đủ, "
    "không chứa đại từ mơ hồ (nó, cái đó, ở trên, trong đó, v.v.) và bổ sung thực thể/chủ đề chính đã nhắc từ đầu phiên nếu cần.\n"
    "Nguyên tắc cân bằng:\n"
    "- Nếu câu hỏi đã đầy đủ, rõ ràng thì giữ nguyên để tiết kiệm thời gian.\n"
    "- Nếu câu hỏi ngắn, thiếu chủ ngữ hoặc tiếp nối ý trước, hãy làm giàu bằng ngữ cảnh phiên.\n"
    "Chỉ trả về câu hỏi đã viết lại, không thêm giải thích, không thêm tiền tố."
)

# Prompt dùng để tóm tắt lịch sử hội thoại dài (nhớ từ đầu phiên)
SUMMARY_SYSTEM_PROMPT = (
    "Bạn là trợ lý tóm tắt hội thoại. "
    "Hãy tóm tắt lịch sử hội thoại sau thành 4-6 câu ngắn gọn bằng tiếng Việt, "
    "giữ lại theo thứ tự thời gian từ đầu phiên: chủ đề chính, các thực thể quan trọng (tên quy chế, điều khoản, đối tượng hỏi), "
    "các câu hỏi và kết luận chính, và ý định hiện tại của người dùng. "
    "Phải giữ thông tin để câu hỏi tiếp theo vẫn hiểu được ngữ cảnh từ đầu phiên. Không bịa thêm."
)


def build_messages(context: list[dict], question: str, style: str | None = None) -> list[dict]:
    """Dựng messages cho LLM từ context đã retrieve (chế độ đơn lượt, không kèm lịch sử)."""
    ctx = "\n\n".join(
        f"[{i+1}] {c['text']} (Nguồn: {c['metadata'].get('filename', '')})"
        for i, c in enumerate(context)
    ) if context else "(Không có ngữ cảnh phù hợp - hãy trả lời: Không đủ nguồn trong tài liệu để trả lời chắc chắn.)"
    user_text = f"Ngữ cảnh:\n{ctx}\n\nCâu hỏi: {question}\nTrả lời:"
    system = apply_style_to_system(SYSTEM_PROMPT, style)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]


def build_messages_with_history(
    context: list[dict],
    question: str,
    history: list[dict] | None = None,
    summary: str | None = None,
    style: str | None = None,
) -> list[dict]:
    """Dựng messages có kèm lịch sử hội thoại và tóm tắt (chế độ hội thoại - nhớ từ đầu phiên, cân bằng)."""
    system = apply_style_to_system(SYSTEM_PROMPT, style)
    messages: list[dict] = [{"role": "system", "content": system}]

    # Thêm tóm tắt hội thoại từ đầu phiên nếu có (để nhớ dù đã cắt cửa sổ)
    if summary:
        messages.append(
            {
                "role": "system",
                "content": f"Tóm tắt toàn phiên từ đầu (đã nén để nhớ lâu): {summary}",
            }
        )

    # Đưa lịch sử hội thoại gần nhất vào prompt (đã giữ 8 turns gần nhất nguyên văn)
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
    ) if context else "(Không có ngữ cảnh phù hợp - hãy trả lời: Không đủ nguồn trong tài liệu để trả lời chắc chắn.)"
    hint = ""
    if history or summary:
        hint = "\n(Lưu ý: Hãy suy luận có bước: xác định ý định thực sự -> đối chiếu ngữ cảnh -> tổng hợp. Nếu lịch sử mâu thuẫn với ngữ cảnh mới, ưu tiên ngữ cảnh. Nếu không đủ nguồn, nói thẳng: Không đủ nguồn trong tài liệu để trả lời chắc chắn.)"
    user_text = f"Ngữ cảnh:\n{ctx}\n\nCâu hỏi: {question}{hint}\nTrả lời:"
    messages.append({"role": "user", "content": user_text})
    return messages


def build_social_messages(
    question: str,
    history: list[dict] | None = None,
    summary: str | None = None,
    style: str | None = None,
) -> list[dict]:
    """Dựng messages cho nhánh xã giao mở rộng - mặc định casual, không cần nguồn."""
    # Dùng SOCIAL_SYSTEM_PROMPT làm nền, cộng thêm style nếu có (mặc định casual)
    effective_style = style or "casual"
    system = apply_style_to_system(SOCIAL_SYSTEM_PROMPT, effective_style)
    messages: list[dict] = [{"role": "system", "content": system}]
    if summary:
        messages.append({"role": "system", "content": f"Tóm tắt toàn phiên từ đầu (để nhớ chuyên môn): {summary}"})
    if history:
        for msg in history:
            role = msg.get("role", "user")
            if role not in ("user", "assistant", "system"):
                role = "user"
            content = msg.get("content", "")
            if content and content.strip():
                messages.append({"role": role, "content": content.strip()})
    # Với xã giao, không cần ngữ cảnh retrieve, chỉ cần câu hỏi và lịch sử
    messages.append({"role": "user", "content": question})
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


def build_prompt_text(context: list[dict], question: str, style: str | None = None) -> str:
    """Dựng prompt dạng text đơn giản cho transformers pipeline (đơn lượt)."""
    messages = build_messages(context, question, style=style)
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


def build_prompt_text_with_history(
    context: list[dict], question: str, history: list[dict] | None = None, summary: str | None = None, style: str | None = None
) -> str:
    """Dựng prompt dạng text đơn giản có kèm lịch sử (cho transformers pipeline)."""
    messages = build_messages_with_history(context, question, history, summary, style=style)
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
