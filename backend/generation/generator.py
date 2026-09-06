"""
backend/generation/generator.py - Sinh câu trả lời bằng mô hình ngôn ngữ
====================================================================
Nhiệm vụ:
- Gọi LLM để sinh câu trả lời dựa trên ngữ cảnh đã retrieve.
- Hỗ trợ hai backend (chọn qua config.LLM_BACKEND):
  * ollama: gọi Ollama đang chạy local (mặc định, dùng qwen2.5)
  * transformers: load trực tiếp bằng transformers
- Cung cấp các hàm:
  * generate(): sinh câu trả lời đơn lượt
  * generate_with_history(): sinh câu trả lời có kèm lịch sử hội thoại và tóm tắt
  * rewrite_query(): viết lại câu hỏi mơ hồ thành dạng độc lập (phục vụ retrieval)
  * summarize_history(): tóm tắt lịch sử hội thoại dài
"""
import requests

from backend.config import (
    DEVICE,
    GEN_MAX_TOKENS,
    LLM_BACKEND,
    LLM_MODEL,
    LLM_THINK,
    OLLAMA_BASE,
    REWRITE_MAX_TOKENS,
    REWRITE_MODEL,
    SUMMARY_MAX_TOKENS,
    SUMMARY_MODEL,
)
from .prompts import (
    SYSTEM_PROMPT,
    build_messages,
    build_messages_with_history,
    build_prompt_text,
    build_rewrite_messages,
    build_summary_messages,
)

_gen = None


def _call_ollama(messages: list[dict], model: str, max_tokens: int, think: bool = False) -> str:
    """Gọi Ollama /api/chat với danh sách messages và trả về nội dung trả lời."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": {"num_predict": max_tokens},
    }
    try:
        resp = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=600)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Không kết nối được Ollama tại {OLLAMA_BASE}. Hãy chạy 'ollama serve' và kiểm tra 'ollama list'. Chi tiết: {e}"
        ) from e

    if resp.status_code != 200:
        try:
            err_json = resp.json()
            err_msg = err_json.get("error", resp.text)
        except Exception:
            err_msg = resp.text
        err_low = err_msg.lower()
        if "out-of-memory" in err_low or "failed to allocate" in err_low or "memory" in err_low:
            raise RuntimeError(
                f"Ollama báo out-of-memory khi load model '{model}' (~4.7GB cho qwen2.5:7b). "
                f"RAM/VRAM không đủ. Gợi ý: 1) Chạy 'ollama pull qwen2.5:1.5b' hoặc 'qwen2.5:0.5b' (nhẹ hơn), "
                f"2) Đổi LLM_MODEL trong backend/config.py sang model nhỏ hơn, "
                f"3) Hoặc thử 'qwen2.5:3b' nếu cần cân bằng. Chi tiết gốc: {err_msg}"
            )
        if "not found" in err_low or "model" in err_low and "not" in err_low:
            raise RuntimeError(
                f"Model '{model}' không tìm thấy trên Ollama. Chạy 'ollama list' để kiểm tra, "
                f"và 'ollama pull {model}' hoặc đổi LLM_MODEL trong backend/config.py. Chi tiết: {err_msg}"
            )
        resp.raise_for_status()

    return resp.json()["message"]["content"].strip()


def _generate_ollama(context: list[dict], question: str) -> str:
    """Sinh câu trả lời đơn lượt qua Ollama."""
    messages = build_messages(context, question)
    return _call_ollama(messages, LLM_MODEL, GEN_MAX_TOKENS, LLM_THINK)


def _generate_ollama_with_history(
    context: list[dict], question: str, history: list[dict] | None, summary: str | None
) -> str:
    """Sinh câu trả lời có kèm lịch sử hội thoại qua Ollama."""
    messages = build_messages_with_history(context, question, history, summary)
    return _call_ollama(messages, LLM_MODEL, GEN_MAX_TOKENS, LLM_THINK)


def _get_transformers_gen():
    global _gen
    if _gen is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        tok = AutoTokenizer.from_pretrained(LLM_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL, torch_dtype="auto", device_map=DEVICE
        )
        _gen = pipeline("text-generation", model=model, tokenizer=tok)
    return _gen


def _generate_transformers(context: list[dict], question: str) -> str:
    """Sinh câu trả lời đơn lượt qua transformers pipeline."""
    gen = _get_transformers_gen()
    prompt = "\n\n".join(
        f"{m['role']}: {m['content']}" for m in build_messages(context, question)
    )
    out = gen(prompt, max_new_tokens=GEN_MAX_TOKENS, do_sample=False, return_full_text=False)
    return out[0]["generated_text"].strip()


def _generate_transformers_with_history(
    context: list[dict], question: str, history: list[dict] | None, summary: str | None
) -> str:
    """Sinh câu trả lời có kèm lịch sử qua transformers pipeline."""
    gen = _get_transformers_gen()
    from .prompts import build_prompt_text_with_history

    prompt = build_prompt_text_with_history(context, question, history, summary)
    out = gen(prompt, max_new_tokens=GEN_MAX_TOKENS, do_sample=False, return_full_text=False)
    return out[0]["generated_text"].strip()


def generate(context: list[dict], question: str) -> str:
    """Sinh câu trả lời đơn lượt (tự chọn backend theo cấu hình)."""
    if LLM_BACKEND == "ollama":
        return _generate_ollama(context, question)
    return _generate_transformers(context, question)


def generate_with_history(
    context: list[dict],
    question: str,
    history: list[dict] | None = None,
    summary: str | None = None,
) -> str:
    """Sinh câu trả lời có kèm lịch sử hội thoại và tóm tắt (tự chọn backend)."""
    if LLM_BACKEND == "ollama":
        return _generate_ollama_with_history(context, question, history, summary)
    return _generate_transformers_with_history(context, question, history, summary)


# ── Viết lại câu hỏi ──

# Tập từ khóa đại từ mơ hồ dùng để phát hiện câu hỏi phụ thuộc ngữ cảnh
_PRONOUNS = [
    "nó ",
    "nó?",
    "nó.",
    "nó,",
    "cái đó",
    "cái này",
    "ở trên",
    "trong đó",
    "trong đấy",
    "như trên",
    "đó là gì",
    "này là gì",
    "của nó",
    "cho nó",
    "về nó",
    "nói rõ hơn",
    "chi tiết hơn",
    "giải thích thêm",
    "còn ",
    # Dạng không dấu để bắt cả khi gõ không dấu
    "no ",
    "no?",
    "no.",
    "no,",
    "cai do",
    "cai nay",
    "trong do",
    "trong day",
    "nhu tren",
    "cua no",
    "cho no",
    "ve no",
    "noi ro hon",
    "chi tiet hon",
]
# Các câu hỏi ngắn thường là câu nối tiếp cần ngữ cảnh
_SHORT_FOLLOWUP = ["là gì?", "là sao?", "tại sao?", "thế nào?", "bao nhiêu?", "khi nào?", "ở đâu?", "la gi?", "la sao?", "the nao?", "bao nhieu?"]


def _needs_rewrite(question: str, history: list[dict] | None) -> bool:
    """Kiểm tra câu hỏi có cần viết lại để thành dạng độc lập hay không."""
    if not history:
        return False
    q_low = question.strip().lower()
    # Câu rất ngắn thường là câu nối tiếp
    if len(q_low) < 12:
        return True
    for kw in _SHORT_FOLLOWUP:
        if q_low == kw or q_low.startswith(kw):
            return True
    # Chứa đại từ mơ hồ
    for p in _PRONOUNS:
        if p in q_low:
            return True
    # Đại từ dạng token riêng biệt
    tokens = [t.strip('.,?:;!\"\'') for t in q_low.split()]
    if "nó" in tokens or "no" in tokens:
        return True
    # Câu quá ngắn, thiếu danh từ cụ thể
    if len(q_low.split()) <= 4:
        return True
    return False


def rewrite_query(question: str, history: list[dict] | None) -> str:
    """Viết lại câu hỏi cuối thành dạng độc lập, đầy đủ ngữ nghĩa dựa trên lịch sử.

    Sử dụng LLM để viết lại. Nếu không cần thiết hoặc gặp lỗi thì trả về câu gốc.
    """
    if not history or not question or not question.strip():
        return question

    # Tối ưu: bỏ qua gọi LLM nếu câu hỏi đã đầy đủ
    if not _needs_rewrite(question, history):
        return question

    model = REWRITE_MODEL or LLM_MODEL
    messages = build_rewrite_messages(history, question)
    try:
        if LLM_BACKEND == "ollama":
            rewritten = _call_ollama(messages, model, REWRITE_MAX_TOKENS, False)
        else:
            gen = _get_transformers_gen()
            prompt = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
            out = gen(prompt, max_new_tokens=REWRITE_MAX_TOKENS, do_sample=False, return_full_text=False)
            rewritten = out[0]["generated_text"].strip()

        # Làm sạch kết quả
        rewritten = rewritten.strip().strip('"').strip("'").strip()
        if "\n" in rewritten:
            lines = [l.strip() for l in rewritten.split("\n") if l.strip()]
            if lines:
                q_lines = [l for l in lines if "?" in l]
                rewritten = q_lines[0] if q_lines else lines[0]
        for prefix in ["câu hỏi viết lại:", "câu hỏi độc lập:", "rewritten:", "question:"]:
            if rewritten.lower().startswith(prefix):
                rewritten = rewritten[len(prefix) :].strip()
        if not rewritten or len(rewritten) < 3:
            return question
        return rewritten
    except Exception:
        return question


# ── Tóm tắt hội thoại ──


def summarize_history(history: list[dict]) -> str:
    """Tóm tắt lịch sử hội thoại dài thành 3-5 câu ngắn gọn."""
    if not history or len(history) <= 4:
        return ""
    model = SUMMARY_MODEL or LLM_MODEL
    messages = build_summary_messages(history)
    try:
        if LLM_BACKEND == "ollama":
            summary = _call_ollama(messages, model, SUMMARY_MAX_TOKENS, False)
        else:
            gen = _get_transformers_gen()
            prompt = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
            out = gen(prompt, max_new_tokens=SUMMARY_MAX_TOKENS, do_sample=False, return_full_text=False)
            summary = out[0]["generated_text"].strip()
        return summary.strip()[:800]
    except Exception:
        # Fallback: nối các message đầu thành tóm tắt thô
        parts = [f"{m.get('role')}: {m.get('content','')[:80]}" for m in history[:6]]
        return " | ".join(parts)[:600]
