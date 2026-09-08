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
    AVAILABLE_STYLES,
    DEFAULT_STYLE,
    STYLE_KEYWORDS,
    STYLE_TEMPERATURE,
    ENABLE_SOCIAL,
    SOCIAL_ALLOWLIST,
)
from .prompts import (
    SYSTEM_PROMPT,
    build_messages,
    build_messages_with_history,
    build_prompt_text,
    build_rewrite_messages,
    build_summary_messages,
    build_social_messages,
)

_gen = None


def _call_ollama(messages: list[dict], model: str, max_tokens: int, think: bool = False, temperature: float | None = None) -> str:
    """Gọi Ollama /api/chat với danh sách messages và trả về nội dung trả lời."""
    options = {"num_predict": max_tokens}
    if temperature is not None:
        options["temperature"] = temperature
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": options,
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


def _generate_ollama(context: list[dict], question: str, style: str | None = None) -> str:
    """Sinh câu trả lời đơn lượt qua Ollama."""
    messages = build_messages(context, question, style=style)
    temp = STYLE_TEMPERATURE.get(style) if style else None
    return _call_ollama(messages, LLM_MODEL, GEN_MAX_TOKENS, LLM_THINK, temperature=temp)


def _generate_ollama_with_history(
    context: list[dict], question: str, history: list[dict] | None, summary: str | None, style: str | None = None
) -> str:
    """Sinh câu trả lời có kèm lịch sử hội thoại qua Ollama."""
    messages = build_messages_with_history(context, question, history, summary, style=style)
    temp = STYLE_TEMPERATURE.get(style) if style else None
    return _call_ollama(messages, LLM_MODEL, GEN_MAX_TOKENS, LLM_THINK, temperature=temp)


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


def _generate_transformers(context: list[dict], question: str, style: str | None = None) -> str:
    """Sinh câu trả lời đơn lượt qua transformers pipeline."""
    gen = _get_transformers_gen()
    prompt = "\n\n".join(
        f"{m['role']}: {m['content']}" for m in build_messages(context, question, style=style)
    )
    out = gen(prompt, max_new_tokens=GEN_MAX_TOKENS, do_sample=False, return_full_text=False)
    return out[0]["generated_text"].strip()


def _generate_transformers_with_history(
    context: list[dict], question: str, history: list[dict] | None, summary: str | None, style: str | None = None
) -> str:
    """Sinh câu trả lời có kèm lịch sử qua transformers pipeline."""
    gen = _get_transformers_gen()
    from .prompts import build_prompt_text_with_history

    prompt = build_prompt_text_with_history(context, question, history, summary, style=style)
    out = gen(prompt, max_new_tokens=GEN_MAX_TOKENS, do_sample=False, return_full_text=False)
    return out[0]["generated_text"].strip()


def generate(context: list[dict], question: str, style: str | None = None) -> str:
    """Sinh câu trả lời đơn lượt (tự chọn backend theo cấu hình)."""
    if LLM_BACKEND == "ollama":
        return _generate_ollama(context, question, style=style)
    return _generate_transformers(context, question, style=style)


def generate_with_history(
    context: list[dict],
    question: str,
    history: list[dict] | None = None,
    summary: str | None = None,
    style: str | None = None,
) -> str:
    """Sinh câu trả lời có kèm lịch sử hội thoại và tóm tắt (tự chọn backend)."""
    if LLM_BACKEND == "ollama":
        return _generate_ollama_with_history(context, question, history, summary, style=style)
    return _generate_transformers_with_history(context, question, history, summary, style=style)


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
    """Kiểm tra câu hỏi có cần viết lại để thành dạng độc lập hay không - cân bằng."""
    if not history:
        return False
    q_low = question.strip().lower()
    # Câu rất ngắn thường là câu nối tiếp -> cần nhớ từ đầu phiên
    if len(q_low) < 15:
        return True
    for kw in _SHORT_FOLLOWUP:
        if q_low == kw or q_low.startswith(kw):
            return True
    # Chứa đại từ mơ hồ
    for p in _PRONOUNS:
        if p in q_low:
            return True
    tokens = [t.strip('.,?:;!\"\'') for t in q_low.split()]
    if "nó" in tokens or "no" in tokens:
        return True
    # Câu ngắn, thiếu danh từ cụ thể -> cần làm giàu bằng ngữ cảnh phiên (cân bằng: 4 từ)
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
            raise ValueError("rewrite too short")
        # Nếu rewrite vẫn chứa đại từ mơ hồ thì coi như chưa thành công -> fallback heuristic
        low_re = rewritten.lower()
        if any(p.strip() in low_re for p in [" nó ", " cái đó", " cái này", " trong đó", " ở trên"]):
            raise ValueError("rewrite still ambiguous")
        return rewritten
    except Exception:
        # Fallback heuristic: thay đại từ "nó", "cái đó"... bằng thực thể từ lịch sử đầu phiên để không mất câu đầu
        try:
            q_low = question.strip().lower()
            # Tìm câu hỏi người dùng gần nhất có nội dung cụ thể (ưu tiên câu đầu tiên)
            last_user_q = None
            for m in reversed(history):
                if m.get("role") == "user" and m.get("content", "").strip():
                    # Bỏ qua câu hiện tại nếu trùng
                    if m["content"].strip().lower() != q_low:
                        last_user_q = m["content"].strip()
                        break
            # Ưu tiên câu đầu tiên để đảm bảo nhớ từ đầu phiên
            first_user_q = None
            for m in history:
                if m.get("role") == "user" and m.get("content", "").strip():
                    first_user_q = m["content"].strip()
                    break
            # Chọn thực thể tham chiếu: nếu câu hiện tại ngắn và mơ hồ, dùng câu đầu
            referent = None
            if any(pron in q_low for pron in [" nó", "cái đó", "cái này", "trong đó", "ở trên", "như trên"]):
                # Nếu lịch sử có >1 câu, thử dùng câu đầu làm gốc để không quên
                referent = first_user_q or last_user_q
            else:
                referent = last_user_q or first_user_q
            if referent:
                # Làm giàu câu hỏi bằng cách bổ sung thực thể
                # Ví dụ: "nó là gì?" -> "Quy chế đào tạo là gì?"
                # Nếu câu gốc chứa "nó", thay thế
                heuristic = question
                for pron in ["nó", "cái đó", "cái này", "trong đó", "ở trên"]:
                    if pron in q_low:
                        # Thay thế đơn giản: thêm referent vào
                        # Nếu câu chỉ là "nó là gì?" thì trả về referent dạng câu hỏi
                        if len(question.strip().split()) <= 4:
                            # Câu ngắn mơ hồ -> trả về dạng "Về [referent], " + question
                            # Hoặc thay "nó" bằng referent rút gọn
                            short_ref = referent[:80].rstrip("?.")
                            heuristic = question.lower().replace(pron, short_ref).strip()
                            # Viết hoa lại
                            if heuristic:
                                heuristic = heuristic[0].upper() + heuristic[1:]
                        else:
                            heuristic = heuristic.replace("nó", referent[:60]).replace("Nó", referent[:60])
                        break
                # Nếu heuristic khác gốc và dài hơn thì dùng
                if heuristic.strip().lower() != q_low and len(heuristic) > len(question):
                    return heuristic
                # Nếu không thay được pronoun, nối thêm ngữ cảnh đầu phiên
                if len(question.split()) <= 6 and first_user_q:
                    return f"{question} (liên quan đến: {first_user_q[:80]})"
        except Exception:
            pass
        return question


# ── Tóm tắt hội thoại ──


def summarize_history(history: list[dict]) -> str:
    """Tóm tắt lịch sử hội thoại dài thành 4-6 câu ngắn gọn - nhớ từ đầu phiên, cân bằng."""
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
        # Tăng giới hạn để giữ ý từ đầu phiên
        return summary.strip()[:900]
    except Exception:
        # Fallback: nối các message đầu thành tóm tắt thô, giữ thứ tự từ đầu
        parts = [f"{m.get('role')}: {m.get('content','')[:90]}" for m in history[:8]]
        return " | ".join(parts)[:700]


# ── Phát hiện đổi phong cách qua câu tự nhiên ──
_STYLE_INTENT_KEYWORDS = ["đổi", "chuyển", "phong cách", "giọng", "kiểu", "giải thích", "trả lời", "dùng", "không dùng", "đừng", "hãy", "theo", "style", "giong", "kieu"]

def detect_style_change(question: str) -> str | None:
    """Phát hiện yêu cầu đổi phong cách trong câu hỏi tự nhiên.
    Trả về tên style nếu phát hiện, None nếu không.
    Yêu cầu có cả từ khóa phong cách và từ khóa ý định để tránh nhầm."""
    if not question or not question.strip():
        return None
    q_low = question.lower()
    # kiểm tra có ý định đổi phong cách không
    has_intent = any(kw in q_low for kw in _STYLE_INTENT_KEYWORDS)
    if not has_intent:
        # vẫn kiểm tra trường hợp câu chỉ chứa kiểu plain đặc biệt như "không gọi hàm" dù không có từ đổi
        if "không gọi hàm" in q_low or "không dùng hàm" in q_low or "không giải thích bằng gọi hàm" in q_low:
            return "plain"
        return None
    # tìm style khớp keyword
    for style, kws in STYLE_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in q_low:
                # đảm bảo style hợp lệ
                if style in AVAILABLE_STYLES:
                    return style
    return None

def strip_style_instruction(question: str) -> str:
    """Tách phần chỉ thị đổi phong cách khỏi câu hỏi để lấy nội dung thực cần trả lời.
    Nếu câu chỉ là chỉ thị đổi phong cách không kèm câu hỏi thì trả về rỗng."""
    if not question:
        return question
    q_low = question.lower()
    for style, kws in STYLE_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in q_low:
                idx = q_low.find(kw.lower())
                after_raw = question[idx + len(kw):]
                after = after_raw.strip(" :,-.")
                lower_after = after.lower()
                for sep in [" và ", " rồi ", ", ", "; ", " và cho ", " và hãy ", "và ", "rồi ", ","]:
                    if sep.strip() and sep.strip() in lower_after:
                        sep_idx = lower_after.find(sep.strip())
                        if sep_idx != -1:
                            parts = after.split(sep.strip(), 1)
                            if len(parts) > 1 and len(parts[1].strip()) > 5:
                                return parts[1].strip()
                        if sep in lower_after:
                            parts = after.lower().split(sep.strip(), 1)
                            orig_idx = after.lower().find(sep.strip())
                            if orig_idx != -1:
                                cand = after[orig_idx + len(sep.strip()):].strip()
                                if len(cand) > 5:
                                    return cand
                if len(after) < 15 or lower_after.startswith("đi") or lower_after.startswith("nhé") or lower_after.startswith("nha"):
                    return ""
                return question
    return question


# ── Xã giao mở rộng ──
_SOCIAL_PROFESSIONAL_HINTS = ["quy chế", "quy định", "quyết định", "thông báo", "tài liệu", "học viện", "kỹ thuật mật mã", "đào tạo", "khảo thí", "học bổng", "tốt nghiệp", "tín chỉ"]

def is_history_recall_question(question: str) -> bool:
    """Phát hiện câu hỏi yêu cầu nhớ lại lịch sử hội thoại (câu đầu tiên, vừa hỏi gì...).
    Những câu này cần ưu tiên lịch sử thay vì ngữ cảnh retrieve."""
    if not question or not question.strip():
        return False
    q_low = question.strip().lower()
    recall_keywords = [
        "vừa hỏi", "vua hoi",
        "câu đầu", "cau dau",
        "câu thứ nhất", "cau thu nhat",
        "câu trước", "cau truoc",
        "hỏi gì trước", "hoi gi truoc",
        "nhớ lại", "nho lai",
        "tóm tắt lại hội thoại", "tom tat lai hoi thoai",
        "lịch sử hội thoại", "lich su hoi thoai",
        "đã hỏi gì", "da hoi gi",
        "câu hỏi đầu tiên",
        "ban đầu hỏi", "ban dau hoi",
    ]
    for kw in recall_keywords:
        if kw in q_low:
            return True
    # Mẫu "tôi đã hỏi gì" / "tôi vừa hỏi gì"
    if ("tôi" in q_low or "toi" in q_low) and ("hỏi" in q_low or "hoi" in q_low) and ("gì" in q_low or "gi" in q_low):
        # Nếu câu chứa đại từ hỏi về lịch sử thì coi là recall
        if any(x in q_low for x in ["vừa", "vua", "đã", "da", "trước", "truoc", "đầu", "dau"]):
            return True
    return False

def is_social_question(question: str, history: list[dict] | None = None) -> bool:
    """Nhận diện câu xã giao đời thường để đi nhánh không cần nguồn.
    Chỉ cung cấp ngoài lề khi có nguồn chính xác hoặc kiến thức huấn luyện -> nhánh xã giao cho phép dùng kiến thức chung."""
    if not ENABLE_SOCIAL:
        return False
    if not question or not question.strip():
        return False
    q_low = question.strip().lower()
    # Nếu là câu recall lịch sử thì không coi là xã giao (cần trả lời từ lịch sử)
    if is_history_recall_question(question):
        return False
    # Nếu chứa từ chuyên môn rõ ràng thì ưu tiên chuyên môn, không coi là xã giao
    for hint in _SOCIAL_PROFESSIONAL_HINTS:
        if hint in q_low:
            return False
    # Nếu câu cần viết lại do chứa đại từ mơ hồ + có lịch sử => likely là follow-up RAG, không phải xã giao
    # Kiểm tra đại từ tham chiếu
    if history and _needs_rewrite(question, history):
        # chứa "nó", "cái đó", "trong đó" v.v. và lịch sử tồn tại => không phải xã giao
        return False
    # Kiểm tra allowlist -> chắc chắn là xã giao
    for kw in SOCIAL_ALLOWLIST:
        if kw.lower() in q_low:
            return True
    # Các câu rất ngắn: chỉ coi là xã giao nếu khớp mẫu chào hỏi cụ thể, không phải bất kỳ câu ngắn nào
    # Đã thắt chặt: yêu cầu phải chứa từ khóa xã giao, không tự động cho mọi câu ngắn
    if len(q_low.split()) <= 4 and len(q_low) < 30:
        # Chỉ coi là xã giao nếu chứa từ chào hỏi/cảm ơn/tạm biệt trong allowlist hoặc là câu chào ngắn
        # Đã kiểm tra allowlist ở trên, nếu không khớp thì không phải xã giao
        # Loại bỏ logic broad "mọi câu ngắn đều là xã giao" gây nhầm với "nó là gì?" hoặc "đồ súc vật"
        return False
    return False

def generate_social(question: str, history: list[dict] | None = None, summary: str | None = None, style: str | None = None) -> str:
    """Sinh câu trả lời xã giao - mặc định casual, không cần nguồn, vẫn nhớ từ đầu phiên."""
    effective_style = style or "casual"
    if effective_style not in AVAILABLE_STYLES:
        effective_style = "casual"
    messages = build_social_messages(question, history, summary, style=effective_style)
    temp = STYLE_TEMPERATURE.get(effective_style, 0.7)
    try:
        if LLM_BACKEND == "ollama":
            return _call_ollama(messages, LLM_MODEL, GEN_MAX_TOKENS, LLM_THINK, temperature=temp)
        else:
            gen = _get_transformers_gen()
            prompt = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
            out = gen(prompt, max_new_tokens=GEN_MAX_TOKENS, do_sample=False, return_full_text=False)
            return out[0]["generated_text"].strip()
    except Exception as e:
        # Fallback khi Ollama chưa chạy: trả lời chào hỏi đơn giản, không cần LLM, để tránh Failed to fetch
        q_low = (question or "").lower()
        if any(kw in q_low for kw in ["xin chào", "chào bạn", "chào", "hello", "hi", "hey"]):
            return "Xin chào! Mình khỏe, cảm ơn bạn đã hỏi thăm. Bạn cần mình hỗ trợ gì về quy chế, quy định hay thông tin nào khác không?"
        if "khỏe" in q_low:
            return "Mình khỏe, cảm ơn bạn! Bạn đang cần tìm hiểu thêm về vấn đề gì?"
        if "cảm ơn" in q_low or "cam on" in q_low:
            return "Không có gì, rất vui được giúp bạn! Bạn cần hỏi thêm gì nữa không?"
        if "tạm biệt" in q_low or "bye" in q_low:
            return "Tạm biệt bạn nhé! Hẹn gặp lại khi bạn cần hỗ trợ thêm."
        if "bạn là ai" in q_low or "gioi thieu" in q_low:
            return "Mình là trợ lý ảo của Học viện Kỹ thuật Mật mã, luôn sẵn sàng hỗ trợ bạn tra cứu quy chế, quy định và các thông tin liên quan."
        # Fallback chung cho xã giao khác
        return "Chào bạn! Mình ở đây để trò chuyện và hỗ trợ bạn. Bạn muốn hỏi về vấn đề gì?"
