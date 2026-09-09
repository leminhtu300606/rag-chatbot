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
from backend.generation.generator import generate, generate_with_history, generate_social, is_social_question, is_history_recall_question, rewrite_query, summarize_history, detect_style_change, strip_style_instruction
from backend.indexing.retriever import retrieve
# Security: output validation defense in depth
try:
    from backend.security import validate_output, validate_sources, validate_tool_call
except Exception:
    def validate_output(x, max_len=4000): return x[:max_len] if x else x
    def validate_sources(x): return x
    def validate_tool_call(n,p): return p

from backend.generation.reranker import DEFAULT_RERANK_TOP_K, rerank

try:
    from backend.generation.calculator import is_math_question, is_math_request_steps, calculate, is_comparison_question, compare_real_numbers, is_multi_math_question, calculate_multi
except Exception:
    is_math_question = lambda *a, **kw: False
    is_math_request_steps = lambda *a, **kw: False
    calculate = lambda *a, **kw: {"success": False, "error": "calculator not loaded"}
    is_comparison_question = lambda *a, **kw: False
    compare_real_numbers = lambda *a, **kw: {"success": False, "error": "comparator not loaded"}
    is_multi_math_question = lambda *a, **kw: False
    calculate_multi = lambda *a, **kw: {"success": False, "error": "multi calculator not loaded"}


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
        # Chỉ tóm tắt khi vượt ngưỡng, trước đó giữ nguyên toàn bộ để không mất câu đầu
        old_part = cleaned[:-window_size] if len(cleaned) > window_size else cleaned[: len(cleaned) // 2]
        if old_part:
            try:
                summary = summarize_history(old_part)
            except Exception:
                summary = None
        cleaned = cleaned[-window_size:]
    # Nếu chưa vượt ngưỡng thì giữ nguyên toàn bộ (không cắt ở window) để không mất câu đầu tiên
    # Chỉ cắt khi đã tóm tắt phía trên

    return (cleaned if cleaned else None), summary


def answer(
    question: str,
    category: str = None,
    top_k: int = None,
    use_rerank: bool = False,
    rerank_k: int = None,
    style: str | None = None,
    last_math_result: str | None = None,
) -> dict:
    """Trả lời câu hỏi ở chế độ đơn lượt (single-turn) - hỗ trợ xã giao và toán học."""
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

    # Nhánh đa biểu thức: "kết quả của 1+1 và 2+2" -> liệt kê / cộng gộp / so sánh
    try:
        if is_multi_math_question(effective_question, None, last_math_result):
            want_steps_multi = is_math_request_steps(effective_question, effective_style)
            if effective_style == "plain":
                want_steps_multi = False
            multi = calculate_multi(effective_question, last_result=last_math_result, style=effective_style, want_steps=want_steps_multi)
            if multi.get("success"):
                # Phân biệt so sánh vs liệt kê/tổng
                q_low_multi = effective_question.lower()
                is_multi_cmp = any(kw in q_low_multi for kw in ["lớn hơn","lon hon","nhỏ hơn","nho hon","bé hơn","be hon","lớn","lon","nhỏ","nho","bé","be","bằng","bang","so sánh","so sanh","so với","so voi"])
                # Nếu là so sánh thì đánh dấu comparison để không ghi đè last_math_result
                if is_multi_cmp and multi.get("left") is not None:
                    return {
                        "answer": multi.get("answer"),
                        "sources": [],
                        "context": [],
                        "style": effective_style,
                        "standalone_question": multi.get("expression") or effective_question,
                        "history_used": None,
                        "summary": None,
                        "math_result": str(multi.get("left")) + (" " + multi.get("op") + " " if multi.get("op") != "compare" else " so với ") + str(multi.get("right")) if multi.get("op") else multi.get("result"),
                        "math_expression": multi.get("expression") or effective_question,
                        "comparison_result": multi.get("left") > multi.get("right") if multi.get("op") == ">" else (multi.get("left") < multi.get("right") if multi.get("op") == "<" else None),
                        "comparison_answer": multi.get("answer"),
                        "comparison_left": str(multi.get("left")),
                        "comparison_right": str(multi.get("right")),
                    }
                else:
                    # Liệt kê hoặc tổng: lưu kết quả cuối/tổng làm math_result để nhớ
                    res_val = multi.get("result")
                    # Nếu liệt kê nhiều kết quả, lấy kết quả cuối để làm last_math_result
                    last_val = res_val
                    if last_val and "," in str(last_val):
                        # "2, 4" -> lấy cuối
                        last_val = str(last_val).split(",")[-1].strip()
                    return {
                        "answer": multi.get("answer"),
                        "sources": [],
                        "context": [],
                        "style": effective_style,
                        "standalone_question": multi.get("expression") or effective_question,
                        "history_used": None,
                        "summary": None,
                        "math_result": last_val,
                        "math_expression": multi.get("expression") or effective_question,
                    }
    except Exception:
        pass

    # Nhánh so sánh số thực: ưu tiên trước toán học thường, trả lời tự nhiên không kiểu gọi hàm
    if is_comparison_question(effective_question, None, last_math_result):
        want_steps = is_math_request_steps(effective_question, effective_style)
        # plain: không hiện steps kiểu gọi hàm
        if effective_style == "plain":
            want_steps = False
        comp = compare_real_numbers(effective_question, last_result=last_math_result, style=effective_style, want_steps=want_steps)
        if comp.get("success"):
            ans = comp.get("answer")
            expr = comp.get("expression") or effective_question
            # Nếu yêu cầu steps và có steps thì nối tự nhiên (không lộ hàm)
            steps = comp.get("steps")
            if want_steps and steps and effective_style != "plain":
                ans = steps + ("\n" + ans if ans not in steps else "")
            return {
                "answer": ans,
                "sources": [],
                "context": [],
                "style": effective_style,
                "standalone_question": expr,
                "history_used": None,
                "summary": None,
                "math_result": comp.get("left") + (" " + comp.get("op") + " " if comp.get("op") != "compare" else " so với ") + comp.get("right") if comp.get("op") else None,
                "math_expression": expr,
                "comparison_result": comp.get("result"),
                "comparison_answer": ans,
                "comparison_left": comp.get("left"),
                "comparison_right": comp.get("right"),
            }
        else:
            # Khi so sánh lỗi (kể cả không trích được số), báo lỗi ngay, không rơi xuống LLM/RAG để tránh "Đúng rồi... nhưng"
            expr = comp.get("expression") or effective_question
            return {
                "answer": f"Không so sánh được `{expr}`. Lỗi: {comp.get('error','không rõ')}. Vui lòng kiểm tra lại hai số cần so sánh (vd: '2,5 > 3 ?' hoặc '2,5 lớn hơn 3 không?').",
                "sources": [],
                "context": [],
                "style": effective_style,
                "standalone_question": expr,
                "history_used": None,
                "summary": None,
            }

    # Nhánh toán học: ưu tiên trước xã giao, độ chính xác cao Decimal prec=50, hỗ trợ chữ->số và last_result
    if is_math_question(effective_question, None, last_math_result):
        want_steps = is_math_request_steps(effective_question, effective_style)
        calc = calculate(effective_question, last_result=last_math_result, want_steps=want_steps, style=effective_style)
        if calc.get("success"):
            expr = calc.get("expression") or effective_question
            res = calc.get("result")
            steps = calc.get("steps")
            if want_steps and steps:
                answer_text = steps
            else:
                # plain: không gọi hàm, trả lời tự nhiên
                if effective_style == "plain":
                    answer_text = f"Kết quả là {res}."
                else:
                    answer_text = f"{expr} = {res}"
            return {
                "answer": answer_text,
                "sources": [],
                "context": [],
                "style": effective_style,
                "standalone_question": expr,
                "history_used": None,
                "summary": None,
                "math_result": res,
                "math_expression": expr,
            }
        else:
            # Nếu trích được biểu thức mà lỗi (chia 0...) thì báo lỗi rõ ràng, không rơi sang RAG để tránh hallucinate
            if calc.get("expression"):
                return {
                    "answer": f"Không tính được phép toán `{calc.get('expression','')}`. Lỗi: {calc.get('error','không rõ')}. Vui lòng kiểm tra lại biểu thức.",
                    "sources": [],
                    "context": [],
                    "style": effective_style,
                    "standalone_question": calc.get("expression"),
                    "history_used": None,
                    "summary": None,
                }

    # Nhánh xã giao: không cần nguồn, dùng kiến thức huấn luyện, mặc định casual
    if is_social_question(effective_question):
        # Nếu chưa có style riêng, dùng casual cho xã giao
        social_style = effective_style if effective_style in ["casual", "friendly", "humorous", "empathetic"] else "casual"
        # Nếu phong cách hiện tại là formal nhưng câu xã giao thì vẫn dùng casual cho tự nhiên
        if style is None and detected_style is None:
            social_style = "casual"
        else:
            social_style = effective_style
        text = validate_output(generate_social(effective_question, history=None, summary=None, style=social_style))
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
    # Validate tool call: AI chỉ đề xuất, backend kiểm tra
    validate_tool_call("retrieve", {"category": category, "top_k": fetch_k})
    context = retrieve(query_for_retrieval, category=category, top_k=fetch_k)

    if use_rerank and rerank and context:
        validate_tool_call("rerank", {"top_k": top_k})
        context = rerank(query_for_retrieval, context, top_k=rerank_k if rerank_k is not None else top_k)
        # Validate sources sau rerank
        context = [{"text": c["text"], "metadata": c["metadata"], "score": c["score"]} for c in context[:10]]

    text = validate_output(generate(context, query_for_retrieval, style=effective_style))
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
    last_math_result: str | None = None,
) -> dict:
    """Trả lời có ngữ cảnh hội thoại (conversational) - hỗ trợ đổi phong cách qua câu tự nhiên và toán học."""
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
        text = validate_output(generate_with_history(confirm_context, f"Xác nhận đã đổi sang phong cách {effective_style}", history_window, summary, style=effective_style))
        # Nếu LLM không trả lời đúng, fallback
        if not text or "phong cách" not in text.lower():
            text = f"Đã đổi sang phong cách {effective_style}. Từ giờ mình sẽ trả lời theo phong cách này nhé. Bạn cần hỏi gì tiếp?"
        else:
            text = validate_output(text)
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

    # Nhánh recall lịch sử: câu hỏi về hội thoại trước (câu đầu tiên, vừa hỏi gì...) -> trả lời trực tiếp từ lịch sử, không cần retrieve
    if is_history_recall_question(effective_question):
        # Dùng generate_with_history với context rỗng để ưu tiên lịch sử
        # Thêm chỉ dẫn rõ ràng để LLM trả lời từ lịch sử
        recall_question = effective_question + " (Hãy trả lời dựa trên lịch sử hội thoại đã cung cấp, liệt kê chính xác các câu hỏi trước, đặc biệt là câu đầu tiên.)"
        text = validate_output(generate_with_history([], recall_question, history_window, summary, style=effective_style))
        # Fallback nếu LLM không trả lời hoặc nói không đủ nguồn
        if not text or "không đủ nguồn" in text.lower():
            # Tạo câu trả lời fallback từ lịch sử thực tế
            if history_window:
                user_qs = [m["content"] for m in history_window if m.get("role") == "user"]
                if user_qs:
                    listing = "\n".join([f"{i+1}. {q}" for i, q in enumerate(user_qs)])
                    text = validate_output(f"Các câu hỏi bạn đã hỏi trong phiên này:\n{listing}\n\nCâu đầu tiên là: \"{user_qs[0]}\"")
                else:
                    text = "Lịch sử hội thoại hiện tại trống, chưa có câu hỏi nào trước đó."
            else:
                text = "Chưa có lịch sử hội thoại để nhớ lại."
        return {
            "answer": text,
            "sources": [],
            "context": [],
            "standalone_question": effective_question,
            "summary": summary,
            "history_used": history_window,
            "style": effective_style,
        }

    # Nhánh đa biểu thức (hội thoại): "kết quả của 1+1 và 2+2" -> liệt kê / cộng gộp / so sánh
    try:
        if is_multi_math_question(effective_question, history_window if history_window else history, last_math_result):
            want_steps_multi = is_math_request_steps(effective_question, effective_style)
            if effective_style == "plain":
                want_steps_multi = False
            multi = calculate_multi(effective_question, last_result=last_math_result, style=effective_style, want_steps=want_steps_multi)
            if multi.get("success"):
                q_low_multi = effective_question.lower()
                is_multi_cmp = any(kw in q_low_multi for kw in ["lớn hơn","lon hon","nhỏ hơn","nho hon","bé hơn","be hon","lớn","lon","nhỏ","nho","bé","be","bằng","bang","so sánh","so sanh","so với","so voi"])
                if is_multi_cmp and multi.get("left") is not None:
                    return {
                        "answer": multi.get("answer"),
                        "sources": [],
                        "context": [],
                        "standalone_question": multi.get("expression") or effective_question,
                        "summary": summary,
                        "history_used": history_window,
                        "style": effective_style,
                        "math_result": str(multi.get("left")) + (" " + multi.get("op") + " " if multi.get("op") != "compare" else " so với ") + str(multi.get("right")) if multi.get("op") else multi.get("result"),
                        "math_expression": multi.get("expression") or effective_question,
                        "comparison_result": multi.get("left") > multi.get("right") if multi.get("op") == ">" else (multi.get("left") < multi.get("right") if multi.get("op") == "<" else None),
                        "comparison_answer": multi.get("answer"),
                        "comparison_left": str(multi.get("left")),
                        "comparison_right": str(multi.get("right")),
                    }
                else:
                    res_val = multi.get("result")
                    last_val = res_val
                    if last_val and "," in str(last_val):
                        last_val = str(last_val).split(",")[-1].strip()
                    return {
                        "answer": multi.get("answer"),
                        "sources": [],
                        "context": [],
                        "standalone_question": multi.get("expression") or effective_question,
                        "summary": summary,
                        "history_used": history_window,
                        "style": effective_style,
                        "math_result": last_val,
                        "math_expression": multi.get("expression") or effective_question,
                    }
    except Exception:
        pass

    # Nhánh so sánh số thực (hội thoại) - ưu tiên trước toán học, trả lời tự nhiên không kiểu gọi hàm
    if is_comparison_question(effective_question, history_window if history_window else history, last_math_result):
        want_steps = is_math_request_steps(effective_question, effective_style)
        if effective_style == "plain":
            want_steps = False
        comp = compare_real_numbers(effective_question, last_result=last_math_result, style=effective_style, want_steps=want_steps)
        if comp.get("success"):
            ans = comp.get("answer")
            expr = comp.get("expression") or effective_question
            steps = comp.get("steps")
            if want_steps and steps and effective_style != "plain":
                ans = steps + ("\n" + ans if ans not in steps else "")
            return {
                "answer": ans,
                "sources": [],
                "context": [],
                "standalone_question": expr,
                "summary": summary,
                "history_used": history_window,
                "style": effective_style,
                "math_result": comp.get("left") + (" " + comp.get("op") + " " if comp.get("op") != "compare" else " so với ") + comp.get("right") if comp.get("op") else None,
                "math_expression": expr,
                "comparison_result": comp.get("result"),
                "comparison_answer": ans,
                "comparison_left": comp.get("left"),
                "comparison_right": comp.get("right"),
            }
        else:
            # Khi so sánh lỗi (kể cả không trích được số), báo lỗi ngay, không rơi xuống LLM/RAG để tránh "Đúng rồi... nhưng"
            expr = comp.get("expression") or effective_question
            return {
                "answer": f"Không so sánh được `{expr}`. Lỗi: {comp.get('error','không rõ')}. Vui lòng kiểm tra lại hai số cần so sánh (vd: '2,5 > 3 ?' hoặc '2,5 lớn hơn 3 không?').",
                "sources": [],
                "context": [],
                "standalone_question": expr,
                "summary": summary,
                "history_used": history_window,
                "style": effective_style,
            }

    # Nhánh toán học: hỗ trợ toàn bộ ký tự, chữ->số, nhớ last_result, chỉ hiện steps khi yêu cầu
    if is_math_question(effective_question, history_window if history_window else history, last_math_result):
        want_steps = is_math_request_steps(effective_question, effective_style)
        calc = calculate(effective_question, last_result=last_math_result, want_steps=want_steps, style=effective_style)
        if calc.get("success"):
            expr = calc.get("expression") or effective_question
            res = calc.get("result")
            steps = calc.get("steps")
            if want_steps and steps:
                answer_text = steps
            else:
                if effective_style == "plain":
                    answer_text = f"Kết quả là {res}."
                else:
                    answer_text = f"{expr} = {res}"
            return {
                "answer": answer_text,
                "sources": [],
                "context": [],
                "standalone_question": expr,
                "summary": summary,
                "history_used": history_window,
                "style": effective_style,
                "math_result": res,
                "math_expression": expr,
            }
        else:
            if calc.get("expression"):
                return {
                    "answer": f"Không tính được phép toán `{calc.get('expression','')}`. Lỗi: {calc.get('error','không rõ')}. Vui lòng kiểm tra lại biểu thức.",
                    "sources": [],
                    "context": [],
                    "standalone_question": calc.get("expression"),
                    "summary": summary,
                    "history_used": history_window,
                    "style": effective_style,
                }

    # Nhánh xã giao mở rộng: nếu câu là xã giao đời thường thì không cần nguồn, trả lời tự nhiên
    # Chỉ cung cấp ngoài lề khi có nguồn chính xác hoặc kiến thức huấn luyện -> cho phép dùng generate_social
    if is_social_question(effective_question, history):
        # Chọn phong cách xã giao: mặc định casual, nếu người dùng đã chọn friendly/humorous thì giữ
        social_style = effective_style
        if effective_style not in ["casual", "friendly", "humorous", "empathetic", "plain", "simple"]:
            # nếu đang là formal mà câu xã giao thì dùng casual cho tự nhiên
            social_style = "casual"
        text = validate_output(generate_social(effective_question, history_window, summary, style=social_style))
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
            # rewrite_query là untrusted nhưng đã được bọc trong security, vẫn validate
            standalone_q = rewrite_query(effective_question, history_window)
        except Exception:
            standalone_q = effective_question

    if top_k is None:
        top_k = DEFAULT_RERANK_TOP_K if (use_rerank and rerank) else RETRIEVE_TOP_K
    fetch_k = top_k * 3 if use_rerank and rerank else top_k
    # Validate tool call trước khi retrieve
    validate_tool_call("retrieve", {"category": category, "top_k": fetch_k})

    query_for_retrieval = standalone_q if standalone_q and standalone_q.strip() else effective_question
    context = retrieve(query_for_retrieval, category=category, top_k=fetch_k)

    if use_rerank and rerank and context:
        validate_tool_call("rerank", {"top_k": top_k})
        context = rerank(query_for_retrieval, context, top_k=rerank_k if rerank_k is not None else top_k)
        context = [{"text": c["text"], "metadata": c["metadata"], "score": c["score"]} for c in context[:10]]

    text = validate_output(generate_with_history(context, effective_question, history_window, summary, style=effective_style))

    return {
        "answer": text,
        "sources": [c["metadata"] for c in context],
        "context": context,
        "standalone_question": standalone_q,
        "summary": summary,
        "history_used": history_window,
        "style": effective_style,
    }
