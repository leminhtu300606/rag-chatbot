"""Agent da buoc (gop tu agent/state.py + tools.py + nodes.py + graph.py: 1 dict duy nhat di qua moi node).

StateGraph: rewrite -> retrieve -> rerank -> generate -> selfcheck -loop-> retrieve (max AGENT_MAX_STEPS).
Tools boc lai logic cu (khong doi thuat toan), de node goi deterministically.
"""
from __future__ import annotations

from typing import TypedDict


class ChatState(TypedDict, total=False):
    question: str            # cau goc (effective_question, da tach style)
    standalone_question: str # cau viet lai dung cho retrieval
    history: list            # history_window (da chuan hoa)
    summary: str | None
    style: str | None
    category: str | None
    top_k: int
    use_rerank: bool
    session_id: str | None
    context: list            # [{text, metadata, score}]
    answer: str
    sources: list
    attempts: int            # so lan generate (max 3)
    needs_retry: bool
    retry_reason: str


# ── Tools ──

def retrieve_tool(query: str, category=None, top_k: int = 5, session_id: str | None = None) -> list:
    from backend.generation.rag import _do_retrieve, _ensure_trang_phuc_coverage
    ctx = _do_retrieve(query, category=category, top_k=top_k, session_id=session_id)
    ctx = _ensure_trang_phuc_coverage(query, ctx)
    try:
        from backend.common.utils import deduplicate_docs
        ctx = deduplicate_docs(ctx, threshold=0.92)
    except Exception:
        pass
    return ctx or []


def rerank_tool(query: str, context: list, top_k: int | None = None) -> list:
    if not context:
        return []
    try:
        from backend.generation.reranker import rerank, DEFAULT_RERANK_TOP_K
        k = top_k if top_k is not None else DEFAULT_RERANK_TOP_K
        out = rerank(query, context, top_k=k)
        return out or context[:k]
    except Exception:
        return context


def generate_tool(context: list, question: str, history=None, summary=None, style=None) -> str:
    from backend.generation.generator import generate_with_history
    from backend.config.settings import validate_output
    text = generate_with_history(context, question, history, summary, style=style)
    return validate_output(text)


def self_check_tool(question: str, answer: str, context: list) -> dict:
    """Kiem tra dap co du nguon khong. Heuristic nhanh (khong ton LLM), du cho CPU.

    Tra ve {"sufficient": bool, "reason": str}.
    """
    ans = (answer or "").strip()
    low = ans.lower()
    if not ans:
        return {"sufficient": False, "reason": "empty_answer"}
    if not context:
        return {"sufficient": False, "reason": "no_context"}
    # Model tu thu nhan thieu nguon -> dung, khong can retry vo han
    if "không đủ nguồn" in low or "khong du nguon" in low:
        return {"sufficient": True, "reason": "admitted_lack"}
    # Can trich dan cho cau chuyen mon
    if "[" not in ans or "]" not in ans:
        # Cau ngan xa giao thi ok, cau dai chuyen mon thieu citation -> retry 1 lan
        if len(ans) > 300:
            return {"sufficient": False, "reason": "missing_citation"}
    # Tu khoa chinh cua cau hoi co xuat hien trong context khong?
    try:
        import re as _re
        toks = [w for w in _re.findall(r"\w+", question.lower()) if len(w) > 4][:4]
        if toks:
            joined = " ".join((c.get("text", "") or "").lower() for c in context)
            hits = sum(1 for t in toks if t in joined)
            if hits == 0:
                return {"sufficient": False, "reason": "keyword_miss"}
    except Exception:
        pass
    return {"sufficient": True, "reason": "ok"}


# ── Nodes: moi node nhan state, tra ve partial update ──

def _max_steps() -> int:
    try:
        import backend.config.settings as cfg
        return max(1, min(5, int(getattr(cfg, "AGENT_MAX_STEPS", 3))))
    except Exception:
        return 3


def n_rewrite(state: dict) -> dict:
    q = state.get("question", "")
    hist = state.get("history")
    standalone = q
    if hist:
        try:
            from backend.generation.generator import rewrite_query
            standalone = rewrite_query(q, hist) or q
        except Exception:
            standalone = q
    return {"standalone_question": standalone}


def _fetch_k(top_k: int, use_rerank: bool) -> int:
    if use_rerank:
        return min(max(1, top_k) * 5, 15)
    return max(1, top_k)


def n_retrieve(state: dict) -> dict:
    q = state.get("standalone_question") or state.get("question", "")
    # Lan thu lai: mo rong query bang tu khoa goc de tang recall
    if state.get("attempts", 0) > 0 and state.get("question"):
        q = f"{q} {state.get('question', '')}".strip()[:1000]
    ctx = retrieve_tool(q, category=state.get("category"), top_k=_fetch_k(int(state.get("top_k", 5)), bool(state.get("use_rerank"))), session_id=state.get("session_id"))
    return {"context": ctx}


def n_rerank(state: dict) -> dict:
    if not bool(state.get("use_rerank")):
        k = int(state.get("top_k", 5))
        return {"context": (state.get("context") or [])[:k]}
    q = state.get("standalone_question") or state.get("question", "")
    ctx = rerank_tool(q, state.get("context") or [], top_k=int(state.get("top_k", 5)))
    flat = [{"text": c.get("text", ""), "metadata": c.get("metadata", {}), "score": c.get("score", 0.0)} for c in ctx[:10]]
    return {"context": flat}


def n_generate(state: dict) -> dict:
    ctx = state.get("context") or []
    ans = generate_tool(ctx, state.get("question", ""), history=state.get("history"), summary=state.get("summary"), style=state.get("style"))
    sources = [c.get("metadata", {}) for c in ctx] if ctx else []
    return {"answer": ans, "sources": sources, "attempts": int(state.get("attempts", 0)) + 1}


def n_selfcheck(state: dict) -> dict:
    try:
        max_steps = _max_steps()
    except Exception:
        max_steps = 3
    res = self_check_tool(state.get("question", ""), state.get("answer", ""), state.get("context") or [])
    attempts = int(state.get("attempts", 0))
    # Chi retry khi thieu that (missing_citation/keyword_miss/no_context/empty), toi da max_steps lan generate
    retryable = res.get("reason") in ("missing_citation", "keyword_miss", "no_context", "empty_answer")
    needs = (not res.get("sufficient")) and retryable and attempts < max_steps
    return {"needs_retry": bool(needs), "retry_reason": res.get("reason", "")}


# ── Graph ──

_compiled = None


def is_available() -> bool:
    try:
        import langgraph  # noqa: F401
        return True
    except Exception:
        return False


def is_enabled() -> bool:
    try:
        import backend.config.settings as cfg
        if not bool(getattr(cfg, "USE_AGENT", True)):
            return False
    except Exception:
        pass
    return is_available()


def build_graph():
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver

    g = StateGraph(ChatState)
    g.add_node("rewrite", n_rewrite)
    g.add_node("retrieve", n_retrieve)
    g.add_node("rerank", n_rerank)
    g.add_node("generate", n_generate)
    g.add_node("selfcheck", n_selfcheck)

    g.add_edge(START, "rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "generate")
    g.add_edge("generate", "selfcheck")

    def _route(s: dict) -> str:
        if s.get("needs_retry"):
            return "retrieve"
        return END

    g.add_conditional_edges("selfcheck", _route, {"retrieve": "retrieve", END: END})
    return g.compile(checkpointer=MemorySaver())


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


def run_agent(question: str, history=None, summary=None, style=None, category=None, top_k: int = 5, use_rerank: bool = True, session_id: str | None = None) -> dict:
    """Chay agent 1 luot, tra ve dict tuong thich rag.answer_with_history."""
    import uuid
    graph = get_graph()
    init = {
        "question": question,
        "standalone_question": question,
        "history": history,
        "summary": summary,
        "style": style,
        "category": category,
        "top_k": int(top_k or 5),
        "use_rerank": bool(use_rerank),
        "session_id": session_id,
        "context": [],
        "answer": "",
        "sources": [],
        "attempts": 0,
        "needs_retry": False,
        "retry_reason": "",
    }
    cfg_invoke = {"configurable": {"thread_id": session_id or f"agent-{uuid.uuid4().hex[:8]}"}}
    out = graph.invoke(init, config=cfg_invoke)
    return {
        "answer": out.get("answer", ""),
        "sources": out.get("sources", []),
        "context": out.get("context", []),
        "standalone_question": out.get("standalone_question", question),
        "attempts": int(out.get("attempts", 1)),
        "retry_reason": out.get("retry_reason", ""),
    }


def run_agent_stream(question: str, history=None, summary=None, style=None, category=None, top_k: int = 5, use_rerank: bool = True, session_id: str | None = None):
    """Stream theo node (dung cho SSE sau nay). Yield (node_name, state_update)."""
    import uuid
    graph = get_graph()
    init = {
        "question": question,
        "standalone_question": question,
        "history": history,
        "summary": summary,
        "style": style,
        "category": category,
        "top_k": int(top_k or 5),
        "use_rerank": bool(use_rerank),
        "session_id": session_id,
        "context": [],
        "answer": "",
        "sources": [],
        "attempts": 0,
        "needs_retry": False,
        "retry_reason": "",
    }
    cfg_invoke = {"configurable": {"thread_id": session_id or f"agent-{uuid.uuid4().hex[:8]}"}}
    yield from graph.stream(init, config=cfg_invoke, stream_mode="updates")


__all__ = [
    "ChatState",
    "retrieve_tool", "rerank_tool", "generate_tool", "self_check_tool",
    "n_rewrite", "n_retrieve", "n_rerank", "n_generate", "n_selfcheck",
    "build_graph", "get_graph", "is_available", "is_enabled",
    "run_agent", "run_agent_stream",
]
