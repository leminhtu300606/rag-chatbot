"""
backend/app.py - FastAPI cho RAG Chatbot
=========================================
Nhiệm vụ:
- Cung cấp API cho frontend và quản lý hội thoại có ngữ cảnh.
Endpoints:
  GET  /api/health
  GET  /api/stats
  GET  /api/categories
  POST /api/chat                 # hỏi đáp, hỗ trợ lịch sử hội thoại và session
  GET  /api/sessions             # danh sách phiên
  GET  /api/sessions/{id}        # chi tiết phiên
  DELETE /api/sessions/{id}      # xóa phiên

Chạy: uvicorn backend.app:app --reload --port 8000
      hoặc: python -m backend.app
"""
from pathlib import Path
import sys

# Fix import khi chay truc tiep
if __package__ in (None, ""):
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import time
import uuid
import threading

import backend.config as cfg

def _print_banner(host: str = "0.0.0.0", port: int = 8000):
    """In link chay ra console cho de thay."""
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    base = f"http://{display_host}:{port}"
    bar = "=" * 60
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("\n" + bar, flush=True)
    print(" RAG Chatbot - Hoc vien Ky thuat Mat ma - DA SAN SANG", flush=True)
    print(bar, flush=True)
    print(f"  Frontend  : {base}/", flush=True)
    print(f"  API Docs  : {base}/docs", flush=True)
    print(f"  API Health: {base}/api/health", flush=True)
    print(f"  API Stats : {base}/api/stats", flush=True)
    print(f"  API Chat  : POST {base}/api/chat", flush=True)
    print(bar, flush=True)
    print(f"  LLM       : {cfg.LLM_BACKEND}/{cfg.LLM_MODEL} @ {cfg.OLLAMA_BASE}", flush=True)
    print(f"  Embed     : {cfg.EMBED_MODEL}", flush=True)
    print("  Data      : xem /api/health de biet so files/vectors", flush=True)
    print(bar + "\n", flush=True)


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # In banner khi chay bang `uvicorn backend.app:app` (khong qua __main__)
    try:
        _print_banner(host="0.0.0.0", port=8000)
    except Exception:
        pass
    yield

app = FastAPI(
    title="RAG Chatbot - Hoc vien Ky thuat Mat ma",
    description="API hỏi đáp trên kho tài liệu nội bộ (quy chế, quy định, thông báo...) - hỗ trợ hội thoại có ngữ cảnh",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Quản lý phiên hội thoại (session memory) ──
# Lưu trữ in-memory: {session_id: {"history": [...], "summary": str|None, "updated_at": float, "created_at": float}}
_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()
MAX_SESSIONS = 200
SESSION_TTL_SECONDS = 24 * 3600  # 24h

def _cleanup_sessions():
    now = time.time()
    with _sessions_lock:
        expired = [sid for sid, v in _sessions.items() if now - v.get("updated_at", 0) > SESSION_TTL_SECONDS]
        for sid in expired:
            _sessions.pop(sid, None)
        # gioi han so luong: xoa cu nhat
        if len(_sessions) > MAX_SESSIONS:
            sorted_sids = sorted(_sessions.items(), key=lambda x: x[1].get("updated_at", 0))
            for sid, _ in sorted_sids[: len(_sessions) - MAX_SESSIONS]:
                _sessions.pop(sid, None)

def _get_or_create_session(session_id: Optional[str]) -> str:
    _cleanup_sessions()
    if session_id and isinstance(session_id, str) and session_id.strip():
        sid = session_id.strip()
        with _sessions_lock:
            if sid not in _sessions:
                _sessions[sid] = {"history": [], "summary": None, "created_at": time.time(), "updated_at": time.time()}
        return sid
    # tao moi
    sid = str(uuid.uuid4())
    with _sessions_lock:
        _sessions[sid] = {"history": [], "summary": None, "created_at": time.time(), "updated_at": time.time()}
    return sid

def _append_to_session(session_id: str, user_q: str, assistant_a: str, summary: Optional[str] = None):
    with _sessions_lock:
        sess = _sessions.get(session_id)
        if not sess:
            _sessions[session_id] = {"history": [], "summary": summary, "created_at": time.time(), "updated_at": time.time()}
            sess = _sessions[session_id]
        sess["history"].append({"role": "user", "content": user_q})
        sess["history"].append({"role": "assistant", "content": assistant_a})
        if summary:
            sess["summary"] = summary
        sess["updated_at"] = time.time()
        # gioi han lich su luu toi da 100 messages de tranh phinh
        if len(sess["history"]) > 100:
            sess["history"] = sess["history"][-100:]

# --- Models ---
class ChatMessage(BaseModel):
    role: str = Field(..., description="user | assistant | system")
    content: str

class ChatRequest(BaseModel):
    question: str
    category: Optional[str] = None
    top_k: Optional[int] = None
    use_rerank: bool = True
    show_context: bool = False
    # Lịch sử hội thoại
    history: Optional[List[ChatMessage]] = None
    session_id: Optional[str] = None
    use_history: bool = True  # cho phép tắt lịch sử để hỏi đơn lượt

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    context: Optional[List[Dict[str, Any]]] = None
    # Thông tin hội thoại trả về
    session_id: Optional[str] = None
    standalone_question: Optional[str] = None
    summary: Optional[str] = None
    history_used: Optional[List[Dict[str, Any]]] = None

# --- Helpers ---
def _get_stats():
    try:
        from backend.preprocessing.storage import get_processed_files, count_chunks
        from backend.indexing.vectorstore import count as vec_count, get_stats as vec_stats
        files = get_processed_files()
        return {
            "processed_files": len(files),
            "total_chunks": count_chunks(),
            "chroma_count": vec_count(),
            "chroma_stats": vec_stats(),
            "embed_model": cfg.EMBED_MODEL,
            "llm_backend": cfg.LLM_BACKEND,
            "llm_model": cfg.LLM_MODEL,
            "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "conversation_window": cfg.CONVERSATION_WINDOW,
            "conversation_threshold": cfg.CONVERSATION_SUMMARY_THRESHOLD,
            "active_sessions": len(_sessions),
        }
    except Exception as e:
        return {"error": str(e)}

# --- Routes ---
@app.get("/api")
async def api_root():
    return {"message": "RAG Chatbot API", "docs": "/docs", "health": "/api/health"}

@app.get("/api/health")
async def health():
    try:
        from backend.indexing.vectorstore import count
        from backend.preprocessing.storage import get_processed_files
        c = count()
        files = get_processed_files()
        ready = c > 0 and len(files) > 0
        return {
            "status": "ready" if ready else "not_ready",
            "chroma_count": c,
            "processed_files": len(files),
            "message": "OK" if ready else "Chua co data. Hay chay: python -m backend.clean && python -m backend.index --rebuild",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/api/stats")
async def stats():
    return _get_stats()

@app.get("/api/categories")
async def categories():
    return {"categories": cfg.CATEGORY_ORDER}

# Endpoint quản lý phiên hội thoại
@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    with _sessions_lock:
        sess = _sessions.get(session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "session_id": session_id,
            "history": sess["history"],
            "summary": sess.get("summary"),
            "created_at": sess.get("created_at"),
            "updated_at": sess.get("updated_at"),
            "turns": len(sess["history"]) // 2,
        }

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    with _sessions_lock:
        if session_id in _sessions:
            _sessions.pop(session_id)
            return {"deleted": session_id}
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/api/sessions")
async def list_sessions():
    with _sessions_lock:
        return {
            "sessions": [
                {
                    "session_id": sid,
                    "turns": len(v["history"]) // 2,
                    "updated_at": v.get("updated_at"),
                    "created_at": v.get("created_at"),
                    "summary": v.get("summary"),
                }
                for sid, v in _sessions.items()
            ],
            "total": len(_sessions),
        }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Cau hoi rong")

    # Validate category
    if req.category and req.category not in cfg.CATEGORY_ORDER:
        raise HTTPException(status_code=400, detail=f"Category khong hop le. Chon trong: {cfg.CATEGORY_ORDER}")

    # Kiem tra co data truoc khi goi LLM (tranh goi Ollama khi chua co data)
    try:
        from backend.indexing.vectorstore import count as vec_count
        from backend.preprocessing.storage import get_processed_files
        if vec_count() == 0 or len(get_processed_files()) == 0:
            sid = _get_or_create_session(req.session_id) if req.session_id or req.history else None
            return ChatResponse(
                answer="Chua co du lieu de tra loi. Vui long chay: python -m backend.clean && python -m backend.index --rebuild, sau do thu lai.",
                sources=[],
                context=[] if req.show_context else None,
                session_id=sid,
            )
    except Exception:
        pass

    try:
        # Chọn pipeline có/không có lịch sử hội thoại
        use_history = req.use_history and (req.history or req.session_id)

        # Hợp nhất lịch sử từ session lưu trên server và request từ client
        effective_history: Optional[List[Dict[str, Any]]] = None
        session_id: Optional[str] = None

        if use_history:
            # Lay hoac tao session
            if req.session_id or req.history:
                session_id = _get_or_create_session(req.session_id)
                # Lay history tu session store
                with _sessions_lock:
                    sess_hist = list(_sessions[session_id]["history"]) if session_id in _sessions else []
                # Merge voi history tu client (client history uu tien neu dai hon)
                if req.history:
                    # req.history la List[ChatMessage], convert sang dict
                    client_hist = [{"role": m.role, "content": m.content} for m in req.history]
                    # neu client gui day du hon server thi dung client, nguoc lai merge
                    if len(client_hist) > len(sess_hist):
                        effective_history = client_hist
                    else:
                        # merge: server history + client history moi nhat (tranh duplicate)
                        # Diem don gian: neu client history khac rong thi dung client, con khong dung server
                        effective_history = client_hist if client_hist else sess_hist
                    # dong bo lai session store neu client dai hon
                    if len(client_hist) > len(sess_hist):
                        with _sessions_lock:
                            if session_id in _sessions:
                                _sessions[session_id]["history"] = client_hist
                                _sessions[session_id]["updated_at"] = time.time()
                else:
                    effective_history = sess_hist if sess_hist else None
            else:
                # khong co session_id nhung co history: dung truc tiep
                if req.history:
                    effective_history = [{"role": m.role, "content": m.content} for m in req.history]
        else:
            # single-turn
            effective_history = None
            session_id = req.session_id  # giu nguyen neu client gui

        # Goi RAG tuong ung
        if effective_history:
            from backend.generation.rag import answer_with_history
            res = answer_with_history(
                question=req.question,
                history=effective_history,
                category=req.category,
                top_k=req.top_k,
                use_rerank=req.use_rerank,
            )
            # Luu vao session sau khi co ket qua
            if session_id:
                _append_to_session(session_id, req.question, res["answer"], res.get("summary"))

            if not res.get("context"):
                return ChatResponse(
                    answer="Chua co du lieu de tra loi. Vui long chay: python -m backend.clean && python -m backend.index --rebuild, sau do thu lai.",
                    sources=[],
                    context=[] if req.show_context else None,
                    session_id=session_id,
                    standalone_question=res.get("standalone_question"),
                    summary=res.get("summary"),
                    history_used=res.get("history_used"),
                )
            context = res.get("context", []) if req.show_context else None
            return ChatResponse(
                answer=res["answer"],
                sources=res.get("sources", []),
                context=context,
                session_id=session_id,
                standalone_question=res.get("standalone_question"),
                summary=res.get("summary"),
                history_used=res.get("history_used"),
            )
        else:
            # Chế độ đơn lượt (không dùng lịch sử)
            from backend.generation.rag import answer
            res = answer(
                question=req.question,
                category=req.category,
                top_k=req.top_k,
                use_rerank=req.use_rerank,
            )
            if not res.get("context"):
                return ChatResponse(
                    answer="Chua co du lieu de tra loi. Vui long chay: python -m backend.clean && python -m backend.index --rebuild, sau do thu lai.",
                    sources=[],
                    context=[] if req.show_context else None,
                    session_id=session_id,
                )
            context = None
            if req.show_context:
                context = res.get("context", [])

            # Van luu vao session neu co session_id du la single-turn
            if session_id:
                _append_to_session(session_id, req.question, res["answer"])

            return ChatResponse(
                answer=res["answer"],
                sources=res.get("sources", []),
                context=context,
                session_id=session_id,
            )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        msg = str(e).lower()
        # Handle dimension mismatch hint
        if "dimension" in msg:
            raise HTTPException(
                status_code=500,
                detail="Loi dimension embedding. Hay chay: python -m backend.index --rebuild (hoac backend.build_index). Chi tiet: " + str(e),
            )
        # Ollama OOM - RAM/VRAM không đủ
        if any(k in msg for k in ["out-of-memory", "failed to allocate", "ggml", "memory"]):
            raise HTTPException(
                status_code=503,
                detail=str(e),
            )
        # Ollama / connection errors (bao gom 500 Server Error tu Ollama)
        if any(k in msg for k in ["ollama", "connection", "http://localhost:11434", "500 server error", "connection refused", "failed to connect", "not found"]):
            raise HTTPException(
                status_code=503,
                detail="Khong ket noi duoc Ollama (http://localhost:11434) hoac model loi. Kiem tra: ollama serve & ollama list & ollama pull. Chi tiet: " + str(e),
            )
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clean")
async def trigger_clean():
    """Trigger clean (preprocessing) - async demo, thuc te nen chay background task"""
    try:
        from backend.preprocessing.clean_service import show_stats
        # Just return stats, not actually run clean (can be long)
        return {"message": "Dung: python -m backend.clean", "stats": _get_stats()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount frontend static files if exists (phuc vu giao dien)
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    # Serve frontend tai / (index.html, style.css, app.js) - API routes uu tien hon
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="Chay RAG Chatbot API + Frontend")
    parser.add_argument("--host", default="0.0.0.0", help="Host (mac dinh 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (mac dinh 8000)")
    parser.add_argument("--no-reload", action="store_true", help="Tat reload")
    args, _ = parser.parse_known_args()

    # Banner se duoc in boi lifespan khi server startup, khong can in o day de tranh duplicate
    # Tuy nhien in them 1 lan o day de nguoi dung thay ngay khi go lenh (truoc khi reload)
    # lifespan se in lan 2 sau khi reload xong - chap nhan duplicate 1 lan de hien thi som

    uvicorn.run("backend.app:app", host=args.host, port=args.port, reload=not args.no_reload)
