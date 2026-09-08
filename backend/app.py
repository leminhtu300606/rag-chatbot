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
import json

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

# ── Quản lý phiên hội thoại (session memory + persistent) ──
# Lưu trữ in-memory + persist ra file JSON để sống qua restart: {session_id: {"history": [...], "summary": str|None, "title": str|None, "style": str, "updated_at": float, "created_at": float}}
_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()
MAX_SESSIONS = 200
SESSION_TTL_SECONDS = 24 * 3600  # 24h
SESSIONS_FILE = cfg.BASE_DIR / "data" / "sessions.json"

def _save_sessions():
    """Ghi _sessions ra file JSON (gọi trong lock hoặc sẽ tự lock)."""
    try:
        SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        # copy dưới lock để tránh race
        with _sessions_lock:
            data = dict(_sessions)
        # ghi atomically
        tmp = SESSIONS_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(SESSIONS_FILE)
    except Exception as e:
        print(f"[sessions] save failed: {e}")

def _load_sessions():
    """Load sessions từ file nếu có."""
    try:
        if SESSIONS_FILE.exists():
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                with _sessions_lock:
                    for sid, v in data.items():
                        if not isinstance(v, dict):
                            continue
                        # đảm bảo field
                        v.setdefault("history", [])
                        v.setdefault("summary", None)
                        v.setdefault("title", None)
                        v.setdefault("style", cfg.DEFAULT_STYLE)
                        v.setdefault("last_math_result", None)
                        v.setdefault("created_at", time.time())
                        v.setdefault("updated_at", time.time())
                        # chuẩn hóa history
                        if not isinstance(v["history"], list):
                            v["history"] = []
                        _sessions[sid] = v
                print(f"[sessions] loaded {len(_sessions)} from {SESSIONS_FILE}")
    except Exception as e:
        print(f"[sessions] load failed: {e}")

# load ngay khi import
_load_sessions()

def _cleanup_sessions():
    now = time.time()
    removed = False
    with _sessions_lock:
        expired = [sid for sid, v in _sessions.items() if now - v.get("updated_at", 0) > SESSION_TTL_SECONDS]
        for sid in expired:
            _sessions.pop(sid, None)
            removed = True
        # gioi han so luong: xoa cu nhat
        if len(_sessions) > MAX_SESSIONS:
            sorted_sids = sorted(_sessions.items(), key=lambda x: x[1].get("updated_at", 0))
            for sid, _ in sorted_sids[: len(_sessions) - MAX_SESSIONS]:
                _sessions.pop(sid, None)
                removed = True
    if removed:
        _save_sessions()

def _get_or_create_session(session_id: Optional[str], title: Optional[str] = None, style: Optional[str] = None) -> str:
    _cleanup_sessions()
    # chuẩn hóa style
    if style and style not in cfg.AVAILABLE_STYLES:
        style = cfg.DEFAULT_STYLE
    if session_id and isinstance(session_id, str) and session_id.strip():
        sid = session_id.strip()
        created = False
        with _sessions_lock:
            if sid not in _sessions:
                _sessions[sid] = {"history": [], "summary": None, "title": title, "style": style or cfg.DEFAULT_STYLE, "last_math_result": None, "created_at": time.time(), "updated_at": time.time()}
                created = True
            else:
                if title and not _sessions[sid].get("title"):
                    _sessions[sid]["title"] = title
                if style and _sessions[sid].get("style") != style:
                    # nếu caller truyền style mới thì cập nhật
                    _sessions[sid]["style"] = style
                # đảm bảo field last_math_result
                _sessions[sid].setdefault("last_math_result", None)
        if created:
            _save_sessions()
        return sid
    # tao moi
    sid = str(uuid.uuid4())
    with _sessions_lock:
        _sessions[sid] = {"history": [], "summary": None, "title": title, "style": style or cfg.DEFAULT_STYLE, "last_math_result": None, "created_at": time.time(), "updated_at": time.time()}
    _save_sessions()
    return sid

def _append_to_session(session_id: str, user_q: str, assistant_a: str, summary: Optional[str] = None, style: Optional[str] = None, last_math_result: Optional[str] = None):
    need_save = False
    with _sessions_lock:
        sess = _sessions.get(session_id)
        if not sess:
            _sessions[session_id] = {"history": [], "summary": summary, "title": None, "style": style or cfg.DEFAULT_STYLE, "last_math_result": last_math_result, "created_at": time.time(), "updated_at": time.time()}
            sess = _sessions[session_id]
            need_save = True
        sess["history"].append({"role": "user", "content": user_q})
        sess["history"].append({"role": "assistant", "content": assistant_a})
        if summary:
            sess["summary"] = summary
        if style and style in cfg.AVAILABLE_STYLES:
            sess["style"] = style
        if last_math_result is not None:
            sess["last_math_result"] = str(last_math_result)
        # tự đặt title từ câu hỏi đầu tiên nếu chưa có
        if not sess.get("title") and user_q:
            sess["title"] = user_q.strip()[:50]
        sess["updated_at"] = time.time()
        # gioi han lich su luu toi da 100 messages de tranh phinh
        if len(sess["history"]) > 100:
            sess["history"] = sess["history"][-100:]
    _save_sessions()

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
    style: Optional[str] = None  # phong cách do frontend gửi hoặc để trống sẽ tự phát hiện qua câu tự nhiên

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    context: Optional[List[Dict[str, Any]]] = None
    # Thông tin hội thoại trả về
    session_id: Optional[str] = None
    standalone_question: Optional[str] = None
    summary: Optional[str] = None
    history_used: Optional[List[Dict[str, Any]]] = None
    style: Optional[str] = None  # phong cách hiện tại của phiên
    math_result: Optional[str] = None
    math_expression: Optional[str] = None

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
            "title": sess.get("title"),
            "style": sess.get("style", cfg.DEFAULT_STYLE),
            "last_math_result": sess.get("last_math_result"),
            "created_at": sess.get("created_at"),
            "updated_at": sess.get("updated_at"),
            "turns": len(sess["history"]) // 2,
        }

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    with _sessions_lock:
        if session_id in _sessions:
            _sessions.pop(session_id)
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    _save_sessions()
    return {"deleted": session_id}

@app.get("/api/sessions")
async def list_sessions():
    with _sessions_lock:
        sessions = list(_sessions.items())
    # sort mới nhất trước
    sessions.sort(key=lambda x: x[1].get("updated_at", 0), reverse=True)
    return {
        "sessions": [
            {
                "session_id": sid,
                "title": v.get("title") or (v["history"][0]["content"][:40] if v["history"] else "Cuộc trò chuyện mới"),
                "preview": (v["history"][-2]["content"][:80] if len(v["history"]) >= 2 else (v["history"][-1]["content"][:80] if v["history"] else "")),
                "turns": len(v["history"]) // 2,
                "updated_at": v.get("updated_at"),
                "created_at": v.get("created_at"),
                "summary": v.get("summary"),
                "style": v.get("style", cfg.DEFAULT_STYLE),
                "last_math_result": v.get("last_math_result"),
            }
            for sid, v in sessions
        ],
        "total": len(sessions),
    }

class CreateSessionRequest(BaseModel):
    title: Optional[str] = None
    session_id: Optional[str] = None
    style: Optional[str] = None

@app.post("/api/sessions")
async def create_session(req: CreateSessionRequest = None):
    title = (req.title.strip()[:50] if req and req.title and req.title.strip() else None)
    style = (req.style.strip().lower() if req and req.style and req.style.strip().lower() in cfg.AVAILABLE_STYLES else None)
    sid = _get_or_create_session(req.session_id if req and req.session_id else None, title=title, style=style)
    with _sessions_lock:
        sess = _sessions[sid]
        return {
            "session_id": sid,
            "title": sess.get("title"),
            "style": sess.get("style", cfg.DEFAULT_STYLE),
            "last_math_result": sess.get("last_math_result"),
            "created_at": sess.get("created_at"),
            "updated_at": sess.get("updated_at"),
            "history": sess.get("history", []),
        }

class UpdateSessionRequest(BaseModel):
    title: str

@app.patch("/api/sessions/{session_id}")
async def update_session(session_id: str, req: UpdateSessionRequest):
    with _sessions_lock:
        sess = _sessions.get(session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        sess["title"] = req.title.strip()[:50]
        sess["updated_at"] = time.time()
    _save_sessions()
    return {"session_id": session_id, "title": req.title.strip()[:50]}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Cau hoi rong")

    # Validate category
    if req.category and req.category not in cfg.CATEGORY_ORDER:
        raise HTTPException(status_code=400, detail=f"Category khong hop le. Chon trong: {cfg.CATEGORY_ORDER}")

    # Kiem tra co data truoc khi goi LLM (tranh goi Ollama khi chua co data)
    # Nhưng với câu xã giao, recall lịch sử, toán học hoặc so sánh thì không cần data, cho phép trả lời ngay
    is_social_early = False
    try:
        from backend.generation.generator import is_social_question, is_history_recall_question
        from backend.generation.calculator import is_math_question, is_comparison_question
        _lr_early = None
        if req.session_id and req.session_id.strip():
            with _sessions_lock:
                _lr_early = _sessions.get(req.session_id.strip(), {}).get("last_math_result")
        is_social_early = is_social_question(req.question, None) or is_history_recall_question(req.question) or is_math_question(req.question, None, _lr_early) or is_comparison_question(req.question, None, _lr_early)
    except Exception:
        pass
    if not is_social_early:
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
        # Xác định phong cách: ưu tiên req.style -> phát hiện qua câu tự nhiên -> style lưu trong phiên -> mặc định
        effective_style = None
        if req.style and req.style.strip().lower() in cfg.AVAILABLE_STYLES:
            effective_style = req.style.strip().lower()
        else:
            try:
                from backend.generation.generator import detect_style_change
                detected = detect_style_change(req.question)
                if detected:
                    effective_style = detected
            except Exception:
                pass

        # Chọn pipeline có/không có lịch sử hội thoại — ưu tiên server làm source of truth
        use_history = req.use_history and (req.history or req.session_id)

        effective_history: Optional[List[Dict[str, Any]]] = None
        session_id: Optional[str] = None
        session_style = None

        if use_history:
            if req.session_id:
                # Có session_id -> tái sử dụng phiên đó (không tạo mới mỗi câu hỏi)
                # Lấy style hiện tại của phiên nếu chưa có effective_style
                with _sessions_lock:
                    existing_style = _sessions.get(req.session_id.strip(), {}).get("style") if req.session_id and req.session_id.strip() else None
                if not effective_style and existing_style and existing_style in cfg.AVAILABLE_STYLES:
                    effective_style = existing_style
                if not effective_style:
                    effective_style = cfg.DEFAULT_STYLE
                session_id = _get_or_create_session(req.session_id, style=effective_style)
                with _sessions_lock:
                    sess_hist = list(_sessions[session_id]["history"]) if session_id in _sessions else []
                    session_style = _sessions[session_id].get("style", effective_style) if session_id in _sessions else effective_style
                # Nếu phát hiện đổi phong cách, cập nhật ngay vào session
                if effective_style and session_style != effective_style:
                    with _sessions_lock:
                        if session_id in _sessions:
                            _sessions[session_id]["style"] = effective_style
                            _sessions[session_id]["updated_at"] = time.time()
                    _save_sessions()
                    session_style = effective_style
                else:
                    effective_style = session_style or effective_style

                if sess_hist:
                    effective_history = sess_hist
                elif req.history:
                    # server trống (sau restart trước khi có persist hoặc phiên mới) -> lấy client
                    client_hist = [{"role": m.role, "content": m.content} for m in req.history]
                    effective_history = client_hist if client_hist else None
                    # đồng bộ client lên server
                    if client_hist:
                        with _sessions_lock:
                            if session_id in _sessions:
                                _sessions[session_id]["history"] = client_hist
                                _sessions[session_id]["updated_at"] = time.time()
                        _save_sessions()
                else:
                    effective_history = None
            elif req.history:
                # Có history nhưng chưa có session_id -> tạo phiên mới từ history client
                if not effective_style:
                    effective_style = cfg.DEFAULT_STYLE
                client_hist = [{"role": m.role, "content": m.content} for m in req.history]
                session_id = _get_or_create_session(None, style=effective_style)
                effective_history = client_hist if client_hist else None
                if client_hist:
                    with _sessions_lock:
                        if session_id in _sessions:
                            _sessions[session_id]["history"] = client_hist
                            _sessions[session_id]["updated_at"] = time.time()
                    _save_sessions()
                session_style = effective_style
            else:
                # use_history true nhưng không có gì -> tạo phiên mới rỗng
                if not effective_style:
                    effective_style = cfg.DEFAULT_STYLE
                session_id = _get_or_create_session(None, style=effective_style)
                effective_history = None
                session_style = effective_style
        else:
            # single-turn: không dùng lịch sử, nhưng vẫn giữ session_id nếu client gửi
            effective_history = None
            session_id = req.session_id.strip() if req.session_id and req.session_id.strip() else None
            if not effective_style:
                # lấy style từ session nếu có
                with _sessions_lock:
                    existing_style = _sessions.get(session_id, {}).get("style") if session_id else None
                effective_style = existing_style or cfg.DEFAULT_STYLE
            # nếu client gửi use_history=false nhưng vẫn có session_id thì đảm bảo phiên tồn tại
            if session_id:
                _get_or_create_session(session_id, style=effective_style)
                session_style = effective_style
            else:
                session_style = effective_style

        # Đảm bảo effective_style luôn có giá trị
        if not effective_style:
            effective_style = session_style or cfg.DEFAULT_STYLE

        # Lấy last_math_result để phục vụ "nó + 2" hoặc "cộng thêm 5"
        last_math_result = None
        if session_id:
            with _sessions_lock:
                last_math_result = _sessions.get(session_id, {}).get("last_math_result")

        # Goi RAG tuong ung (truyền phong cách)
        if effective_history:
            from backend.generation.rag import answer_with_history
            res = answer_with_history(
                question=req.question,
                history=effective_history,
                category=req.category,
                top_k=req.top_k,
                use_rerank=req.use_rerank,
                style=effective_style,
                last_math_result=last_math_result,
            )
            # Cập nhật style từ kết quả nếu có (phát hiện đổi phong cách)
            final_style = res.get("style", effective_style)
            # Xử lý so sánh trước toán học: giữ last_math_result cũ (không ghi đè bằng biểu thức so sánh)
            if "comparison_answer" in res:
                comp_math_res = res.get("math_result")
                if session_id:
                    _append_to_session(session_id, req.question, res["answer"], res.get("summary"), style=final_style, last_math_result=None)
                    with _sessions_lock:
                        if session_id in _sessions and final_style in cfg.AVAILABLE_STYLES:
                            _sessions[session_id]["style"] = final_style
                    _save_sessions()
                return ChatResponse(
                    answer=res["answer"],
                    sources=[],
                    context=[] if req.show_context else None,
                    session_id=session_id,
                    standalone_question=res.get("math_expression") or res.get("standalone_question"),
                    summary=res.get("summary"),
                    history_used=res.get("history_used"),
                    style=final_style,
                    math_result=comp_math_res,
                    math_expression=res.get("math_expression") or res.get("standalone_question"),
                )
            # Lưu vào session sau khi có kết quả (kèm style, summary và last_math_result nếu là toán)
            math_res = res.get("math_result")
            if session_id:
                _append_to_session(session_id, req.question, res["answer"], res.get("summary"), style=final_style, last_math_result=math_res if math_res is not None else None)
                # đảm bảo session style đồng bộ
                with _sessions_lock:
                    if session_id in _sessions and final_style in cfg.AVAILABLE_STYLES:
                        _sessions[session_id]["style"] = final_style
                    # nếu là toán học, cập nhật last_math_result ngay (tránh bị _append None ghi đè)
                    if math_res is not None and session_id in _sessions:
                        _sessions[session_id]["last_math_result"] = str(math_res)
                _save_sessions()

            # Toán học: trả về ngay không cần context, hỗ trợ độ chính xác cao
            if math_res is not None:
                return ChatResponse(
                    answer=res["answer"],
                    sources=[],
                    context=[] if req.show_context else None,
                    session_id=session_id,
                    standalone_question=res.get("math_expression") or res.get("standalone_question"),
                    summary=res.get("summary"),
                    history_used=res.get("history_used"),
                    style=final_style,
                    math_result=math_res,
                    math_expression=res.get("math_expression") or res.get("standalone_question"),
                )

            if not res.get("context"):
                # Trường hợp chỉ đổi phong cách không cần context, vẫn trả về style
                if res.get("style"):
                    return ChatResponse(
                        answer=res["answer"],
                        sources=[],
                        context=[] if req.show_context else None,
                        session_id=session_id,
                        standalone_question=res.get("standalone_question"),
                        summary=res.get("summary"),
                        history_used=res.get("history_used"),
                        style=final_style,
                    )
                return ChatResponse(
                    answer="Chua co du lieu de tra loi. Vui long chay: python -m backend.clean && python -m backend.index --rebuild, sau do thu lai.",
                    sources=[],
                    context=[] if req.show_context else None,
                    session_id=session_id,
                    standalone_question=res.get("standalone_question"),
                    summary=res.get("summary"),
                    history_used=res.get("history_used"),
                    style=final_style,
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
                style=final_style,
            )
        else:
            # Chế độ đơn lượt (không dùng lịch sử) - vẫn hỗ trợ toán học với last_math_result
            from backend.generation.rag import answer
            # last_math_result đã lấy ở trên, dùng lại
            res = answer(
                question=req.question,
                category=req.category,
                top_k=req.top_k,
                use_rerank=req.use_rerank,
                style=effective_style,
                last_math_result=last_math_result if 'last_math_result' in locals() else None,
            )
            final_style = res.get("style", effective_style)
            # So sánh đơn lượt: không ghi đè last_math_result bằng biểu thức so sánh
            if "comparison_answer" in res:
                comp_math_res = res.get("math_result")
                if session_id:
                    _append_to_session(session_id, req.question, res["answer"], style=final_style, last_math_result=None)
                    _save_sessions()
                return ChatResponse(
                    answer=res["answer"],
                    sources=[],
                    context=[] if req.show_context else None,
                    session_id=session_id,
                    standalone_question=res.get("math_expression") or res.get("standalone_question"),
                    style=final_style,
                    math_result=comp_math_res,
                    math_expression=res.get("math_expression") or res.get("standalone_question"),
                )
            math_res_single = res.get("math_result")
            if not res.get("context"):
                # Nếu là toán học hoặc social/style thì vẫn trả về ngay (không cần context)
                if res.get("math_result") is not None:
                    if session_id:
                        _append_to_session(session_id, req.question, res["answer"], style=final_style, last_math_result=math_res_single)
                        with _sessions_lock:
                            if session_id in _sessions:
                                _sessions[session_id]["last_math_result"] = str(math_res_single)
                        _save_sessions()
                    return ChatResponse(
                        answer=res["answer"],
                        sources=[],
                        context=[] if req.show_context else None,
                        session_id=session_id,
                        standalone_question=res.get("math_expression") or res.get("standalone_question"),
                        style=final_style,
                        math_result=math_res_single,
                        math_expression=res.get("math_expression") or res.get("standalone_question"),
                    )
                # Nếu chỉ đổi phong cách
                if res.get("style") and not res.get("context"):
                    # trả về luôn nếu có answer dù không có context (trường hợp đổi style)
                    if res.get("answer"):
                        if session_id:
                            _append_to_session(session_id, req.question, res["answer"], style=final_style)
                        return ChatResponse(
                            answer=res["answer"],
                            sources=[],
                            context=[] if req.show_context else None,
                            session_id=session_id,
                            style=final_style,
                        )
                return ChatResponse(
                    answer="Chua co du lieu de tra loi. Vui long chay: python -m backend.clean && python -m backend.index --rebuild, sau do thu lai.",
                    sources=[],
                    context=[] if req.show_context else None,
                    session_id=session_id,
                    style=final_style,
                )
            context = None
            if req.show_context:
                context = res.get("context", [])

            # Van luu vao session neu co session_id du la single-turn
            # Nếu là toán học mà chưa được xử lý early-return (hiếm), vẫn lưu last_math_result
            # So sánh không ghi đè last_math_result (giữ số trước đó cho "nó")
            if "comparison_answer" in res:
                _math_single_tail = None
            else:
                _math_single_tail = res.get("math_result")
            if session_id:
                _append_to_session(session_id, req.question, res["answer"], style=final_style, last_math_result=_math_single_tail if _math_single_tail is not None else None)
                with _sessions_lock:
                    if session_id in _sessions and final_style in cfg.AVAILABLE_STYLES:
                        _sessions[session_id]["style"] = final_style
                    if _math_single_tail is not None and session_id in _sessions:
                        _sessions[session_id]["last_math_result"] = str(_math_single_tail)
                _save_sessions()

            return ChatResponse(
                answer=res["answer"],
                sources=res.get("sources", []),
                context=context,
                session_id=session_id,
                standalone_question=res.get("math_expression") or res.get("standalone_question"),
                style=final_style,
                math_result=res.get("math_result"),
                math_expression=res.get("math_expression"),
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
