"""
backend/api.py - FastAPI cho RAG Chatbot
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

Chạy: uvicorn backend.api:app --reload --port 8000
      hoặc: python -m backend.api
"""
from pathlib import Path
import sys

# Fix import khi chay truc tiep
if __package__ in (None, ""):
    _ROOT = Path(__file__).resolve().parent.parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from typing import Optional, List, Dict, Any
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
import time
import uuid
import threading
import json
import re
import shutil
import os

import backend.config.settings as cfg
# Security (gop trong backend.config.settings)
from backend.config.settings import (
    validate_question, validate_category, validate_top_k, validate_session_id,
    validate_tool_call, validate_output, validate_sources,
    sanitize_input,
)
from backend.security import auth as auth_mod
from backend.security.auth import require_user, require_admin, get_current_user_opt

# Rate limiting: N requests / phút / identity (user đã login, ngược lại theo IP)
_rate_limit: Dict[str, List[float]] = {}
_rate_lock = threading.Lock()

def _rate_limit_cfg():
    try:
        return int(cfg.RATE_LIMIT_MAX), int(cfg.RATE_LIMIT_WINDOW)
    except Exception:
        return 30, 60

def _check_rate_limit(key: str):
    rmax, rwin = _rate_limit_cfg()
    now = time.time()
    with _rate_lock:
        lst = _rate_limit.get(key, [])
        # giữ lại trong window
        lst = [t for t in lst if now - t < rwin]
        if len(lst) >= rmax:
            raise HTTPException(status_code=429, detail="Quá nhiều yêu cầu, vui lòng thử lại sau 1 phút")
        lst.append(now)
        _rate_limit[key] = lst

def _identity_key(request: Request, user: Optional[Dict] = None) -> str:
    if user and user.get("username") not in (None, "anonymous"):
        return f"user:{user['username']}"
    try:
        ip = request.client.host if request.client else "unknown"
    except Exception:
        ip = "unknown"
    return f"ip:{ip}"

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
import asyncio
from starlette.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    # In banner khi chay bang `uvicorn backend.api:app` (khong qua __main__)
    try:
        _print_banner(host="0.0.0.0", port=8000)
    except Exception:
        pass
    # Warmup model nhẹ để request đầu không chậm
    try:
        print(f"[warmup] DEVICE={cfg.DEVICE} USE_GPU={getattr(cfg,'USE_GPU',False)} EMBED_BATCH={getattr(cfg,'EMBED_BATCH_SIZE',32)}")
        # Chạy warmup trong threadpool để không chặn startup
        def _warm():
            try:
                from backend.indexing.embedder import warmup as em_warm
                from backend.generation.reranker import warmup as rr_warm
                em_warm()
                rr_warm()
                print("[warmup] embed + reranker ready")
            except Exception as e:
                print(f"[warmup] skip: {e}")
        await run_in_threadpool(_warm)
    except Exception as e:
        print(f"[warmup] failed: {e}")
    # GĐ2: bootstrap admin + GĐ3: job dọn phiên/upload hết hạn định kỳ
    try:
        auth_mod.ensure_bootstrap_admin()
    except Exception as e:
        print(f"[auth] bootstrap failed: {e}")

    async def _janitor():
        while True:
            try:
                await asyncio.sleep(int(cfg.CLEANUP_INTERVAL_SECONDS))
                await run_in_threadpool(_cleanup_sessions)
                await run_in_threadpool(_sweep_orphan_uploads)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[janitor] {e}")

    _janitor_task = asyncio.create_task(_janitor())
    try:
        yield
    finally:
        try:
            _janitor_task.cancel()
        except Exception:
            pass

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
# Lưu trữ in-memory + persist ra file JSON để sống qua restart: {session_id: {"history": [...], "summary": str|None, "title": str|None, "style": str, "owner": str|None, "updated_at": float, "created_at": float}}
# owner: username sở hữu phiên (GĐ2 RBAC). Dùng ContextVar để impl tự gắn owner mà không đổi chữ ký các hàm gọi.
from contextvars import ContextVar
_current_user: ContextVar = ContextVar("api_user", default=None)

_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.RLock()  # RLock để _require_session_owner gọi lồng được
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
                        v.setdefault("owner", None)
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
    expired_ids: list[str] = []
    with _sessions_lock:
        expired = [sid for sid, v in _sessions.items() if now - v.get("updated_at", 0) > SESSION_TTL_SECONDS]
        for sid in expired:
            _sessions.pop(sid, None)
            expired_ids.append(sid)
            removed = True
        # gioi han so luong: xoa cu nhat
        if len(_sessions) > MAX_SESSIONS:
            sorted_sids = sorted(_sessions.items(), key=lambda x: x[1].get("updated_at", 0))
            for sid, _ in sorted_sids[: len(_sessions) - MAX_SESSIONS]:
                _sessions.pop(sid, None)
                expired_ids.append(sid)
                removed = True
    if removed:
        _save_sessions()
    # Xóa vector + file upload của phiên hết hạn (không chặn)
    if expired_ids:
        try:
            from backend.indexing.vectorstore import delete_by_session
            for sid in expired_ids:
                try:
                    delete_by_session(sid)
                except Exception:
                    pass
                try:
                    import shutil as _sh2
                    up_dir = cfg.UPLOAD_DIR / sid
                    if up_dir.exists():
                        _sh2.rmtree(up_dir, ignore_errors=True)
                except Exception:
                    pass
        except Exception:
            pass


def _sweep_orphan_uploads() -> int:
    """GĐ3: xóa thư mục upload của phiên đã mất (không còn trong _sessions)."""
    removed = 0
    try:
        base = cfg.UPLOAD_DIR
        if not base.is_dir():
            return 0
        with _sessions_lock:
            alive = set(_sessions.keys())
        for p in base.iterdir():
            try:
                if p.is_dir() and p.name not in alive:
                    import shutil as _sh3
                    _sh3.rmtree(p, ignore_errors=True)
                    removed += 1
            except Exception:
                pass
    except Exception as e:
        print(f"[janitor] sweep uploads failed: {e}")
    if removed:
        print(f"[janitor] đã xóa {removed} thư mục upload mồ côi")
    return removed

def _current_username() -> Optional[str]:
    try:
        u = _current_user.get() or {}
        name = u.get("username")
        return name if name and name != "anonymous" else None
    except Exception:
        return None


def _require_session_owner(sid: str, user: Optional[Dict]) -> None:
    """Chặn user đọc phiên của user khác (GĐ2). Tự nhận phiên mồ côi (owner None)."""
    if not cfg.AUTH_ENABLED:
        return
    if (user or {}).get("is_admin"):
        return
    uname = (user or {}).get("username")
    with _sessions_lock:
        sess = _sessions.get(sid)
        if not sess:
            return
        owner = sess.get("owner")
        if owner and uname and owner != uname:
            raise HTTPException(status_code=403, detail="Không có quyền truy cập phiên này")
        if not owner and uname and uname != "anonymous":
            sess["owner"] = uname
            sess["updated_at"] = time.time()
            _save_sessions()


def _get_or_create_session(session_id: Optional[str], title: Optional[str] = None, style: Optional[str] = None) -> str:
    _cleanup_sessions()
    owner = _current_username()
    # chuẩn hóa style
    if style and style not in cfg.AVAILABLE_STYLES:
        style = cfg.DEFAULT_STYLE
    if session_id and isinstance(session_id, str) and session_id.strip():
        sid = session_id.strip()
        created = False
        with _sessions_lock:
            if sid not in _sessions:
                _sessions[sid] = {"history": [], "summary": None, "title": title, "owner": owner, "style": style or cfg.DEFAULT_STYLE, "last_math_result": None, "created_at": time.time(), "updated_at": time.time()}
                created = True
            else:
                if title and not _sessions[sid].get("title"):
                    _sessions[sid]["title"] = title
                if owner and not _sessions[sid].get("owner"):
                    _sessions[sid]["owner"] = owner
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
        _sessions[sid] = {"history": [], "summary": None, "title": title, "owner": owner, "style": style or cfg.DEFAULT_STYLE, "last_math_result": None, "created_at": time.time(), "updated_at": time.time()}
    _save_sessions()
    return sid

def _append_to_session(session_id: str, user_q: str, assistant_a: str, summary: Optional[str] = None, style: Optional[str] = None, last_math_result: Optional[str] = None):
    need_save = False
    with _sessions_lock:
        sess = _sessions.get(session_id)
        if not sess:
            _sessions[session_id] = {"history": [], "summary": summary, "title": None, "owner": _current_username(), "style": style or cfg.DEFAULT_STYLE, "last_math_result": last_math_result, "created_at": time.time(), "updated_at": time.time()}
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

    @field_validator('content')
    @classmethod
    def check_content_len(cls, v):
        if len(v) > 2000:
            raise ValueError('content quá dài (max 2000)')
        # Loại bỏ ký tự điều khiển
        v = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", v)
        return v[:2000]

    @field_validator('role')
    @classmethod
    def check_role(cls, v):
        if v not in ("user", "assistant", "system"):
            return "user"
        # System trong history chỉ được backend tạo, không cho client gửi tuỳ ý
        if v == "system":
            raise ValueError('role system không được phép từ client')
        return v

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

    @field_validator('question')
    @classmethod
    def check_question(cls, v):
        if not v or not v.strip():
            raise ValueError('Câu hỏi rỗng')
        if len(v) > 2000:
            raise ValueError('Câu hỏi quá dài (max 2000)')
        return v[:2000]

    @field_validator('top_k')
    @classmethod
    def check_top_k(cls, v):
        if v is None:
            return v
        if not (1 <= int(v) <= 10):
            raise ValueError('top_k phải từ 1 đến 10')
        return int(v)

    @field_validator('category')
    @classmethod
    def check_category(cls, v):
        if v is None:
            return v
        if v not in cfg.CATEGORY_ORDER:
            raise ValueError(f'Category không hợp lệ')
        return v

    @field_validator('style')
    @classmethod
    def check_style(cls, v):
        if v is None:
            return v
        if v not in cfg.AVAILABLE_STYLES:
            raise ValueError('style không hợp lệ')
        return v

    @field_validator('session_id')
    @classmethod
    def check_session(cls, v):
        if v is None:
            return v
        if len(v) > 128:
            raise ValueError('session_id quá dài')
        if not re.match(r"^[a-zA-Z0-9\-_]+$", v):
            raise ValueError('session_id không hợp lệ')
        return v

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
    is_social: bool = False  # True nếu là câu xã giao, không cần tài liệu chứng minh
    needs_clarification: bool = False  # True nếu câu hỏi chưa rõ ý cần hỏi lại
    clarify_reason: Optional[str] = None

# --- Helpers ---
def _get_stats():
    try:
        from backend.preprocessing.storage import get_processed_files, count_chunks
        from backend.indexing.vectorstore import count as vec_count, get_stats as vec_stats
        files = get_processed_files()
        _n = vec_count()
        _st = vec_stats()
        return {
            "processed_files": len(files),
            "total_chunks": count_chunks(),
            "chroma_count": _n,
            "vector_count": _n,
            "chroma_stats": _st,
            "vector_stats": _st,
            "embed_model": cfg.EMBED_MODEL,
            "device": getattr(cfg, "DEVICE", "cpu"),
            "use_gpu": getattr(cfg, "USE_GPU", False),
            "embed_batch": getattr(cfg, "EMBED_BATCH_SIZE", 32),
            "rerank_batch": getattr(cfg, "RERANK_BATCH_SIZE", 16),
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
@app.middleware("http")
async def _auth_context(request: Request, call_next):
    """Gắn user (nếu có Bearer token hợp lệ) vào ContextVar cho mọi request."""
    user = None
    if cfg.AUTH_ENABLED:
        try:
            h = request.headers.get("authorization", "")
            if h.lower().startswith("bearer "):
                user = auth_mod.get_user_by_token(h[7:].strip())
        except Exception:
            user = None
    tok = _current_user.set(user)
    try:
        return await call_next(request)
    finally:
        _current_user.reset(tok)


_ollama_cache: Dict[str, Any] = {"ok": None, "ts": 0.0}

def _ollama_ok() -> Optional[bool]:
    """Kiểm tra Ollama sống không (cache 30s, timeout ngắn). None = không cấu hình check."""
    now = time.time()
    if now - _ollama_cache.get("ts", 0) < 30 and _ollama_cache.get("ok") is not None:
        return _ollama_cache["ok"]
    ok = False
    try:
        import urllib.request
        base = (cfg.OLLAMA_BASE or "").rstrip("/")
        with urllib.request.urlopen(base + "/api/tags", timeout=3) as r:
            ok = r.status == 200
    except Exception:
        ok = False
    _ollama_cache.update(ok=ok, ts=now)
    return ok


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
        out = {
            "status": "ready" if ready else "not_ready",
            "chroma_count": c,
            "vector_count": c,
            "processed_files": len(files),
            "message": "OK" if ready else "Chua co data. Hay chay: python -m backend.cli clean && python -m backend.cli index --rebuild",
            "auth_enabled": bool(cfg.AUTH_ENABLED),
            "llm": f"{cfg.LLM_BACKEND}/{cfg.LLM_MODEL}",
        }
        # GĐ3: kiểm tra phụ (không làm fail readiness cơ bản)
        try:
            out["ollama_ok"] = _ollama_ok()
        except Exception:
            out["ollama_ok"] = False
        try:
            import shutil as _shd
            du = _shd.disk_usage(str(cfg.BASE_DIR))
            out["disk_free_gb"] = round(du.free / 1024 / 1024 / 1024, 2)
        except Exception:
            pass
        return out
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/api/metrics")
async def metrics(admin: Dict = Depends(require_admin)):
    """GĐ3: tổng hợp từ data/logs/chat.jsonl (count, latency p50/p95, no-source rate, cache hit)."""
    import statistics
    rows = []
    try:
        p = cfg.LOG_DIR / "chat.jsonl"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
            rows = rows[-5000:]
    except Exception as e:
        return {"error": str(e)}
    if cfg.AUTH_ENABLED and not (admin or {}).get("is_admin"):
        raise HTTPException(status_code=403, detail="Cần quyền admin")
    if not rows:
        return {"count": 0}
    lats = sorted(r.get("latency", 0) for r in rows if isinstance(r.get("latency"), (int, float)))
    def _pct(q):
        if not lats:
            return 0
        i = min(len(lats) - 1, int(q * len(lats)))
        return round(lats[i], 3)
    from backend.infra import cache as _cache
    return {
        "count": len(rows),
        "latency_p50": _pct(0.5),
        "latency_p95": _pct(0.95),
        "no_source_rate": round(sum(1 for r in rows if r.get("no_source")) / len(rows), 4),
        "cached_rate": round(sum(1 for r in rows if r.get("cached")) / len(rows), 4),
        "stream_rate": round(sum(1 for r in rows if r.get("stream")) / len(rows), 4),
        "by_category": {k: sum(1 for r in rows if (r.get("category") or "all") == k)
                        for k in set((r.get("category") or "all") for r in rows)},
        "cache_items": _cache.stats().get("items", 0),
        "active_sessions": len(_sessions),
    }

@app.get("/api/stats")
async def stats():
    return _get_stats()

@app.get("/api/categories")
async def categories(user: Optional[Dict] = Depends(get_current_user_opt)):
    if cfg.AUTH_ENABLED and user and "*" not in (user.get("categories") or ["*"]):
        return {"categories": [c for c in cfg.CATEGORY_ORDER if c in (user.get("categories") or [])]}
    return {"categories": cfg.CATEGORY_ORDER}

# ── Auth (GĐ2): login/logout/me/quản trị users ──
class LoginRequest(BaseModel):
    username: str
    password: str

class CreateUserRequest(BaseModel):
    username: str
    password: str
    is_admin: bool = False
    categories: Optional[List[str]] = None

@app.post("/api/auth/login")
async def login(req: LoginRequest):
    try:
        return auth_mod.verify_login(req.username, req.password)
    except ValueError as ve:
        raise HTTPException(status_code=401, detail=str(ve))

@app.post("/api/auth/logout")
async def logout(user: Dict = Depends(require_user),
                 creds=Depends(auth_mod._bearer)):
    if creds and creds.credentials:
        auth_mod.revoke_token(creds.credentials)
    return {"ok": True}

@app.get("/api/auth/me")
async def me(user: Dict = Depends(require_user)):
    return {"username": user.get("username"), "is_admin": user.get("is_admin"),
            "categories": user.get("categories")}

@app.get("/api/auth/users")
async def users_list(admin: Dict = Depends(require_admin)):
    return {"users": auth_mod.list_users(), "auth_enabled": cfg.AUTH_ENABLED}

@app.post("/api/auth/users")
async def users_create(req: CreateUserRequest, admin: Dict = Depends(require_admin)):
    try:
        return auth_mod.create_user(req.username, req.password, req.is_admin, req.categories)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

@app.delete("/api/auth/users/{username}")
async def users_delete(username: str, admin: Dict = Depends(require_admin)):
    if username == admin.get("username"):
        raise HTTPException(status_code=400, detail="Không tự xóa chính mình")
    try:
        auth_mod.delete_user(username)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    return {"deleted": username}

# Endpoint quản lý phiên hội thoại
@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str, user: Dict = Depends(require_user)):
    with _sessions_lock:
        sess = _sessions.get(session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        _require_session_owner(session_id, user)
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
async def delete_session(session_id: str, user: Dict = Depends(require_user)):
    _require_session_owner(session_id, user)
    with _sessions_lock:
        if session_id in _sessions:
            _sessions.pop(session_id)
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    _save_sessions()
    # Xóa vector + file upload của phiên (chỉ lưu tại phiên)
    try:
        from backend.indexing.vectorstore import delete_by_session
        await run_in_threadpool(lambda: delete_by_session(session_id))
    except Exception as e:
        print(f"[upload] delete vectors lỗi {session_id}: {e}")
    try:
        import shutil as _sh
        up_dir = cfg.UPLOAD_DIR / session_id
        if up_dir.exists():
            _sh.rmtree(up_dir, ignore_errors=True)
    except Exception as e:
        print(f"[upload] delete files lỗi {session_id}: {e}")
    return {"deleted": session_id}

@app.get("/api/sessions")
async def list_sessions(user: Dict = Depends(require_user)):
    with _sessions_lock:
        sessions = list(_sessions.items())
    # RBAC: user thường chỉ thấy phiên của mình
    if cfg.AUTH_ENABLED and not (user or {}).get("is_admin"):
        uname = (user or {}).get("username")
        sessions = [(sid, v) for sid, v in sessions if not v.get("owner") or v.get("owner") == uname]
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
async def create_session(req: CreateSessionRequest = None, user: Dict = Depends(require_user)):
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
async def update_session(session_id: str, req: UpdateSessionRequest, user: Dict = Depends(require_user)):
    _require_session_owner(session_id, user)
    with _sessions_lock:
        sess = _sessions.get(session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        sess["title"] = req.title.strip()[:50]
        sess["updated_at"] = time.time()
    _save_sessions()
    return {"session_id": session_id, "title": req.title.strip()[:50]}


# ── Upload theo phiên (chỉ lưu tại phiên) ──
# Hỗ trợ dung lượng lớn: streaming ghi file, không load hết RAM
# ── Upload hardening (GĐ2): magic bytes + quota user + ClamAV hook ──
def _sniff_upload(path: Path, ext: str) -> Optional[str]:
    """Kiểm tra magic bytes khớp đuôi file. None = OK, str = lý do từ chối."""
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except Exception:
        return "Không đọc được file"
    if ext == ".pdf":
        return None if head.startswith(b"%PDF-") else "File không phải PDF hợp lệ (thiếu header %PDF)"
    if ext == ".docx":
        if not head.startswith(b"PK\x03\x04"):
            return "File không phải DOCX hợp lệ"
        return None
    if ext in (".txt", ".md", ".csv", ".log"):
        try:
            head.decode("utf-8")
            return None
        except Exception:
            return "File text phải ở định dạng UTF-8"
    return None


def _clamav_scan(path: Path) -> Optional[str]:
    """Quét ClamAV qua INSTREAM. None = sạch/bỏ qua; str = lý do chặn."""
    if not cfg.CLAMAV_ENABLED:
        return None
    import socket
    try:
        s = socket.create_connection((cfg.CLAMAV_HOST, int(cfg.CLAMAV_PORT)), timeout=15)
        s.sendall(b"zINSTREAM\0")
        with open(path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                s.sendall(len(chunk).to_bytes(4, "big") + chunk)
        s.sendall((0).to_bytes(4, "big"))
        resp = b""
        while True:
            d = s.recv(4096)
            if not d:
                break
            resp += d
        try:
            s.close()
        except Exception:
            pass
        txt = resp.decode("utf-8", errors="ignore")
        if "FOUND" in txt:
            return f"Phát hiện mã độc: {txt.strip()[:200]}"
        return None
    except Exception as e:
        if cfg.CLAMAV_FAIL_CLOSED:
            return f"Không quét virus được (fail-closed): {e}"
        print(f"[clamav] bỏ qua (không kết nối được daemon): {e}")
        return None


def _user_upload_bytes(username: Optional[str]) -> int:
    """Tổng dung lượng upload của 1 user (duyệt thư mục các phiên sở hữu)."""
    if not username:
        return 0
    total = 0
    try:
        with _sessions_lock:
            sids = [sid for sid, v in _sessions.items() if v.get("owner") == username]
        for sid in sids:
            d = cfg.UPLOAD_DIR / sid
            if not d.is_dir():
                continue
            for p in d.iterdir():
                try:
                    if p.is_file():
                        total += p.stat().st_size
                except Exception:
                    pass
    except Exception:
        pass
    return total


@app.get("/api/sessions/{session_id}/uploads")
async def list_uploads(session_id: str, user: Dict = Depends(require_user)):
    try:
        session_id = validate_session_id(session_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    _require_session_owner(session_id, user)
    _get_or_create_session(session_id)
    try:
        from backend.indexing.vectorstore import list_uploads_by_session, count_by_session
        files = await run_in_threadpool(lambda: list_uploads_by_session(session_id))
        total_chunks = await run_in_threadpool(lambda: count_by_session(session_id))
        # Lấy danh sách file trên đĩa
        up_dir = cfg.UPLOAD_DIR / session_id
        disk_files = []
        if up_dir.exists():
            for p in up_dir.iterdir():
                if p.is_file():
                    try:
                        disk_files.append({"filename": p.name, "size": p.stat().st_size, "path": str(p)})
                    except Exception:
                        pass
        return {"session_id": session_id, "files": files, "total_chunks": total_chunks, "disk_files": disk_files}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/sessions/{session_id}/uploads/{filename}")
async def delete_upload_file(session_id: str, filename: str, user: Dict = Depends(require_user)):
    try:
        session_id = validate_session_id(session_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    _require_session_owner(session_id, user)
    # sanitize filename
    safe_name = re.sub(r"[^a-zA-Z0-9._\- ]", "_", filename).strip()[:200]
    if not safe_name:
        raise HTTPException(status_code=400, detail="Tên file không hợp lệ")
    try:
        from backend.indexing.vectorstore import delete_chunks
        # Xóa vector theo filename + session_id
        try:
            delete_chunks(session_id, safe_name)
        except Exception:
            pass
        # Xóa file đĩa
        up_path = cfg.UPLOAD_DIR / session_id / safe_name
        if up_path.exists():
            try:
                up_path.unlink()
            except Exception:
                pass
        return {"deleted": safe_name, "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sessions/{session_id}/upload")
async def upload_session_file(session_id: str, file: UploadFile = File(...), user: Dict = Depends(require_user)):
    # Rate limit nhẹ
    # Validate session_id
    try:
        session_id = validate_session_id(session_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"session_id lỗi: {e}")
    _require_session_owner(session_id, user)
    _get_or_create_session(session_id)
    # Kiểm tra số lượng file hiện tại
    try:
        from backend.indexing.vectorstore import count_by_session
        cur_cnt = await run_in_threadpool(lambda: count_by_session(session_id))
        # count files not chunks - dùng list
        from backend.indexing.vectorstore import list_uploads_by_session
        cur_files = await run_in_threadpool(lambda: list_uploads_by_session(session_id))
        if len(cur_files) >= getattr(cfg, "MAX_UPLOADS_PER_SESSION", 10):
            raise HTTPException(status_code=400, detail=f"Phiên đã đạt giới hạn {cfg.MAX_UPLOADS_PER_SESSION} file. Xóa bớt file cũ trước khi tải thêm.")
    except HTTPException:
        raise
    except Exception:
        pass

    # Validate filename + ext
    orig_name = file.filename or "upload"
    # sanitize
    safe_name = re.sub(r"[^a-zA-Z0-9._\-\(\) ]", "_", orig_name).strip()
    if not safe_name:
        safe_name = f"file_{int(time.time())}"
    # giữ đuôi
    ext = Path(safe_name).suffix.lower()
    if not ext:
        # đoán từ orig
        ext = Path(orig_name).suffix.lower()
        safe_name += ext
    allowed = getattr(cfg, "UPLOAD_ALLOWED_EXTS", {".pdf", ".docx", ".txt", ".md"})
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Định dạng {ext} không hỗ trợ. Chỉ hỗ trợ: {', '.join(sorted(allowed))}")
    # Streaming ghi file để hỗ trợ dung lượng lớn (100MB)
    up_dir = cfg.UPLOAD_DIR / session_id
    up_dir.mkdir(parents=True, exist_ok=True)
    # Tránh trùng tên: thêm timestamp nếu đã tồn tại
    target_path = up_dir / safe_name
    if target_path.exists():
        stem = target_path.stem
        target_path = up_dir / f"{stem}_{int(time.time())}{ext}"
        safe_name = target_path.name
    # Ghi streaming 8KB chunk
    max_size = getattr(cfg, "MAX_UPLOAD_SIZE", 100*1024*1024)
    written = 0
    try:
        with open(target_path, "wb") as out_f:
            while True:
                chunk = await file.read(8192)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_size:
                    out_f.close()
                    try:
                        target_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    raise HTTPException(status_code=413, detail=f"File vượt quá giới hạn {max_size//1024//1024}MB")
                out_f.write(chunk)
        if written == 0:
            raise HTTPException(status_code=400, detail="File rỗng")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi ghi file: {e}")
    finally:
        try:
            await file.close()
        except Exception:
            pass

    # GĐ2: kiểm tra nội dung thật (magic bytes) + quét virus + quota user
    _sniff_err = _sniff_upload(target_path, ext)
    if _sniff_err:
        try:
            target_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=_sniff_err)
    _virus_err = await run_in_threadpool(lambda: _clamav_scan(target_path))
    if _virus_err:
        try:
            target_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=_virus_err)
    _uname = (user or {}).get("username")
    if _uname and _uname != "anonymous":
        _quota = int(cfg.MAX_UPLOAD_MB_PER_USER) * 1024 * 1024
        if _user_upload_bytes(_uname) > _quota:
            try:
                target_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise HTTPException(status_code=413, detail=f"Vượt quota upload {cfg.MAX_UPLOAD_MB_PER_USER}MB/user. Xóa bớt file cũ.")

    # Xử lý chunk + embed trong threadpool (không chặn event loop)
    def _process_and_index():
        from backend.preprocessing.pipeline import process_file_to_chunks
        from backend.indexing.embedder import embed as _embed
        from backend.indexing.vectorstore import add_chunks as _add
        # 1. Chunk
        chunks = process_file_to_chunks(target_path, session_id=session_id, filename_override=safe_name)
        if not chunks:
            raise RuntimeError("Không trích được nội dung từ file (file rỗng hoặc không đọc được)")
        # 2. Embed
        texts = [c["text"] for c in chunks]
        # Batch embedding
        embs = _embed(texts)
        items = []
        for c, emb in zip(chunks, embs):
            # metadata: giữ session_id để cô lập
            meta = {
                "category": c.get("category", "uploaded"),
                "subcategory": c.get("subcategory", ""),
                "filename": c.get("filename", safe_name),
                "source": c.get("source", str(target_path)),
                "page": c.get("page", 0),
                "chunk_index": c.get("chunk_index", 0),
                "chunk_mode": c.get("chunk_mode", "uploaded"),
                "chunk_type": c.get("chunk_type", "text"),
                "block_type": c.get("block_type", "text"),
                "session_id": session_id,
                "text_length": len(c.get("text", "")),
            }
            # thêm extra nếu có
            if c.get("has_table"):
                meta["has_table"] = True
            if c.get("has_image"):
                meta["has_image"] = True
            items.append({"id": c["id"], "text": c["text"], "embedding": emb, "metadata": meta})
        _add(items)
        return chunks, len(texts)

    try:
        chunks, n_chunks = await run_in_threadpool(_process_and_index)
    except Exception as e:
        # cleanup file nếu lỗi index
        try:
            target_path.unlink(missing_ok=True)
        except Exception:
            pass
        import traceback as _tb
        _tb.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý file: {e}")

    # Lấy text raw để phát hiện ý định (tối đa 8000 chars)
    file_text = ""
    try:
        # ghép từ chunks
        file_text = "\n".join(c.get("text","") for c in chunks[:10])[:8000]
        if not file_text and target_path.suffix.lower() in (".txt", ".md"):
            file_text = target_path.read_text(encoding="utf-8", errors="ignore")[:8000]
    except Exception:
        file_text = ""

    # Tự động xử lý: nếu file có yêu cầu rõ ràng -> trả lời ngay, không thì hỏi lại
    auto_message = None
    auto_sources = []
    is_auto = False
    try:
        if getattr(cfg, "UPLOAD_AUTO_ANSWER", True) and file_text:
            from backend.generation.generator import detect_file_intent, auto_answer_file
            intent = await run_in_threadpool(lambda: detect_file_intent(file_text))
            has_req = intent.get("has_request", False)
            extracted = intent.get("extracted_query", "") or ""
            if has_req and extracted:
                # tự trả lời
                ans = await run_in_threadpool(lambda: auto_answer_file(chunks, file_text, extracted, style=None))
                # validate
                try:
                    ans = validate_output(ans, max_len=4000)
                except Exception:
                    ans = ans[:4000]
                auto_message = ans
                auto_sources = [{"filename": safe_name, "page": c.get("page", 0), "category": "uploaded"} for c in chunks[:3]]
                is_auto = True
                # Lưu vào lịch sử phiên để lần sau chat có ngữ cảnh
                _append_to_session(session_id, f"[Đã tải file: {safe_name}]", ans)
            else:
                # hỏi lại
                ask = getattr(cfg, "UPLOAD_AUTO_ASKBACK", "Bạn muốn làm gì với file này?")
                pages = len(set(c.get("page", 0) for c in chunks))
                ask_filled = ask.format(filename=safe_name, pages=pages or 1, chunks=n_chunks)
                auto_message = ask_filled
                auto_sources = []
                is_auto = False
                _append_to_session(session_id, f"[Đã tải file: {safe_name}]", ask_filled)
        else:
            ask = getattr(cfg, "UPLOAD_AUTO_ASKBACK", "Bạn muốn làm gì với file này?")
            pages = len(set(c.get("page", 0) for c in chunks))
            auto_message = ask.format(filename=safe_name, pages=pages or 1, chunks=n_chunks)
            _append_to_session(session_id, f"[Đã tải file: {safe_name}]", auto_message)
    except Exception as e:
        print(f"[upload] auto intent lỗi: {e}")
        # fallback ask
        try:
            ask = getattr(cfg, "UPLOAD_AUTO_ASKBACK", "Bạn muốn làm gì với file này?")
            auto_message = ask.format(filename=safe_name, pages=1, chunks=n_chunks)
            _append_to_session(session_id, f"[Đã tải file: {safe_name}]", auto_message)
        except Exception:
            pass

    # Xác định số trang
    pages = len(set(c.get("page", 0) for c in chunks))
    return {
        "session_id": session_id,
        "filename": safe_name,
        "size": written,
        "pages": pages,
        "chunks": n_chunks,
        "auto_message": auto_message,
        "is_auto_answered": is_auto,
        "sources": auto_sources,
    }

# ── Guard + cache + log cho /api/chat (GĐ2/GĐ4) ──
def _guard_rate_quota(request: Request, user: Optional[Dict], count_quota: bool = True) -> None:
    _check_rate_limit(_identity_key(request, user))
    uname = (user or {}).get("username")
    if cfg.AUTH_ENABLED and count_quota and uname and uname != "anonymous":
        try:
            auth_mod.bump_daily(uname, int(cfg.CHAT_DAILY_LIMIT))
        except ValueError as ve:
            raise HTTPException(status_code=429, detail=str(ve))


def _guard_rbac(req: ChatRequest, user: Optional[Dict]) -> None:
    if req.category and not auth_mod.user_can_category(user, req.category):
        raise HTTPException(status_code=403, detail=f"Tài khoản không được xem category '{req.category}'")


def _guard_owner(req: ChatRequest, user: Optional[Dict]) -> None:
    if req.session_id and req.session_id.strip():
        with _sessions_lock:
            exists = req.session_id.strip() in _sessions
        if exists:
            _require_session_owner(req.session_id.strip(), user)


def _cacheable(req: ChatRequest):
    """Key cache cho câu đơn lượt giống hệt (không session/history)."""
    if req.session_id or req.history or req.use_history:
        return None
    from backend.infra import cache as _cache
    return _cache.make_key(req.question, req.category, req.top_k, req.style)


def _log_chat(user: Optional[Dict], req: ChatRequest, resp, latency: float, cached: bool = False, stream: bool = False) -> None:
    if not cfg.LOG_CHAT_JSONL:
        return
    try:
        cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
        uname = (user or {}).get("username") if user else None
        entry = {
            "ts": time.time(),
            "user": uname,
            "q_len": len(req.question or ""),
            "category": req.category,
            "top_k": req.top_k,
            "latency": round(latency, 3),
            "sources": len(resp.sources) if resp and getattr(resp, "sources", None) is not None else 0,
            "no_source": not (resp.sources if resp and getattr(resp, "sources", None) else []),
            "cached": cached,
            "stream": stream,
            "standalone": (getattr(resp, "standalone_question", None) or "")[:200] if resp else "",
        }
        with open(cfg.LOG_DIR / "chat.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[metrics] log failed: {e}")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request, user: Dict = Depends(require_user)):
    from backend.infra import cache as _cache
    t0 = time.time()
    _guard_rate_quota(request, user)
    _guard_rbac(req, user)
    _guard_owner(req, user)
    ckey = _cacheable(req)
    if ckey:
        hit = _cache.get(ckey)
        if hit is not None:
            _log_chat(user, req, hit, time.time() - t0, cached=True)
            return hit
    try:
        resp = await _chat_impl(req, request, user)
    except Exception:
        _log_chat(user, req, None, time.time() - t0)
        raise
    if ckey and getattr(resp, "sources", None):
        _cache.put(ckey, resp)
    _log_chat(user, req, resp, time.time() - t0)
    return resp


def _validate_chat_request(req: ChatRequest) -> None:
    """Validate chung cho /api/chat và /api/chat/stream (Least Privilege).

    Chuẩn hóa question/category/top_k/session_id, giới hạn history,
    cho phép AI chỉ retrieve với category/top_k đã kiểm duyệt.
    """
    try:
        req.question = validate_question(req.question)
        if req.category:
            req.category = validate_category(req.category)
        if req.top_k is not None:
            req.top_k = validate_top_k(req.top_k)
        if req.session_id:
            req.session_id = validate_session_id(req.session_id)
        # Validate tool call: AI chỉ được phép retrieve với category/top_k đã kiểm duyệt
        validate_tool_call("retrieve", {"category": req.category, "top_k": req.top_k})
        # Giới hạn history client gửi: tối đa 20 items, mỗi item 2000 chars đã được validator ở trên
        if req.history and len(req.history) > 20:
            raise HTTPException(status_code=400, detail="history quá dài (max 20)")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {e}")
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Cau hoi rong")


async def _chat_impl(req: ChatRequest, request: Request, user: Optional[Dict]):
    # Backend validation: AI chỉ đề xuất, backend kiểm tra lại (Least Privilege)
    _validate_chat_request(req)

    # Kiem tra co data truoc khi goi LLM (tranh goi Ollama khi chua co data)
    # Nhưng với câu xã giao, recall lịch sử, toán học hoặc so sánh thì không cần data, cho phép trả lời ngay
    is_social_early = False
    try:
        from backend.generation.generator import is_social_question, is_history_recall_question
        from backend.generation.calculator import is_math_question, is_comparison_question, is_multi_math_question
        _lr_early = None
        if req.session_id and req.session_id.strip():
            with _sessions_lock:
                _lr_early = _sessions.get(req.session_id.strip(), {}).get("last_math_result")
        is_social_early = is_social_question(req.question, None) or is_history_recall_question(req.question) or is_math_question(req.question, None, _lr_early) or is_comparison_question(req.question, None, _lr_early) or is_multi_math_question(req.question, None, _lr_early)
    except Exception:
        pass
    if not is_social_early:
        try:
            from backend.indexing.vectorstore import count as vec_count
            from backend.preprocessing.storage import get_processed_files
            c = await run_in_threadpool(vec_count)
            files = await run_in_threadpool(get_processed_files)
            if c == 0 or len(files) == 0:
                sid = _get_or_create_session(req.session_id) if req.session_id or req.history else None
                return ChatResponse(
                    answer="Chua co du lieu de tra loi. Vui long chay: python -m backend.cli clean && python -m backend.cli index --rebuild, sau do thu lai.",
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

        # Goi RAG tuong ung (truyền phong cách) - chạy trong threadpool để không chặn event loop
        if effective_history:
            from backend.generation.rag import answer_with_history
            res = await run_in_threadpool(
                lambda: answer_with_history(
                    question=req.question,
                    history=effective_history,
                    category=req.category,
                    top_k=req.top_k,
                    use_rerank=req.use_rerank,
                    style=effective_style,
                    last_math_result=last_math_result,
                    session_id=session_id,
                )
            )
            # Output validation: kiểm tra kết quả AI trước khi đưa vào hệ thống
            try:
                res["answer"] = validate_output(res.get("answer", ""))
                if "sources" in res:
                    res["sources"] = validate_sources(res.get("sources", []))
                if "context" in res and res["context"]:
                    # Giới hạn context trả về
                    res["context"] = res["context"][: res.get("top_k", 3) if isinstance(res.get("top_k"), int) else 10]
            except Exception:
                pass
            # Cập nhật style từ kết quả nếu có (phát hiện đổi phong cách)
            final_style = res.get("style", effective_style)
            is_social_res = res.get("is_social", False)
            # Hỏi lại nếu câu hỏi chưa rõ ý
            if res.get("needs_clarification"):
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
                    standalone_question=res.get("standalone_question"),
                    summary=res.get("summary"),
                    history_used=res.get("history_used"),
                    style=final_style,
                    needs_clarification=True,
                    clarify_reason=res.get("clarify_reason"),
                    is_social=False,
                )
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
                    is_social=is_social_res,
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
                    is_social=is_social_res,
                )

            # Xã giao: không cần tài liệu chứng minh
            if is_social_res:
                return ChatResponse(
                    answer=res["answer"],
                    sources=[],
                    context=[] if req.show_context else None,
                    session_id=session_id,
                    standalone_question=res.get("standalone_question"),
                    summary=res.get("summary"),
                    history_used=res.get("history_used"),
                    style=final_style,
                    is_social=True,
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
                        is_social=is_social_res,
                    )
                return ChatResponse(
                    answer="Chua co du lieu de tra loi. Vui long chay: python -m backend.cli clean && python -m backend.cli index --rebuild, sau do thu lai.",
                    sources=[],
                    context=[] if req.show_context else None,
                    session_id=session_id,
                    standalone_question=res.get("standalone_question"),
                    summary=res.get("summary"),
                    history_used=res.get("history_used"),
                    style=final_style,
                    is_social=is_social_res,
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
                is_social=is_social_res,
            )
        else:
            # Chế độ đơn lượt (không dùng lịch sử) - vẫn hỗ trợ toán học với last_math_result
            from backend.generation.rag import answer
            # last_math_result đã lấy ở trên, dùng lại
            res = await run_in_threadpool(
                lambda: answer(
                    question=req.question,
                    category=req.category,
                    top_k=req.top_k,
                    use_rerank=req.use_rerank,
                    style=effective_style,
                    last_math_result=last_math_result if 'last_math_result' in locals() else None,
                    session_id=session_id,
                )
            )
            # Output validation cho nhánh đơn lượt
            try:
                res["answer"] = validate_output(res.get("answer", ""))
                if "sources" in res:
                    res["sources"] = validate_sources(res.get("sources", []))
            except Exception:
                pass
            final_style = res.get("style", effective_style)
            is_social_res_single = res.get("is_social", False)
            # Hỏi lại nếu chưa rõ ý
            if res.get("needs_clarification"):
                if session_id:
                    _append_to_session(session_id, req.question, res["answer"], style=final_style, last_math_result=None)
                    _save_sessions()
                return ChatResponse(
                    answer=res["answer"],
                    sources=[],
                    context=[] if req.show_context else None,
                    session_id=session_id,
                    standalone_question=res.get("standalone_question"),
                    style=final_style,
                    needs_clarification=True,
                    clarify_reason=res.get("clarify_reason"),
                    is_social=False,
                )
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
                    is_social=is_social_res_single,
                )
            math_res_single = res.get("math_result")
            if is_social_res_single:
                if session_id:
                    _append_to_session(session_id, req.question, res["answer"], style=final_style)
                    _save_sessions()
                return ChatResponse(
                    answer=res["answer"],
                    sources=[],
                    context=[] if req.show_context else None,
                    session_id=session_id,
                    standalone_question=res.get("standalone_question"),
                    style=final_style,
                    is_social=True,
                )
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
                        is_social=is_social_res_single,
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
                            is_social=is_social_res_single,
                        )
                return ChatResponse(
                    answer="Chua co du lieu de tra loi. Vui long chay: python -m backend.cli clean && python -m backend.cli index --rebuild, sau do thu lai.",
                    sources=[],
                    context=[] if req.show_context else None,
                    session_id=session_id,
                    style=final_style,
                    is_social=is_social_res_single,
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
                is_social=res.get("is_social", False),
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
                detail="Loi dimension embedding. Hay chay: python -m backend.cli index --rebuild. Chi tiet: " + str(e),
            )
        # Ollama OOM - RAM/VRAM không đủ
        if any(k in msg for k in ["out-of-memory", "failed to allocate", "ggml", "memory"]):
            raise HTTPException(
                status_code=503,
                detail=str(e),
            )
        # Ollama / connection / timeout errors (bao gom 500 Server Error tu Ollama)
        if any(k in msg for k in ["ollama", "connection", "http://localhost:11434", "localhost", "500 server error", "connection refused", "failed to connect", "not found", "timed out", "timeout", "read timed out"]):
            # Giữ nguyên message gốc nếu đã là RuntimeError chi tiết từ generator (chứa gợi ý fix)
            detail = str(e) if "ollama" in msg and ("timeout" in msg or "gợi ý" in msg.lower()) else "Khong ket noi duoc Ollama (http://localhost:11434) hoac model loi/timeout. Kiem tra: ollama serve & ollama list & ollama pull. Nếu DEVICE=cpu + qwen2.5:7b thì cần tăng timeout hoặc đổi sang qwen2.5:1.5b. Chi tiet: " + str(e)
            raise HTTPException(
                status_code=503,
                detail=detail,
            )
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, request: Request, user: Dict = Depends(require_user)):
    """Streaming chat - trả chữ dần dần để người dùng thấy phản hồi nhanh hơn."""
    t0 = time.time()
    _guard_rate_quota(request, user)
    _guard_rbac(req, user)
    _guard_owner(req, user)
    # Validate như /api/chat
    _validate_chat_request(req)
    # Tận dụng logic chat thường nhưng stream phần generate
    # Với câu toán/xã giao thì trả ngay không cần stream
    try:
        from backend.generation.generator import is_social_question, is_history_recall_question
        from backend.generation.calculator import is_math_question, is_comparison_question, is_multi_math_question
        # Kiểm tra nhanh có cần RAG không
        is_simple = False
        try:
            is_simple = is_social_question(req.question, None) or is_history_recall_question(req.question) or is_math_question(req.question, None, None) or is_comparison_question(req.question, None, None)
        except Exception:
            pass
        if is_simple:
            # Dùng luôn logic chat thường
            res = await _chat_impl(req, request, user)
            _log_chat(user, req, res, time.time() - t0, stream=True)
            async def _simple_gen():
                import json as _j
                yield f"data: {_j.dumps({'token': res.answer}, ensure_ascii=False)}\n\n"
                yield f"data: {_j.dumps({'done': True, 'answer': res.answer, 'session_id': res.session_id, 'sources': list(getattr(res, 'sources', []) or []), 'standalone_question': getattr(res, 'standalone_question', None), 'summary': getattr(res, 'summary', None), 'style': getattr(res, 'style', None), 'math_result': getattr(res, 'math_result', None), 'math_expression': getattr(res, 'math_expression', None), 'is_social': bool(getattr(res, 'is_social', False)), 'needs_clarification': bool(getattr(res, 'needs_clarification', False)), 'clarify_reason': getattr(res, 'clarify_reason', None)}, ensure_ascii=False)}\n\n"
            return StreamingResponse(_simple_gen(), media_type="text/event-stream")
        # Với RAG cần stream từ Ollama
        # Chuẩn bị history/session như chat thường (rút gọn)
        session_id = req.session_id.strip() if req.session_id and req.session_id.strip() else None
        if not session_id and req.history:
            session_id = _get_or_create_session(None)
        effective_style = req.style.strip().lower() if req.style and req.style.strip().lower() in cfg.AVAILABLE_STYLES else cfg.DEFAULT_STYLE
        # Lấy history từ session nếu có
        eff_hist = None
        if session_id and session_id in _sessions:
            with _sessions_lock:
                eff_hist = list(_sessions[session_id]["history"])
        elif req.history:
            eff_hist = [{"role": m.role, "content": m.content} for m in req.history]

        from backend.generation.rag import (
            _do_retrieve, _ensure_trang_phuc_coverage, _prepare_history as _prep,
        )
        from backend.generation.generator import _call_ollama_stream, rewrite_query, detect_style_change
        from backend.generation.reranker import rerank, DEFAULT_RERANK_TOP_K
        from backend.generation.prompts import build_messages_with_history, build_messages
        from backend.config.settings import RETRIEVE_TOP_K
        from backend.common.utils import deduplicate_docs
        from backend.infra import cache as _cache

        # Giải session/style/history giống _chat_impl (frontend luôn use_history=true).
        # Stream single-pass: không agent retry để time bounded.
        style = effective_style
        if not (req.style and req.style.strip().lower() in cfg.AVAILABLE_STYLES):
            try:
                _det = detect_style_change(req.question)
                if _det:
                    style = _det
            except Exception:
                pass
        use_hist = req.use_history and (req.history or req.session_id)
        sid = req.session_id.strip() if req.session_id and req.session_id.strip() else None
        eff_hist = None
        if use_hist:
            if sid:
                sid = _get_or_create_session(sid, style=style)
                with _sessions_lock:
                    _sh = list(_sessions[sid]["history"]) if sid in _sessions else []
                    _ss = _sessions[sid].get("style", style) if sid in _sessions else style
                if style and _ss != style:
                    with _sessions_lock:
                        if sid in _sessions:
                            _sessions[sid]["style"] = style
                            _sessions[sid]["updated_at"] = time.time()
                    _save_sessions()
                else:
                    style = _ss or style
                if _sh:
                    eff_hist = _sh
                elif req.history:
                    _ch = [{"role": m.role, "content": m.content} for m in req.history]
                    eff_hist = _ch or None
                    if _ch:
                        with _sessions_lock:
                            if sid in _sessions:
                                _sessions[sid]["history"] = _ch
                                _sessions[sid]["updated_at"] = time.time()
                        _save_sessions()
            elif req.history:
                _ch = [{"role": m.role, "content": m.content} for m in req.history]
                sid = _get_or_create_session(None, style=style)
                eff_hist = _ch or None
                if _ch:
                    with _sessions_lock:
                        if sid in _sessions:
                            _sessions[sid]["history"] = _ch
                            _sessions[sid]["updated_at"] = time.time()
                    _save_sessions()
            else:
                sid = _get_or_create_session(None, style=style)
        elif sid:
            _get_or_create_session(sid, style=style)
        if not style:
            style = cfg.DEFAULT_STYLE

        # Chuẩn bị context nhanh (single-pass)
        async def _gen():
            import json as _j

            def _status(msg: str) -> str:
                return f"data: {_j.dumps({'status': msg}, ensure_ascii=False)}\n\n"

            def _done(payload: dict) -> str:
                return f"data: {_j.dumps({'done': True, **payload}, ensure_ascii=False)}\n\n"

            try:
                # (0) Cache: câu đơn lượt giống hệt -> trả ngay từng đoạn
                ckey = _cacheable(req)
                if ckey:
                    hit = _cache.get(ckey)
                    if hit is not None:
                        _log_chat(user, req, hit, time.time() - t0, stream=True)
                        _ans = getattr(hit, "answer", "") or ""
                        for i in range(0, len(_ans), 200):
                            yield f"data: {_j.dumps({'token': _ans[i:i + 200]}, ensure_ascii=False)}\n\n"
                        yield _done({
                            "answer": _ans,
                            "sources": list(getattr(hit, "sources", []) or []),
                            "standalone_question": getattr(hit, "standalone_question", None),
                            "summary": getattr(hit, "summary", None),
                            "session_id": getattr(hit, "session_id", None),
                            "style": getattr(hit, "style", None),
                        })
                        return
                # (1) Viết lại câu hỏi cho retrieval
                yield _status("Đang hiểu câu hỏi...")
                standalone_q = req.question
                if eff_hist:
                    try:
                        standalone_q = await run_in_threadpool(lambda: rewrite_query(req.question, eff_hist))
                    except Exception:
                        pass
                # (2) Retrieve hybrid + dedup (ngang /api/chat)
                yield _status("Đang tìm tài liệu...")
                _top = req.top_k or (DEFAULT_RERANK_TOP_K if req.use_rerank else RETRIEVE_TOP_K)
                fetch_k = min(_top * 5, 15) if req.use_rerank else _top
                validate_tool_call("retrieve", {"category": req.category, "top_k": _top})
                ctx = await run_in_threadpool(lambda: _do_retrieve(standalone_q, category=req.category, top_k=fetch_k, session_id=sid))
                ctx = _ensure_trang_phuc_coverage(standalone_q, ctx)
                try:
                    ctx = deduplicate_docs(ctx)
                except Exception:
                    pass
                if not ctx:
                    from backend.generation.rag import _make_clarification
                    clar = validate_output(_make_clarification(req.question, style=style) + " (Hiện mình chưa tìm thấy tài liệu nào khớp với câu hỏi này, bạn có thể cho thêm chi tiết được không?)")
                    if sid:
                        try:
                            _append_to_session(sid, sanitize_input(req.question, max_len=2000), clar, style=style, last_math_result=None)
                        except Exception:
                            pass
                    for i in range(0, len(clar), 200):
                        yield f"data: {_j.dumps({'token': clar[i:i + 200]}, ensure_ascii=False)}\n\n"
                    yield _done({
                        "answer": clar, "sources": [], "standalone_question": standalone_q,
                        "summary": None, "session_id": sid, "style": style,
                        "needs_clarification": True, "clarify_reason": "no_context",
                    })
                    return
                # (3) Rerank
                if req.use_rerank and ctx:
                    yield _status("Đang xếp hạng tài liệu...")
                    ctx = await run_in_threadpool(lambda: rerank(standalone_q, ctx, top_k=_top))
                sources = [c.get("metadata", {}) for c in ctx]
                # (4) Dựng messages (kèm cửa sổ lịch sử + tóm tắt như /api/chat)
                hist_win, summ = None, None
                if eff_hist:
                    hist_win, summ = await run_in_threadpool(lambda: _prep(eff_hist))
                    messages = build_messages_with_history(ctx, req.question, hist_win, summ, style=style)
                else:
                    messages = build_messages(ctx, req.question, style=style)
                # (5) Stream từng token từ Ollama trên thread riêng để không chặn event loop
                yield _status("Đang sinh câu trả lời...")
                import queue as _queue
                import threading as _threading
                _tok_q: _queue.Queue = _queue.Queue()
                _END = object()

                def _pump():
                    try:
                        for _tok in _call_ollama_stream(messages, cfg.LLM_MODEL, cfg.GEN_MAX_TOKENS, cfg.LLM_THINK):
                            _tok_q.put(_tok)
                    except Exception as _pe:
                        _tok_q.put(_pe)
                    finally:
                        _tok_q.put(_END)

                _th = _threading.Thread(target=_pump, daemon=True)
                _th.start()
                full = ""
                _n_tokens = 0
                while True:
                    item = await run_in_threadpool(_tok_q.get)
                    if item is _END:
                        break
                    if isinstance(item, Exception):
                        raise item
                    # Loại bỏ delimiter giả mạo trong token streaming
                    token_safe = item.replace("<<<UNTRUSTED_DATA>>>", "").replace("<<<END_UNTRUSTED_DATA>>>", "")
                    if not token_safe:
                        continue
                    full += token_safe
                    _n_tokens += 1
                    if _n_tokens % 25 == 0 and await request.is_disconnected():
                        break
                    yield f"data: {_j.dumps({'token': token_safe}, ensure_ascii=False)}\n\n"
                # Output validation cho toàn bộ trước khi lưu
                try:
                    full = validate_output(full)
                    sources = validate_sources(sources)
                except Exception:
                    pass
                # Lưu session sau khi xong (đã validated)
                if sid:
                    try:
                        _append_to_session(sid, sanitize_input(req.question, max_len=2000), full, summary=summ, style=style)
                    except Exception:
                        pass
                resp_obj = ChatResponse(
                    answer=full, sources=sources, context=None, session_id=sid,
                    standalone_question=standalone_q, summary=summ, style=style,
                )
                if ckey and sources:
                    try:
                        _cache.put(ckey, resp_obj)
                    except Exception:
                        pass
                _log_chat(user, req, resp_obj, time.time() - t0, stream=True)
                yield _done({
                    "answer": full, "sources": sources, "standalone_question": standalone_q,
                    "summary": summ, "session_id": sid, "style": style,
                })
            except Exception as e:
                import json as _j2
                yield f"data: {_j2.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        return StreamingResponse(_gen(), media_type="text/event-stream")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clean")
async def trigger_clean():
    """Trigger clean (preprocessing) - async demo, thuc te nen chay background task"""
    try:
        from backend.preprocessing.pipeline import show_stats
        # Just return stats, not actually run clean (can be long)
        return {"message": "Dung: python -m backend.cli clean", "stats": _get_stats()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount frontend: ưu tiên bản build React+Vite (frontend/dist), fallback thư mục frontend/
_frontend_root = Path(__file__).resolve().parent.parent.parent / "frontend"
_dist_dir = _frontend_root / "dist"
frontend_dir = _dist_dir if (_dist_dir / "index.html").exists() else _frontend_root
if frontend_dir.exists():
    # Luon doc index.html hien tai va khong de trinh duyet giu giao dien cu.
    @app.get("/", include_in_schema=False)
    async def frontend_index():
        return FileResponse(
            frontend_dir / "index.html",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    @app.middleware("http")
    async def disable_frontend_cache(request: Request, call_next):
        response = await call_next(request)
        if not request.url.path.startswith("/api/") and request.url.path not in {"/docs", "/openapi.json", "/redoc"}:
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response

    # API routes uu tien hon static files; tai nguyen frontend cung khong bi cache cu.
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

    uvicorn.run("backend.api:app", host=args.host, port=args.port, reload=not args.no_reload)
