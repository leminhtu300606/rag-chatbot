"""
backend/auth.py - Xac thuc + phan quyen production (GĐ2)
=========================================================
- Không thêm dependency: PBKDF2-HMAC-SHA256 (hashlib) + token ngẫu nhiên (secrets).
- Users lưu ở data/users.json (gitignore). Mỗi user:
    {"pw": "<salt>$<hash>", "is_admin": bool, "categories": ["*"] | [...],
     "daily": {"date": "YYYY-MM-DD", "count": int}}
- Token: {"token": {"username": str, "exp": epoch}} giữ RAM + không persist
  (restart server thì login lại — chấp nhận được ở quy mô <20 user).
- Tắt auth: AUTH_ENABLED=false (mặc định) → mọi dependency cho qua, giữ
  nguyên hành vi demo hiện tại.

Dùng trong backend/api.py qua get_current_user / require_admin.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from datetime import date
from typing import Dict, List, Optional

import backend.config.settings as cfg
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)

_users_lock = threading.Lock()
_users: Dict[str, Dict] = {}
_tokens: Dict[str, Dict] = {}  # token -> {"username":..., "exp":...}

_PBKDF2_ROUNDS = 200_000


def _hash_password(password: str, salt: Optional[str] = None) -> str:
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ROUNDS)
    return f"{salt}${dk.hex()}"


def _check_password(password: str, stored: str) -> bool:
    try:
        salt, _ = stored.split("$", 1)
        return hmac.compare_digest(_hash_password(password, salt), stored)
    except Exception:
        return False


def _save_users() -> None:
    try:
        p = cfg.USERS_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with _users_lock:
            data = dict(_users)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(p)
    except Exception as e:
        print(f"[auth] save users failed: {e}")


def _load_users() -> None:
    try:
        p = cfg.USERS_FILE
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                with _users_lock:
                    _users.update({k: v for k, v in data.items() if isinstance(v, dict)})
    except Exception as e:
        print(f"[auth] load users failed: {e}")


def ensure_bootstrap_admin() -> None:
    """Tạo admin đầu tiên từ ENV (chỉ khi chưa có user nào)."""
    _load_users()
    with _users_lock:
        has_any = bool(_users)
    if has_any:
        return
    pw = (cfg.ADMIN_PASSWORD or "").strip()
    if not pw:
        pw = secrets.token_urlsafe(16)
        print(f"[auth] sinh ADMIN_PASSWORD ngẫu nhiên (lưu lại!): {pw}")
    with _users_lock:
        _users[cfg.ADMIN_USERNAME] = {
            "pw": _hash_password(pw),
            "is_admin": True,
            "categories": ["*"],
            "daily": {"date": date.today().isoformat(), "count": 0},
        }
    _save_users()
    print(f"[auth] bootstrap admin '{cfg.ADMIN_USERNAME}'")


def list_users() -> List[Dict]:
    with _users_lock:
        return [
            {"username": u, "is_admin": bool(v.get("is_admin")), "categories": v.get("categories", ["*"])}
            for u, v in _users.items()
        ]


def create_user(username: str, password: str, is_admin: bool = False, categories: Optional[List[str]] = None) -> Dict:
    username = (username or "").strip()
    if not username or len(username) > 64:
        raise ValueError("username không hợp lệ")
    if not password or len(password) < 6:
        raise ValueError("password tối thiểu 6 ký tự")
    cats = categories if categories else (["*"] if is_admin else list(cfg.CATEGORY_ORDER))
    # chỉ cho category hợp lệ hoặc "*"
    ok = set(cfg.CATEGORY_ORDER) | {"*"}
    cats = [c for c in cats if c in ok] or list(cfg.CATEGORY_ORDER)
    with _users_lock:
        if username in _users:
            raise ValueError("user đã tồn tại")
        _users[username] = {
            "pw": _hash_password(password),
            "is_admin": bool(is_admin),
            "categories": cats,
            "daily": {"date": date.today().isoformat(), "count": 0},
        }
    _save_users()
    return {"username": username, "is_admin": bool(is_admin), "categories": cats}


def delete_user(username: str) -> None:
    with _users_lock:
        if username not in _users:
            raise ValueError("user không tồn tại")
        _users.pop(username, None)
        # thu hồi token của user
        for t in [t for t, v in _tokens.items() if v.get("username") == username]:
            _tokens.pop(t, None)
    _save_users()


def verify_login(username: str, password: str) -> Dict:
    with _users_lock:
        rec = _users.get(username or "")
    if not rec or not _check_password(password or "", rec.get("pw", "")):
        raise ValueError("sai tên đăng nhập hoặc mật khẩu")
    token = secrets.token_urlsafe(32)
    with _users_lock:
        _tokens[token] = {"username": username, "exp": time.time() + int(cfg.AUTH_TOKEN_TTL_SECONDS)}
    return {"username": username, "is_admin": bool(rec.get("is_admin")),
            "categories": rec.get("categories", ["*"]), "token": token}


def revoke_token(token: str) -> None:
    with _users_lock:
        _tokens.pop(token, None)


def _cleanup_tokens() -> None:
    now = time.time()
    with _users_lock:
        for t in [t for t, v in _tokens.items() if v.get("exp", 0) < now]:
            _tokens.pop(t, None)


def get_user_by_token(token: str) -> Optional[Dict]:
    _cleanup_tokens()
    with _users_lock:
        rec = _tokens.get(token or "")
        if not rec:
            return None
        u = _users.get(rec["username"])
        if not u:
            return None
        return {"username": rec["username"], "is_admin": bool(u.get("is_admin")),
                "categories": u.get("categories", ["*"])}


def bump_daily(username: str, limit: int) -> int:
    """Tăng đếm lượt chat trong ngày, trả về số còn lại. Hết quota → ValueError."""
    today = date.today().isoformat()
    with _users_lock:
        rec = _users.get(username)
        if not rec:
            raise ValueError("user không tồn tại")
        d = rec.get("daily") or {}
        if d.get("date") != today:
            d = {"date": today, "count": 0}
        if int(d.get("count", 0)) >= int(limit):
            raise ValueError(f"Đã hết {limit} lượt hỏi hôm nay, vui lòng quay lại ngày mai")
        d["count"] = int(d.get("count", 0)) + 1
        rec["daily"] = d
        left = int(limit) - int(d["count"])
    _save_users()
    return left


def user_can_category(user: Optional[Dict], category: Optional[str]) -> bool:
    if user is None or category is None:
        return True
    cats = user.get("categories", ["*"]) or ["*"]
    return "*" in cats or category in cats


# ── FastAPI dependencies ──

async def get_current_user_opt(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[Dict]:
    """User hiện tại hoặc None (dùng cho endpoint public khi tắt auth)."""
    if not cfg.AUTH_ENABLED:
        return None
    if not creds or not creds.credentials:
        return None
    return get_user_by_token(creds.credentials)


async def require_user(user: Optional[Dict] = Depends(get_current_user_opt)) -> Dict:
    if not cfg.AUTH_ENABLED:
        return {"username": "anonymous", "is_admin": True, "categories": ["*"]}
    if not user:
        raise HTTPException(status_code=401, detail="Chưa đăng nhập (thiếu/không đúng Bearer token)")
    return user


async def require_admin(user: Dict = Depends(require_user)) -> Dict:
    if not cfg.AUTH_ENABLED:
        return user
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Cần quyền admin")
    return user
