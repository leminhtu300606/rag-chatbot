# RAG Chatbot - Học viện Kỹ thuật Mật mã

Hệ thống **Retrieval-Augmented Generation (RAG)** cho phép hỏi-đáp trên kho tài liệu nội bộ (quy chế, quy định, quyết định, tài liệu hướng dẫn, thông báo) với khả năng **hội thoại có ngữ cảnh** (hiểu câu hỏi nối tiếp như “nó là gì?”, “nói rõ hơn”).

> **Đặc điểm chính:** Truy xuất bằng embedding tiếng Việt (lưu ở PostgreSQL/pgvector) → rerank bằng cross-encoder → sinh câu trả lời bằng LLM (Ollama `qwen2.5` mặc định), có viết lại câu hỏi (query rewriting), quản lý lịch sử hội thoại dạng cửa sổ + tóm tắt, lưu phiên (session) phía server và localStorage.

---

## 1. Kiến trúc tổng quan

```
[PDF/DOCX] ──► Preprocessing (loader → cleaner → chunker) ──► data/processed/*.parquet
                                                                   │
                                                                   ▼
                                                           Indexing (embedder → PostgreSQL/pgvector) ──► bảng rag_chunks
                                                                   │
                                                                   ▼
User ──► Frontend (React+Vite) ──► FastAPI (/api/chat) ──► Generation (retrieve → rerank → prompt → LLM)
                                              ▲                              │
                                              │                              ▼
                                         Session Store                  answer + sources
                                        (history, summary)
```

**Luồng hội thoại (conversational):**

```
Câu hỏi mới + Lịch sử
       │
       ├─► _prepare_history(): cắt cửa sổ CONVERSATION_WINDOW*2, tóm tắt nếu > CONVERSATION_SUMMARY_THRESHOLD
       │
       ├─► rewrite_query(): phát hiện đại từ mơ hồ → LLM viết lại thành câu độc lập (dùng cho retrieval)
       │
       ├─► retrieve(standalone_question) → rerank(standalone_question)
       │
       └─► generate_with_history(context, question_gốc, history_window, summary) → trả lời tự nhiên
```

---

## 2. Cấu trúc dự án

```
E:\rag\
├── data/
│   ├── classified/                 # Dữ liệu thô (PDF, DOCX) theo category
│   │   ├── quyet_dinh/thanh_tra/
│   │   ├── quy_che/cong_tac_hsv / dao_tao/
│   │   ├── quy_dinh/khao_thi / van_hoa_hoc_duong/
│   │   ├── tai_lieu_huong_dan/dao_tao / ky_nang_sv/
│   │   └── thong_bao/hoc_bong / hoc_phi / ky_thi/
│   ├── processed/                  # Artifact sau tiền xử lý: *.parquet
│   │   └── .indexed.json           # manifest mtime (key posix) cho build incremental
│   ├── uploads/                    # File upload theo phiên (không commit)
├── backend/
│   ├── api/app.py                  # Giao diện HTTP: FastAPI + serve frontend, quản lý session
│   ├── cli/main.py                 # Giao diện CLI: clean / index (python -m backend.cli <clean|index>)
│   ├── config/settings.py          # Cấu hình chung + security dùng chung
│   ├── common/utils.py             # dedup (chuẩn hóa + loại trùng lặp)
│   ├── security/auth.py            # Xác thực + phân quyền
│   ├── infra/cache.py              # Cache câu hỏi lặp
│   ├── preprocessing/              # PHẦN 1: Tiền xử lý
│   │   ├── loader.py               # PDF (pymupdf + OCR easyocr) & DOCX
│   │   ├── cleaner.py              # Bỏ header/footer, số trang, quốc huy
│   │   ├── chunker.py              # Semantic + Contextual chunking
│   │   ├── pipeline.py             # process_file/process_all → *.parquet (+ clean_*, show_stats, CLI)
│   │   ├── storage.py              # Helper parquet (đọc/ghi, hỗ trợ jsonl cũ)
│   ├── indexing/                   # PHẦN 2: Indexing
│   │   ├── embedder.py             # dangvantuan/vietnamese-embedding
│   │   ├── vectorstore.py          # PostgreSQL/pgvector (bảng rag_chunks)
│   │   └── retrieval.py            # Cụm truy xuất: vector + BM25 + hybrid
│   ├── generation/                 # PHẦN 3: Generation
│   │   ├── agent.py                # Agent đa bước (LangGraph self-check, max AGENT_MAX_STEPS)
│   │   ├── prompts.py              # SYSTEM_PROMPT, build_messages, rewrite/summary prompts
│   │   ├── generator.py            # Ollama / transformers, rewrite_query, summarize_history
│   │   ├── langchain_chain.py      # Adapter LangChain (ChatOllama + LCEL)
│   │   ├── reranker.py             # cross-encoder ms-marco-MiniLM-L-6-v2
│   │   ├── rag.py                  # Điều phối retrieve → rerank → generate (agent-first, fallback manual)
│   │   ├── calculator.py           # Tính toán chính xác bằng Decimal
├── tools/                          # Vận hành (trước đây scripts/)
│   ├── data/rebuild_data.py        # Rebuild toàn bộ: clean + index --rebuild + verify
│   ├── eval/eval_quick.py          # Đánh giá nhanh qua HTTP API
│   └── ops/backup.py               # Backup/restore pgvector + processed + sessions + users
├── infra/nginx/
│   └── nginx.conf                  # Reverse proxy + TLS nội bộ (trước đây deploy/nginx.conf)
├── frontend/                       # React + TypeScript + Vite (build ra dist/, FastAPI serve)
│   ├── index.html
│   ├── package.json / vite.config.ts / tsconfig.json
│   └── src/ (App.tsx, components.tsx, api/client.ts, types.ts, hooks.ts, utils.ts, styles.css)
├── requirements.txt
└── README.md
```

| Chạy nhanh | Lệnh |
|---|---|
| Rebuild toàn bộ data | `python tools/data/rebuild_data.py` |
| Chỉ clean | `python -m backend.cli clean` |
| Chỉ index | `python -m backend.cli index --rebuild` |
| Kiểm tra | `python -m backend.cli clean --stats` (xem /api/health để biết số files/vectors) |
| Chạy API | `uvicorn backend.api:app --reload --port 8000` |

| Phần | Nhiệm vụ | Input → Output | Thư mục |
|------|----------|----------------|---------|
| **Tiền xử lý** | Làm sạch & chunk | `data/classified` → `data/processed/*.parquet` | `backend/preprocessing/` |
| **Indexing** | Embed → Vector Store | `data/processed/*.parquet` → PostgreSQL (`rag_chunks`) | `backend/indexing/` + `backend/cli/main.py` |
| **Generation** | Rerank + Prompt + LLM | `query (+ history)` → `{answer, sources}` | `backend/generation/` |
| **API** | FastAPI + Session | `HTTP` → `JSON` | `backend/api/app.py` |
| **Frontend** | Giao diện chat | Người dùng → API | `frontend/` |

---

## 3. Tính năng chính

*   **Tiền xử lý:** Hỗ trợ PDF (bản quét OCR) và DOCX, làm sạch header/footer, chunking ngữ nghĩa + ngữ cảnh (giữ tiêu đề Chương/Điều, overlap câu).
*   **Indexing:** Embedding `dangvantuan/vietnamese-embedding`, pgvector cosine (HNSW, distance 0~2 như cũ), build incremental theo `mtime` + manifest.
*   **Rerank:** Cross-encoder `ms-marco-MiniLM-L-6-v2`, mặc định `top_k=3`.
*   **Hội thoại có ngữ cảnh:**
    *   Viết lại câu hỏi mơ hồ thành dạng độc lập (heuristic + LLM) để retrieval chính xác.
    *   Quản lý cửa sổ lịch sử `CONVERSATION_WINDOW=6` turns, cắt mỗi message `300` ký tự.
    *   Tự động tóm tắt khi vượt `CONVERSATION_SUMMARY_THRESHOLD=12` turns.
    *   Lưu phiên in-memory trên server (`session_id`) + `localStorage` trên client, có endpoint `GET/DELETE /api/sessions/{id}`.
*   **Giao diện:** Bỏ phần cấu hình RAG trên UI (dùng mặc định `top_k=3`, `rerank=true`, luôn nhớ hội thoại), chỉ hiển thị trạng thái health/history/session, hỗ trợ gõ `quy_che: câu hỏi` để lọc danh mục, hiệu ứng typewriter, panel nguồn/rewrite/summary.
*   **CLI:** `backend/answer.py` hỗ trợ hội thoại tương tác (`clear`/`history`), `backend/test.py` kiểm tra chunking/embedding.

---

## 4. Yêu cầu hệ thống

*   Python 3.10+, Node.js 20+ (build frontend)
*   PostgreSQL 16 + extension pgvector (có sẵn qua `docker compose up db`, image `pgvector/pgvector`)
*   Ollama (khuyến nghị) hoặc transformers local
*   RAM/VRAM: `qwen2.5:7b` ~4.7GB cần ~8GB; có thể dùng `qwen2.5:1.5b/0.5b/3b` nếu hạn chế

---

## 5. Cài đặt

```powershell
# 1. Tạo venv & kích hoạt
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Cài dependencies
pip install -r requirements.txt

# 3. Cài Ollama (https://ollama.com)
ollama pull qwen2.5        # hoặc qwen2.5:1.5b nếu máy yếu
ollama serve               # chạy ở terminal riêng
```

---

## 6. Sử dụng nhanh

### 6.1 Tiền xử lý
```powershell
python -m backend.cli clean                 # toàn bộ data/classified → data/processed/*.parquet
python -m backend.cli clean --stats         # xem thống kê
python -m backend.cli clean --dry-run       # chỉ thử, không ghi

# Trong Python
from backend.preprocessing.pipeline import process_all
from backend.config.settings import DATA_DIR
process_all(DATA_DIR)
```

### 6.2 Indexing
```powershell
python -m backend.cli index                 # build incremental
python -m backend.cli index --rebuild       # build lại toàn bộ
```

### 6.3 Hỏi-đáp CLI

**Đơn lượt (trong Python hoặc qua API — không còn CLI riêng):**
```powershell
# Hỏi qua API đang chạy:
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{"question":"Quy chế đào tạo là gì?","use_rerank":true,"show_context":true}'
```

**Hội thoại nhiều lượt:** dùng giao diện web (`http://localhost:8000/`, mỗi phiên có `session_id` riêng, hỗ trợ `New chat`) hoặc truyền `history` + `session_id` qua `POST /api/chat`.

Trong Python:
```python
from backend.generation.rag import answer, answer_with_history

# Đơn lượt
res = answer("Học bổng KKHT xét thế nào?", use_rerank=True)

# Có lịch sử
history = [
    {"role": "user", "content": "Quy chế đào tạo là gì?"},
    {"role": "assistant", "content": "Là quy định về đào tạo..."}
]
res = answer_with_history("Điều kiện trong nó là gì?", history=history, use_rerank=True)
print(res["standalone_question"])  # câu đã viết lại
print(res["answer"])
```

### 6.4 Chạy Fullstack (API + Frontend)

```powershell
# 1. Chuẩn bị dữ liệu
python -m backend.cli clean
python -m backend.cli index --rebuild

# 2. Chạy API (phục vụ cả frontend)
uvicorn backend.api:app --reload --port 8000
# hoặc
python -m backend.api

# 3. Mở trình duyệt
# Frontend:   http://localhost:8000/
# API Docs:   http://localhost:8000/docs
# Health:     http://localhost:8000/api/health
```

---

## 7. API

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/` | Frontend |
| GET | `/docs` | Swagger UI |
| GET | `/api/health` | Kiểm tra `chroma_count` + `processed_files` |
| GET | `/api/stats` | Thống kê chi tiết + `active_sessions` |
| GET | `/api/categories` | Danh sách category |
| GET | `/api/sessions` | Danh sách phiên |
| GET | `/api/sessions/{id}` | Chi tiết phiên (history, summary) |
| DELETE | `/api/sessions/{id}` | Xóa phiên |
| POST | `/api/chat` | Hỏi đáp hội thoại |
| POST | `/api/clean` | Trả về stats (không chạy clean) |

**POST /api/chat:**

Request:
```json
{
  "question": "Điều kiện trong nó là gì?",
  "category": "quy_che",
  "top_k": 3,
  "use_rerank": true,
  "show_context": true,
  "history": [{"role": "user", "content": "Quy chế đào tạo là gì?"}, {"role": "assistant", "content": "..."}],
  "session_id": "uuid-optional",
  "use_history": true
}
```

Response:
```json
{
  "answer": "...",
  "sources": [{"filename": "quyche.pdf", "page": 2, "category": "quy_che"}],
  "context": [{"text": "...", "metadata": {}, "score": 0.85}],
  "session_id": "uuid",
  "standalone_question": "Điều kiện tốt nghiệp trong quy chế đào tạo là gì?",
  "summary": "Tóm tắt hội thoại cũ...",
  "history_used": [{"role": "user", "content": "..."}]
}
```

Ví dụ curl:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Quy chế đào tạo là gì?","use_rerank":true,"show_context":true}'

# Hội thoại
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Trong nó, học bổng xét thế nào?","history":[{"role":"user","content":"Quy chế đào tạo là gì?"},{"role":"assistant","content":"..."}],"session_id":"my-session-1"}'
```

---

## 8. Frontend (React + TypeScript + Vite)

`frontend/` là project Vite. Dev: `cd frontend && npm install && npm run dev` (proxy `/api` → `http://127.0.0.1:8000`, mở `http://localhost:5173`). Prod: `npm run build` → `frontend/dist/`, được `backend/api/app.py` serve tại `/` (ưu tiên `dist`, fallback thư mục `frontend/`).

*   `src/App.tsx` — Điều phối state: sessions, hội thoại, upload, auth, health polling.
*   `src/components.tsx` — Layout 3 cột (sidebar lịch sử + chat chính + panel nguồn right drawer), welcome + examples, typewriter, uploads bar, login/stats modal.
*   `src/api/client.ts` — Typed client cho mọi endpoint (`/api/chat`, sessions, uploads, auth...), tự gắn Bearer `rag_token`, 401 mở form login.
*   Dùng mặc định `top_k=3`, `rerank=true`, luôn nhớ hội thoại. Hỗ trợ gõ `quy_che: câu hỏi` để lọc danh mục ngay trong ô chat. Giữ nguyên keys `localStorage` cũ (`rag_sessions_v2`, `rag_active_session_v2`, `rag_token`) nên tương thích phiên/chat cũ.

---

## 9. Cấu hình (`backend/config/settings.py`)

```python
DATA_DIR = BASE_DIR / "data" / "classified"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATABASE_URL = "postgresql://rag:rag@localhost:5432/rag"  # local; trong compose: ...@db:5432/rag
PGVECTOR_TABLE = "rag_chunks"
PROCESSED_EXT = ".parquet"

EMBED_MODEL = "dangvantuan/vietnamese-embedding"
LLM_BACKEND = "ollama"          # ollama | transformers
LLM_MODEL = "qwen2.5"           # qwen2.5:0.5b/1.5b/3b/7b/14b/32b
OLLAMA_BASE = "http://localhost:11434"
GEN_MAX_TOKENS = 512

CHUNK = {
    "min_chars": 200, "max_chars": 1200,
    "overlap_sentences": 1, "break_threshold": 0.5,
    "contextual": True, "context_mode": "heading+overlap",
    "context_window": 1, "heading_enrich": True, "use_llm_context": False
}
RETRIEVE_TOP_K = 3
COLLECTION_NAME = "rag_hvm"

# Hội thoại
CONVERSATION_WINDOW = 6                  # giữ 6 turns gần nhất
CONVERSATION_SUMMARY_THRESHOLD = 12      # vượt 12 turns thì tóm tắt
CONVERSATION_MAX_HISTORY_CHARS = 300
REWRITE_MODEL = None                     # None = dùng LLM_MODEL
SUMMARY_MODEL = None
REWRITE_MAX_TOKENS = 256
SUMMARY_MAX_TOKENS = 256
```

---

## 10. Lưu ý

*   **Parquet:** `storage.py` đọc được cả `parquet` và `jsonl` cũ để migrate: `python -m backend.preprocessing.storage --migrate` (cần `pandas` + `pyarrow` đã có trong `requirements.txt`).
*   **Ollama lỗi:** Nếu gặp `out-of-memory` → đổi `LLM_MODEL` sang `qwen2.5:1.5b`/`0.5b`; nếu `connection refused` → kiểm tra `ollama serve` và `ollama list`.
*   **Dimension mismatch:** Chạy `python -m backend.cli index --rebuild` sau khi đổi `EMBED_MODEL`.
*   **Session:** In-memory, TTL 24h, tối đa 200 phiên, mỗi phiên tối đa 100 messages; xóa bằng `DELETE /api/sessions/{id}` hoặc nút `New chat` (xóa cả server + localStorage).

---

## 11. Vận hành production (server nội bộ)

### 11.1 Chuẩn bị server

*   Docker Engine + Compose plugin, ~16GB RAM (CPU-only), 50GB đĩa trống.
*   Mở port 80/443 nội bộ. Ollama + PostgreSQL/pgvector chạy trong compose (không cần cài riêng).

### 11.2 Cấu hình

```bash
cp .env.example .env
# Sửa bắt buộc: ADMIN_PASSWORD, POSTGRES_PASSWORD, LLM_MODEL (phải khớp model sẽ pull)
nano .env
```

### 11.3 Build & chạy

```bash
docker compose build
docker compose up -d            # service db (pgvector) lên trước, app chờ db healthy
docker compose exec ollama ollama pull ${LLM_MODEL:-qwen2.5:1.5b}
# Nạp dữ liệu lần đầu vào pgvector (mount data/classified read-only từ host):
docker compose exec app python -m backend.cli clean
docker compose exec app python -m backend.cli index --rebuild
curl http://localhost:8000/api/health   # vector_count > 0 là đạt
```

### 11.4 Đăng nhập (khuyến nghị trước khi mở cho user)

```bash
# trong .env: AUTH_ENABLED=true  (admin bootstrap từ ADMIN_USERNAME/ADMIN_PASSWORD)
docker compose up -d
# Đăng nhập lấy token:
curl -X POST http://localhost:8000/api/auth/login \
  -H 'Content-Type: application/json' -d '{"username":"admin","password":"..."}'
# Tạo user phòng ban (chỉ thấy category cho phép):
curl -X POST http://localhost:8000/api/auth/users -H "Authorization: Bearer <TOKEN>" \
  -H 'Content-Type: application/json' \
  -d '{"username":"daotao","password":"...","categories":["quy_che","quy_dinh","thong_bao"]}'
```

Frontend tự hiện form login khi gặp 401.

### 11.5 TLS nội bộ (nginx profile `edge`, chi tiết: `infra/nginx/nginx.conf`)

```bash
# Tự ký cho tên nội bộ (ví dụ rag.noibo.local), CA nội bộ ký sẽ chuẩn hơn:
openssl req -x509 -newkey rsa:2048 -keyout infra/nginx/certs/privkey.pem \
  -out infra/nginx/certs/fullchain.pem -days 825 -nodes -subj "/CN=rag.noibo.local"
# Mở block 443 trong infra/nginx/nginx.conf rồi:
docker compose --profile edge up -d
```

### 11.6 Backup / restore (chạy trên host, nên cron hàng đêm)

```bash
python tools/ops/backup.py --backup --keep 7     # gồm pg_dump bảng vector + processed + sessions + users
python tools/ops/backup.py --list
python tools/ops/backup.py --restore backups/rag-YYYYMMDD-HHMMSS.tar.gz
# Nạp lại DB từ bản backup: psql "$DATABASE_URL" -f <thư-mục-bung>/db/rag.sql
# Backup trực tiếp Postgres: docker compose exec db pg_dump -U rag rag > pg-rag.sql
```

### 11.7 Quy trình cập nhật tài liệu (ai cũng làm được)

1.  Người phụ trách bỏ PDF/DOCX đã duyệt vào `data/classified/<category>/...`.
2.  Chạy: `docker compose exec app python -m backend.cli clean` rồi `... index` (incremental, không `--rebuild`).
3.  Kiểm tra: `/api/health` (số files/vectors tăng), hỏi thử 2–3 câu đặc trưng.
4.  Ghi log thay đổi (ngày, ai, file nào) vào sổ/quy định đơn vị.

### 11.8 Giám sát & đánh giá

*   `GET /api/health`: `status`, `ollama_ok`, `disk_free_gb`, `auth_enabled`.
*   `GET /api/metrics` (admin): count, latency p50/p95, no-source rate, cache hit; log chi tiết mỗi hỏi đáp ở `data/logs/chat.jsonl`.
*   `python tools/eval/eval_quick.py --base http://localhost:8000` sau mỗi lần đổi model/tham số (báo lỗi khi no-source > 40%).

### 11.9 ClamAV (tùy chọn)

Chạy clamav-daemon đâu đó nội bộ rồi đặt trong `.env`:
`CLAMAV_ENABLED=true`, `CLAMAV_HOST`, `CLAMAV_PORT`, `CLAMAV_FAIL_CLOSED=false`.

---

## 12. Giấy phép

Dự án nội bộ Học viện Kỹ thuật Mật mã.
