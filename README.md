# RAG Chatbot - Học viện Kỹ thuật Mật mã

Hệ thống **Retrieval-Augmented Generation (RAG)** cho phép hỏi-đáp trên kho tài liệu nội bộ (quy chế, quy định, quyết định, tài liệu hướng dẫn, thông báo) với khả năng **hội thoại có ngữ cảnh** (hiểu câu hỏi nối tiếp như “nó là gì?”, “nói rõ hơn”).

> **Đặc điểm chính:** Truy xuất bằng embedding tiếng Việt → rerank bằng cross-encoder → sinh câu trả lời bằng LLM (Ollama `qwen2.5` mặc định), có viết lại câu hỏi (query rewriting), quản lý lịch sử hội thoại dạng cửa sổ + tóm tắt, lưu phiên (session) phía server và localStorage.

---

## 1. Kiến trúc tổng quan

```
[PDF/DOCX] ──► Preprocessing (loader → cleaner → chunker) ──► data/processed/*.parquet
                                                                  │
                                                                  ▼
                                                          Indexing (embedder → ChromaDB) ──► chroma_db/
                                                                  │
                                                                  ▼
User ──► Frontend (HTML/JS) ──► FastAPI (/api/chat) ──► Generation (retrieve → rerank → prompt → LLM)
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
E:\test\
├── data/
│   ├── classified/                 # Dữ liệu thô (PDF, DOCX) theo category
│   │   ├── quyet_dinh/thanh_tra/
│   │   ├── quy_che/cong_tac_hsv / dao_tao/
│   │   ├── quy_dinh/khao_thi / van_hoa_hoc_duong/
│   │   ├── tai_lieu_huong_dan/dao_tao / ky_nang_sv/
│   │   └── thong_bao/hoc_bong / hoc_phi / ky_thi/
│   ├── processed/                  # Artifact sau tiền xử lý: *.parquet
│   │   └── .indexed.json           # manifest mtime cho build incremental
│   └── chroma_db/                  # (thực tế ở repo root: chroma_db/)
├── backend/
│   ├── config.py                   # Cấu hình chung (đường dẫn, model, chunk, hội thoại)
│   ├── app.py                      # FastAPI - API + serve frontend, quản lý session
│   ├── answer.py                   # CLI hỏi-đáp (hỗ trợ lịch sử tương tác)
│   ├── test.py                     # CLI kiểm tra chunking/embedding
│   ├── clean.py                    # Alias cho preprocessing/clean_service
│   ├── index.py                    # Entry indexing (build incremental / --rebuild)
│   ├── preprocessing/              # PHẦN 1: Tiền xử lý
│   │   ├── loader.py               # PDF (pymupdf + OCR easyocr) & DOCX
│   │   ├── cleaner.py              # Bỏ header/footer, số trang, quốc huy
│   │   ├── chunker.py              # Semantic + Contextual chunking
│   │   ├── pipeline.py             # process_file/process_all → *.parquet
│   │   ├── storage.py              # Helper parquet (đọc/ghi, hỗ trợ jsonl cũ)
│   │   ├── clean_service.py        # Logic cho clean.py
│   │   └── preprocess.py           # Bí danh cho pipeline
│   ├── indexing/                   # PHẦN 2: Indexing
│   │   ├── embedder.py             # dangvantuan/vietnamese-embedding
│   │   ├── vectorstore.py          # ChromaDB persistent
│   │   └── retriever.py            # Truy xuất top-k
│   └── generation/                 # PHẦN 3: Generation
│       ├── prompts.py              # SYSTEM_PROMPT, build_messages, rewrite/summary prompts
│       ├── generator.py            # Ollama / transformers, rewrite_query, summarize_history
│       ├── reranker.py             # cross-encoder ms-marco-MiniLM-L-6-v2 (top_k=3)
│       ├── rag.py                  # Điều phối retrieve → rerank → generate (answer / answer_with_history)
│       └── test_service.py         # Logic cho test.py
├── frontend/
│   ├── index.html                  # Giao diện chat 3 cột (sidebar - chat - nguồn)
│   ├── style.css                   # Dark theme, responsive
│   └── app.js                      # Gọi /api/chat, quản lý localStorage + session + typewriter
├── chroma_db/                      # ChromaDB persistent
├── requirements.txt
└── README.md
```

| Phần | Nhiệm vụ | Input → Output | Thư mục |
|------|----------|----------------|---------|
| **Tiền xử lý** | Làm sạch & chunk | `data/classified` → `data/processed/*.parquet` | `backend/preprocessing/` |
| **Indexing** | Embed → Vector Store | `data/processed/*.parquet` → `chroma_db/` | `backend/indexing/` + `backend/index.py` |
| **Generation** | Rerank + Prompt + LLM | `query (+ history)` → `{answer, sources}` | `backend/generation/` |
| **API** | FastAPI + Session | `HTTP` → `JSON` | `backend/app.py` |
| **Frontend** | Giao diện chat | Người dùng → API | `frontend/` |

---

## 3. Tính năng chính

*   **Tiền xử lý:** Hỗ trợ PDF (bản quét OCR) và DOCX, làm sạch header/footer, chunking ngữ nghĩa + ngữ cảnh (giữ tiêu đề Chương/Điều, overlap câu).
*   **Indexing:** Embedding `dangvantuan/vietnamese-embedding`, ChromaDB cosine, build incremental theo `mtime` + manifest.
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

*   Python 3.10+
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
python -m backend.clean                 # toàn bộ data/classified → data/processed/*.parquet
python -m backend.clean --stats         # xem thống kê
python -m backend.clean --dry-run       # chỉ thử, không ghi

# Trong Python
from backend.preprocessing.pipeline import process_all
from backend.config import DATA_DIR
process_all(DATA_DIR)
```

### 6.2 Indexing
```powershell
python -m backend.index                 # build incremental
python -m backend.index --rebuild       # build lại toàn bộ
```

### 6.3 Hỏi-đáp CLI

**Đơn lượt:**
```powershell
python -m backend.answer --question "Quy chế đào tạo là gì?" --show-context
python -m backend.test --chunks --embeddings
python -m backend.test -q "Học bổng xét thế nào?" --show-context
```

**Hội thoại (CLI tương tác):**
```powershell
python -m backend.answer
# Ban: Quy chế đào tạo là gì?
# Ban: Điều kiện tốt nghiệp trong nó là gì?   # sẽ tự rewrite
# Ban: history   # xem lịch sử
# Ban: clear     # xóa lịch sử
```

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
python -m backend.clean
python -m backend.index --rebuild

# 2. Chạy API (phục vụ cả frontend)
uvicorn backend.app:app --reload --port 8000
# hoặc
python -m backend.app

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

## 8. Frontend

`frontend/` là trang tĩnh thuần HTML/CSS/JS, không cần build, được serve bởi `backend/app.py` qua `StaticFiles(html=True)`.

*   `index.html` — Layout 3 cột: sidebar lịch sử + chat chính + panel nguồn (right drawer). Đã **loại bỏ khối cấu hình RAG** (category/top_k/rerank) khỏi giao diện; các giá trị dùng mặc định (`top_k=3`, `rerank=true`, `luôn nhớ hội thoại`). Hỗ trợ gõ `quy_che: câu hỏi` để lọc danh mục ngay trong ô chat.
*   `style.css` — Dark theme, responsive (sidebar thành drawer trên mobile).
*   `app.js` — Quản lý `localStorage` (`rag_session_id`, `rag_conversation_v2`), gọi `fetch("/api/chat")` với `history` + `session_id`, hiển thị rewrite/summary, hiệu ứng typewriter, health polling.

---

## 9. Cấu hình (`backend/config.py`)

```python
DATA_DIR = BASE_DIR / "data" / "classified"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHROMA_DIR = BASE_DIR / "chroma_db"
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
*   **Dimension mismatch:** Chạy `python -m backend.index --rebuild` sau khi đổi `EMBED_MODEL`.
*   **Session:** In-memory, TTL 24h, tối đa 200 phiên, mỗi phiên tối đa 100 messages; xóa bằng `DELETE /api/sessions/{id}` hoặc nút `New chat` (xóa cả server + localStorage).

---

## 11. Giấy phép

Dự án nội bộ Học viện Kỹ thuật Mật mã.
