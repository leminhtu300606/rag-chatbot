# RAG Chatbot — Học viện Kỹ thuật Mật mã

Hệ thống **hỏi-đáp tiếng Việt trên kho tài liệu nội bộ** (quy chế, quy định, quyết định, thông báo, tài liệu hướng dẫn) theo mô hình **Retrieval-Augmented Generation (RAG) có hội thoại**.

Trọng tâm: trả lời **chỉ dựa trên tài liệu truy xuất được**, trích dẫn nguồn, hiểu câu hỏi nối tiếp (“nó là gì?”, “nói rõ hơn”), nói thẳng khi không đủ nguồn thay vì bịa.

## 1. Hướng giải quyết

```
[PDF/DOCX] → Preprocessing (clean + chunk) → data/processed/*.parquet
                                                    ↓ embed
User → React → FastAPI /api/chat → [rewrite query → hybrid retrieve → rerank → LLM sinh trả lời] → answer + sources
                                    ↑ session (window 8 turns + summary) lưu server + localStorage
```

1. **Tiền xử lý:** đọc PDF (PyMuPDF + OCR EasyOCR) / DOCX, bỏ header/footer, chunk ngữ nghĩa + ngữ cảnh (giữ tiêu đề Chương/Điều, overlap 1 câu).
2. **Indexing:** embed từng chunk → PostgreSQL/pgvector. Truy xuất **hybrid BM25 (0.6) + vector (0.4)**, build incremental theo `mtime`.
3. **Generation:** viết lại câu hỏi mơ hồ thành câu độc lập → retrieve → **rerank cross-encoder** → prompt grounded + lịch sử (8 turns gần nhất + tóm tắt phần cũ) → LLM sinh trả lời kèm `sources`. Agent LangGraph tự kiểm tra nguồn (tối đa 3 bước), lỗi thì fallback manual. Streaming SSE mặc định.

## 2. Công nghệ, framework, model

| Lớp | Công nghệ / Model |
|---|---|
| **Backend** | Python 3.10+, **FastAPI** + Uvicorn, `psycopg` + `pgvector` (PostgreSQL 16, bảng `rag_chunks`, HNSW cosine) |
| **Preprocessing** | `pymupdf`, `easyocr`, `python-docx`, `underthesea`, `pandas` + `pyarrow` (parquet) |
| **Embedding** | `dangvantuan/vietnamese-embedding` (768d, CPU) / `BAAI/bge-m3` (GPU, tự chọn theo `DEVICE`) + `rank-bm25` |
| **Rerank** | Cross-encoder `ms-marco-MiniLM-L-6-v2`, `top_k=4` |
| **LLM** | **Ollama `qwen2.5:1.5b`** (mặc định, đổi được `0.5b/3b/7b/14b/32b` qua `LLM_MODEL`), bọc bằng **LangChain** (ChatOllama + LCEL) + **LangGraph** agent |
| **Frontend** | **React 18 + TypeScript + Vite**, `react-markdown`, SSE streaming, sidebar lịch sử theo ngày |
| **Infra** | Docker Compose (`app` + `db` pgvector + `ollama` + `nginx` TLS nội bộ) |

## 3. Cấu trúc dự án

```
backend/
  preprocessing/  # loader (PDF/OCR/DOCX) → cleaner → chunker → pipeline → *.parquet
  indexing/       # embedder → vectorstore (pgvector) → retrieval (hybrid)
  generation/     # rag.py (điều phối) → reranker → generator/prompts → agent (LangGraph) + calculator
  api/app.py      # FastAPI + session + SSE /api/chat/stream
  cli/main.py     # python -m backend.cli <clean|index>
frontend/         # React+Vite (src/App.tsx, components.tsx, api/client.ts)
tools/            # data/rebuild_data.py, eval/eval_quick.py, ops/backup.py
data/classified/  # PDF/DOCX thô theo category | data/processed/ # *.parquet
```

## 4. Chạy nhanh

```powershell
python -m venv venv; .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull qwen2.5:1.5b; ollama serve   # terminal riêng

python -m backend.cli clean               # classified → processed/*.parquet
python -m backend.cli index --rebuild     # embed → pgvector (lần sau bỏ --rebuild để incremental)
.\start-server.bat                        # uvicorn :8000 (API + serve frontend)
# Mở http://127.0.0.1:8000/ | Docs http://127.0.0.1:8000/docs | Health /api/health
```

Docker (server nội bộ):

```bash
cp .env.example .env   # sửa ADMIN_PASSWORD, POSTGRES_PASSWORD, LLM_MODEL
docker compose up -d --build
docker compose exec ollama ollama pull ${LLM_MODEL:-qwen2.5:1.5b}
docker compose exec app python -m backend.cli clean
docker compose exec app python -m backend.cli index --rebuild
```

## 5. API chính

| Method | Endpoint | Mô tả |
|---|---|---|
| POST | `/api/chat` | Hỏi-đáp hội thoại (blocking, fallback) |
| POST | `/api/chat/stream` | Streaming SSE (UI dùng mặc định) |
| GET | `/api/health` | `vector_count`, `processed_files`, `ollama_ok` |
| GET/DELETE | `/api/sessions`, `/api/sessions/{id}` | Quản lý phiên hội thoại |

Request `POST /api/chat`:
```json
{"question": "Học bổng KKHT xét thế nào?", "category": "thong_bao", "top_k": 4, "use_rerank": true, "session_id": "uuid", "history": []}
```
Response: `{answer, sources[{filename, page, category}], standalone_question, session_id, summary}`.

## 6. Cấu hình chính (`backend/config/settings.py` + `.env`)

`EMBED_MODEL`, `LLM_MODEL=qwen2.5:1.5b`, `OLLAMA_BASE`, `DATABASE_URL`, `RETRIEVE_TOP_K=5`, `CONVERSATION_WINDOW=8`, `CONVERSATION_SUMMARY_THRESHOLD=10`, `USE_AGENT=true`, `AGENT_MAX_STEPS=3`. Đổi model embedding phải `index --rebuild` lại.

## 7. Lưu ý

* 1 port = 1 server: lỗi `10048` thì chạy `.\stop-server.bat` trước.
* Đổi `LLM_MODEL` khi hết RAM: `qwen2.5:0.5b/1.5b`; lỗi Ollama `connection refused` thì kiểm tra `ollama serve`.
* Session in-memory (TTL 24h, tối đa 200 phiên); upload PDF/DOCX/TXT/MD theo phiên, không vào kho chung.
* Giấy phép: dự án nội bộ Học viện Kỹ thuật Mật mã.
