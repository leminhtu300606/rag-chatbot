# RAG Chatbot - Học viện Kỹ thuật Mật mã

Chatbot hỏi đáp tiếng Việt trên kho tài liệu nội bộ như quy chế, quy định, quyết
định, thông báo và tài liệu hướng dẫn. Hệ thống sử dụng **Retrieval-Augmented
Generation (RAG)**: tìm các đoạn tài liệu liên quan, đưa chúng vào prompt, rồi
gọi LLM để tạo câu trả lời có trích dẫn nguồn.

> Đây là hệ thống hỗ trợ tra cứu, không thay thế văn bản pháp quy hoặc quyết
> định chính thức. Khi câu trả lời quan trọng, hãy mở nguồn tài liệu được trả
> về để kiểm tra.

## 1. Tính năng và cách hoạt động

- Đọc PDF, PDF scan (OCR), DOCX và TXT.
- Làm sạch header/footer, số trang và nội dung nhiễu; chia tài liệu thành các
  chunk theo cấu trúc/ngữ cảnh.
- Tạo embedding tiếng Việt và lưu trong PostgreSQL + pgvector.
- Truy hồi kết hợp vector search và BM25; có thể rerank bằng cross-encoder.
- Viết lại câu hỏi nối tiếp, giữ cửa sổ lịch sử và tóm tắt hội thoại dài.
- Sinh câu trả lời bằng Ollama (mặc định `qwen2.5:1.5b`).
- Hỗ trợ streaming SSE, phiên hội thoại, upload tài liệu theo phiên và giao diện
  React.
- Có giới hạn tốc độ, quota, cache, log hội thoại và tùy chọn ClamAV.

### Luồng RAG

```text
PDF/DOCX/TXT
    -> loader -> cleaner -> chunker
    -> data/processed/*.parquet
    -> embedding -> PostgreSQL/pgvector (rag_chunks)

Người dùng -> React/Vite -> FastAPI
    -> viết lại câu hỏi (nếu cần)
    -> hybrid retrieval (vector + BM25)
    -> rerank (tùy chọn)
    -> prompt + lịch sử -> Ollama
    -> answer + sources
```

Với câu hỏi có ngữ cảnh, hệ thống dùng lịch sử để tạo một câu hỏi độc lập cho
khâu truy hồi, nhưng vẫn dùng câu hỏi gốc và lịch sử phù hợp khi sinh câu trả
lời. Khi không dùng được vector store, benchmark offline có thể chạy nhánh
BM25 trên dữ liệu đã xử lý.

## 2. Kiến trúc và cấu trúc thư mục

```text
E:\rag\
├── data/
│   ├── classified/                 # PDF/DOCX/TXT đầu vào, phân theo category
│   ├── processed/                  # parquet và .indexed.json
│   └── uploads/                    # upload theo phiên, không commit
├── backend/
│   ├── api/app.py                  # FastAPI, chat, session, upload, auth
│   ├── cli/main.py                 # python -m backend.cli clean|index
│   ├── config/settings.py          # cấu hình runtime
│   ├── security/auth.py            # xác thực và phân quyền
│   ├── infra/cache.py              # cache câu hỏi
│   ├── preprocessing/              # loader, cleaner, chunker, storage
│   ├── indexing/                   # embedder, vectorstore, retrieval
│   └── generation/                 # RAG, agent, prompt, reranker, calculator
├── frontend/                       # React + TypeScript + Vite
├── tools/
│   ├── data/rebuild_data.py        # clean + rebuild index + verify
│   ├── eval/eval_rag.py            # benchmark 10 câu ground-truth
│   ├── eval/eval_quick.py          # smoke test qua HTTP API
│   └── ops/backup.py               # backup/restore dữ liệu vận hành
├── infra/nginx/nginx.conf          # reverse proxy/TLS tùy chọn
├── docker-compose.yml              # app, PostgreSQL/pgvector, Ollama, nginx
├── requirements.txt
├── .env.example
├── start-server.bat
└── stop-server.bat
```

| Thành phần | Công nghệ | Vai trò |
|---|---|---|
| API | Python, FastAPI | Chat, session, upload, health và auth |
| Tiền xử lý | PyMuPDF/OCR, DOCX, Parquet | Chuẩn hóa và chia nhỏ tài liệu |
| Truy hồi | PostgreSQL/pgvector, BM25 | Tìm context liên quan |
| Sinh câu trả lời | Ollama, LangChain/LangGraph | Prompt, agent, LLM và lịch sử |
| Giao diện | React 18, TypeScript, Vite | Chat, streaming, session, nguồn |
| Reverse proxy | Nginx | Truy cập nội bộ, giới hạn upload/rate và TLS |

## 3. Yêu cầu

### Chạy local

- Python 3.10 trở lên.
- Node.js 20 trở lên nếu phát triển frontend riêng.
- PostgreSQL 16 có extension pgvector.
- Ollama đang chạy và có model LLM.
- Đủ RAM/VRAM cho model đã chọn; `qwen2.5:1.5b` phù hợp máy CPU hạn chế
  hơn các model 7B.

### Chạy Docker Compose

Docker Compose cung cấp sẵn PostgreSQL/pgvector, Ollama và ứng dụng. Cổng
`8000`, `5432`, `11434` mặc định chỉ bind vào loopback; muốn truy cập qua mạng
 nội bộ thì bật Nginx profile `edge` và cấu hình TLS.

## 4. Cài đặt và chạy local

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Tạo `.env` từ [.env.example](.env.example), sau đó khởi động PostgreSQL và
Ollama:

```powershell
ollama pull qwen2.5:1.5b
ollama serve
```

Nếu dùng PostgreSQL local, đặt `DATABASE_URL` trong `.env` trỏ tới
`localhost:5432`. Nạp dữ liệu và tạo index:

```powershell
python -m backend.cli clean
python -m backend.cli index --rebuild
```

Chạy API (API cũng phục vụ frontend đã build):

```powershell
.\start-server.bat
# hoặc:
uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Mở `http://127.0.0.1:8000/`; Swagger ở
`http://127.0.0.1:8000/docs`. Dùng `.\stop-server.bat` để dừng server chạy
trên cổng 8000.

### Phát triển frontend riêng

```powershell
Set-Location frontend
npm install
npm run dev
```

Mở `http://localhost:5173`. Vite proxy các request `/api` tới backend
`http://127.0.0.1:8000`. Build production bằng `npm run build`; backend sẽ
serve thư mục `frontend/dist`.

## 5. Chạy bằng Docker Compose

```powershell
Copy-Item .env.example .env
# Sửa tối thiểu POSTGRES_PASSWORD và ADMIN_PASSWORD trong .env

docker compose up -d --build
docker compose exec ollama ollama pull qwen2.5:1.5b
docker compose exec app python -m backend.cli clean
docker compose exec app python -m backend.cli index --rebuild
```

Kiểm tra:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/api/health
```

Bật reverse proxy (sau khi đặt certificate vào `infra/nginx/certs/` và kiểm
tra `infra/nginx/nginx.conf`):

```powershell
docker compose --profile edge up -d
```

Dữ liệu PostgreSQL, Ollama và dữ liệu ứng dụng nằm trong các named volume
`pgdata`, `ollama_data`, `app_data`. Thư mục `data/classified` được mount
read-only vào container.

## 6. Tiền xử lý và indexing

```powershell
# Xem trợ giúp
python -m backend.cli --help

# Làm sạch toàn bộ data/classified
python -m backend.cli clean

# Chỉ xem thống kê, không thay đổi dữ liệu
python -m backend.cli clean --stats

# Chạy thử không ghi file
python -m backend.cli clean --dry-run

# Index tăng dần, bỏ qua file không đổi theo mtime
python -m backend.cli index

# Xóa collection cũ và embed lại toàn bộ
python -m backend.cli index --rebuild
```

Pipeline tạo các chunk trong `data/processed`. Lệnh `index` dùng manifest
`data/processed/.indexed.json` để chỉ embed file mới hoặc đã thay đổi. Sau khi
đổi model embedding hoặc thay đổi cách chunking, cần dùng `--rebuild`.

## 7. Cấu hình quan trọng

Các biến đầy đủ nằm trong [.env.example](.env.example). Một số biến thường dùng:

| Nhóm | Biến | Mặc định | Ý nghĩa |
|---|---|---:|---|
| LLM | `OLLAMA_BASE` | `http://ollama:11434` | Địa chỉ Ollama |
| LLM | `LLM_MODEL` | `qwen2.5:1.5b` | Model sinh câu trả lời |
| Runtime | `LLM_BACKEND` | `ollama` | Backend sinh |
| Runtime | `DEVICE` | `cpu` | Thiết bị chạy model |
| DB | `DATABASE_URL` | xem `.env` | Kết nối PostgreSQL |
| DB | `PGVECTOR_TABLE` | `rag_chunks` | Bảng vector |
| Auth | `AUTH_ENABLED` | `false` | Bật xác thực API |
| Limit | `RATE_LIMIT_MAX` | `30` | Số request trong window |
| Limit | `CHAT_DAILY_LIMIT` | `200` | Số lượt chat/user/ngày |
| Cache | `CACHE_ENABLED` | `true` | Bật cache câu hỏi |
| Cache | `CACHE_TTL_SECONDS` | `600` | TTL cache |
| Retrieval | `RETRIEVAL_FUSION` | `weighted` | Cách gộp vector+BM25: `weighted` hoặc `rrf` |
| Retrieval | `RETRIEVAL_RRF_K` | `60` | Hằng số RRF khi `RETRIEVAL_FUSION=rrf` |
| Rerank | `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Model rerank (đổi sang model tiếng Việt nếu có) |
| Rerank | `RERANK_HEURISTIC_BOOST` | `1` | Bật boost thủ công (đặt `0` khi dùng reranker tiếng Việt) |
| Concurrency | `OLLAMA_MAX_CONCURRENCY` | `1` | Số request Ollama song song (CPU nên để 1) |
| Concurrency | `TORCH_MAX_CONCURRENCY` | `1` | Số suy luận embedding/rerank song song |
| Upload | `MAX_UPLOAD_MB_PER_USER` | `500` | Hạn mức upload/user |
| Security | `CLAMAV_ENABLED` | `false` | Quét file qua ClamAV |

Không commit `.env`, mật khẩu, token hoặc certificate vào repository. Khi triển
khai thật, phải đổi các giá trị mẫu `POSTGRES_PASSWORD` và `ADMIN_PASSWORD`.

## 8. API chính

| Method | Endpoint | Mục đích |
|---|---|---|
| GET | `/api/health` | Trạng thái app, số file và vector (readiness) |
| GET | `/api/healthz` | Liveness nhanh cho HEALTHCHECK container |
| GET | `/api/stats` | Thống kê index và session |
| GET | `/api/categories` | Danh sách category |
| POST | `/api/chat` | Hỏi đáp blocking |
| POST | `/api/chat/stream` | Hỏi đáp streaming SSE |
| GET/POST | `/api/sessions` | Liệt kê/tạo phiên |
| GET/PATCH/DELETE | `/api/sessions/{id}` | Đọc/sửa/xóa phiên |
| GET/POST/DELETE | `/api/sessions/{id}/upload...` | Quản lý file theo phiên |
| POST | `/api/auth/login` | Đăng nhập khi bật auth |
| GET | `/docs` | Swagger UI |

Ví dụ hỏi đáp:

```powershell
$body = @{
  question = "Sinh viên khi đến trường phải mặc trang phục thế nào?"
  top_k = 5
  use_rerank = $true
  show_context = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/chat `
  -Method Post -ContentType "application/json" -Body $body
```

`/api/chat` trả về `answer`, `sources`, `session_id`,
`standalone_question` và context nếu yêu cầu. Khi dùng UI, streaming được ưu
tiên và client có fallback về endpoint blocking.

## 9. Đánh giá và số liệu hiện tại

### 9.1 Benchmark RAG 10 câu

Chạy từ thư mục gốc:

```powershell
python tools/eval/eval_rag.py --top-k 5                 # BM25 + sinh câu trả lời
python tools/eval/eval_rag.py --no-gen                  # chỉ đo retrieval (không cần Ollama/DB)
python tools/eval/eval_rag.py --retrieval hybrid        # dùng pgvector + BM25 (cần DB có index)
python tools/eval/eval_rag.py --rerank 1                # đo trước/sau rerank
```

Bộ ground-truth nằm ở `tools/eval/golden.json`; kết quả đầy đủ được ghi vào
`tools/eval/last_eval.json`. Ngoài Hit/MRR/Precision/Recall, benchmark còn đo
**nDCG@k**, **độ trễ từng tầng** (retrieval/generation/rerank) và **tỉ lệ từ
chối** trên các câu `"unanswerable": true`. Lần đo gần nhất dùng `qwen2.5:1.5b`,
`n=10` câu trả lời được (+2 câu không đáp án), `top_k=5`:

| Tầng | Chỉ số | Kết quả |
|---|---|---:|
| Retrieval | Hit@5 | **1.00** |
| Retrieval | MRR@5 | **1.00** |
| Retrieval | Context precision@5 | **0.50** |
| Retrieval | Context recall | **1.00** |
| Retrieval | nDCG@5 (đo `--no-gen`) | **0.97** |
| Generation | Lexical F1 | **0.5459** |
| Generation | Keyword coverage | **0.7250** |
| Generation | Faithfulness heuristic | **1.00** |


Diễn giải: cả 10 câu đều tìm được tài liệu liên quan và tài liệu vàng luôn ở
hạng đầu. Precision 0.50 nghĩa là trung bình khoảng một nửa context top-5
được chấm là relevant theo bộ golden; đây không phải tỷ lệ câu trả lời đúng.
F1 0.5459 và keyword coverage 0.725 phản ánh chất lượng diễn đạt của model,
không phủ định kết quả retrieval. Faithfulness 1.00 là heuristic dựa trên
độ trùng token với context, không phải đánh giá độc lập của con người.

`--judge 1` bật LLM-as-judge, nhưng model 1.5B thường không trả đúng format
chấm điểm; vì vậy kết quả hiện tại để `null`. Muốn dùng chỉ số này cần model
lớn hơn và kiểm tra lại bằng đánh giá thủ công.

### 9.2 Thống kê dữ liệu đã xử lý

Lần thống kê gần nhất có **11 file parquet và 503 chunks**, gồm:

- `chunk_mode`: contextual 149, table_struct 9, figure 1.
- Category: `thong_bao` 16, `tai_lieu_huong_dan` 23, `quy_dinh` 60,
  `quy_che` 40, `quyet_dinh` 20.

### 9.3 Smoke test HTTP

```powershell
python tools/eval/eval_quick.py --base http://127.0.0.1:8000 --top-k 5
```

Smoke test đo latency, nguồn trả về và keyword qua API. Không được dùng kết quả
smoke test khi PostgreSQL chưa có index để kết luận chất lượng RAG: nếu
`vector_count=0`, API có thể chờ DB rồi trả fallback không có nguồn. Trước khi
đo cần đảm bảo `/api/health` ở trạng thái sẵn sàng và `vector_count > 0`.

### 9.4 Kiểm thử hồi quy

Bộ test hồi quy cho các bản sửa bảo mật (RBAC, fail-closed, RRF, concurrency)
chạy không cần DB/Ollama:

```powershell
python -m unittest discover -s tests -v
```

## 10. Giới hạn và hướng cải thiện

- Benchmark hiện tại chủ yếu đo BM25 offline; nhánh hybrid vector + pgvector
  cần được đo lại trên môi trường có database và index đầy đủ.
- Bộ golden mới có 10 câu trả lời được + 2 câu không đáp án, chưa đại diện toàn
  bộ tài liệu và chưa bao phủ đầy đủ câu hỏi ngoài miền.
- Model 1.5B có thể bỏ sót chi tiết, đặc biệt câu hỏi nhiều điều kiện; nên
  thử model lớn hơn, prompt kiểm chứng và đánh giá thủ công.
- Reranker `ms-marco-MiniLM-L-6-v2` là model tiếng Anh, có thể kém phù hợp
  tiếng Việt. Đổi qua `RERANK_MODEL`; cần benchmark (`--rerank 1`) trước khi bật
  mặc định và đặt `RERANK_HEURISTIC_BOOST=0` khi dùng reranker tiếng Việt.
- Fusion `weighted` là mặc định; đo `--retrieval hybrid` với `RETRIEVAL_FUSION=rrf`
  trên golden set trước khi đổi mặc định.
- Các chỉ số lexical không đo đầy đủ tính đúng pháp lý, tính mới của văn bản
  hoặc mức độ hữu ích của câu trả lời.

## 11. Vận hành và sao lưu

- `/api/health` là readiness (kèm `vector_count`, `ollama_ok`); `/api/healthz`
  là liveness dùng cho `HEALTHCHECK` container. Cả app, db và ollama đều có
  healthcheck trong `docker-compose.yml`.
- Theo dõi `/api/health`, `/api/stats`, log ứng dụng và dung lượng các volume.
- Sau khi thêm/sửa tài liệu: chạy `clean`, kiểm tra thống kê, rồi chạy
  `index` (hoặc `--rebuild` khi thay đổi embedding/chunking). `index` tự dọn
  chunk của file đã gỡ và chỉ lưu manifest sau khi ghi vector thành công.
- Giới hạn tải: `OLLAMA_MAX_CONCURRENCY` và `TORCH_MAX_CONCURRENCY` (mặc định 1)
  tránh nhiều request cùng lúc làm quá tải CPU/RAM trên một worker.
- Dùng `tools/ops/backup.py` để backup/restore theo hướng dẫn của script.
- Chỉ mở API ra ngoài qua reverse proxy có TLS; giữ PostgreSQL và Ollama ở
  loopback hoặc network nội bộ.
- Bật `AUTH_ENABLED=true`, đổi mật khẩu mẫu, giới hạn rate/quota và cân nhắc
  `CLAMAV_ENABLED=true` trước khi cho phép upload từ người dùng không tin cậy.
