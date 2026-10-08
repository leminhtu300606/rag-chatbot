# Triển khai production (server nội bộ)
# Chi tiết TLS/nginx: infra/nginx/nginx.conf

## 1. Chuẩn bị server
- Docker Engine + Compose plugin, ~16GB RAM (CPU-only), 50GB đĩa trống.
- Mở port 80/443 nội bộ. Ollama chạy trong compose (không cần cài riêng).

## 2. Cấu hình
```bash
cp .env.example .env
# Sửa bắt buộc: ADMIN_PASSWORD, LLM_MODEL (phải khớp model sẽ pull)
nano .env
```

## 3. Build & chạy
```bash
docker compose build
docker compose up -d
docker compose exec ollama ollama pull ${LLM_MODEL:-qwen2.5:1.5b}
# Nạp dữ liệu lần đầu (mount data/classified read-only từ host):
docker compose exec app python -m backend.cli clean
docker compose exec app python -m backend.cli index --rebuild
curl http://localhost:8000/api/health
```

## 4. Bật đăng nhập (khuyến nghị trước khi mở cho user)
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

## 5. TLS nội bộ (nginx profile `edge`)
```bash
# Tự ký cho tên nội bộ (ví dụ rag.noibo.local), CA nội bộ ký sẽ chuẩn hơn:
openssl req -x509 -newkey rsa:2048 -keyout infra/nginx/certs/privkey.pem \
  -out infra/nginx/certs/fullchain.pem -days 825 -nodes -subj "/CN=rag.noibo.local"
# Mở block 443 trong infra/nginx/nginx.conf rồi:
docker compose --profile edge up -d
```

## 6. Backup / restore (chạy trên host, cron hàng đêm)
```bash
python tools/ops/backup.py --backup --keep 7     # BACKUP_SRC/BACKUP_DST trỏ vào volume khi cần
python tools/ops/backup.py --list
python tools/ops/backup.py --restore backups/rag-YYYYMMDD-HHMMSS.tar.gz
```

## 7. Quy trình cập nhật tài liệu (ai cũng làm được)
1. Người phụ trách bỏ PDF/DOCX đã duyệt vào `data/classified/<category>/...`.
2. Chạy: `docker compose exec app python -m backend.cli clean` rồi `... index` (incremental, không `--rebuild`).
3. Kiểm tra: `/api/health` (số files/vectors tăng), hỏi thử 2–3 câu đặc trưng.
4. Ghi log thay đổi (ngày, ai, file nào) vào sổ/quy định đơn vị.

## 8. Giám sát nhanh
- `GET /api/health`: `status`, `ollama_ok`, `disk_free_gb`, `auth_enabled`.
- `GET /api/metrics` (admin): count, latency p50/p95, no-source rate, cache hit.
- `python tools/eval/eval_quick.py --base http://localhost:8000` sau mỗi lần đổi model/tham số.

## 9. ClamAV (tùy chọn)
Chạy clamav-daemon đâu đó nội bộ rồi đặt trong `.env`:
`CLAMAV_ENABLED=true`, `CLAMAV_HOST`, `CLAMAV_PORT`, `CLAMAV_FAIL_CLOSED=false`.
