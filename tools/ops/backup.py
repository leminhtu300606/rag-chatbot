"""Backup/khoi phuc du lieu RAG (GĐ1/GĐ3 van hanh).

Bao gồm: pgvector dump (db/rag.sql) + data/processed/ (+ .indexed.json)
+ data/sessions.json + data/users.json.
KHÔNG backup: data/classified (nguồn gốc, giữ riêng), data/uploads (tái tạo được), venv.

Chạy từ repo root (hoặc trong container với BACKUP_SRC=/app, BACKUP_DST=/app/backups):
    python tools/ops/backup.py --backup [--keep 7]
    python tools/ops/backup.py --list
    python tools/ops/backup.py --restore backups/rag-20240101-120000.tar.gz

Khôi phục DB: psql "$DATABASE_URL" -f <thư-mục-bung>/db/rag.sql
"""
import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = Path(os.getenv("BACKUP_SRC", str(ROOT)))
DST = Path(os.getenv("BACKUP_DST", str(ROOT / "backups")))

INCLUDE = ["data/processed", "data/sessions.json", "data/users.json"]


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _dump_db(tmpdir: Path) -> Path | None:
    """pg_dump bảng vector ra db/rag.sql. Trả về path hoặc None nếu bỏ qua."""
    dsn = os.getenv("DATABASE_URL", "")
    if not dsn:
        print("[backup] bỏ qua dump DB (chưa có DATABASE_URL)")
        return None
    if shutil.which("pg_dump") is None:
        print("[backup] bỏ qua dump DB (không tìm thấy pg_dump)")
        return None
    out = tmpdir / "db" / "rag.sql"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Chỉ dump bảng vector + extension (nhẹ, đủ rebuild truy vấn)
    table = os.getenv("PGVECTOR_TABLE", "rag_chunks")
    r = subprocess.run(
        ["pg_dump", dsn, "--no-owner", "--no-privileges", "-t", table],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0 or not r.stdout.strip():
        print(f"[backup] dump DB thất bại, bỏ qua: {(r.stderr or '').strip()[:200]}")
        return None
    out.write_text(r.stdout, encoding="utf-8")
    print(f"[backup] dump DB: {out} ({out.stat().st_size / 1024:.0f} KB)")
    return out


def do_backup(keep: int = 7) -> int:
    DST.mkdir(parents=True, exist_ok=True)
    name = f"rag-{_stamp()}.tar.gz"
    out = DST / name
    print(f"[backup] {SRC} -> {out}")
    import tempfile
    with tempfile.TemporaryDirectory(prefix="rag-backup-") as tmp:
        db_dump = _dump_db(Path(tmp))
        with tarfile.open(out, "w:gz") as tar:
            for rel in INCLUDE:
                p = SRC / rel
                if not p.exists():
                    print(f"[backup] bỏ qua (không có): {rel}")
                    continue
                tar.add(p, arcname=rel)
            if db_dump is not None:
                tar.add(db_dump, arcname="db/rag.sql")
    print(f"[backup] xong: {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")
    # xoay vòng giữ N bản mới nhất
    olds = sorted(DST.glob("rag-*.tar.gz"))
    for o in olds[:-max(1, keep)]:
        o.unlink()
        print(f"[backup] xóa bản cũ: {o.name}")
    return 0


def do_list() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    for p in sorted(DST.glob("rag-*.tar.gz")):
        print(f"{p.name}  {p.stat().st_size / 1024 / 1024:.1f} MB")
    return 0


def do_restore(archive: str) -> int:
    p = Path(archive)
    if not p.is_absolute():
        p = DST / archive if (DST / archive).exists() else ROOT / archive
    if not p.exists():
        print(f"[restore] không thấy file: {archive}", file=sys.stderr)
        return 1
    print(f"[restore] giải nén {p} vào {SRC} (ghi đè file trùng tên)")
    with tarfile.open(p, "r:gz") as tar:
        # an toàn: chặn path traversal
        for m in tar.getmembers():
            mp = (SRC / m.name).resolve()
            if SRC.resolve() not in mp.parents and mp != SRC.resolve():
                print(f"[restore] bỏ qua path nguy hiểm: {m.name}", file=sys.stderr)
                continue
            tar.extract(m, path=SRC)
    print("[restore] xong. Nếu app đang chạy, restart để load lại sessions/users.")
    print("[restore] DB: nếu có db/rag.sql trong bản backup, nạp bằng: psql \"$DATABASE_URL\" -f <thư-mục-bung>/db/rag.sql")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Backup/khôi phục dữ liệu RAG")
    ap.add_argument("--backup", action="store_true", help="Tạo bản backup mới")
    ap.add_argument("--keep", type=int, default=7, help="Giữ N bản mới nhất (mặc định 7)")
    ap.add_argument("--list", action="store_true", help="Liệt kê các bản backup")
    ap.add_argument("--restore", metavar="FILE", help="Khôi phục từ file backup")
    a = ap.parse_args(argv)
    if a.restore:
        return do_restore(a.restore)
    if a.list:
        return do_list()
    return do_backup(keep=a.keep)


if __name__ == "__main__":
    raise SystemExit(main())
