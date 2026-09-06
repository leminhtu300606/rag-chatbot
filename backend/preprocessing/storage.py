"""
backend/preprocessing/storage.py - Quản lý lưu trữ chunk
========================================================
Nhiệm vụ:
- Lưu danh sách chunk ra file parquet và đọc lại.
- Hỗ trợ đọc định dạng jsonl để tương thích ngược.
- Cung cấp helper liệt kê file, đếm chunk và migrate jsonl sang parquet.
"""
import json
from pathlib import Path
from typing import List, Dict, Generator

import backend.config as cfg

# Try import parquet libs
try:
    import pandas as pd
    import pyarrow  # noqa: F401
    HAS_PARQUET = True
except Exception:
    HAS_PARQUET = False
    pd = None


def _get_ext() -> str:
    return getattr(cfg, "PROCESSED_EXT", ".parquet")


def get_processed_files() -> List[Path]:
    """Lay tat ca file chunk da xu ly (parquet chinh, jsonl fallback cho file chua migrate)."""
    ext = _get_ext()
    # Lay ca parquet va jsonl, uu tien parquet neu cung base name
    parquet_files = list(cfg.PROCESSED_DIR.rglob("*.parquet")) if ext == ".parquet" else []
    jsonl_files = [p for p in cfg.PROCESSED_DIR.rglob("*.jsonl") if p.suffix == ".jsonl" and p.name != ".indexed.json"]
    if ext == ".parquet":
        # Neu co parquet, van giu jsonl chua co parquet tuong ung (ho tro migrate tung phan)
        parquet_stems = {p.with_suffix("").as_posix() for p in parquet_files}
        remaining_jsonl = [p for p in jsonl_files if p.with_suffix("").as_posix() not in parquet_stems]
        return parquet_files + remaining_jsonl
    else:
        return list(cfg.PROCESSED_DIR.rglob(f"*{ext}"))


def save_chunks(path: Path, chunks: List[Dict]) -> None:
    """Luu list chunk ra parquet (hoac jsonl neu thieu lib)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    # Neu path la .parquet va co pyarrow
    if ext == ".parquet" and HAS_PARQUET:
        if not chunks:
            # Tao file rong voi schema co san de tranh loi
            df = pd.DataFrame(columns=["id","text","category","subcategory","filename","source","page","chunk_index","chunk_mode"])
            df.to_parquet(path, index=False)
            return
        df = pd.DataFrame(chunks)
        # Dam bao thu tu cot on dinh
        cols = ["id","text","category","subcategory","filename","source","page","chunk_index","chunk_mode"]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        df = df[cols]
        df.to_parquet(path, index=False)
        # Xoa file jsonl cu neu co (migrate)
        try:
            old_jl = path.with_suffix(".jsonl")
            if old_jl.exists():
                old_jl.unlink()
        except Exception:
            pass
    else:
        # Fallback jsonl
        with path.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> List[Dict]:
    """Doc 1 file parquet/jsonl tra ve list chunk dict."""
    path = Path(path)
    if not path.exists():
        # Thu tim file .jsonl tuong ung neu parquet chua co
        alt = path.with_suffix(".jsonl")
        if alt.exists():
            path = alt
        else:
            return []
    ext = path.suffix.lower()
    if ext == ".parquet":
        if HAS_PARQUET:
            try:
                df = pd.read_parquet(path)
                # Chuyen NaN -> None / int
                df = df.where(pd.notnull(df), None)
                return df.to_dict(orient="records")
            except Exception:
                return []
        else:
            return []
    elif ext == ".jsonl":
        try:
            lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            return [json.loads(l) for l in lines]
        except Exception:
            return []
    return []


def iter_chunks() -> Generator[Dict, None, None]:
    """Iterate qua tat ca chunk trong PROCESSED_DIR (parquet + fallback jsonl)."""
    for p in get_processed_files():
        for ch in load_chunks(p):
            yield ch


def count_chunks() -> int:
    """Dem tong so chunk."""
    total = 0
    for p in get_processed_files():
        # Nhanh hon neu dung parquet metadata
        if p.suffix == ".parquet" and HAS_PARQUET:
            try:
                import pyarrow.parquet as pq
                total += pq.read_table(p).num_rows
                continue
            except Exception:
                pass
        total += len(load_chunks(p))
    return total


def migrate_jsonl_to_parquet(delete_old: bool = True) -> int:
    """Chuyen tat ca file jsonl con lai sang parquet."""
    if not HAS_PARQUET:
        print("[migrate] Thieu pyarrow/pandas, khong the migrate")
        return 0
    jsonl_files = [p for p in cfg.PROCESSED_DIR.rglob("*.jsonl") if p.suffix == ".jsonl" and p.name != ".indexed.json"]
    if not jsonl_files:
        print("[migrate] Khong co file jsonl can migrate")
        return 0
    print(f"[migrate] Tim thay {len(jsonl_files)} file jsonl -> parquet")
    migrated = 0
    for jl in jsonl_files:
        pq_path = jl.with_suffix(".parquet")
        if pq_path.exists():
            print(f"  Skip {jl.relative_to(cfg.PROCESSED_DIR)} (da co parquet)")
            continue
        chunks = load_chunks(jl)
        if not chunks:
            print(f"  Skip {jl} (rong)")
            continue
        save_chunks(pq_path, chunks)
        print(f"  Migrated {jl.relative_to(cfg.PROCESSED_DIR)} -> {pq_path.relative_to(cfg.PROCESSED_DIR)} ({len(chunks)} chunks)")
        migrated += 1
        if delete_old and pq_path.exists():
            try:
                jl.unlink()
            except Exception:
                pass
    print(f"[migrate] Hoan tat: {migrated} files")
    return migrated


if __name__ == "__main__":
    import sys
    if "--migrate" in sys.argv or "migrate" in sys.argv:
        migrate_jsonl_to_parquet()
    else:
        print(f"Storage: ext={_get_ext()} HAS_PARQUET={HAS_PARQUET}")
        print(f"Processed files: {len(get_processed_files())} files, {count_chunks()} chunks")
        for p in get_processed_files()[:3]:
            print(f"  - {p.relative_to(cfg.PROCESSED_DIR) if p.is_relative_to(cfg.PROCESSED_DIR) else p} ({len(load_chunks(p))} chunks)")

