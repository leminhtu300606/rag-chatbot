"""
preprocessing/clean_service.py - Logic lam sach cho backend/clean.py
================================================================
Chua toan bo logic xu ly: clean_single_file, clean_all, show_stats, v.v.
backend/clean.py chi goi den file nay (khong xu ly logic truc tiep).
"""
import argparse
import json
import os
import sys
import warnings
import logging
# An warning CPU/GPU/HF, chi hien thi tien trinh
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from pathlib import Path
from collections import Counter

import backend.config as cfg
from backend.preprocessing.loader import load_file
from backend.preprocessing.cleaner import clean_text
from backend.preprocessing.chunker import contextual_chunk, semantic_chunk, _effective_min_chars
# Import embed sau khi da tat warning
from backend.indexing.embedder import embed


def _show_clean_sample(raw_text: str, cleaned: str, preview: int = 600):
    """Hien thi truoc/sau khi lam sach de kiem tra."""
    print("\n--- RAW (truoc clean) preview ---")
    print(raw_text[:preview] + ("..." if len(raw_text) > preview else ""))
    print(f"  [raw chars: {len(raw_text)} | lines: {len(raw_text.splitlines())}]")
    print("\n--- CLEANED (sau clean) preview ---")
    print(cleaned[:preview] + ("..." if len(cleaned) > preview else ""))
    print(f"  [cleaned chars: {len(cleaned)} | lines: {len(cleaned.splitlines())}]")
    print(f"  -> Giam {len(raw_text)-len(cleaned)} chars ({(len(raw_text)-len(cleaned))/max(len(raw_text),1)*100:.1f}%)")
    print("-" * 60)


def clean_single_file(path: Path, dry_run: bool = False, show_sample: bool = False, preview: int = 600) -> int:
    """Lam sach 1 file tho -> data sach. Tra ve so chunks neu ghi, 0 neu dry_run."""
    path = Path(path)
    if not path.exists():
        print(f"[clean] Khong ton tai: {path}", file=sys.stderr)
        return 0

    records = load_file(path)
    if not records:
        print(f"[clean] Khong doc duoc gi tu {path.name} (co the file rong/khong ho tro)")
        return 0

    try:
        rel = path.resolve().relative_to(cfg.DATA_DIR.resolve())
        category = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        subcategory = "/".join(rel.parts[1:-1]) if len(rel.parts) > 2 else ""
    except ValueError:
        category, subcategory = "unknown", ""
    filename = path.name

    use_contextual = cfg.CHUNK.get("contextual", True)
    chunk_fn = contextual_chunk if use_contextual else semantic_chunk

    cleaned_chunks = []

    for rec in records:
        raw = rec.get("text", "")
        cleaned = clean_text(raw)
        if show_sample and raw.strip():
            print(f"\n[File: {path.name} | page {rec.get('page','?')} | kind={rec.get('kind','')}]")
            _show_clean_sample(raw, cleaned, preview=preview)

        if not cleaned:
            continue

        meta_for_chunk = {
            "filename": filename,
            "category": rec.get("category", category),
            "subcategory": rec.get("subcategory", subcategory),
            "source": str(path),
            "page": rec.get("page", 0),
        }
        if use_contextual:
            pieces = chunk_fn(cleaned, embed, metadata=meta_for_chunk)
        else:
            pieces = chunk_fn(cleaned, embed)

        for i, piece in enumerate(pieces):
            # Đồng bộ với ngưỡng hiệu dụng của chunker (max(min_chars, 20), kẹp theo max_chars)
            _eff_min = _effective_min_chars(None)
            if len(piece.strip()) < _eff_min:
                continue
            cleaned_chunks.append({
                "id": f"{filename}-{rec.get('page',0)}-{i}",
                "text": piece,
                "category": meta_for_chunk["category"],
                "subcategory": meta_for_chunk["subcategory"],
                "filename": filename,
                "source": str(path),
                "page": meta_for_chunk["page"],
                "chunk_index": i,
                "chunk_mode": "contextual" if use_contextual else "semantic",
            })

    if dry_run:
        print(f"[dry-run] {path.name}: {len(records)} pages -> {len(cleaned_chunks)} chunks (khong ghi file)")
        return 0

    ext = getattr(cfg, "PROCESSED_EXT", ".parquet")
    try:
        rel = path.resolve().relative_to(cfg.DATA_DIR.resolve())
        out = cfg.PROCESSED_DIR / rel.with_suffix(ext)
    except ValueError:
        out = cfg.PROCESSED_DIR / "_external" / path.with_suffix(ext).name

    from backend.preprocessing.storage import save_chunks
    save_chunks(out, cleaned_chunks)

    try:
        rel_out = out.relative_to(cfg.BASE_DIR)
    except ValueError:
        rel_out = out
    print(f"[clean] {path.name}: {len(records)} pages -> {len(cleaned_chunks)} chunks -> {rel_out} (mode={ 'contextual' if use_contextual else 'semantic'})")
    return len(cleaned_chunks)


def clean_all(data_dir: Path = None, dry_run: bool = False, show_sample: bool = False, preview: int = 600) -> int:
    """Lam sach toan bo data/classified -> data/processed."""
    if data_dir is None:
        data_dir = cfg.DATA_DIR
    else:
        data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"[clean] DATA_DIR khong ton tai: {data_dir}", file=sys.stderr)
        return 0

    print(f"[clean] Quet {data_dir} -> {cfg.PROCESSED_DIR} (dry_run={dry_run})")
    print(f"  CHUNK config: {cfg.CHUNK}")
    total = 0
    file_count = 0
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if fn.startswith("~$"):
                continue
            p = Path(root) / fn
            if p.suffix.lower() not in (".pdf", ".docx", ".txt"):
                continue
            try:
                do_show = show_sample and file_count == 0
                n = clean_single_file(p, dry_run=dry_run, show_sample=do_show, preview=preview)
                total += n
                file_count += 1
            except Exception as e:
                print(f"[clean] Loi {p}: {e}")
                import traceback
                traceback.print_exc()
    print(f"\n[clean] Hoan tat: {file_count} files -> {total} chunks {'(dry-run, chua ghi)' if dry_run else ''}")
    return total


def show_stats():
    """Hien thi thong ke data sach hien co trong data/processed (parquet)."""
    from backend.preprocessing.storage import get_processed_files, load_chunks
    print("=" * 70)
    print("THONG KE DATA SACH - data/processed/")
    print("=" * 70)
    print(f"PROCESSED_DIR: {cfg.PROCESSED_DIR} (exists={cfg.PROCESSED_DIR.exists()})  ext={getattr(cfg,'PROCESSED_EXT','.parquet')}")
    if not cfg.PROCESSED_DIR.exists():
        print("  Chua co data sach. Hay chay: python backend/clean.py")
        return

    files = get_processed_files()
    print(f"Tong file {getattr(cfg,'PROCESSED_EXT','.parquet')}: {len(files)}")
    if not files:
        print("  Chua co chunk nao.")
        return

    total = 0
    mode_cnt = Counter()
    cat_cnt = Counter()
    for jf in files:
        try:
            chunks = load_chunks(jf)
            total += len(chunks)
            for ch in chunks[:20]:
                try:
                    mode_cnt[ch.get("chunk_mode","unknown")] += 1
                    cat_cnt[ch.get("category","unknown")] += 1
                except Exception:
                    pass
        except Exception as e:
            print(f"  Loi doc {jf}: {e}")

    print(f"Tong chunks: {total}")
    print(f"Phan bo chunk_mode: {dict(mode_cnt)}")
    print(f"Phan bo category: {dict(cat_cnt)}")

    print("\n--- Mau data sach (2 chunks dau) ---")
    shown = 0
    for jf in files[:2]:
        try:
            chunks = load_chunks(jf)
            if not chunks:
                continue
            ch = chunks[0]
            print(f"  File: {jf.relative_to(cfg.PROCESSED_DIR)} | id={ch.get('id')} | {ch.get('category')}/{ch.get('subcategory')} p{ch.get('page')} idx{ch.get('chunk_index')} mode={ch.get('chunk_mode')}")
            print(f"    {ch.get('text','')[:250]}...")
            shown += 1
            if shown >= 2:
                break
        except Exception:
            pass
    print()


def build_parser():
    p = argparse.ArgumentParser(
        prog="backend.clean",
        description="Lam sach data tho -> data sach (preprocessing: load -> clean -> chunk -> luu parquet)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Vi du:\n"
            "  python backend/clean.py                          # lam sach toan bo data/classified\n"
            "  python backend/clean.py --show-sample --preview 800\n"
            "  python backend/clean.py data/classified/quy_che --dry-run\n"
            "  python backend/clean.py data/classified/quy_che/dao_tao/file.pdf --show-sample\n"
            "  python backend/clean.py --stats                   # chi xem thong ke data sach\n"
            "  python -m backend.clean --dry-run\n"
        ),
    )
    p.add_argument("path", nargs="?", default=None, help="Duong dan file/thu muc cu the (bo trong = toan bo DATA_DIR)")
    p.add_argument("--data-dir", dest="data_dir", default=None, help="Ghi de DATA_DIR (mac dinh backend/config.py)")
    p.add_argument("--dry-run", action="store_true", help="Chi hien thi, khong ghi file ra data/processed")
    p.add_argument("--show-sample", action="store_true", help="Hien thi truoc/sau khi clean (raw vs cleaned) cho file dau tien")
    p.add_argument("--preview", type=int, default=600, help="So ky tu preview khi show-sample (mac dinh 600)")
    p.add_argument("--stats", action="store_true", help="Chi hien thi thong ke data sach hien co, khong lam sach")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.stats:
        show_stats()
        return 0

    if args.path:
        p = Path(args.path)
        if p.is_file():
            clean_single_file(p, dry_run=args.dry_run, show_sample=args.show_sample or True, preview=args.preview)
            if not args.dry_run:
                show_stats()
            return 0
        elif p.is_dir():
            clean_all(p, dry_run=args.dry_run, show_sample=args.show_sample, preview=args.preview)
            if not args.dry_run:
                show_stats()
            return 0
        else:
            alt = cfg.DATA_DIR / args.path
            if alt.exists():
                if alt.is_file():
                    clean_single_file(alt, dry_run=args.dry_run, show_sample=True, preview=args.preview)
                else:
                    clean_all(alt, dry_run=args.dry_run, show_sample=args.show_sample, preview=args.preview)
                return 0
            print(f"[clean] Khong tim thay: {args.path}", file=sys.stderr)
            return 1

    data_dir = Path(args.data_dir) if args.data_dir else None
    clean_all(data_dir, dry_run=args.dry_run, show_sample=args.show_sample, preview=args.preview)
    if not args.dry_run and not args.show_sample:
        show_stats()
    return 0

