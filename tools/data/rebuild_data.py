"""Rebuild toan bo data: clean (classified -> processed) + index --rebuild (processed -> ChromaDB).

Chay tu repo root:
    python tools/data/rebuild_data.py
    python tools/data/rebuild_data.py --skip-clean
    python tools/data/rebuild_data.py --clean-only
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENV = dict(os.environ)
ENV["PYTHONUTF8"] = "1"
ENV["TOKENIZERS_PARALLELISM"] = "false"


def _run(mod: str, args: list[str]) -> int:
    cmd = [sys.executable, "-m", mod, *args]
    print(f"\n=== {' '.join(cmd)} ===", flush=True)
    return subprocess.call(cmd, cwd=str(ROOT), env=ENV)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Rebuild toan bo data RAG")
    p.add_argument("--skip-clean", action="store_true", help="Bo qua clean, chi index --rebuild")
    p.add_argument("--clean-only", action="store_true", help="Chi clean, khong index")
    p.add_argument("--stats-only", action="store_true", help="Chi in thong ke, khong xu ly")
    a = p.parse_args(argv)

    if a.stats_only:
        _run("backend.cli", ["clean", "--stats"])
        return 0

    if not a.skip_clean:
        rc = _run("backend.cli", ["clean"])
        if rc != 0:
            print(f"[rebuild] clean that bai (exit {rc})", flush=True)
            return rc
    if a.clean_only:
        _run("backend.cli", ["clean", "--stats"])
        return 0

    rc = _run("backend.cli", ["index", "--rebuild"])
    if rc != 0:
        print(f"[rebuild] index that bai (exit {rc})", flush=True)
        return rc

    print("\n=== verify ===", flush=True)
    _run("backend.preprocessing.storage", [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
