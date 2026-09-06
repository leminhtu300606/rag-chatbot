"""
generation/test_service.py - Logic test RAG cho backend/test.py
=============================================================
Chua toan bo logic: show_chunking, show_embeddings, do_ask, main.
backend/test.py chi goi den file nay (khong xu ly logic truc tiep) de dam bao
test chi goi den thu muc generation (dung yeu cau).
"""
import argparse
import json
from collections import Counter

# Top 3 mac dinh do reranker quyet dinh
from backend.generation.reranker import DEFAULT_RERANK_TOP_K

# Cau hinh mac dinh - sua truc tiep tai day (test goi generation)
DEFAULT_QUESTION = "Quy che dao tao la gi?"
DEFAULT_CATEGORY = None
DEFAULT_TOP_K = DEFAULT_RERANK_TOP_K
DEFAULT_USE_RERANK = True
DEFAULT_SHOW_CONTEXT = False
DEFAULT_SHOW_RAW = False


def show_chunking(limit=3, preview=300):
    import backend.config as cfg
    from backend.preprocessing.storage import get_processed_files, load_chunks
    ext = getattr(cfg, "PROCESSED_EXT", ".parquet")
    print("="*70 + f"\nKET QUA CHUNKING - data/processed/ (*{ext})\n" + "="*70)
    print(f"PROCESSED_DIR: {cfg.PROCESSED_DIR} (exists={cfg.PROCESSED_DIR.exists()})")
    print(f"CHUNK: {cfg.CHUNK}\n")
    if not cfg.PROCESSED_DIR.exists():
        print("  Chua co processed. Chay: python backend/main.py preprocess"); return
    files = get_processed_files()
    print(f"Tong file {ext}: {len(files)}")
    if not files:
        print("  Chua co chunk nao."); return
    total, modes, cats, samples = 0, Counter(), Counter(), []
    for jf in files:
        try:
            chunks = load_chunks(jf)
            total += len(chunks)
            for ch in chunks:
                try:
                    modes[ch.get("chunk_mode","?")] += 1
                    cats[ch.get("category","?")] += 1
                    if len(samples) < limit:
                        samples.append((jf, ch))
                except Exception: continue
        except Exception as e: print(f"  Loi doc {jf}: {e}")
    print(f"Tong chunks: {total}\n  chunk_mode: {dict(modes)}\n  category: {dict(cats)}")
    mf = cfg.INDEXED_MANIFEST
    if mf.exists():
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
            print(f"Manifest: {len(m)} nguon")
            for k,v in list(m.items())[:3]: print(f"  - {k} -> {v}")
        except Exception as e: print(f"  Loi manifest: {e}")
    else: print(f"Manifest chua co: {mf}")
    print(f"\n--- {limit} chunk mau (preview {preview}) ---")
    for i,(jf,ch) in enumerate(samples,1):
        rel = jf.relative_to(cfg.PROCESSED_DIR) if jf.is_relative_to(cfg.PROCESSED_DIR) else jf.name
        print(f"\n[{i}] {rel} | id={ch.get('id','')} | {ch.get('category','')}/p{ch.get('page','')} idx={ch.get('chunk_index','')} mode={ch.get('chunk_mode','')}")
        t = ch.get("text","")
        print(f"    {t[:preview]}{'...' if len(t)>preview else ''}")
    print()


def show_embeddings(limit=3, preview=200):
    import backend.config as cfg
    print("="*70 + "\nKET QUA EMBEDDING - ChromaDB (chroma_db/)\n" + "="*70)
    print(f"CHROMA_DIR: {cfg.CHROMA_DIR} (exists={cfg.CHROMA_DIR.exists()})")
    print(f"COLLECTION: {cfg.COLLECTION_NAME} | EMBED: {cfg.EMBED_MODEL} | DEVICE: {cfg.DEVICE}\n")
    try:
        from backend.indexing.vectorstore import count, get_stats, peek
        c = count()
        print(f"Tong vectors: {c}")
        if c == 0:
            print("  Chua co embedding. Chay: python -m backend.indexing.build_index --rebuild"); return
        try: print(f"Stats: {get_stats()}")
        except Exception: pass
        data = peek(limit=limit)
        ids, docs, metas, embs = data.get("ids",[]), data.get("documents",[]), data.get("metadatas",[]), data.get("embeddings",[])
        print(f"\n--- {min(limit,c)} mau ---")
        for i in range(min(limit,len(ids))):
            emb = embs[i] if embs is not None and i < len(embs) else None
            dim = len(emb) if emb is not None else "?"
            if emb is not None:
                try:
                    pv = ", ".join(f"{float(x):.4f}" for x in emb[:5]) + (" ..." if len(emb) > 5 else "")
                except Exception:
                    pv = str(emb[:5])
            else:
                pv = ""
            print(f"\n[{i+1}] id={ids[i]}\n    meta: {metas[i] if i<len(metas) else {}}\n    emb: dim={dim} [{pv}]\n    text: {docs[i][:preview]}{'...' if len(docs[i])>preview else ''}" if i < len(docs) else f"\n[{i+1}] id={ids[i]}")
    except Exception as e:
        print(f"Loi ChromaDB: {e}")
        import traceback; traceback.print_exc()
    print()


def do_ask(question, category=DEFAULT_CATEGORY, top_k=None, use_rerank=DEFAULT_USE_RERANK, show_context=DEFAULT_SHOW_CONTEXT, show_raw=DEFAULT_SHOW_RAW):
    """Hoi 1 cau - mac dinh doc toan bo data embedding, top 3 do reranker.py xu ly"""
    from backend.generation.rag import answer
    if not question or not question.strip():
        print("Cau hoi rong."); return 1
    eff_top_k = top_k
    eff_rerank = use_rerank if use_rerank is not None else DEFAULT_USE_RERANK
    eff_cat = category
    disp_k = eff_top_k if eff_top_k is not None else DEFAULT_RERANK_TOP_K
    print("="*70 + f"\nHOI: {question}\n  top_k={disp_k} (do reranker.py xu ly, mac dinh {DEFAULT_RERANK_TOP_K}) | category={eff_cat or 'ALL (toan bo data)'} | rerank={eff_rerank}\n" + "-"*70)
    try:
        res = answer(question, category=eff_cat, top_k=eff_top_k, use_rerank=eff_rerank)
        print("\n--- TRA LOI ---\n" + res["answer"])
        print("\n--- NGUON ---")
        srcs = res.get("sources",[])
        if srcs:
            for s in sorted({f"{x.get('filename','')} (p{x.get('page','')}, {x.get('category','')})" for x in srcs if x.get("filename")}): print(f"  - {s}")
        else: print("  (khong co nguon)")
        if show_context:
            print("\n--- CONTEXT (3 chunk tot nhat sau rerank) ---")
            for i,c in enumerate(res.get("context",[]),1):
                m, t = c.get("metadata",{}), c.get("text","")
                print(f"\n[{i}] score={c.get('score','?')} | {m.get('filename','')} p{m.get('page','')} {m.get('category','')}/{m.get('subcategory','')} idx={m.get('chunk_index','')} mode={m.get('chunk_mode','')}\n    {t[:500]}{'...' if len(t)>500 else ''}")
        if show_raw:
            print("\n--- RAW ---")
            print(json.dumps({"answer": res["answer"], "sources": res["sources"]}, ensure_ascii=False, indent=2))
        return 0
    except Exception as e:
        print(f"\n[LOI] {e}")
        import traceback; traceback.print_exc()
        if "dimension" in str(e).lower():
            print("\nGoi y: dimension khong khop -> chay: python -m backend.indexing.build_index --rebuild")
        return 1


def build_parser():
    p = argparse.ArgumentParser(prog="backend.test", description=f"Test RAG - chunking & embedding & hoi thu (mac dinh rerank lay {DEFAULT_RERANK_TOP_K}, doc toan bo data - top_k xu ly trong reranker.py)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"Vi du:\n  python backend/test.py                              # sua DEFAULT_QUESTION trong file roi chay\n  python backend/test.py --chunks --embeddings\n  python backend/test.py -q \"Quy che dao tao la gi?\" --show-context  # mac dinh rerank lay {DEFAULT_RERANK_TOP_K}\n  python backend/test.py -q \"Hoc bong?\" -c thong_bao --top-k 3\n  python backend/test.py --no-rerank --top-k 5              # tat rerank")
    p.add_argument("--chunks", action="store_true", help="Hien chunking (data/processed/*.parquet)")
    p.add_argument("--embeddings", action="store_true", help="Hien embedding (ChromaDB)")
    p.add_argument("--question", "-q", nargs="+", default=None, help="Cau hoi (ghi de DEFAULT_QUESTION)")
    p.add_argument("--category", "-c", default=None, help="Loc category (mac dinh None = toan bo data)")
    p.add_argument("--top-k", type=int, default=None, help=f"So chunk sau rerank (mac dinh {DEFAULT_RERANK_TOP_K} xu ly trong reranker.py)")
    p.add_argument("--rerank", dest="use_rerank", action="store_true", help="Bat rerank (mac dinh bat)")
    p.add_argument("--no-rerank", dest="use_rerank", action="store_false", help="Tat rerank")
    p.set_defaults(use_rerank=DEFAULT_USE_RERANK)
    p.add_argument("--show-context", action="store_true", help=f"Hien context {DEFAULT_RERANK_TOP_K} chunk tot nhat (do reranker cat)")
    p.add_argument("--show-raw", action="store_true", help="Hien raw JSON")
    p.add_argument("--limit", type=int, default=3, help="So mau hien thi (mac dinh 3)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.question:
        q = " ".join(args.question)
        return do_ask(q, category=args.category, top_k=args.top_k, use_rerank=args.use_rerank, show_context=args.show_context, show_raw=args.show_raw)

    has_diag = args.chunks or args.embeddings
    if DEFAULT_QUESTION and DEFAULT_QUESTION.strip() and not has_diag:
        print(f"[Mac dinh] Dung cau hoi trong code: \"{DEFAULT_QUESTION}\" (sua DEFAULT_QUESTION trong backend/generation/test_service.py de doi)")
        print(f"           Doc toan bo data embedding (category={DEFAULT_CATEGORY or 'ALL'}), top_k={args.top_k or DEFAULT_TOP_K}, rerank={args.use_rerank}\n")
        return do_ask(DEFAULT_QUESTION, category=args.category if args.category is not None else DEFAULT_CATEGORY, top_k=args.top_k, use_rerank=args.use_rerank, show_context=args.show_context, show_raw=args.show_raw)

    if has_diag:
        if args.chunks: show_chunking(limit=args.limit)
        if args.embeddings: show_embeddings(limit=args.limit)
        return 0

    show_chunking(limit=args.limit)
    show_embeddings(limit=args.limit)
    print("="*70 + "\nCHAT TUONG TAC - go 'exit' de thoat (mac dinh rerank lay 3 tot nhat, doc toan bo data)\n" + "="*70)
    print("Goi y: sua DEFAULT_QUESTION trong backend/generation/test_service.py de hoi truc tiep ma khong can go CLI")
    while True:
        try: q = input("\nBan: ").strip()
        except (EOFError, KeyboardInterrupt): print("\nTam biet!"); break
        if not q or q.lower() in ("exit","thoat","quit","q"): print("Tam biet!"); break
        cat, ques = args.category if args.category is not None else DEFAULT_CATEGORY, q
        try:
            from backend.config import CATEGORY_ORDER
            if ":" in q and CATEGORY_ORDER:
                h,_,b = q.partition(":")
                if h.strip().lower() in CATEGORY_ORDER:
                    cat, ques = h.strip().lower(), b.strip()
                    if not ques: print(f"[Da chon category={cat}, nhap cau hoi]"); continue
        except Exception: pass
        do_ask(ques, category=cat, top_k=args.top_k, use_rerank=args.use_rerank, show_context=args.show_context, show_raw=args.show_raw)
    return 0

