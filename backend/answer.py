"""
backend/answer.py - Entry point trả lời cho CLI
================================================
Nhiệm vụ:
- Điều phối pipeline truy xuất → rerank → dựng prompt → sinh câu trả lời.
- Gọi trực tiếp các module generation và indexing.
- Hỗ trợ cả chế độ đơn lượt và hội thoại có lịch sử (viết lại câu hỏi, tóm tắt).
- Mặc định top_k=3, đồng bộ với DEFAULT_RERANK_TOP_K.

Chạy: python backend/answer.py | python -m backend.answer
"""
from pathlib import Path
import sys
if __package__ in (None, ""):
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse

# Helper in-file: hàm hỗ trợ in kết quả
def _print_retrieved(docs, title="KET QUA RETRIEVE"):
    print(f"\n--- {title} ({len(docs)} chunks) ---")
    for i, d in enumerate(docs, 1):
        meta = d.get("metadata", {})
        score = d.get("score", "?")
        try:
            score_str = f"{float(score):.4f}"
        except Exception:
            score_str = str(score)
        print(f"[{i}] score={score_str} | {meta.get('filename','')} p{meta.get('page','')} | {meta.get('category','')}/{meta.get('subcategory','')} | idx={meta.get('chunk_index','')} mode={meta.get('chunk_mode','')}")
        print(f"    {d.get('text','')[:260]}...")


def _print_rerank_comparison(before, after):
    print("\n--- SO SANH TRUOC/SAU RERANK (generation/reranker.py) ---")
    print(f"Truoc rerank: {len(before)} chunks (sap xep theo vector distance)")
    for i, d in enumerate(before[:3], 1):
        print(f"  [{i}] score_vec={d.get('score', '?'):.4f} | {d.get('metadata',{}).get('filename','')} -> {d.get('text','')[:80]}...")
    print(f"Sau rerank: {len(after)} chunks (sap xep theo cross-encoder, cao hon = lien quan hon)")
    for i, d in enumerate(after, 1):
        print(f"  [{i}] score_rerank={d.get('score', '?'):.4f} | {d.get('metadata',{}).get('filename','')} -> {d.get('text','')[:80]}...")
    before_ids = [d.get("metadata",{}).get("filename","")+str(d.get("metadata",{}).get("chunk_index","")) for d in before]
    after_ids = [d.get("metadata",{}).get("filename","")+str(d.get("metadata",{}).get("chunk_index","")) for d in after]
    if before_ids[:len(after)] != after_ids:
        print("  -> Thu tu da thay doi sau rerank (chung to reranker da ho tro chon context tot hon cho prompt)")
    else:
        print("  -> Thu tu khong doi (co the query don gian hoac chunks da dung thu tu)")


def do_answer(question: str, category=None, top_k=None, use_rerank=True, show_rerank=False, show_prompt=False, show_context=False, history=None):
    """Chạy pipeline đầy đủ: retrieve → rerank → dựng prompt → sinh câu trả lời.

    Hỗ trợ lịch sử hội thoại: nếu có history thì viết lại câu hỏi và sinh câu trả lời
    có kèm ngữ cảnh hội thoại. Mặc định top_k=3.
    """
    from backend.config import RETRIEVE_TOP_K
    from backend.indexing.retriever import retrieve
    from backend.generation.prompts import build_messages, build_messages_with_history
    from backend.generation.generator import generate, generate_with_history, rewrite_query, summarize_history
    from backend.generation.reranker import rerank
    from backend.generation.rag import _prepare_history

    if not question or not question.strip():
        print("Cau hoi rong.")
        return None

    # Mac dinh 3 theo yeu cau
    top_k = top_k or 3
    if top_k is None:
        top_k = RETRIEVE_TOP_K
    fetch_k = top_k * 3 if use_rerank and rerank else top_k

    print("=" * 70)
    print(f"HOI: {question}")
    print(f"  category={category or 'tat ca'} | top_k={top_k} | fetch_k={fetch_k} | rerank={use_rerank}")
    print("-" * 70)

    # Chuẩn bị lịch sử: cắt cửa sổ, tóm tắt và viết lại câu hỏi
    history_window, summary = (None, None)
    standalone_q = question
    use_history_flag = bool(history)
    if use_history_flag:
        try:
            history_window, summary = _prepare_history(history)
            if history_window:
                # Phát hiện câu phụ thuộc ngữ cảnh và viết lại thành dạng độc lập
                try:
                    rewritten = rewrite_query(question, history_window)
                    if rewritten and rewritten.strip() and rewritten.strip() != question.strip():
                        standalone_q = rewritten
                        print(f"[0] Rewrite (C): '{question}' -> '{standalone_q}' (history {len(history_window)} msgs)")
                    else:
                        print(f"[0] Rewrite: giu nguyen (khong can viet lai)")
                except Exception as e:
                    print(f"[0] Rewrite loi, dung cau goc: {e}")
                    standalone_q = question
                if summary:
                    print(f"[0] Summary (D): {summary[:120]}...")
                print(f"    History window: {len(history_window)} msgs, summary={'co' if summary else 'khong'}")
        except Exception as e:
            print(f"[0] History prepare loi: {e}")
            history_window, summary = None, None
            standalone_q = question

    query_for_retrieval = standalone_q if use_history_flag and standalone_q and standalone_q.strip() else question

    print(f"[1] Retrieve: tim {fetch_k} chunks lien quan nhat trong ChromaDB... (query='{query_for_retrieval[:60]}...')")
    try:
        context = retrieve(query_for_retrieval, category=category, top_k=fetch_k)
    except Exception as e:
        print(f"[Loi retrieve] {e}")
        import traceback
        traceback.print_exc()
        if "dimension" in str(e).lower():
            print("\nGoi y: dimension khong khop -> hay chay: python -m backend.indexing.build_index --rebuild")
        return None

    if not context:
        print("  Khong tim thay context nao. Hay kiem tra: python backend/test.py --embeddings")
        return None

    _print_retrieved(context, title="1. RETRIEVE (tu ChromaDB)")

    reranked = context
    if use_rerank:
        if rerank is None:
            print("\n[2] Rerank: bo qua (khong load duoc cross-encoder)")
        else:
            print(f"\n[2] Rerank: dung backend/generation/reranker.py cham diem lai {len(context)} chunks -> chon top {top_k}... (query='{query_for_retrieval[:50]}')")
            before = list(context)
            reranked = rerank(query_for_retrieval, context, top_k=top_k)
            _print_retrieved(reranked, title="2. RERANK (sau khi tinh chinh)")
            if show_rerank:
                _print_rerank_comparison(before, reranked)
    else:
        print(f"\n[2] Rerank: tat (lay truc tiep top {top_k} tu retrieve)")
        reranked = context[:top_k]

    print(f"\n[3] Prompt: dung {len(reranked)} chunks da rerank de tao prompt cho LLM... (history={'co' if history_window else 'khong'})")
    if use_history_flag and history_window:
        messages = build_messages_with_history(reranked, question, history_window, summary)
    else:
        messages = build_messages(reranked, question)
    if show_prompt:
        print("\n--- PROMPT GUI LLM (backend/generation/prompts.py) ---")
        for m in messages:
            print(f"\n[{m['role'].upper()}]\n{m['content'][:900]}{'...' if len(m['content'])>900 else ''}")
    else:
        print(f"  System: {messages[0]['content'][:120]}...")
        if history_window:
            print(f"  History: {len(history_window)} msgs + summary={'co' if summary else 'khong'}")
        print(f"  User preview: Ngu canh gom {len(reranked)} chunks + Cau hoi: {question}")

    print(f"\n[4] Generate: goi LLM ({__import__('backend.config', fromlist=['LLM_BACKEND']).LLM_BACKEND}/{__import__('backend.config', fromlist=['LLM_MODEL']).LLM_MODEL})... (with_history={bool(history_window)})")
    try:
        if use_history_flag and history_window:
            answer_text = generate_with_history(reranked, question, history_window, summary)
        else:
            answer_text = generate(reranked, question)
    except Exception as e:
        print(f"[Loi generate] {e}")
        import traceback
        traceback.print_exc()
        print("\nGoi y: kiem tra Ollama dang chay? ollama serve & ollama list")
        return None

    print("\n" + "=" * 70)
    print("TRA LOI:")
    print("=" * 70)
    print(answer_text)

    print("\n--- NGUON ---")
    sources = sorted({ (c.get("metadata",{}).get("filename",""), c.get("metadata",{}).get("page",""), c.get("metadata",{}).get("category","")) for c in reranked})
    for fn, page, cat in sources:
        if fn:
            print(f"  - {fn} (p{page}, {cat})")

    if show_context:
        _print_retrieved(reranked, title="CONTEXT CUOI CUNG DUA VAO PROMPT")

    # Trả về nội dung trả lời để caller có thể lưu vào lịch sử hội thoại
    return answer_text


def check_ready():
    """Kiem tra data sach va ChromaDB da san sang chua."""
    import backend.config as cfg
    from backend.preprocessing.storage import get_processed_files, load_chunks

    print("Kiem tra data sach...")
    ok = True
    files = get_processed_files()
    if not cfg.PROCESSED_DIR.exists() or not files:
        print(f"  [X] Chua co data sach trong {cfg.PROCESSED_DIR}. Hay chay: python backend/clean.py")
        ok = False
    else:
        cnt = len(files)
        total_chunks = sum(len(load_chunks(p)) for p in files)
        print(f"  [OK] data sach: {cnt} files, {total_chunks} chunks (parquet)")

    try:
        from backend.indexing.vectorstore import count, get_stats
        c = count()
        if c == 0:
            print(f"  [X] ChromaDB rong ({cfg.CHROMA_DIR}). Hay chay: python -m backend.indexing.build_index --rebuild")
            ok = False
        else:
            print(f"  [OK] ChromaDB: {c} vectors")
            try:
                stats = get_stats()
                print(f"       {stats}")
            except Exception:
                pass
    except Exception as e:
        print(f"  [X] Loi ChromaDB: {e}")
        ok = False

    if not ok:
        print("\nHay hoan tat buoc clean -> build_index truoc khi hoi.")
    return ok


def build_parser():
    p = argparse.ArgumentParser(
        prog="backend.answer",
        description="Dung data sach ho tro rerank va tra loi (retrieve -> rerank (generation) -> prompt -> LLM) - mac dinh top_k=3",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Vi du:\n"
            "  python backend/answer.py\n"
            "  python backend/answer.py --question \"Quy che dao tao la gi?\" --rerank --show-rerank\n"
            "  python backend/answer.py -q \"Hoc bong xet the nao?\" -c thong_bao --top-k 3 --show-prompt --show-context\n"
            "  python -m backend.answer --question \"Thi lai quy dinh the nao?\" --no-rerank\n"
        ),
    )
    p.add_argument("--question", "-q", nargs="+", default=None, help="Cau hoi (nhieu tu se duoc noi lai)")
    p.add_argument("--category", "-c", default=None, help="Loc theo category (quy_che, quy_dinh, ...)")
    p.add_argument("--top-k", type=int, default=None, help="So chunk dua vao LLM (mac dinh 3)")
    p.add_argument("--rerank", dest="use_rerank", action="store_true", help="Bat rerank (mac dinh bat)")
    p.add_argument("--no-rerank", dest="use_rerank", action="store_false", help="Tat rerank")
    p.set_defaults(use_rerank=True)
    p.add_argument("--show-rerank", action="store_true", help="Hien thi so sanh truoc/sau rerank")
    p.add_argument("--show-prompt", action="store_true", help="Hien thi prompt day du gui LLM")
    p.add_argument("--show-context", action="store_true", help="Hien thi context cuoi cung")
    p.add_argument("--check", action="store_true", help="Chi kiem tra data sach + ChromaDB, khong hoi")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.check:
        check_ready()
        return 0

    if args.question:
        q = " ".join(args.question)
        ok = check_ready()
        if not ok:
            print("\nVan co gang hoi (co the khong co context)...")
        res = do_answer(q, category=args.category, top_k=args.top_k, use_rerank=args.use_rerank, show_rerank=args.show_rerank, show_prompt=args.show_prompt, show_context=args.show_context)
        return 0 if res is not None else 1

    print("=" * 70)
    print("CHAT HOI DAP - Dung data sach + Rerank (backend/generation/reranker.py)")
    print("=" * 70)
    check_ready()
    print("\nGo 'exit' de thoat. Goi y: 'quy_che: dieu kien tot nghiep?'")
    print(f"Che do: top_k={args.top_k or '3 (mac dinh)'} | rerank={args.use_rerank} | hoi thoai=bat (co nho lich su)")
    print("Lenh: 'clear' de xoa lich su hoi thoai, 'history' de xem lich su")
    print("-" * 70)
    # Lịch sử hội thoại cho chế độ CLI tương tác
    cli_history: list[dict] = []
    while True:
        try:
            q = input("\nBan: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTam biet!")
            break
        if not q:
            continue
        if q.lower() in ("exit", "thoat", "quit", "q"):
            print("Tam biet!")
            break
        if q.lower() in ("clear", "reset", "new"):
            cli_history.clear()
            print("[Da xoa lich su hoi thoai]")
            continue
        if q.lower() == "history":
            if not cli_history:
                print("[Lich su rong]")
            else:
                print(f"[Lich su {len(cli_history)//2} turns]")
                for m in cli_history:
                    print(f"  {m['role']}: {m['content'][:80]}")
            continue
        category = args.category
        question = q
        try:
            from backend.config import CATEGORY_ORDER
        except Exception:
            CATEGORY_ORDER = []
        if ":" in q and CATEGORY_ORDER:
            head, _, body = q.partition(":")
            if head.strip().lower() in CATEGORY_ORDER:
                category = head.strip().lower()
                question = body.strip()
                if not question:
                    print(f"[Da chon category={category}, hay nhap cau hoi]")
                    continue
        # Gọi pipeline có kèm lịch sử hội thoại
        answer_text = do_answer(question, category=category, top_k=args.top_k, use_rerank=args.use_rerank, show_rerank=args.show_rerank, show_prompt=args.show_prompt, show_context=args.show_context, history=cli_history if cli_history else None)
        # Lưu vào lịch sử nếu thành công
        if answer_text is not None:
            cli_history.append({"role": "user", "content": question})
            # Luu answer that neu co, neu khong thi fallback
            if isinstance(answer_text, str) and answer_text.strip():
                cli_history.append({"role": "assistant", "content": answer_text})
            else:
                cli_history.append({"role": "assistant", "content": f"[Da tra loi cho: {question[:60]}]"})
            if len(cli_history) > 30:
                cli_history = cli_history[-30:]
            print(f"[History] Da luu {len(cli_history)//2} turns (window {min(len(cli_history), 12)} msgs se dung cho cau sau)")
        else:
            print("[History] Khong luu do loi tra loi")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
