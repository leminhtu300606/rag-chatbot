"""Đánh giá tổng thể mô hình RAG (chuẩn RAGAS rút gọn, chạy offline không cần pgvector).
================================================================================
Đo 2 tầng:
  RETRIEVAL (BM25 mặc định; --retrieval hybrid dùng pgvector khi có DB):
    - Hit@k, MRR@k, Context Precision@k, Context Recall, nDCG@k
    - (tuỳ chọn) trước/sau rerank + độ trễ từng tầng
  GENERATION (top-3 làm context -> Ollama LLM sinh trả lời):
    - Lexical F1 vs đáp án chuẩn, Keyword coverage
    - Faithfulness heuristic (câu trả lời có được context chống lưng không)
    - LLM-as-judge: faithfulness / answer relevancy / answer correctness (thang 0-5)
    - Câu "không có đáp án" (unanswerable): đo tỉ lệ từ chối đúng

Bộ ground-truth: tools/eval/golden.json.
  - Item thường: cần question/reference/golden_files/golden_keywords (+ref_keywords).
  - Item unanswerable: thêm "unanswerable": true (không cần reference).

Chạy:
    python tools/eval/eval_rag.py [--top-k 5] [--judge 1] [--retrieval bm25|hybrid] [--rerank 1] [--no-gen]
Kết quả JSON lưu ở tools/eval/last_eval.json.
"""
import argparse
import json
import math
import os
import re
import statistics
import sys
import time
import unicodedata
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
os.environ.setdefault("EVAL_FAST_TOKENIZE", "1")

# Đảm bảo stdout/stderr UTF-8 trên Windows (tránh UnicodeEncodeError khi in tiếng Việt)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

GOLDEN = os.path.join(BASE, "tools", "eval", "golden.json")
OUT = os.path.join(BASE, "tools", "eval", "last_eval.json")
OLLAMA = os.getenv("OLLAMA_BASE", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b")

REFUSAL_MARKERS = [
    "khong co thong tin", "khong tim thay", "khong de cap", "khong co trong tai lieu",
    "khong ro", "khong du thong tin", "chua co thong tin", "khong duoc de cap",
    "khong co du lieu", "khong the tra loi", "khong lien quan",
]


def norm(s: str) -> str:
    s = (s or "").lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return s


def toks(s: str):
    return [w for w in re.findall(r"[a-z0-9_]+", norm(s)) if len(w) >= 2]


def f1(pred: str, ref: str) -> float:
    pt, rt = toks(pred), toks(ref)
    if not pt or not rt:
        return 0.0
    from collections import Counter
    pc, rc = Counter(pt), Counter(rt)
    overlap = sum(min(pc[w], rc[w]) for w in pc if w in rc)
    if not overlap:
        return 0.0
    p, r = overlap / len(pt), overlap / len(rt)
    return 2 * p * r / (p + r)


def _percentile(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[idx]


def latency_stats(vals_ms):
    if not vals_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "n": 0}
    return {
        "mean_ms": sum(vals_ms) / len(vals_ms),
        "p50_ms": statistics.median(vals_ms),
        "p95_ms": _percentile(vals_ms, 95),
        "n": len(vals_ms),
    }


def ollama_generate(prompt: str, num_predict: int = 150) -> str:
    body = json.dumps({"model": LLM_MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.1, "num_predict": num_predict}}).encode()
    req = urllib.request.Request(OLLAMA + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode()).get("response", "").strip()


GEN_PROMPT = ("Dựa vào TÀI LIỆU dưới đây, trả lời câu hỏi ngắn gọn (tối đa 3 câu), "
              "chỉ dùng thông tin trong tài liệu.\n\nTÀI LIỆU:\n{ctx}\n\nCÂU HỎI: {q}\nTRẢ LỜI:")

JUDGE_PROMPT = """Ban la giam khao. Cham 3 tieu chi thang 0-5 (so nguyen).
CAU HOI: {q}
DAP AN CHUAN: {ref}
TAI LIEU: {ctx}
CAU TRA LOI CAN CHAM: {a}
- F (faithfulness): cau tra loi co dung voi TAI LIEU khong? Bia dat = 0.
- R (relevancy): co tra loi dung CAU HOI khong?
- C (correctness): co khop DAP AN CHUAN khong?
Tra loi dung 3 dong:
F: <so>
R: <so>
C: <so>"""


def judge(q, ref, ctx, ans):
    try:
        raw = ollama_generate(JUDGE_PROMPT.format(q=q, ref=ref, ctx=ctx[:2000], a=ans), num_predict=60)
        def grab(k):
            m = re.search(k + r"\s*:\s*(\d)", raw)
            return float(m.group(1)) if m else None
        vals = {"faithfulness": grab("F"), "relevancy": grab("R"), "correctness": grab("C")}
        if all(v is None for v in vals.values()):
            vals["raw"] = raw[:200]
        return vals
    except Exception as e:
        return {"faithfulness": None, "relevancy": None, "correctness": None, "error": str(e)[:100]}


def faithfulness_heuristic(answer: str, contexts) -> float:
    sents = [s.strip() for s in re.split(r"[.?!;\n]+", answer or "") if len(s.strip()) > 5]
    if not sents:
        return 0.0
    ctx_toks = [set(toks(c)) for c in contexts]
    ok = 0
    for s in sents:
        st = set(toks(s))
        if not st:
            continue
        best = max((len(st & ct) / len(st) for ct in ctx_toks), default=0)
        if best >= 0.4:
            ok += 1
    return ok / len(sents)


# ── Retrieval metrics ──

def _doc_is_relevant(doc, gfs, gk) -> bool:
    fn = norm(str(doc.get("metadata", {}).get("filename", "")))
    if not any(g in fn for g in gfs):
        return False
    txt = norm(doc.get("text", ""))
    return any(k in txt for k in gk)


def _ndcg(relevances, k) -> float:
    rels = relevances[:k]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    ideal = sorted(relevances, reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def retrieval_metrics(docs, gfs, gk, k) -> dict:
    relevances = [1.0 if _doc_is_relevant(d, gfs, gk) else 0.0 for d in docs[:k]]
    ranks = [i + 1 for i, r in enumerate(relevances) if r > 0]
    hit = bool(ranks)
    return {
        "hit": hit,
        "mrr": (1.0 / ranks[0]) if ranks else 0.0,
        "prec": (len(ranks) / len(docs)) if docs else 0.0,
        "recall": 1.0 if hit else 0.0,
        "ndcg": _ndcg(relevances, k),
    }


def make_retriever(kind: str):
    if kind == "hybrid":
        from backend.indexing.retrieval import hybrid_retrieve

        def _r(q, k):
            try:
                docs = hybrid_retrieve(q, top_k=k)
                if docs:
                    return docs
            except Exception as e:
                print(f"[eval-rag] hybrid lỗi ({e}); dùng BM25")
            from backend.indexing.retrieval import bm25_search
            return bm25_search(q, top_k=k)

        return _r
    from backend.indexing.retrieval import bm25_search
    return lambda q, k: bm25_search(q, top_k=k)


def make_reranker():
    try:
        from backend.generation.reranker import rerank
        return rerank
    except Exception as e:
        print(f"[eval-rag] Bỏ qua rerank (không nạp được): {e}")
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--judge", type=int, default=0, help="1=bật LLM-as-judge (thực nghiệm, cần model >=7b), 0=tắt")
    ap.add_argument("--retrieval", choices=["bm25", "hybrid"], default="bm25")
    ap.add_argument("--rerank", type=int, default=0, help="1=đo trước/sau rerank (cần model cross-encoder)")
    ap.add_argument("--no-gen", action="store_true", help="Chỉ đo retrieval, bỏ qua sinh câu trả lời (không cần Ollama)")
    a = ap.parse_args(argv)

    retriever = make_retriever(a.retrieval)
    rerank_fn = make_reranker() if a.rerank else None
    with open(GOLDEN, encoding="utf-8") as f:
        items = json.load(f)
    answerable = [it for it in items if not it.get("unanswerable")]
    unanswerable = [it for it in items if it.get("unanswerable")]
    n = len(answerable)
    print(f"[eval-rag] {len(items)} câu ({n} trả lời được, {len(unanswerable)} không đáp án), "
          f"top_k={a.top_k}, retrieval={a.retrieval}, rerank={bool(rerank_fn)}, gen={not a.no_gen}, judge={a.judge}")

    rows = []
    ret_lat, gen_lat, rrk_lat = [], [], []
    for i, it in enumerate(items, 1):
        q = it["question"]
        t0 = time.perf_counter()
        docs = retriever(q, a.top_k)
        ret_lat.append((time.perf_counter() - t0) * 1000.0)
        gfs = [norm(g) for g in it.get("golden_files", [it.get("golden_file", "")])]
        gk = [norm(k) for k in it.get("golden_keywords", [])]

        if it.get("unanswerable"):
            ctxs = [d.get("text", "") for d in docs[:3]]
            if a.no_gen:
                refused = None
                ans = ""
            else:
                ans = ollama_generate(GEN_PROMPT.format(ctx="\n---\n".join(c[:1200] for c in ctxs), q=q))
                refused = any(m in norm(ans) for m in REFUSAL_MARKERS)
            rows.append({"q": q, "unanswerable": True, "refused": refused,
                         "n_ctx": len(docs), "answer": ans[:200]})
            print(f"  [{i}/{len(items)}] unanswerable refused={refused} n_ctx={len(docs)} :: {q[:50]}")
            continue

        before = retrieval_metrics(docs, gfs, gk, a.top_k)

        after = None
        docs_ctx = docs
        if rerank_fn:
            t0 = time.perf_counter()
            try:
                docs_ctx = rerank_fn(q, docs, top_k=a.top_k)
                if not docs_ctx:
                    docs_ctx = docs
            except Exception as e:
                print(f"[eval-rag] rerank lỗi ({e}); bỏ qua")
                docs_ctx = docs
            rrk_lat.append((time.perf_counter() - t0) * 1000.0)
            after = retrieval_metrics(docs_ctx, gfs, gk, a.top_k)

        if a.no_gen:
            rows.append({"q": q, "hit": before["hit"], "mrr": before["mrr"], "prec": before["prec"],
                         "recall": before["recall"], "ndcg": before["ndcg"],
                         "after": after, "n_ctx": len(docs)})
            ds = "" if after is None else f" after_ndcg={after['ndcg']:.2f}"
            print(f"  [{i}/{len(items)}] hit={before['hit']} mrr={before['mrr']:.2f} ndcg={before['ndcg']:.2f}{ds} :: {q[:50]}")
            continue

        ctxs = [d.get("text", "") for d in docs_ctx[:3]]
        t0 = time.perf_counter()
        ans = ollama_generate(GEN_PROMPT.format(ctx="\n---\n".join(c[:1200] for c in ctxs), q=q))
        gen_lat.append((time.perf_counter() - t0) * 1000.0)
        lex = f1(ans, it["reference"])
        kw = it.get("ref_keywords", [])
        kwc = sum(1 for k in kw if norm(k) in norm(ans)) / len(kw) if kw else 0.0
        fh = faithfulness_heuristic(ans, ctxs)
        j = judge(q, it["reference"], "\n".join(ctxs)[:2000], ans) if a.judge else {}
        rows.append({"q": q, "hit": before["hit"], "mrr": before["mrr"], "prec": before["prec"],
                     "recall": before["recall"], "ndcg": before["ndcg"], "after": after,
                     "f1": lex, "kw": kwc, "faith_h": fh, "judge": j,
                     "answer": ans[:300], "n_ctx": len(docs)})
        js = "" if not a.judge else f" judge={j.get('faithfulness')}/{j.get('relevancy')}/{j.get('correctness')}"
        print(f"  [{i}/{len(items)}] hit={before['hit']} mrr={before['mrr']:.2f} ndcg={before['ndcg']:.2f} "
              f"f1={lex:.2f} kw={kwc:.2f} fh={fh:.2f}{js} :: {q[:50]}")

    def avg(key, sub=None, src=None):
        src = src if src is not None else rows
        vs = []
        for r in src:
            if r.get("unanswerable"):
                continue
            v = (r.get(sub) or {}).get(key) if sub else r.get(key)
            if v is not None:
                vs.append(v)
        return sum(vs) / len(vs) if vs else 0.0

    def retrieval_block(sub=None):
        return {
            "hit_at_k": avg("hit", sub),
            "mrr_at_k": avg("mrr", sub),
            "context_precision": avg("prec", sub),
            "context_recall": avg("recall", sub),
            "ndcg_at_k": avg("ndcg", sub),
        }

    unans_rows = [r for r in rows if r.get("unanswerable")]
    refused_vals = [1.0 if r["refused"] else 0.0 for r in unans_rows if r["refused"] is not None]

    res = {
        "n": len(items),
        "n_answerable": n,
        "top_k": a.top_k,
        "model": LLM_MODEL,
        "retrieval_method": a.retrieval,
        "retrieval": retrieval_block(),
        "retrieval_latency": latency_stats(ret_lat),
        "generation": {
            "lexical_f1": avg("f1"),
            "keyword_coverage": avg("kw"),
            "faithfulness_heuristic": avg("faith_h"),
            "judge_faithfulness": avg("faithfulness", "judge") / 5 if a.judge else None,
            "judge_relevancy": avg("relevancy", "judge") / 5 if a.judge else None,
            "judge_correctness": avg("correctness", "judge") / 5 if a.judge else None,
        },
        "generation_latency": latency_stats(gen_lat),
        "rows": rows,
    }
    if rerank_fn:
        res["retrieval_rerank"] = retrieval_block(sub="after")
        res["rerank_latency"] = latency_stats(rrk_lat)
    if unans_rows:
        res["unanswerable"] = {
            "n": len(unans_rows),
            "refusal_rate": (sum(refused_vals) / len(refused_vals)) if refused_vals else None,
        }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("\n--- TỔNG HỢP ---")
    r = res["retrieval"]
    print(f"  Retrieval  : Hit@{a.top_k}={r['hit_at_k']:.2f} MRR={r['mrr_at_k']:.2f} "
          f"P={r['context_precision']:.2f} R={r['context_recall']:.2f} nDCG={r['ndcg_at_k']:.2f} "
          f"(ret {res['retrieval_latency']['mean_ms']:.0f}ms)")
    if rerank_fn:
        rr = res["retrieval_rerank"]
        print(f"  Sau rerank : Hit={rr['hit_at_k']:.2f} MRR={rr['mrr_at_k']:.2f} "
              f"nDCG={rr['ndcg_at_k']:.2f} (rerank {res['rerank_latency']['mean_ms']:.0f}ms)")
    if not a.no_gen:
        print(f"  Generation : F1={res['generation']['lexical_f1']:.2f} KW={res['generation']['keyword_coverage']:.2f} "
              f"Faith_h={res['generation']['faithfulness_heuristic']:.2f} "
              f"(gen {res['generation_latency']['mean_ms']:.0f}ms)", end="")
        if a.judge:
            print(f" Judge F={res['generation']['judge_faithfulness']:.2f} "
                  f"R={res['generation']['judge_relevancy']:.2f} C={res['generation']['judge_correctness']:.2f}")
        else:
            print()
    if "unanswerable" in res:
        rr = res["unanswerable"]["refusal_rate"]
        print(f"  Không đáp án: n={res['unanswerable']['n']} refusal_rate={rr if rr is None else round(rr, 2)}")
    print(f"[eval-rag] đã lưu {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
