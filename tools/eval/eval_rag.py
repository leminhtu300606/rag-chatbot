"""Đánh giá tổng thể mô hình RAG (chuẩn RAGAS rút gọn, chạy offline không cần pgvector).
================================================================================
Đo 2 tầng:
  RETRIEVAL (dùng BM25 trên data/processed — nhánh vẫn chạy được khi thiếu DB):
    - Hit@k, MRR@k, Context Precision@k, Context Recall
  GENERATION (BM25 top-3 làm context -> Ollama LLM sinh trả lời):
    - Lexical F1 vs đáp án chuẩn, Keyword coverage
    - Faithfulness heuristic (câu trả lời có được context chống lưng không)
    - LLM-as-judge: faithfulness / answer relevancy / answer correctness (thang 0-5)

Bộ ground-truth: tools/eval/golden.json (10 cặp hỏi-đáp gắn golden_file + keywords).

Chạy:
    python tools/eval/eval_rag.py [--top-k 5] [--judge 1]
Kết quả JSON lưu ở tools/eval/last_eval.json.
"""
import argparse
import json
import os
import re
import sys
import unicodedata
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
os.environ.setdefault("EVAL_FAST_TOKENIZE", "1")

GOLDEN = os.path.join(BASE, "tools", "eval", "golden.json")
OUT = os.path.join(BASE, "tools", "eval", "last_eval.json")
OLLAMA = os.getenv("OLLAMA_BASE", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b")


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--judge", type=int, default=0, help="1=bật LLM-as-judge (thực nghiệm, cần model >=7b), 0=tắt")
    a = ap.parse_args(argv)

    from backend.indexing.retrieval import bm25_search
    with open(GOLDEN, encoding="utf-8") as f:
        items = json.load(f)
    n = len(items)
    print(f"[eval-rag] {n} câu, top_k={a.top_k}, judge={a.judge}")

    rows = []
    for i, it in enumerate(items, 1):
        q = it["question"]
        docs = bm25_search(q, top_k=a.top_k)
        gfs = [norm(g) for g in it.get("golden_files", [it.get("golden_file", "")])]
        gk = [norm(k) for k in it["golden_keywords"]]
        ranks = [r + 1 for r, d in enumerate(docs)
                 if any(g in norm(str(d.get("metadata", {}).get("filename", ""))) for g in gfs)
                 and any(k in norm(d.get("text", "")) for k in gk)]
        hit = bool(ranks)
        rr = 1.0 / ranks[0] if hit else 0.0
        prec = len(ranks) / len(docs) if docs else 0.0

        ctxs = [d.get("text", "") for d in docs[:3]]
        ans = ollama_generate(GEN_PROMPT.format(ctx="\n---\n".join(c[:1200] for c in ctxs), q=q))
        lex = f1(ans, it["reference"])
        kw = it.get("ref_keywords", [])
        kwc = sum(1 for k in kw if norm(k) in norm(ans)) / len(kw) if kw else 0.0
        fh = faithfulness_heuristic(ans, ctxs)
        j = judge(q, it["reference"], "\n".join(ctxs)[:2000], ans) if a.judge else {}
        rows.append({"q": q, "hit": hit, "mrr": rr, "prec": prec, "recall": 1.0 if hit else 0.0,
                     "f1": lex, "kw": kwc, "faith_h": fh, "judge": j,
                     "answer": ans[:300], "n_ctx": len(docs)})
        js = "" if not a.judge else f" judge={j.get('faithfulness')}/{j.get('relevancy')}/{j.get('correctness')}"
        print(f"  [{i}/{n}] hit={hit} mrr={rr:.2f} f1={lex:.2f} kw={kwc:.2f} fh={fh:.2f}{js} :: {q[:55]}")

    def avg(k, sub=None):
        vs = [(r[sub][k] if sub else r[k]) for r in rows
              if (r[sub].get(k) if sub else r.get(k)) is not None]
        return sum(vs) / len(vs) if vs else 0.0

    res = {
        "n": n, "top_k": a.top_k, "model": LLM_MODEL,
        "retrieval": {"hit_at_k": sum(r["hit"] for r in rows) / n,
                      "mrr_at_k": avg("mrr"),
                      "context_precision": avg("prec"),
                      "context_recall": avg("recall")},
        "generation": {"lexical_f1": avg("f1"), "keyword_coverage": avg("kw"),
                       "faithfulness_heuristic": avg("faith_h"),
                       "judge_faithfulness": avg("faithfulness", "judge") / 5 if a.judge else None,
                       "judge_relevancy": avg("relevancy", "judge") / 5 if a.judge else None,
                       "judge_correctness": avg("correctness", "judge") / 5 if a.judge else None},
        "rows": rows,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("\n--- TỔNG HỢP ---")
    print(f"  Retrieval  : Hit@{a.top_k}={res['retrieval']['hit_at_k']:.2f} MRR={res['retrieval']['mrr_at_k']:.2f} "
          f"P={res['retrieval']['context_precision']:.2f} R={res['retrieval']['context_recall']:.2f}")
    print(f"  Generation : F1={res['generation']['lexical_f1']:.2f} KW={res['generation']['keyword_coverage']:.2f} "
          f"Faith_h={res['generation']['faithfulness_heuristic']:.2f}", end="")
    if a.judge:
        print(f" Judge F={res['generation']['judge_faithfulness']:.2f} "
              f"R={res['generation']['judge_relevancy']:.2f} C={res['generation']['judge_correctness']:.2f}")
    else:
        print()
    print(f"[eval-rag] đã lưu {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
