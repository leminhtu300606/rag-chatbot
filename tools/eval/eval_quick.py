"""Danh gia nhanh RAG qua HTTP API (GĐ4 chat luong).
====================================================
Không load model local — bắn câu hỏi vào server đang chạy (docker/local),
đo: latency, có nguồn không, từ khóa kỳ vọng có trong đáp án không.

Câu hỏi mẫu mặc định (sửa cho khớp kho tài liệu của đơn vị) hoặc --file JSON:
    [{"question": "...", "keywords": ["tín chỉ", "..."]}, ...]

Chạy:
    python tools/eval/eval_quick.py [--base http://localhost:8000] [--file cau_hoi.json]
                                 [--token <bearer>] [--top-k 5]
Ra exit 1 nếu no-source rate > 40%% (ngưỡng cảnh báo pipeline có vấn đề).
"""
import argparse
import json
import sys
import time
import urllib.request

DEFAULT_SET = [
    {"question": "Quy chế đào tạo là gì?", "keywords": ["đào tạo", "tín chỉ"]},
    {"question": "Học bổng khuyến khích học tập xét thế nào?", "keywords": ["học bổng"]},
    {"question": "Quy định khảo thí như thế nào?", "keywords": ["khảo thí", "thi"]},
    {"question": "Điều kiện tốt nghiệp?", "keywords": ["tốt nghiệp"]},
    {"question": "Quy định trang phục đối với sinh viên?", "keywords": ["trang phục"]},
]


def post_chat(base: str, q: str, token: str, top_k: int, timeout: int = 300) -> dict:
    body = json.dumps({"question": q, "top_k": top_k, "use_rerank": True, "show_context": False,
                       "use_history": False}).encode("utf-8")
    req = urllib.request.Request(base.rstrip("/") + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return {"ok": True, "latency": time.time() - t0, **data}
    except Exception as e:
        try:
            detail = e.read().decode("utf-8", errors="ignore")[:300] if hasattr(e, "read") else str(e)
        except Exception:
            detail = str(e)
        return {"ok": False, "latency": time.time() - t0, "error": detail}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Đánh giá nhanh RAG qua HTTP")
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--file", default=None, help="JSON list [{question, keywords[]}]")
    ap.add_argument("--token", default="", help="Bearer token (khi AUTH_ENABLED=true)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args(argv)

    if a.file:
        with open(a.file, "r", encoding="utf-8") as f:
            items = json.load(f)
    else:
        items = DEFAULT_SET

    n = len(items)
    lat, no_src, kw_hit, fails = [], 0, 0, 0
    print(f"[eval] {n} câu -> {a.base}")
    for i, it in enumerate(items, 1):
        q = it["question"]
        r = post_chat(a.base, q, a.token, a.top_k, a.timeout)
        if not r.get("ok"):
            fails += 1
            print(f"  [{i}/{n}] FAIL {q[:60]} :: {r.get('error', '')[:120]}")
            continue
        lat.append(r["latency"])
        srcs = r.get("sources") or []
        if not srcs:
            no_src += 1
        ans = (r.get("answer") or "").lower()
        kws = [k.lower() for k in it.get("keywords", [])]
        hit = sum(1 for k in kws if k in ans) if kws else 0
        if kws and hit == len(kws):
            kw_hit += 1
        print(f"  [{i}/{n}] {r['latency']:.1f}s src={len(srcs)} kw={hit}/{len(kws)} :: {q[:60]}")

    ok_n = n - fails
    avg = sum(lat) / len(lat) if lat else 0
    print("\n--- KẾT QUẢ ---")
    print(f"  thành công: {ok_n}/{n} | latency TB: {avg:.1f}s | max: {max(lat) if lat else 0:.1f}s")
    print(f"  no-source: {no_src}/{ok_n} | đủ keywords: {kw_hit}/{ok_n}")
    if fails:
        print("[eval] CÓ request lỗi -> kiểm tra server/ollama", file=sys.stderr)
        return 1
    if ok_n and no_src / ok_n > 0.4:
        print("[eval] no-source rate > 40% -> pipeline/index có vấn đề", file=sys.stderr)
        return 1
    print("[eval] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
