"""Extended tests: session, API, edge VN formats"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from backend.generation.calculator import calculate, normalize_expression, is_math_question
from decimal import Decimal

fail=0
pas=0
def test(desc, q, expected=None, last=None, should_success=True):
    global fail, pas
    r = calculate(q, last_result=last)
    ok = r["success"] == should_success
    if should_success and ok and expected is not None:
        if str(r["result"]) != str(expected):
            print(f"FAIL [{desc}] {q!r} -> {r['result']} expr={r['expression']} expected {expected}")
            fail+=1; return
    if not ok:
        print(f"FAIL [{desc}] {q!r} success {r['success']} exp {should_success} err {r['error']} expr {r['expression']}")
        fail+=1; return
    pas+=1
    print(f"PASS [{desc}] {q!r} -> {r['expression']} = {r['result']}")

print("=== EXTENDED: VN formats & real numbers ===")
test("1.000,5 VN", "1.000,5 + 0.5", "1001")
test("1,000 . decimal", "1,000 + 2", "1002")  # 1,000 as 1000? our heuristic removes comma thousand
test("decimal comma single", "2,5 * 2", "5")
test("mix dot comma", "1.000,5 * 2", "2001")
test("số âm phẩy", "âm hai phẩy năm cộng năm", "2.5")
test("lẻ VN", "một trăm lẻ năm cộng năm", "110")
test("mươi VN", "hai mươi mốt cộng hai", "23")
test("nghìn VN", "hai nghìn ba trăm cộng bảy trăm", "3000")
test("triệu", "một triệu cộng một nghìn", "1001000")
test("tỷ", "một tỷ chia hai", "500000000")
test("phẩy chữ chi tiết", "0,1 cộng 0,2", "0.3")
test("nhiều phẩy", "1,5 + 2,5 + 3", "7")
test("âm thập phân", "-2,5 * 4", "-10")
test("ngoặc và chữ", "ba nhân (hai cộng bốn)", "18")
test("ưu tiên", "10 - 2 * 3", "4")
test("chia có dư decimal", "5 / 2", "2.5")
test("chia lẻ high prec", "1 / 6", "0.16666666666666666666666666666666666666666666666667")
test("pow decimal", "2.5 ^ 2", "6.25")
test("factorial 0", "0!", "1")
test("phần trăm decimal", "12,5% * 200", "25")
test("sqrt decimal", "sqrt(2,25)", "1.5")  # sqrt 2.25 =1.5 with comma
test("cbrt decimal", "cbrt(8)", "2")
test("log10", "log10(1000)", "3")
test("sin pi/2", "sin(pi/2)", "1")
# last_result edge
test("cộng thêm với last 0", "cộng thêm 1", "1", last="0")
test("nhân đôi last", "nhân 2", "20", last="10")  # need? "nhân 2" bare
test("trừ bớt 5", "trừ bớt 5", "5", last="10")
test("chia cho 2", "chia cho 2", "5", last="10")

# is_math edge
print("\n=== is_math edge ===")
def im(q, exp, last=None):
    global fail,pas
    got=is_math_question(q,None,last)
    if got!=exp:
        print(f"FAIL is_math {q!r} got {got} exp {exp}")
        fail+=1
    else:
        pas+=1
        print(f"PASS is_math {q!r} -> {got}")
im("quy chế 3", False)
im("điều 3 quy chế", False)
im("2,5+3", True)
im("hai phẩy năm nhân ba", True)
im("xin chào 123", False)  # no op
im("tính 123", False)  # single number not math
im("5!", True)
im("5 × 2", True)
im("sqrt 9", True)
im("pi * 2", True)
im("cộng thêm 2", True, last="10")
im("cộng thêm 2", True, last=None)  # with no last_result, still math because has op word + digit
im("cộng thêm 2", True)  # corrected

print("\n=== RAG integration ===")
try:
    from backend.generation.rag import answer, answer_with_history
    # Mock retrieve/generate? But calculator branch doesn't need retrieve. So just call answer with math
    # Patch retrieve to avoid needing chroma
    import backend.generation.rag as rag_mod
    orig_retrieve = rag_mod.retrieve
    orig_rerank = rag_mod.rerank
    orig_generate = rag_mod.generate
    # If calculator branch is taken, retrieve not called. So we can test without mock? But answer still calls retrieve after math check fail.
    # For math question, it should return early without retrieve. We'll test.
    r = answer("2.5 + 3.7")
    print(f"answer 2.5+3.7 -> {r}")
    if r.get("math_result") == "6.2":
        print("PASS RAG answer math bypass")
        pas+=1
    else:
        print(f"FAIL RAG answer math bypass got {r.get('math_result')}")
        fail+=1

    r2 = answer_with_history("cộng thêm 5", history=[{"role":"user","content":"5 * 2"},{"role":"assistant","content":"10"}], last_math_result="10")
    print(f"answer_with_history cộng thêm 5 last 10 -> {r2}")
    if r2.get("math_result") == "15":
        print("PASS RAG history math")
        pas+=1
    else:
        print(f"FAIL RAG history expected 15 got {r2.get('math_result')}")
        fail+=1

    # Test non-math still goes to RAG but we mock generate to avoid Ollama
    # Mock generate to return dummy
    rag_mod.retrieve = lambda *a,**k: [{"text":"dummy context","metadata":{"filename":"test.pdf"}}]
    rag_mod.rerank = lambda q,ctx,**k: ctx
    rag_mod.generate = lambda ctx,q,style=None: "dummy answer"
    rag_mod.generate_with_history = lambda ctx,q,h,s,style=None: "dummy history answer"
    # Now test non-math
    r3 = answer("quy chế là gì")
    print(f"answer non-math -> {r3.get('answer')}")
    if r3.get("answer") == "dummy answer":
        print("PASS RAG non-math mocked")
        pas+=1
    else:
        print("FAIL RAG non-math")
        fail+=1
    # Restore
    rag_mod.retrieve = orig_retrieve
    rag_mod.rerank = orig_rerank
    rag_mod.generate = orig_generate
    # generate_with_history restore via import
    from backend.generation.generator import generate_with_history as _gwh
    rag_mod.generate_with_history = _gwh

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"FAIL RAG integration exception {e}")
    fail+=1

print("\n=== normalize edge ===")
cases = [
    ("1.000,5 + 1", "1000.5 + 1"),
    ("2,5 × 2", "2.5 * 2"),
    ("5! + 2", "factorial(5) + 2"),
    ("căn bậc hai của 2,25", "sqrt(2.25)"),
    ("hai mươi ba phẩy năm chia hai", "23.5 / 2"),
]
for q, exp_sub in cases:
    expr = normalize_expression(q)
    ok = exp_sub in expr or expr==exp_sub
    if not ok:
        print(f"FAIL normalize {q!r} -> {expr!r} expected contains {exp_sub!r}")
        fail+=1
    else:
        pas+=1
        print(f"PASS normalize {q!r} -> {expr!r}")

print(f"\nTOTAL ext: {pas} PASS {fail} FAIL")
sys.exit(1 if fail else 0)
