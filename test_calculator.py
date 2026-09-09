"""
Test suite tự tạo cho calculator số thực - 4 phép cơ bản + đầy đủ (option A)
Chạy: python test_calculator.py  hoặc pytest
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.generation.calculator import (
    calculate, normalize_expression, is_math_question,
    replace_vn_numbers, parse_vn_integer, format_decimal, safe_eval
)
from decimal import Decimal

def assert_eq(a, b, msg=""):
    if str(a) != str(b):
        print(f"  FAIL {msg}: got {a!r} expected {b!r}")
        return False
    return True

failures = 0
passes = 0

def test(desc, question, expected_result=None, should_success=True, last_result=None, want_steps=False, style=None):
    global failures, passes
    res = calculate(question, last_result=last_result, want_steps=want_steps, style=style)
    ok = res.get("success") == should_success
    if should_success and ok:
        if expected_result is not None:
            if str(res.get("result")) != str(expected_result):
                print(f"FAIL [{desc}] '{question}' -> {res.get('result')} (expr={res.get('expression')}) expected {expected_result} | error={res.get('error')}")
                failures+=1
                return
    if not ok:
        print(f"FAIL [{desc}] '{question}' success={res.get('success')} expr={res.get('expression')} result={res.get('result')} error={res.get('error')} expected success={should_success}")
        failures+=1
        return
    # also check is_math detection should be true for success cases
    if should_success:
        if not is_math_question(question, None, last_result):
            # some may be detected via normalize only, but we still want to know
            print(f"  WARN [{desc}] is_math_question False but calculate succeeded for '{question}'")
    passes+=1
    print(f"PASS [{desc}] '{question}' => {res.get('expression')} = {res.get('result')}")

def test_is_math(desc, question, expected, last_result=None):
    global failures, passes
    got = is_math_question(question, None, last_result)
    if got != expected:
        print(f"FAIL [is_math:{desc}] '{question}' got {got} expected {expected}")
        failures+=1
    else:
        passes+=1
        print(f"PASS [is_math:{desc}] '{question}' -> {got}")

print("="*70)
print("TEST 1: Bốn phép cơ bản với số thực (Decimal)")
print("="*70)
test("cộng số thực", "2.5 + 3.7", "6.2")
test("trừ số thực", "10 - 4.2", "5.8")
test("nhân số thực", "3 * 2.5", "7.5")
test("chia số thực", "7 / 2", "3.5")
test("chia lẻ", "10 / 3", "3.3333333333333333333333333333333333333333333333333")  # prec 50 ~ 49 digits after?
test("số âm cộng", "-5.2 + 3", "-2.2")
test("số âm nhân", "-2.5 * -4", "10")
test("nhân âm", "5 * -2", "-10")
test("ngoặc ưu tiên", "(2+3)*4", "20")
test("ngoặc lồng", "((2.5+2.5)*2)/5", "2")
test("thập phân nhiều chữ số", "0.1 + 0.2", "0.3")  # Decimal should be exact
test("số 0", "0 + 0", "0")
test("số lớn", "123456789.123 + 0.877", "123456790")
test("dạng dấu phẩy VN", "2,5 + 3,5", "6")  # normalize có xử lý comma? kiểm tra
test("dấu × ÷", "5 × 2", "10")
test("dấu ÷", "10 ÷ 2", "5")
test("dấu × với số thực", "2.5 × 4", "10")
test("prioritize */ over +-", "2 + 3 * 4", "14")
test("prioritize ngoặc", "2 * (3 + 4)", "14")

print("\n"+"="*70)
print("TEST 2: Chữ tiếng Việt -> số")
print("="*70)
# Test replace_vn_numbers trực tiếp
print(f"replace 'hai phẩy năm' -> {replace_vn_numbers('hai phẩy năm')}")
print(f"replace 'một trăm hai mươi ba' -> {replace_vn_numbers('một trăm hai mươi ba')}")
print(f"replace 'âm năm' -> {replace_vn_numbers('âm năm')}")
print(f"parse_vn_integer(['hai','mươi','ba']) -> {parse_vn_integer(['hai','mươi','ba'])}")

test("hai cộng ba", "hai cộng ba", "5")
test("hai phẩy năm cộng ba", "hai phẩy năm cộng ba", "5.5")
test("năm trừ hai phẩy năm", "năm trừ hai phẩy năm", "2.5")
test("âm năm cộng mười", "âm năm cộng mười", "5")
test("một trăm hai mươi ba cộng bảy", "một trăm hai mươi ba cộng bảy", "130")
test("không phẩy năm cộng không phẩy năm", "không phẩy năm cộng không phẩy năm", "1")
test("hai mươi cộng ba mươi", "hai mươi cộng ba mươi", "50")
test("một trăm cộng hai mươi ba phẩy năm", "một trăm cộng hai mươi ba phẩy năm", "123.5")
test("tính hai cộng ba", "tính hai cộng ba", "5")
test("phép tính chữ: ba nhân bốn", "ba nhân bốn", "12")
test("phép chia chữ: mười chia hai", "mười chia hai", "5")

print("\n"+"="*70)
print("TEST 3: Các phép mở rộng (giữ theo option A)")
print("="*70)
test("lũy thừa", "2 ^ 3", "8")
test("mũ **", "2 ** 3", "8")
test("mũ chữ", "hai mũ ba", "8")
test("căn bậc hai", "căn bậc hai của 16", "4")
test("sqrt", "sqrt(16)", "4")
test("sqrt không ngoặc", "sqrt 16", "4")
test("phần trăm", "5% * 200", "10")  # 5% =0.05 *200=10
test("giai thừa", "5!", "120")
test("giai thừa chữ", "năm giai thừa", "120")
test("log10", "log10(100)", "2")
test("pi", "pi * 2", str(Decimal("3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421070679")*2).rstrip("0").rstrip(".") if False else None)  # approximate
# pi test just check success
test("pi nhân", "pi * 2", None)  # just check parse
test("sin", "sin(0)", "0")
test("cbrt", "cbrt(27)", "3")

print("\n"+"="*70)
print("TEST 4: Edge cases & lỗi")
print("="*70)
test("chia 0", "5 / 0", should_success=False)
test("chia chữ cho 0", "năm chia 0", should_success=False)
test("căn âm", "sqrt(-4)", should_success=False)
test("biểu thức rỗng chữ thường", "quy chế đào tạo là gì", should_success=False)
test("số đơn không toán tử", "2.5", should_success=False)
test("dài quá 200 ký tự", "1+" + "1+"*150 + "1", should_success=False)
test("ngoặc thiếu", "(2+3", should_success=False)
test("hai dấu ++", "2 ++ 3", "5")  # -- should be handled? check
test("trừ âm kép", "2 - -3", "5")

print("\n"+"="*70)
print("TEST 5: is_math_question detection")
print("="*70)
test_is_math("cộng số", "2+3", True)
test_is_math("câu RAG không toán", "quy chế đào tạo là gì", False)
test_is_math("quy chế 3 (số nhưng không op)", "quy chế 3", False)  # should not be math
test_is_math("hai cộng ba (chữ)", "hai cộng ba", True)
test_is_math("xin chào", "xin chào", False)
test_is_math("tính 2+2", "tính 2+2", True)
test_is_math("cộng thêm với last_result", "cộng thêm 2", True, last_result="10")
test_is_math("căn bậc hai", "căn bậc hai của 16", True)
test_is_math("phần trăm", "5% của 200", True)

print("\n"+"="*70)
print("TEST 6: last_result & follow-up")
print("="*70)
test("follow-up cộng thêm", "cộng thêm 5", "15", last_result="10")
test("nó + 2", "nó + 2", "12", last_result="10")
test("nó nhân 2", "nó nhân 2", "20", last_result="10")
test("kết quả chia 2", "kết quả chia 2", "5", last_result="10")
test("ans + 1", "ans + 1", "11", last_result="10")
test("6 cộng kết quả", "6 cộng kết quả", "16", last_result="10")

print("\n"+"="*70)
print("TEST 7: normalize_expression inspection")
print("="*70)
cases = [
    ("2,5 + 3,5", None),
    ("5 × 2", None),
    ("hai cộng ba", None),
    ("10 / 0", None),
    ("cộng thêm 5", "10"),
    ("nó * 2", "5"),
    ("tính 2 + 2 bằng bao nhiêu", None),
]
for q, lr in cases:
    expr = normalize_expression(q, lr)
    print(f"  normalize '{q}' (last={lr}) -> '{expr}'")

print("\n"+"="*70)
print("TEST 8: format_decimal & safe_eval")
print("="*70)
for val in ["0.30000000000000004", "10.000", "-0.000", "123.45000"]:
    print(f"  format {val} -> {format_decimal(Decimal(val))}")
# safe_eval injection attempt
try:
    safe_eval("__import__('os').system('echo hacked')")
    print("FAIL injection should not succeed")
    failures+=1
except Exception as e:
    print(f"PASS injection blocked: {e}")
    passes+=1
try:
    safe_eval("2 + 3")
    print(f"PASS safe_eval 2+3 = {safe_eval('2+3')}")
    passes+=1
except Exception as e:
    print(f"FAIL safe_eval: {e}")
    failures+=1

print("\n"+"="*70)
print(f"TỔNG: {passes} PASS, {failures} FAIL")
print("="*70)
if failures>0:
    sys.exit(1)
else:
    sys.exit(0)

