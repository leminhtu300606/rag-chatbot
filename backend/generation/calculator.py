"""
backend/generation/calculator.py - Tính toán số thực với độ chính xác cao
====================================================================
Hỗ trợ:
- Tất cả ký tự toán học: + - * / % ^ ** () . , sqrt sin cos tan log ln exp abs factorial pi e
- Chữ -> số: "hai phẩy năm" -> 2.5, "một trăm hai mươi ba" -> 123, "âm năm" -> -5
- Độ chính xác cao: Decimal prec=50
- Nhớ last_result (kết quả trước)
- Chỉ hiện steps khi user yêu cầu (chi tiết/từng bước/bước)
"""
import ast
import math
import re
from decimal import Decimal, getcontext, InvalidOperation

# Độ chính xác cao nhất
getcontext().prec = 50
getcontext().Emax = 999999999
getcontext().Emin = -999999999

# ── Hằng số ──
PI_DEC = Decimal("3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421070679")
E_DEC = Decimal("2.7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274")

ALLOWED_NAMES = {
    "pi": PI_DEC,
    "e": E_DEC,
    "tau": PI_DEC * 2,
}

def _sqrt_dec(x: Decimal) -> Decimal:
    if x < 0:
        raise ValueError("căn bậc hai của số âm")
    try:
        return x.sqrt()
    except Exception:
        return Decimal(str(math.sqrt(float(x))))

def _exp_dec(x: Decimal) -> Decimal:
    try:
        return x.exp()
    except Exception:
        return Decimal(str(math.exp(float(x))))

def _ln_dec(x: Decimal) -> Decimal:
    if x <= 0:
        raise ValueError("ln của số không dương")
    try:
        return x.ln()
    except Exception:
        return Decimal(str(math.log(float(x))))

def _log10_dec(x: Decimal) -> Decimal:
    if x <= 0:
        raise ValueError("log10 của số không dương")
    try:
        return x.log10()
    except Exception:
        return Decimal(str(math.log10(float(x))))

def _log_dec(x: Decimal, base: Decimal | None = None) -> Decimal:
    if x <= 0:
        raise ValueError("log của số không dương")
    if base is None:
        return _ln_dec(x)
    if base <= 0 or base == 1:
        raise ValueError("cơ số log không hợp lệ")
    try:
        # log_base(x) = ln(x)/ln(base)
        return _ln_dec(x) / _ln_dec(base)
    except Exception:
        return Decimal(str(math.log(float(x), float(base))))

ALLOWED_FUNCS = {
    "sqrt": _sqrt_dec,
    "cbrt": lambda x: Decimal(str(float(x) ** (1/3))) if x >= 0 else -Decimal(str((-float(x)) ** (1/3))),
    "abs": lambda x: abs(x),
    "sin": lambda x: Decimal(str(math.sin(float(x)))),
    "cos": lambda x: Decimal(str(math.cos(float(x)))),
    "tan": lambda x: Decimal(str(math.tan(float(x)))),
    "asin": lambda x: Decimal(str(math.asin(float(x)))),
    "acos": lambda x: Decimal(str(math.acos(float(x)))),
    "atan": lambda x: Decimal(str(math.atan(float(x)))),
    "sinh": lambda x: Decimal(str(math.sinh(float(x)))),
    "cosh": lambda x: Decimal(str(math.cosh(float(x)))),
    "tanh": lambda x: Decimal(str(math.tanh(float(x)))),
    "log": _log_dec,
    "ln": _ln_dec,
    "log10": _log10_dec,
    "log2": lambda x: Decimal(str(math.log2(float(x)))),
    "exp": _exp_dec,
    "factorial": lambda x: Decimal(math.factorial(int(x))) if x >= 0 and x == int(x) else (_ for _ in ()).throw(ValueError("giai thừa chỉ cho số nguyên không âm")),
    "pow": lambda x, y: _pow_dec(x, y),
    "ceil": lambda x: Decimal(math.ceil(float(x))),
    "floor": lambda x: Decimal(math.floor(float(x))),
}

def _pow_dec(a: Decimal, b: Decimal) -> Decimal:
    try:
        # Decimal pow handles int exponent well
        # For non-int, fallback to float then Decimal
        if b == int(b):
            return a.__pow__(b)
        # Try Decimal ln/exp route
        # a^b = exp(b*ln(a)) if a>0
        if a > 0:
            try:
                return (b * _ln_dec(a)).exp()
            except Exception:
                pass
        return Decimal(str(math.pow(float(a), float(b))))
    except Exception as e:
        return Decimal(str(math.pow(float(a), float(b))))

# ── Vietnamese number maps ──
VN_DIGIT = {
    "không": 0, "khong": 0,
    "một": 1, "mot": 1, "mốt": 1, "mot_": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4, "bon": 4, "tư": 4, "tu": 4,
    "năm": 5, "nam": 5, "lăm": 5, "lam": 5,
    "sáu": 6, "sau": 6,
    "bảy": 7, "bay": 7,
    "tám": 8, "tam": 8,
    "chín": 9, "chin": 9,
}
# For checking number word
VN_NUMBER_WORDS = set(VN_DIGIT.keys()) | {
    "mười", "muoi", "mươi", "muoi", "trăm", "tram",
    "nghìn", "nghin", "ngàn", "ngan", "triệu", "trieu", "tỷ", "ty",
    "lẻ", "le", "linh",
    "âm", "am",
    "phẩy", "phay", "chấm", "cham", "phảy", "cham",
}

LARGE_UNITS = {
    "tỷ": 1_000_000_000, "ty": 1_000_000_000,
    "triệu": 1_000_000, "trieu": 1_000_000,
    "nghìn": 1000, "nghin": 1000, "ngàn": 1000, "ngan": 1000,
}

# ── Helpers for Vietnamese parsing ──
def _parse_tens_units(words: list[str]) -> int:
    if not words:
        return 0
    # single
    if len(words) == 1:
        w = words[0]
        if w in ("mười", "muoi"):
            return 10
        return VN_DIGIT.get(w, 0)
    # "mười X"
    if words[0] in ("mười", "muoi"):
        d = VN_DIGIT.get(words[1], 0) if len(words) > 1 else 0
        # "mười lăm" -> 15 (lăm->5)
        return 10 + d
    # "X mươi Y"
    if "mươi" in words or "muoi" in words:
        try:
            idx = words.index("mươi") if "mươi" in words else words.index("muoi")
        except ValueError:
            idx = -1
        if idx != -1:
            tens_word = words[idx-1] if idx >= 1 else None
            tens_digit = VN_DIGIT.get(tens_word, 0) if tens_word else 0
            tens_val = tens_digit * 10
            rest = words[idx+1:]
            if not rest:
                return tens_val
            # rest could be ["mốt"] etc
            unit_val = VN_DIGIT.get(rest[0], 0)
            return tens_val + unit_val
    # "lẻ"/"linh" X
    for cand in ("lẻ", "le", "linh"):
        if cand in words:
            idx = words.index(cand)
            rest = words[idx+1:]
            if rest:
                return VN_DIGIT.get(rest[0], 0)
            return 0
    # fallback: first digit
    return VN_DIGIT.get(words[0], 0)

def _parse_group(words: list[str]) -> int:
    if not words:
        return 0
    # check trăm
    has_tram = "trăm" in words or "tram" in words
    if has_tram:
        try:
            idx = words.index("trăm") if "trăm" in words else words.index("tram")
        except ValueError:
            idx = -1
        hundred_part = words[:idx] if idx != -1 else []
        rest = words[idx+1:] if idx != -1 else []
        if not hundred_part:
            hundred_val = 1
        else:
            # hundred_part usually single digit
            h_word = hundred_part[-1]
            hundred_val = VN_DIGIT.get(h_word, 1)
        tens_units = _parse_tens_units(rest)
        return hundred_val * 100 + tens_units
    else:
        return _parse_tens_units(words)

def parse_vn_integer(words: list[str]) -> int:
    """Parse integer Vietnamese phrase -> int, handles âm, large units."""
    if not words:
        return 0
    # handle âm
    sign = 1
    if words and words[0] in ("âm", "am"):
        sign = -1
        words = words[1:]
        if not words:
            return 0
    # split by large units
    total = 0
    current_group: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in LARGE_UNITS:
            unit_val = LARGE_UNITS[w]
            group_val = _parse_group(current_group) if current_group else 1
            # Special: if group empty -> 1 * unit (e.g., "một tỷ" -> group "một" =1)
            # but "tỷ" alone -> 1e9
            total += group_val * unit_val
            current_group = []
        else:
            current_group.append(w)
        i += 1
    # leftover
    if current_group:
        total += _parse_group(current_group)
    return sign * total

def _parse_fractional(words: list[str]) -> str:
    if not words:
        return ""
    # if contains large units, treat as integer string
    has_large = any(w in LARGE_UNITS or w in ("trăm","tram","mươi","muoi","mười","muoi") for w in words)
    # prefer digit-by-digit if possible
    # digit-by-digit: each word is 0-9 digit
    digit_seq = []
    can_digit = True
    for w in words:
        if w in VN_DIGIT and VN_DIGIT[w] < 10:
            digit_seq.append(str(VN_DIGIT[w]))
        else:
            can_digit = False
            break
    if can_digit and digit_seq:
        # check if words include "mười" etc -> not digit-by-digit
        return "".join(digit_seq)
    if has_large:
        # parse as integer then string
        # e.g., "mười hai" -> 12
        try:
            val = parse_vn_integer(words)
            # for fractional, need to preserve without sign?
            return str(abs(val))
        except Exception:
            pass
    # fallback digit-by-digit ignoring unknowns
    res = ""
    for w in words:
        if w in VN_DIGIT:
            d = VN_DIGIT[w]
            if d < 10:
                res += str(d)
            elif d == 10:
                res += "10"
    return res

def replace_vn_numbers(text: str) -> str:
    """Replace Vietnamese number phrases with digit strings."""
    lower = text.lower()
    # tokenize keeping words
    tokens = re.findall(r"[a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]+", lower)
    # We'll scan for maximal contiguous number-word sequences
    # Build result by iterating tokens and reconstructing with replacements
    # But we need to preserve original text structure for non-number parts.
    # Simpler: find sequences in lower and replace via parsed values using regex scanning directly on lower string.
    # We'll scan tokens index
    i = 0
    out_parts = []
    # We will work on lower string word by word, but to keep operators etc, we will process token by token
    # For simplicity, we will build a new string by replacing sequences in lower.
    # Approach: iterate i, if tokens[i] is number word or "âm"/"phẩy"/"chấm", start sequence
    # else out_parts append tokens[i] (or keep as is)
    # However tokens list loses operators like + - etc. But those are not in [a-z] anyway, so tokens only contains words.
    # To handle full text replacement, we need to work on original lower text via word-boundary regex per sequence.
    # Alternative: use regex to find all number phrase candidates and replace.
    # Let's collect sequences as list of word lists.
    seqs: list[tuple[int,int,list[str]]] = []  # start_idx, end_idx, words
    n = len(tokens)
    idx = 0
    while idx < n:
        if tokens[idx] in VN_NUMBER_WORDS or tokens[idx] in ("phẩy","phay","chấm","cham"):
            start = idx
            seq = []
            while idx < n and (tokens[idx] in VN_NUMBER_WORDS or tokens[idx] in ("phẩy","phay","chấm","cham")):
                # Note: "phẩy" already in set, but keep
                seq.append(tokens[idx])
                idx += 1
            # Only keep seq if it contains at least one digit word or large unit
            has_digit = any(w in VN_DIGIT or w in LARGE_UNITS or w in ("trăm","tram","mươi","muoi","mười","muoi") for w in seq)
            if has_digit:
                seqs.append((start, idx, seq))
            else:
                # not a valid number phrase
                pass
        else:
            idx += 1
    # Now we have sequences, need to replace in text
    # For each seq, parse to number string
    # Build mapping from tuple(seq words) -> number string
    # To avoid overlapping, process longest first
    # Create lower text copy for replacement
    result_text = lower
    # Sort by length descending to replace longer first
    seqs_sorted = sorted(seqs, key=lambda x: len(x[2]), reverse=True)
    for _, _, seq in seqs_sorted:
        # skip if already replaced? We'll check phrase exists in result_text
        phrase = " ".join(seq)
        # Check if phrase still exists (word boundaries)
        # Use re.escape
        # Only replace first occurrence to avoid duplicate replacement issues, loop until no more
        # But seq may appear multiple times, replace all
        if "phẩy" in seq or "phay" in seq or "chấm" in seq or "cham" in seq:
            # split at first phay/cham
            sep_idx = -1
            sep_word = None
            for cand in ("phẩy","phay","chấm","cham"):
                if cand in seq:
                    sep_idx = seq.index(cand)
                    sep_word = cand
                    break
            int_part = seq[:sep_idx]
            frac_part = seq[sep_idx+1:]
            int_val = parse_vn_integer(int_part) if int_part else 0
            frac_str = _parse_fractional(frac_part)
            if frac_str == "":
                frac_str = "0"
            # handle âm in int_part already
            sign = "-" if int_val < 0 else ""
            abs_int = abs(int_val)
            num_str = f"{sign}{abs_int}.{frac_str}"
        else:
            val = parse_vn_integer(seq)
            num_str = str(val)
        # replace phrase in result_text with num_str using word boundaries
        # Use regex: \bphrase\b
        pattern = r"\b" + re.escape(phrase) + r"\b"
        result_text = re.sub(pattern, num_str, result_text)
    return result_text

# ── Helper dùng chung: chuẩn hoá token số Việt / Anh ──
def _normalize_number_token_global(m):
    token = m.group(0)
    if "." in token and "," in token:
        if token.rfind(",") > token.rfind("."):
            return token.replace(".", "").replace(",", ".")
        else:
            return token.replace(",", "")
    elif "," in token:
        if re.match(r"^\d{1,3}(,\d{3})+(\.\d+)?$", token):
            return token.replace(",", "")
        if re.match(r"^\d+,\d+$", token):
            parts = token.split(",")
            if len(parts) == 2 and len(parts[1]) <= 2:
                return token.replace(",", ".")
            elif len(parts) == 2 and len(parts[1]) == 3:
                return token.replace(",", "")
            else:
                return token.replace(",", ".")
        return token.replace(",", ".")
    elif "." in token:
        if re.match(r"^[1-9]\d{0,2}(\.\d{3})+$", token):
            return token.replace(".", "")
        return token
    return token

# ── So sánh số thực ──
_COMPARE_PHRASES = [
    ("lớn hơn hoặc bằng", ">="),
    ("lon hon hoac bang", ">="),
    ("nhỏ hơn hoặc bằng", "<="),
    ("nho hon hoac bang", "<="),
    ("bé hơn hoặc bằng", "<="),
    ("be hon hoac bang", "<="),
    ("lớn hơn", ">"),
    ("lon hon", ">"),
    ("nhỏ hơn", "<"),
    ("nho hon", "<"),
    ("bé hơn", "<"),
    ("be hon", "<"),
    # Hỗ trợ "lớn/nhỏ/bé" đơn (không có "hơn") như "4 lớn 1 hay không"
    ("lớn", ">"),
    ("lon", ">"),
    ("nhỏ", "<"),
    ("nho", "<"),
    ("bé", "<"),
    ("be", "<"),
    ("nhỏ nhất", "min"),  # placeholder, không dùng op nhưng để detect
    ("nho nhat", "min"),
    ("lớn nhất", "max"),
    ("lon nhat", "max"),
    ("bằng nhau", "=="),
    ("bang nhau", "=="),
    ("không bằng", "!="),
    ("khong bang", "!="),
    ("khác", "!="),
    ("khac", "!="),
    ("bằng", "=="),
    ("bang", "=="),
    ("so sánh", "compare"),
    ("so sanh", "compare"),
    ("so với", "compare"),
    ("so voi", "compare"),
]

# Ký hiệu toán học so sánh
_COMPARE_SYMBOLS = [">=", "<=", "==", "!=", "<>", ">", "<", "="]
_COMPARE_SYMBOL_RE = re.compile(r"(>=|<=|==|!=|<>|>|<|=)")

# Từ khóa kích hoạt so sánh (dùng cho is_comparison_question nhanh)
_COMPARISON_KEYWORDS = ["lớn hơn","lon hon","nhỏ hơn","nho hon","bé hơn","be hon","lớn","lon","nhỏ","nho","bé","be","bằng nhau","bang nhau","bằng","bang","không bằng","khong bang","khác","khac","so sánh","so sanh","so với","so voi", ">", "<", "=", "≥", "≤", "≠"]

# ── Math detection ──
_MATH_OP_WORDS = [
    "cộng", "cong", "trừ", "tru", "nhân", "nhan", "chia",
    "mũ", "mu", "lũy thừa", "luy thua", "căn", "can", "căn bậc hai", "can bac hai",
    "căn bậc ba", "can bac ba", "căn bậc", "can bac", "bình phương", "binh phuong", "lập phương", "lap phuong",
    "phần trăm", "phan tram", "phần", "phan", "giai thừa", "giai thua",
    "log", "ln", "logarit", "log10", "log2", "sin", "cos", "tan", "sqrt", "cbrt", "exp",
]
_MATH_SYMBOLS_RE = re.compile(r"[0-9]+(?:[.,][0-9]+)?\s*[\+\-\*/%\^×÷·•∶:²³]+\s*[0-9]+")
_MATH_SYMBOL_CHARS = set("+-*/%^()")
_NON_MATH_CONTEXT_HINTS = [
    "quy định", "quy chế", "quyết định", "sinh viên", "giờ học",
    "giờ thực tập", "nghỉ học", "đến muộn", "đi muộn", "không phép",
    "học tập", "kỷ luật", "vi phạm",
]

_STEPS_KEYWORDS = ["chi tiết", "chi tiet", "từng bước", "tung buoc", "các bước", "cac buoc", "bước", "buoc", "giải thích", "giai thich", "cách làm", "cach lam", "step by step", "show steps", "hiển thị bước", "hien thi buoc"]

def is_math_question(question: str, history=None, last_result=None) -> bool:
    if not question or not question.strip():
        return False
    q_low = question.strip().lower()
    # Không để câu hỏi nghiệp vụ bị bắt nhầm bởi từ khóa toán hoặc metadata
    # như "top_k:3"; chỉ bỏ qua hàng rào này khi có biểu thức toán rõ ràng.
    has_explicit_expression = bool(_MATH_SYMBOLS_RE.search(q_low)) or bool(
        re.search(r"\d\s*[\+\-\*/%\^×÷·•∶:]\s*\d", q_low)
    )
    if any(hint in q_low for hint in _NON_MATH_CONTEXT_HINTS) and not has_explicit_expression:
        return False
    # quick: contains math op words
    for w in _MATH_OP_WORDS:
        if w in q_low:
            # need also numbers or last_result reference
            if any(ch.isdigit() for ch in q_low) or any(kw in q_low for kw in ["kết quả", "ket qua", "kq", "ans", "nó", "no ", "trước"]):
                return True
            # even without digit, if operator word alone with last_result -> math followup like "cộng thêm 2"
            if last_result is not None and any(kw in q_low for kw in ["cộng","cong","trừ","tru","nhân","nhan","chia","mũ","mu","thêm","them","bớt","bot"]):
                return True
            # "căn bậc hai của 16" etc - cả khi chỉ có pi/e không có số (vd: "ln e", "sin pi")
            if w in ("sqrt","cbrt","log","logarit","ln","log10","log2","sin","cos","tan","asin","acos","atan","exp"):
                if any(k in q_low for k in ["pi"," e","e ", "e,","e.","e?","e!"]) or "pi" in q_low:
                    return True
                if any(ch.isdigit() for ch in q_low):
                    return True
            if w in ("căn","can"):
                # "căn" chỉ tính là toán khi có "bậc" hoặc số hoặc sqrt/cbrt (tránh nhầm "căn nhà")
                if any(k in q_low for k in ["bậc","bac","sqrt","cbrt"]) or any(ch.isdigit() for ch in q_low):
                    return True
                if any(k in q_low for k in ["pi"," e","e ", "e,"]):
                    return True
    # contains math symbols with digits (including × ÷)
    if _MATH_SYMBOLS_RE.search(q_low):
        return True
    # check for Unicode math symbols like × ÷ √ · • ∶ : ²³ and factorial ! with digits
    if any(sym in q_low for sym in ["×", "÷", "√", "·", "•", "∶", "²", "³", "％", "！"]) and any(ch.isdigit() for ch in q_low):
        return True
    if re.search(r"\d\s*!\s*($|[^\d])", q_low):
        return True
    # contains expression like "2+3", "3.5*2", "10/2", "5%","2^3" (including × ÷ · • : )
    if re.search(r"\d\s*[\+\-\*/%\^×÷·•∶:²³]\s*\d", q_low):
        return True
    # riêng ":" giữa hai số: "4:2" là chia
    if re.search(r"\d\s*:\s*\d", q_low):
        return True
    # contains parentheses with numbers
    if re.search(r"\(\s*\d", q_low) and re.search(r"\d\s*\)", q_low):
        return True
    # contains Vietnamese number words + operator context, e.g., "hai cộng ba"
    # we can test normalized: if replace_vn_numbers yields digits and operators
    try:
        norm = replace_vn_numbers(q_low)
        # after replace, check if contains digits and at least one operator (symbol or Vietnamese word)
        has_digit = any(ch.isdigit() for ch in norm)
        # Check for symbols OR Vietnamese operator words still present in norm (e.g., "2 cộng 3")
        has_op_symbol = any(ch in "+-*/%^×÷·•∶:²³" for ch in norm) or any(sym in q_low for sym in ["·","•","∶",":","²","³","×","÷","√"])
        has_op_word = any(w in norm for w in ["cộng","cong","trừ","tru","nhân","nhan","chia","mũ","mu","căn","can","phần","phan","giai","bình phương","binh phuong","lập phương","lap phuong"])
        has_op_func = any(kw in norm for kw in ["sqrt","cbrt","log","ln","log10","log2","sin","cos","tan","factorial","exp","pow","logarit"])
        has_op = has_op_symbol or has_op_word or has_op_func
        if has_digit and has_op:
            # ensure not too short false positive like "quy chế 3" -> has digit but no op
            # require operator present
            return True
        # also check if norm contains standalone number and last_result reference + operator words
        # e.g., "cộng thêm 2" with last_result -> norm will be "+ 2" but has op
        # handle case: question is "nhân đôi" etc
        if last_result is not None and has_digit and any(kw in q_low for kw in ["cộng","cong","trừ","tru","nhân","nhan","chia"]):
            return True
    except Exception:
        pass
    # also detect "tính" + number with operator (need operator, not just single number like "tính 123")
    if ("tính" in q_low or "tinh" in q_low) and any(ch.isdigit() for ch in q_low):
        # "tính 2+2" or "tính hai cộng ba" - require operator symbol or word
        if any(op in q_low for op in ["+", "-", "*", "/", "×", "÷", "^", "%", "!"]) or any(w in q_low for w in _MATH_OP_WORDS):
            return True
        # also check VN number words: if q_low contains VN number + op word, already handled via norm earlier
        # For "tính hai cộng ba" without digit, need VN check
        try:
            _norm_tinh = replace_vn_numbers(q_low)
            if any(ch.isdigit() for ch in _norm_tinh) and (any(ch in _norm_tinh for ch in "+-*/%^") or any(w in _norm_tinh for w in ["cộng","cong","trừ","tru","nhân","nhan","chia","mũ","mu","căn","can"])):
                return True
        except Exception:
            pass
    # "bằng bao nhiêu" with numbers and operators
    if "bằng" in q_low and any(ch.isdigit() for ch in q_low) and any(ch in "+-*/" for ch in q_low):
        return True
    return False

def is_math_request_steps(question: str, style: str | None = None) -> bool:
    if style in ("detailed", "step_by_step"):
        return True
    if not question:
        return False
    q_low = question.lower()
    for kw in _STEPS_KEYWORDS:
        if kw in q_low:
            return True
    return False

# ═══════════════════════════════════════════════════════════════
# ── So sánh số thực (không giải thích kiểu gọi hàm) ──
# ═══════════════════════════════════════════════════════════════
def _has_comparison_keyword(q_low: str) -> bool:
    for kw in _COMPARISON_KEYWORDS:
        if kw.lower() in q_low:
            return True
    return False

def is_comparison_question(question: str, history=None, last_result=None) -> bool:
    """Phát hiện câu hỏi so sánh số thực.
    Ví dụ: '2,5 lớn hơn 3 không?', 'so sánh 2.5 và 3.7', '2.5 > 3.5 ?', 'hai phẩy năm có nhỏ hơn ba không'
    Không nhầm với RAG: 'quy chế 3' -> False
    """
    if not question or not question.strip():
        return False
    q_low = question.strip().lower()
    # Nếu là câu tính dạng "1+1 = mấy" / "1+1 =" / "2+2 = bao nhiêu" (dấu = ở cuối không có số bên phải) thì không phải so sánh
    # Đây là dạng "tính kết quả", không phải "so sánh"
    if re.search(r"=\s*(mấy|may|bao nhiêu|bao nhieu|thì sao|thi sao|gì|gi|sao|\?)?\s*$", q_low):
        before_eq = q_low.split("=")[0]
        if re.search(r"\d\s*[\+\-\*/%×÷\^]\s*\d", before_eq):
            after_eq = q_low.split("=", 1)[-1]
            after_clean = re.sub(r"^\s*(mấy|may|bao nhiêu|bao nhieu|thì sao|thi sao|gì|gi|sao|\?|\s)*", "", after_eq, flags=re.IGNORECASE)
            after_clean = after_clean.strip(" ?.!")
            if not after_clean or not re.search(r"\d", after_clean):
                return False
        # Trường hợp "1+1 =" không có gì sau = cũng không phải so sánh
        if q_low.strip().endswith("="):
            before_eq2 = q_low.rsplit("=", 1)[0]
            if re.search(r"\d\s*[\+\-\*/%×÷\^]\s*\d", before_eq2):
                after_part = q_low.rsplit("=", 1)[-1].strip()
                if not after_part or not re.search(r"\d", after_part):
                    return False
    # Chứa ký hiệu so sánh unicode/ASCII với số
    if any(sym in q_low for sym in ["≥", "≤", "≠"]) and any(ch.isdigit() for ch in q_low):
        return True
    # Regex số + so sánh + số
    if re.search(r"-?\d+(?:[.,]\d+)?\s*(>=|<=|==|!=|<>|>|<|=)\s*-?\d+(?:[.,]\d+)?", q_low):
        return True
    # Regex theo token đã chuẩn hoá VN: cần thay số chữ trước
    try:
        tmp = q_low
        # thay sớm các cụm so sánh để tránh bị replace_vn_numbers làm sai
        for phrase, op in sorted(_COMPARE_PHRASES, key=lambda x: len(x[0]), reverse=True):
            if op == "compare":
                continue
            if phrase in tmp:
                # chỉ check tồn tại + số
                if any(ch.isdigit() for ch in tmp) or any(w in tmp for w in VN_DIGIT):
                    # có phrase và có số (hoặc last_result)
                    if any(ch.isdigit() for ch in tmp) or last_result is not None:
                        # nếu phrase là lớn hơn/nhỏ hơn/bằng... và có số -> là so sánh
                        return True
        # Dùng replace_vn_numbers để đếm số
        norm_vn = replace_vn_numbers(q_low)
        has_digit = any(ch.isdigit() for ch in norm_vn)
        has_cmp_word = any(w in norm_vn for w in ["lớn hơn","lon hon","nhỏ hơn","nho hon","bé hơn","be hon","bằng","bang","không bằng","khong bang","khác","khac","so sánh","so sanh","so với","so voi"])
        has_cmp_sym = any(sym in norm_vn for sym in [">","<","="])
        # Có 2 số và 1 từ so sánh -> so sánh
        if has_digit and (has_cmp_word or has_cmp_sym):
            # cần ít nhất 1 từ so sánh hoặc ký hiệu và có số
            # Đếm số lượng số trong norm_vn
            nums = re.findall(r"-?\d+(?:\.\d+)?", norm_vn)
            if len(nums) >= 2 and has_cmp_word:
                return True
            if len(nums) >= 2 and has_cmp_sym:
                return True
            # 1 số + last_result và từ so sánh -> "nó có lớn hơn 5 không?"
            if last_result is not None and has_cmp_word and has_digit:
                if any(kw in q_low for kw in ["nó","no ","kết quả","ket qua","kq","ans"]):
                    return True
                # chỉ 1 số nhưng có so sánh + last_result implied?
                if len(nums) >= 1 and has_cmp_word:
                    return True
            # 1 số + "so sánh ... và ..." pattern: "so sánh 2,5 và 3"
            if has_digit and "so sánh" in norm_vn or "so sanh" in norm_vn or "so với" in norm_vn or "so voi" in norm_vn:
                if len(nums) >= 2:
                    return True
                if len(nums) >= 1 and last_result is not None:
                    return True
        # Trường hợp "2,5 và 3,5 số nào lớn hơn" -> chứa lớn hơn + 2 số
        if has_digit and has_cmp_word:
            nums = re.findall(r"-?\d+(?:\.\d+)?", norm_vn)
            if len(nums) >= 2:
                return True
        # Trường hợp số VN thuần + so sánh chữ không có digit ban đầu nhưng sau replace có digit
        # ví dụ "hai phẩy năm có lớn hơn ba không" -> norm_vn = "2.5 có lớn hơn 3 không"
        if "lớn hơn" in norm_vn or "lon hon" in norm_vn or "nhỏ hơn" in norm_vn or "nho hon" in norm_vn or "bằng" in norm_vn or "bang" in norm_vn:
            if has_digit:
                return True
    except Exception:
        pass
    # Fallback: q_low chứa từ so sánh + có số (kể cả chữ) và không phải RAG đơn số
    if _has_comparison_keyword(q_low):
        # Kiểm tra có số (digit hoặc từ số VN)
        has_digit_raw = any(ch.isdigit() for ch in q_low)
        has_vn_num = any(w in q_low for w in VN_DIGIT) or any(w in q_low for w in ["mười","muoi","mươi","trăm","tram","nghìn","nghin","triệu","trieu","tỷ","ty"])
        if has_digit_raw or has_vn_num:
            # Loại trừ false positive "quy chế 3 lớn hơn..." không, nhưng nếu không có từ RAG chuyên môn thì ok
            # Nếu câu chứa chuyên môn RAG như "quy chế" thì không phải so sánh số thuần -> không nhận
            rag_hints = ["quy chế","quy định","quyết định","thông báo","tài liệu","học viện"]
            if any(h in q_low for h in rag_hints):
                return False
            # Cần ít nhất 2 thực thể số hoặc 1 số + last_result
            try:
                n = replace_vn_numbers(q_low)
                cnt = len(re.findall(r"-?\d+(?:\.\d+)?", n))
                if cnt >= 2:
                    return True
                if cnt >= 1 and last_result is not None:
                    return True
            except Exception:
                pass
            # Nếu có ký hiệu so sánh rõ ràng thì 1 số cũng đủ khi có last_result
            if any(s in q_low for s in [">","<","=","≥","≤","≠"]):
                return True
            if has_digit_raw and any(kw in q_low for kw in ["lớn hơn","nhỏ hơn","bé hơn","bằng"]):
                return True
    return False

def _extract_decimal(expr_str: str) -> Decimal:
    """Eval biểu thức số đơn giản hoặc số thuần -> Decimal, dùng safe_eval nếu có operator."""
    s = expr_str.strip()
    if not s:
        raise ValueError("biểu thức rỗng")
    # Nếu chỉ là số (có thể âm) thì Decimal trực tiếp
    if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        return Decimal(s)
    # Nếu chứa ký tự toán học thì dùng safe_eval
    if any(ch in s for ch in "+-*/%()"):
        # Loại bỏ khoảng trắng thừa
        return safe_eval(s)
    # Thử Decimal
    try:
        return Decimal(s)
    except Exception:
        return safe_eval(s)

def _normalize_comparison_raw(question: str, last_result=None) -> str:
    """Chuẩn hoá thô câu so sánh trước khi tách toán hạng.
    Trả về chuỗi đã thay cụm so sánh -> ký hiệu, thay số VN, chuẩn hoá ,/. -> .
    """
    q_low = question.strip().lower()
    norm = q_low
    # Unicode compare symbols
    norm = norm.replace("≥", " >= ").replace("≤", " <= ").replace("≠", " != ").replace("＝", " = ")
    # Loại bỏ hạt "không" cuối câu hỏi (tránh bị biến thành số 0) - phải làm trước khi thay số VN
    norm = re.sub(r"\s+đúng\s+không\s*[\?\.\!]*\s*$", " ", norm)
    norm = re.sub(r"\s+dung\s+khong\s*[\?\.\!]*\s*$", " ", norm)
    norm = re.sub(r"\s+phải\s+không\s*[\?\.\!]*\s*$", " ", norm)
    norm = re.sub(r"\s+phai\s+khong\s*[\?\.\!]*\s*$", " ", norm)
    norm = re.sub(r"\s+không\s*[\?\.\!]*\s*$", " ", norm)
    norm = re.sub(r"\s+khong\s*[\?\.\!]*\s*$", " ", norm)
    # Thay số VN trước cụm so sánh để tránh gộp số khi "lớn hơn" -> ">" làm mất delimiter
    try:
        norm = replace_vn_numbers(norm)
    except Exception:
        pass
    # Thay cụm so sánh dài trước (sau khi đã thay số VN)
    for phrase, op in sorted(_COMPARE_PHRASES, key=lambda x: len(x[0]), reverse=True):
        if op in ("min","max"):
            continue
        if op == "compare":
            pattern = r"\b" + re.escape(phrase) + r"\b"
            norm = re.sub(pattern, " compare ", norm)
        else:
            pattern = r"\b" + re.escape(phrase) + r"\b"
            norm = re.sub(pattern, f" {op} ", norm)
    # Xử lý "âm 2,5" / "am 2.5" dạng số chữ + số Ả Rập (replace_vn_numbers chỉ xử chữ)
    norm = re.sub(r"\bâm\s+(-?\d)", r"-\1", norm)
    norm = re.sub(r"\bam\s+(-?\d)", r"-\1", norm)
    norm = re.sub(r"\bâm\s*-\s*(\d)", r"-\1", norm)
    norm = re.sub(r"\bam\s*-\s*(\d)", r"-\1", norm)
    # Thay last_result
    if last_result is not None:
        lr_str = str(last_result).strip()
        for ph in ["kết quả trước","ket qua truoc","kết quả vừa rồi","ket qua vua roi","kết quả","ket qua","đáp án trước","dap an truoc","kq"]:
            if ph in norm:
                norm = norm.replace(ph, f" {lr_str} ")
        # nó / no chỉ khi có ngữ cảnh so sánh
        if any(kw in norm for kw in [">","<","=","compare"]):
            norm = re.sub(r"\bnó\b", f" {lr_str} ", norm)
            norm = re.sub(r"\bno\b", f" {lr_str} ", norm)
        norm = re.sub(r"\bans\b", f" {lr_str} ", norm)
    # Chuẩn hoá ,/. theo token
    norm = re.sub(r"\d[\d\.,]*", _normalize_number_token_global, norm)
    # Chuẩn hoá khoảng trắng
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm

def normalize_comparison(question: str, last_result=None):
    """Tách câu so sánh -> (left_str, op, right_str, norm) hoặc None nếu không tách được.
    left_str/right_str là chuỗi biểu thức số (đã chuẩn hoá dấu chấm), op in {>,<,>=,<=,==,!=,compare}
    """
    norm = _normalize_comparison_raw(question, last_result)
    # Ưu tiên tìm op显式
    m = _COMPARE_SYMBOL_RE.search(norm)
    if m:
        op = m.group(1)
        if op == "=":
            op = "=="
        elif op == "<>":
            op = "!="
        left_raw = norm[:m.start()].strip()
        right_raw = norm[m.end():].strip()
        left_expr = None
        right_expr = None
        # Tìm biểu thức số: hỗ trợ "2.5", "-2.5", "2*3", "2.5*2", "(2+3)"
        expr_pat = r"-?\d+(?:\.\d+)?(?:\s*[\+\-\*/%]\s*-?\d+(?:\.\d+)?)*"
        # left: lấy biểu thức cuối cùng trong left_raw (bỏ filler chữ)
        left_cands = re.findall(expr_pat, left_raw)
        if left_cands:
            left_expr = left_cands[-1].strip()
        else:
            single = re.findall(r"-?\d+(?:\.\d+)?", left_raw)
            if single:
                left_expr = single[-1]
        # right: lấy biểu thức đầu tiên trong right_raw
        right_cands = re.findall(expr_pat, right_raw)
        if right_cands:
            right_expr = right_cands[0].strip()
        else:
            single = re.findall(r"-?\d+(?:\.\d+)?", right_raw)
            if single:
                right_expr = single[0]
        # Fallback vị trí nếu vẫn thiếu
        if not left_expr or not right_expr:
            num_positions = [(mm.group(), mm.start(), mm.end()) for mm in re.finditer(r"-?\d+(?:\.\d+)?", norm)]
            op_pos = m.start()
            left_cands2 = [v for v, s, e in num_positions if e <= op_pos]
            right_cands2 = [v for v, s, e in num_positions if s >= m.end()]
            if not left_expr and left_cands2:
                left_expr = left_cands2[-1]
            if not right_expr and right_cands2:
                right_expr = right_cands2[0]
        if left_expr and right_expr:
            left_expr = left_expr.strip()
            right_expr = right_expr.strip()
            return (left_expr, op, right_expr, norm)
        # Nếu không đủ 2 vế nhưng có thể là dạng "A và B >" (op ở cuối) -> fallback generic
        # Thử tìm 2 số trong norm và op ở cuối
        # Ví dụ "2.5 và 3.5 >" left_raw chứa cả 2 số, right_raw rỗng
        if not right_expr and left_expr:
            # Có thể left_raw chứa 2 biểu thức: "1+1 và 2+2" hoặc "2.5 và 3.5"
            exprs_in_left = re.findall(expr_pat, left_raw)
            if len(exprs_in_left) >= 2:
                return (exprs_in_left[-2].strip(), op, exprs_in_left[-1].strip(), norm)
            nums_in_left = re.findall(r"-?\d+(?:\.\d+)?", left_raw)
            if len(nums_in_left) >= 2:
                return (nums_in_left[-2], op, nums_in_left[-1], norm)
    # Không có op显式 hoặc không tách được -> thử generic "compare" hoặc hai số + từ khóa
    if "compare" in norm:
        expr_pat_cmp = r"-?\d+(?:\.\d+)?(?:\s*[\+\-\*/%]\s*-?\d+(?:\.\d+)?)*"
        exprs = re.findall(expr_pat_cmp, norm)
        if len(exprs) >= 2:
            cmp_pos = norm.find("compare")
            before = norm[:cmp_pos]
            after = norm[cmp_pos+len("compare"):]
            exprs_before = re.findall(expr_pat_cmp, before)
            exprs_after = re.findall(expr_pat_cmp, after)
            if len(exprs_after) >= 2:
                return (exprs_after[0].strip(), "compare", exprs_after[1].strip(), norm)
            if len(exprs_after) == 1 and exprs_before:
                return (exprs_before[-1].strip(), "compare", exprs_after[0].strip(), norm)
            if len(exprs) >= 2:
                return (exprs[0].strip(), "compare", exprs[1].strip(), norm)
        nums = re.findall(r"-?\d+(?:\.\d+)?", norm)
        if len(nums) >= 2:
            # Lấy 2 số đầu sau compare hoặc 2 số cuối nếu không rõ
            # Ưu tiên 2 số gần compare nhất
            # Tìm vị trí compare
            cmp_pos = norm.find("compare")
            before = norm[:cmp_pos]
            after = norm[cmp_pos+len("compare"):]
            nums_before = re.findall(r"-?\d+(?:\.\d+)?", before)
            nums_after = re.findall(r"-?\d+(?:\.\d+)?", after)
            if len(nums_after) >= 2:
                return (nums_after[0], "compare", nums_after[1], norm)
            if len(nums_after) == 1 and nums_before:
                return (nums_before[-1], "compare", nums_after[0], norm)
            if len(nums) >= 2:
                return (nums[0], "compare", nums[1], norm)
    # Generic: có 2 số và có từ so sánh nhưng không có ký hiệu -> dạng "2,5 và 3,5 số nào lớn hơn"
    # Nếu norm chứa số và chứa từ lớn/nhỏ/bằng nhưng không có op symbol, coi như generic compare
    # Thử với biểu thức trước (để bắt "1+1 và 2+2")
    expr_pat_gen = r"-?\d+(?:\.\d+)?(?:\s*[\+\-\*/%]\s*-?\d+(?:\.\d+)?)*"
    exprs_gen = re.findall(expr_pat_gen, norm)
    has_cmp_word = any(w in norm for w in ["lớn hơn","lon hon","nhỏ hơn","nho hon","bé hơn","be hon","lớn","lon","nhỏ","nho","bé","be","bằng","bang","không bằng","khong bang","khác","khac"])
    if len(exprs_gen) >= 2 and has_cmp_word:
        return (exprs_gen[0].strip(), "compare", exprs_gen[1].strip(), norm)
    nums = re.findall(r"-?\d+(?:\.\d+)?", norm)
    has_cmp_word2 = any(w in norm for w in ["lớn hơn","lon hon","nhỏ hơn","nho hon","bé hơn","be hon","bằng","bang","không bằng","khong bang","khác","khac"])
    # Nếu không có has_cmp_word nhưng có "compare" đã xử lý, còn lại chỉ còn trường hợp op bị miss
    if len(nums) >= 2 and has_cmp_word2:
        # Trả về generic để caller tự so và diễn đạt tự nhiên
        # Nhưng thử tìm op ẩn: nếu có "lớn hơn" thì caller đã chuyển thành > và đã xử lý ở nhánh trên
        # Nếu vẫn vào đây nghĩa là op chưa được chuyển (do thiếu khoảng trắng) -> thử lại
        # Fallback generic
        return (nums[0], "compare", nums[1], norm)
    # Trường hợp "2,5 và 3,5" không có từ khóa nhưng có 2 số và câu hỏi "so sánh" đã bị chuyển thành compare ở trên -> đã return
    # Nếu chỉ có 2 số và câu hỏi chứa "?" và có ý so sánh ngầm? Không nhận
    return None

def _describe_relation(left: Decimal, right: Decimal) -> tuple[str, str]:
    """Trả về (vi_text, symbol) cho quan hệ thực tế giữa left và right."""
    if left > right:
        return ("lớn hơn", ">")
    elif left < right:
        return ("nhỏ hơn", "<")
    else:
        return ("bằng", "==")

def _format_comparison_answer(left_str: str, op: str, right_str: str, left_dec: Decimal, right_dec: Decimal, style: str | None, question: str) -> str:
    """Tạo câu trả lời tự nhiên, không kiểu gọi hàm.
    left_str/right_str là chuỗi biểu thức đã chuẩn hoá (hiển thị), left_dec/right_dec là giá trị Decimal.
    op là op yêu cầu (có thể là compare generic).
    style: plain -> cực ngắn, không ký hiệu thừa.
    """
    # Format số hiển thị: dùng format_decimal để bỏ số 0 thừa
    def fmt(d: Decimal) -> str:
        return format_decimal(d)
    left_fmt = fmt(left_dec)
    right_fmt = fmt(right_dec)
    # Giữ nguyên left_str/right_str nếu là biểu thức phức tạp thì hiển thị biểu thức đó, nhưng fmt đã là giá trị
    # Để hiển thị thân thiện, dùng left_fmt/right_fmt
    actual_vi, actual_sym = _describe_relation(left_dec, right_dec)
    # Chuẩn hoá op để so
    op_norm = op
    if op_norm == "=":
        op_norm = "=="
    # Mapping vi cho op yêu cầu
    op_vi_map = {
        ">": "lớn hơn",
        "<": "nhỏ hơn",
        ">=": "lớn hơn hoặc bằng",
        "<=": "nhỏ hơn hoặc bằng",
        "==": "bằng",
        "!=": "khác",
        "compare": "so với",
    }
    op_vi = op_vi_map.get(op_norm, actual_vi)
    # Đánh giá đúng/sai nếu là so sánh có op cụ thể
    is_generic = (op_norm == "compare")
    if is_generic:
        # Câu dạng "so sánh 2,5 và 3,5" -> trả về quan hệ thực tế
        if style == "plain":
            return f"Kết quả là {left_fmt} {actual_vi} {right_fmt}."
        else:
            # Hiển thị thêm ký hiệu cho rõ nhưng vẫn tự nhiên
            if actual_sym == "==":
                return f"{left_fmt} bằng {right_fmt}."
            else:
                return f"{left_fmt} {actual_vi} {right_fmt} ({left_fmt} {actual_sym} {right_fmt})."
    else:
        # So sánh có điều kiện: ví dụ "2,5 lớn hơn 3 không?"
        # Tính kết quả boolean
        res = False
        if op_norm == ">":
            res = left_dec > right_dec
        elif op_norm == "<":
            res = left_dec < right_dec
        elif op_norm == ">=":
            res = left_dec >= right_dec
        elif op_norm == "<=":
            res = left_dec <= right_dec
        elif op_norm == "==":
            res = left_dec == right_dec
        elif op_norm == "!=":
            res = left_dec != right_dec
        if style == "plain":
            if res:
                return f"Kết quả là {left_fmt} {op_vi} {right_fmt} là đúng."
            else:
                # Khi sai: nói là sai ngay và đính chính quan hệ thực tế
                return f"Sai, {left_fmt} {actual_vi} {right_fmt} ({left_fmt} {actual_sym} {right_fmt}), nên \"{left_fmt} {op_norm} {right_fmt}\" là sai."
        else:
            if res:
                return f"Đúng, {left_fmt} {op_vi} {right_fmt} ({left_fmt} {op_norm} {right_fmt})."
            else:
                # Khi sai: nói là sai ngay và đính chính, không nói đúng rồi mới sửa
                return f"Sai, {left_fmt} {actual_vi} {right_fmt} ({left_fmt} {actual_sym} {right_fmt}), không {op_vi} {right_fmt}."

def compare_real_numbers(question: str, last_result=None, style=None, want_steps: bool = False) -> dict:
    """So sánh hai số thực (hoặc biểu thức số) với độ chính xác Decimal.
    Returns dict: {success: bool, left: str, right: str, op: str, result: bool|None, answer: str, expression: str, error: str|None}
    - answer là câu tự nhiên, không kiểu gọi hàm.
    """
    try:
        parsed = normalize_comparison(question, last_result)
        if not parsed:
            return {"success": False, "error": "không trích được cặp số để so sánh", "expression": "", "answer": None, "result": None}
        left_str, op, right_str, norm = parsed
        # Kiểm tra độ dài
        if len(norm) > 200:
            return {"success": False, "error": "biểu thức quá dài", "expression": norm, "answer": None, "result": None}
        # Eval hai vế
        try:
            left_dec = _extract_decimal(left_str)
        except Exception as e:
            return {"success": False, "error": f"vế trái không hợp lệ: {e}", "expression": norm, "answer": None, "result": None}
        try:
            right_dec = _extract_decimal(right_str)
        except Exception as e:
            return {"success": False, "error": f"vế phải không hợp lệ: {e}", "expression": norm, "answer": None, "result": None}
        # Tính boolean nếu có op cụ thể
        op_norm = op
        if op_norm == "=":
            op_norm = "=="
        is_generic = (op_norm == "compare")
        bool_res = None
        if not is_generic:
            if op_norm == ">":
                bool_res = left_dec > right_dec
            elif op_norm == "<":
                bool_res = left_dec < right_dec
            elif op_norm == ">=":
                bool_res = left_dec >= right_dec
            elif op_norm == "<=":
                bool_res = left_dec <= right_dec
            elif op_norm == "==":
                bool_res = left_dec == right_dec
            elif op_norm == "!=":
                bool_res = left_dec != right_dec
            else:
                bool_res = None
        # Tạo answer tự nhiên
        answer = _format_comparison_answer(left_str, op, right_str, left_dec, right_dec, style, question)
        # expression hiển thị cho debug (không lộ hàm)
        if is_generic:
            expr_display = f"{format_decimal(left_dec)} so với {format_decimal(right_dec)}"
        else:
            expr_display = f"{format_decimal(left_dec)} {op_norm} {format_decimal(right_dec)}"
        # steps chỉ khi yêu cầu và không plain
        steps = None
        if want_steps and style != "plain":
            if style in ("detailed","step_by_step"):
                actual_vi, actual_sym = _describe_relation(left_dec, right_dec)
                steps = f"So sánh: {format_decimal(left_dec)} {actual_sym} {format_decimal(right_dec)} ({format_decimal(left_dec)} {actual_vi} {format_decimal(right_dec)}). Độ chính xác Decimal prec=50."
            else:
                actual_vi, actual_sym = _describe_relation(left_dec, right_dec)
                steps = f"{format_decimal(left_dec)} {actual_sym} {format_decimal(right_dec)}"
        return {
            "success": True,
            "left": format_decimal(left_dec),
            "right": format_decimal(right_dec),
            "op": op_norm,
            "result": bool_res,
            "answer": answer,
            "expression": expr_display,
            "steps": steps,
            "error": None,
            "left_raw": left_str,
            "right_raw": right_str,
            "norm": norm,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "expression": "", "answer": None, "result": None}

# ── Expression normalization ──
_OP_MAP_SINGLE = {
    "×": "*",
    "÷": "/",
    "√": "sqrt",
    "²": " **2 ",
    "³": " **3 ",
    "·": " * ",
    "•": " * ",
    "⋅": " * ",
    "∶": " / ",
    "＋": "+",
    "－": "-",
    "–": "-",
    "—": "-",
    "−": "-",
    "％": "%",
    "！": "!",
}

_OP_PHRASES = [
    ("căn bậc hai", "sqrt"),
    ("can bac hai", "sqrt"),
    ("căn bậc ba", "cbrt"),
    ("can bac ba", "cbrt"),
    ("bình phương", "**2"),
    ("binh phuong", "**2"),
    ("lập phương", "**3"),
    ("lap phuong", "**3"),
    ("lũy thừa", "**"),
    ("luy thua", "**"),
    ("phần trăm", "%"),
    ("phan tram", "%"),
    ("giai thừa", "!"),
    ("giai thua", "!"),
    ("mũ", "**"),
    ("mu", "**"),
    ("cộng", "+"),
    ("cong", "+"),
    ("trừ", "-"),
    ("tru", "-"),
    ("nhân", "*"),
    ("nhan", "*"),
    ("chia", "/"),
    ("căn", "sqrt"),
    ("can", "sqrt"),
]

# For handling "x" as multiplication only when between numbers/words
# We'll handle via phrase map after numbers replaced

def normalize_expression(question: str, last_result: Decimal | str | None = None) -> str:
    """Chuẩn hóa câu hỏi thành biểu thức toán học."""
    q = question.strip()
    q_low = q.lower()

    # Early: handle single-char symbols before VN number replacement
    norm = q_low
    for k, v in _OP_MAP_SINGLE.items():
        norm = norm.replace(k, f" {v} ")

    # Handle "căn bậc hai/ba", "bình phương", "lập phương", "lũy thừa", "phần trăm", "giai thừa" BEFORE VN number replacement
    # to avoid "hai" in "căn bậc hai" being replaced to "2"
    _early_phrases = [
        ("căn bậc hai", "sqrt"),
        ("can bac hai", "sqrt"),
        ("căn bậc ba", "cbrt"),
        ("can bac ba", "cbrt"),
        ("căn bậc 2", "sqrt"),
        ("can bac 2", "sqrt"),
        ("căn bậc 3", "cbrt"),
        ("can bac 3", "cbrt"),
        ("bình phương", "**2"),
        ("binh phuong", "**2"),
        ("lập phương", "**3"),
        ("lap phuong", "**3"),
        ("lũy thừa", "**"),
        ("luy thua", "**"),
        ("phần trăm", "%"),
        ("phan tram", "%"),
        ("giai thừa", "!"),
        ("giai thua", "!"),
    ]
    for phrase, repl in sorted(_early_phrases, key=lambda x: len(x[0]), reverse=True):
        pattern = r"\b" + re.escape(phrase) + r"\b"
        norm = re.sub(pattern, f" {repl} ", norm)

    # Xử lý căn bậc N dạng số/chữ còn lại sau khi đã thay chữ -> số (vd: "căn bậc hai" đã thành sqrt, còn "căn bậc 2" đã thành sqrt, nhưng "căn bậc 4 của 16" cần thành pow)
    # Để bắt cả "căn bậc 4 của 16" dạng chữ chưa chuyển: dùng regex sau khi đã replace VN numbers
    # Tạm giữ để xử lý sau khi replace VN numbers

    # Now replace Vietnamese numbers
    try:
        norm = replace_vn_numbers(norm)
    except Exception:
        pass

    # ── Xử lý căn bậc N tổng quát sau khi đã thay số chữ -> số (để bắt cả "căn bậc 4 của 16", "can bac 4 cua 16") ──
    # Ví dụ: "căn bậc 2 của 4" -> "sqrt(4)", "căn bậc 3 của 27" -> "cbrt(27)", "căn bậc 4 của 16" -> "(16)**(1/4)"
    try:
        # Pattern với số N và số X (có thể có dấu ngoặc hoặc số thập phân)
        def _root_repl(m):
            n_str = m.group(1).strip()
            x_str = m.group(2).strip()
            # Chuẩn hoá x_str: thay , -> . nếu cần (để Decimal hiểu)
            # Giữ nguyên để safe_eval xử lý sau, nhưng cần đảm bảo x_str là số
            # Nếu N là 2 -> sqrt, 3 -> cbrt, khác -> pow
            try:
                n_val = int(float(n_str.replace(",", ".")))
            except:
                n_val = None
            if n_val == 2:
                return f" sqrt({x_str}) "
            elif n_val == 3:
                return f" cbrt({x_str}) "
            elif n_val is not None and n_val != 0:
                # Dùng pow: (x)**(1/n)
                return f" ({x_str})**(1/{n_val}) "
            else:
                return m.group(0)
        # Bắt "căn bậc N của X" hoặc "can bac N cua X" với N là số, X là số (có thể có ngoặc, dấu chấm/phẩy) - hỗ trợ cả có dấu và không dấu
        norm = re.sub(r"\b(?:căn|can)\s*(?:bậc|bac)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:của|cua)?\s*\(?\s*(-?\d+(?:[.,]\d+)?)\s*\)?", _root_repl, norm)
        # Bắt "căn bậc N X" không có "của" (vd: "căn bậc 2 4")
        # Đã xử lý ở trên vì "của" là optional, nên cover luôn
        # Xử lý "sqrt bậc N của X" còn sót sau khi "căn bậc 2" đã thành "sqrt" (do early replace): "sqrt bậc 2 của 4" -> "sqrt(4)"
        norm = re.sub(r"\bsqrt\s*(?:bậc|bac)\s*2\s*(?:của|cua)?\s*\(?\s*(-?\d+(?:[.,]\d+)?)\s*\)?", r" sqrt(\1) ", norm)
        norm = re.sub(r"\bsqrt\s*(?:bậc|bac)\s*3\s*(?:của|cua)?\s*\(?\s*(-?\d+(?:[.,]\d+)?)\s*\)?", r" cbrt(\1) ", norm)
        norm = re.sub(r"\bcbrt\s*(?:bậc|bac)\s*3\s*(?:của|cua)?\s*\(?\s*(-?\d+(?:[.,]\d+)?)\s*\)?", r" cbrt(\1) ", norm)
        # Dọn "bậc" còn sót sau khi căn đã thành sqrt: "sqrt bậc 4 của 16" -> "(16)**(1/4)"
        def _sqrt_bac_repl(m):
            n_str = m.group(1).strip()
            x_str = m.group(2).strip()
            try:
                n_val = int(float(n_str.replace(",", ".")))
            except:
                return m.group(0)
            if n_val == 2:
                return f" sqrt({x_str}) "
            elif n_val == 3:
                return f" cbrt({x_str}) "
            elif n_val != 0:
                return f" ({x_str})**(1/{n_val}) "
            return m.group(0)
        norm = re.sub(r"\bsqrt\s*(?:bậc|bac)\s*([0-9]+)\s*(?:của|cua)?\s*\(?\s*(-?\d+(?:[.,]\d+)?)\s*\)?", _sqrt_bac_repl, norm)
    except Exception:
        pass

    # Handle last_result references: "kết quả", "ket qua", "kq", "ans", "nó"
    if last_result is not None:
        lr_str = str(last_result)
        for phrase in ["kết quả trước", "ket qua truoc", "kết quả vừa rồi", "ket qua vua roi", "kết quả", "ket qua", "đáp án trước", "dap an truoc", "kq"]:
            if phrase in norm:
                norm = norm.replace(phrase, lr_str)
        # "nó" as last result only if math context
        if " nó" in norm or norm.strip() == "nó" or norm.startswith("nó ") or " nó " in norm:
            if any(ch in norm for ch in "+-*/%^") or any(w in norm for w in ["cộng","cong","trừ","tru","nhân","nhan","chia","mũ","mu"]):
                norm = re.sub(r"\bnó\b", lr_str, norm)
                norm = re.sub(r"\bno\b", lr_str, norm)
        norm = re.sub(r"\bans\b", lr_str, norm)
        # Also handle "nó" after operator mapping may have become symbol, so also check standalone "nó"
        # Already done; if norm was "cộng thêm 5" with lr, we need to handle later

    # Handle compound "cộng thêm", "trừ bớt", "nhân với", "chia cho" BEFORE individual operators to avoid "++"
    norm = re.sub(r"\b(cộng|cong)\s+(thêm|them|với|voi)\b", " + ", norm)
    norm = re.sub(r"\b(trừ|tru)\s+(đi|di|bớt|bot)\b", " - ", norm)
    norm = re.sub(r"\b(nhân|nhan)\s+(với|voi|thêm|them)\b", " * ", norm)
    norm = re.sub(r"\b(chia|chia)\s+(cho|với|voi)\b", " / ", norm)

    # Replace remaining operator phrases (longest first) - after compounds
    _remaining_phrases = [
        ("mũ", "**"),
        ("mu", "**"),
        ("cộng", "+"),
        ("cong", "+"),
        ("trừ", "-"),
        ("tru", "-"),
        ("nhân", "*"),
        ("nhan", "*"),
        ("chia", "/"),
        ("căn", "sqrt"),
        ("can", "sqrt"),
    ]
    for phrase, repl in sorted(_remaining_phrases, key=lambda x: len(x[0]), reverse=True):
        pattern = r"\b" + re.escape(phrase) + r"\b"
        norm = re.sub(pattern, f" {repl} ", norm)

    # Single "thêm"/"bớt" as operators (only if not already preceded by operator to avoid "++")
    norm = re.sub(r"(?<![\+\-\*/])\s*\bthêm\b\s*", " + ", norm)
    norm = re.sub(r"(?<![\+\-\*/])\s*\bthem\b\s*", " + ", norm)
    norm = re.sub(r"(?<![\+\-\*/])\s*\bbớt\b\s*", " - ", norm)
    norm = re.sub(r"(?<![\+\-\*/])\s*\bbot\b\s*", " - ", norm)

    # Implicit last_result: bare " + 5" , " * 2" etc -> prepend last_result
    if last_result is not None:
        stripped = norm.strip()
        if stripped and stripped[0] in "+-*/%":
            # Check if starts with operator (unary or binary) and last_result exists -> prepend
            # Avoid prepending if it's already like " -5 + 3" (negative number) - but " -5" alone would be unary
            # Heuristic: if starts with + or * / % or - followed by space/digit, prepend
            # For "-" we need to distinguish negative number vs binary op: " - 5" with last_result should be last_result -5, but "-5" alone without following operator is just number.
            # If expression is like "+ 5" or "* 2" or "/ 2" -> prepend
            # If expression is like "- 5" (unary minus) we treat as last_result -5? But "-5" alone would have been handled as number if not for last_result
            # We'll prepend if after stripping, first char is + * / % or - followed by digit/space and expression contains operator
            if re.match(r"^[\+\*/%]\s*\d", stripped) or re.match(r"^-\s*\d+\s*[\+\-\*/%]", stripped) or (stripped.startswith("+") or stripped.startswith("*") or stripped.startswith("/") or stripped.startswith("%")):
                norm = f" {lr_str} {stripped}"
            elif stripped.startswith("-"):
                # Check if it's binary minus like "- 2"? Actually "-5" alone would be unary; we prepend only if it looks like operator not number
                # If stripped is "- 5" (minus with space) and we have last_result, likely user means "last_result -5"?? ambiguous
                # For "cộng thêm 5" now becomes "+ 5" after mapping, which will be caught above. "- -? " etc
                pass
            # Special: norm may be "+ 5" after "cộng thêm 5" (single operator). Ensure we catch "+ 5"
            if re.match(r"^[\+\-\*/%]\s*\d", stripped):
                # For "- 5" case, we already handled, but "+ 5" should have been caught; do generic prepend for "+ 5" style
                if not norm.strip().startswith(lr_str):
                    # Avoid double prepend if already contains lr_str
                    if lr_str not in norm:
                        norm = f" {lr_str} {stripped}" if not stripped.startswith(lr_str) else norm

    # Handle "x" as multiplication when between numbers/brackets
    norm = re.sub(r"(?<=[0-9\)])\s*[xX]\s*(?=[0-9\(])", " * ", norm)
    # Handle ":" as division when between numbers: "4:2" -> "4/2"
    norm = re.sub(r"(?<=\d)\s*:\s*(?=\d)", " / ", norm)
    # Handle "·" "•" already via map, but ensure ":" variant "∶" already mapped
    # Handle prefix forms that were converted to "**2 của X" etc: "bình phương của 5" -> "**2 của 5" -> "(5)**2"
    try:
        # "**2 của X" -> "(X)**2"  (X là số sau khi đã chuẩn hoá VN)
        norm = re.sub(r"\*\*2\s*(?:của|cua)?\s*\(?\s*(-?\d+(?:[.,]\d+)?)\s*\)?", r" (\1)**2 ", norm)
        norm = re.sub(r"\*\*3\s*(?:của|cua)?\s*\(?\s*(-?\d+(?:[.,]\d+)?)\s*\)?", r" (\1)**3 ", norm)
        # "! của X" prefix -> "factorial(X)"  (ví dụ: "giai thừa của 5" -> "! của 5" -> "factorial(5)")
        # Chỉ xử khi "!" ở đầu hoặc sau khoảng trắng và trước số, không phải "5!" hậu tố
        norm = re.sub(r"(?:^|\s)!\s*(?:của|cua)?\s*\(?\s*(-?\d+(?:[.,]\d+)?)\s*\)?", r" factorial(\1) ", norm)
        # "% của X" -> "% * X" để sau này "(5/100) * X" đúng
        norm = re.sub(r"%\s*(?:của|cua)?\s*(?=\d|\()", "% * ", norm)
    except Exception:
        pass

    # Handle Vietnamese / English decimal comma and thousand separators per-number token
    # Instead of global replace, process each number token individually to handle mixed locales like "1.000,5 + 0.5"
    def _normalize_number_token(m):
        token = m.group(0)
        # token like "1.000,5" , "0.5" , "2,5" , "1,000" , "1.000"
        if "." in token and "," in token:
            # Both present: decide by last occurrence
            if token.rfind(",") > token.rfind("."):
                # VN: 1.000,5 -> dots thousand, comma decimal
                return token.replace(".", "").replace(",", ".")
            else:
                # EN: 1,000.5 -> commas thousand, dot decimal
                return token.replace(",", "")
        elif "," in token:
            # Only commas
            # If matches thousand pattern like 1,000 or 1,000,000 -> remove commas
            if re.match(r"^\d{1,3}(,\d{3})+(\.\d+)?$", token):
                return token.replace(",", "")
            if re.match(r"^\d+,\d+$", token):
                # Could be decimal "2,5" or thousand "1,000" ambiguous
                # If after comma has 1-2 digits -> decimal; if 3 digits and token length >4 maybe thousand
                # Prefer decimal for 1-2 digits, but for 3 digits check thousand vs decimal
                # For "1,000" -> after 3 digits and token matches thousand pattern above, already handled
                # For remaining like "2,5" -> decimal
                parts = token.split(",")
                if len(parts) == 2 and len(parts[1]) <= 2:
                    return token.replace(",", ".")
                elif len(parts) == 2 and len(parts[1]) == 3:
                    # Could be "1,000" but not matched thousand? treat as thousand
                    # Check if token is like "12,345" (5 digits) not standard thousand but still
                    # For safety, if integer part 1-3 digits and fraction 3 digits, treat as thousand
                    return token.replace(",", "")
                else:
                    return token.replace(",", ".")
            return token.replace(",", ".")
        elif "." in token:
            # Only dots
            # Check if looks like VN thousand: 1.000 or 1.000.000 (but not 0.877 decimal)
            if re.match(r"^[1-9]\d{0,2}(\.\d{3})+$", token):
                return token.replace(".", "")
            return token
        return token

    # Find number tokens that contain digit + dot/comma
    norm = re.sub(r"\d[\d\.,]*", _normalize_number_token, norm)

    # Chuẩn hoá logarit trước khi xóa filler: "logarit cơ số 2 của 8" -> "log(8,2)"
    try:
        norm = re.sub(r"\blogarit\s+cơ\s+số\s+(-?\d+(?:\.\d+)?)\s+của\s+(-?\d+(?:\.\d+)?)", r"log(\2, \1)", norm)
        norm = re.sub(r"\blogarit\s+co\s+so\s+(-?\d+(?:\.\d+)?)\s+cua\s+(-?\d+(?:\.\d+)?)", r"log(\2, \1)", norm)
        norm = re.sub(r"\blogarit\b", "log", norm)
        norm = re.sub(r"\bloga\b", "log", norm)
    except Exception:
        pass

    # Remove filler words
    filler = ["tính","tinh","bằng","bang","là","la","bao nhiêu","bao nhieu","mấy","may","gì","gi","sao","kết quả","ket qua","hãy","hay","giúp","giup","tôi","toi","cho","với","voi","của","cua","được","duoc","ra","đi","di","một","mot","cái","cai","phép","phep","toán","toan","thì","thi","nhiêu","nhieu","b nhiêu","b nhieu"]
    for fw in filler:
        # Cho phép xóa cả từ 2 ký tự như "là", "mấy" khi trong ngữ cảnh toán
        if len(fw) < 2:
            continue
        norm = re.sub(r"\b" + re.escape(fw) + r"\b", " ", norm)
    # Giữ "và"/"va" làm dấu tách đa biểu thức, không xóa ở đây — xử lý tách ở hàm multi riêng

    # Handle power ^ -> **
    norm = norm.replace("^", "**")
    norm = re.sub(r"\*{3,}", "**", norm)

    # Keep valid chars including "!" for factorial, then handle factorial after filtering
    allowed_re = re.compile(r"[^0-9\.\+\-\*/%\(\)\!\, a-zA-Z_]")
    tmp = allowed_re.sub(" ", norm)
    tmp = re.sub(r"\s+", " ", tmp).strip()

    # Handle % as percent: "5%" -> "(5/100)"
    def _percent_repl(m):
        num = m.group(1)
        after = m.group(2) if m.lastindex >=2 else ""
        return f"({num}/100){after}"
    tmp = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*([\+\-\*/\)\, ]|$)", _percent_repl, tmp)

    # Handle factorial "!" : "5!" -> "factorial(5)"
    tmp = re.sub(r"(\d+(?:\.\d+)?|\))\s*!", r"factorial(\1)", tmp)

    # Handle implicit function calls: "log 100" -> "log(100)", "sin 0" -> "sin(0)", "logarit 100" -> "log(100)"
    # Đầu tiên chuẩn hoá "logarit" -> "log"
    tmp = re.sub(r"\blogarit\b", "log", tmp)
    tmp = re.sub(r"\bloga\b", "log", tmp)
    # Xử lý "log cơ số 2 của 8" -> "log(8,2)"
    tmp = re.sub(r"\blog\s+cơ\s+số\s+(\d+(?:\.\d+)?)\s+của\s+(\d+(?:\.\d+)?)", r"log(\2, \1)", tmp)
    tmp = re.sub(r"\blog\s+co\s+so\s+(\d+(?:\.\d+)?)\s+cua\s+(\d+(?:\.\d+)?)", r"log(\2, \1)", tmp)
    # Xử lý "log 100" dạng không ngoặc -> "log(100)", áp dụng cho mọi hàm toán
    for _func in ["sqrt","cbrt","log10","log2","log","ln","sin","cos","tan","asin","acos","atan","sinh","cosh","tanh","exp","abs","ceil","floor"]:
        # có số hoặc hằng pi/e ngay sau hàm mà không có ngoặc
        tmp = re.sub(rf"\b{_func}\s+(\d+(?:\.\d+)?)", rf"{_func}(\1)", tmp)
        tmp = re.sub(rf"\b{_func}\s*\(\s*(\d+(?:\.\d+)?)\s*\)", rf"{_func}(\1)", tmp)  # chuẩn hoá khoảng trắng
        # với hằng pi/e: "ln e" -> "ln(e)", "sin pi" -> "sin(pi)"
        tmp = re.sub(rf"\b{_func}\s+(pi|e)\b", rf"{_func}(\1)", tmp)

    tmp = re.sub(r"\s+", " ", tmp).strip()
    if not any(ch.isdigit() for ch in tmp) and "pi" not in tmp and "e" not in tmp:
        return ""

    has_op = any(op in tmp for op in ["+", "-", "*", "/", "%", "**", "sqrt", "cbrt", "log", "ln", "log10", "log2", "sin", "cos", "tan", "asin", "acos", "atan", "factorial", "pow", "exp", "abs", "ceil", "floor"])
    if not has_op:
        if tmp.replace(".","").replace(" ","").isdigit():
            return ""
        return tmp if has_op else ""

    return tmp.strip()

# ── Đa biểu thức: "kết quả của 1+1 và 2+2" -> tách thành ["1+1","2+2"] ──
def _split_multi_expressions(question: str) -> list[str]:
    """Tách câu có nhiều biểu thức nối bằng 'và', ',' , 'với'."""
    q_low = question.lower()
    # Chỉ tách khi có 'và'/'va'/','/'với' và mỗi phần chứa số
    # Ví dụ: "kết quả của 1+1 và 2+2" -> ["kết quả của 1+1", "2+2"]
    # Tách thô bằng regex giữ nguyên chữ
    parts = re.split(r"\s+và\s+|\s+va\s+|,\s+|\s+với\s+|\s+voi\s+", q_low)
    if len(parts) < 2:
        return []
    # Lọc: mỗi phần phải chứa số và (có toán tử hoặc là số đơn nhưng trong ngữ cảnh đa biểu thức)
    # Yêu cầu ít nhất 2 phần có số
    valid_parts = []
    for p in parts:
        # loại bỏ filler mở đầu như "kết quả của", "kết quả", "của"
        p_clean = re.sub(r"^\s*(kết quả của|ket qua cua|kết quả|ket qua|của|cua)\s*", "", p).strip()
        if re.search(r"\d", p_clean):
            valid_parts.append(p_clean)
    if len(valid_parts) >= 2:
        # Trả về dạng gốc đã làm sạch nhẹ, giữ số + toán tử
        return valid_parts
    return []

def is_multi_math_question(question: str, history=None, last_result=None) -> bool:
    """Phát hiện câu hỏi chứa nhiều biểu thức như '1+1 và 2+2'."""
    if not question or not question.strip():
        return False
    q_low = question.strip().lower()
    # Cần có từ nối "và"/"," (với khoảng trắng) và ít nhất 2 số - tránh nhầm dấu phẩy thập phân "2,5"
    if not any(sep in q_low for sep in [" và ", " va ", ", ", " với ", " voi "]):
        return False
    parts = _split_multi_expressions(question)
    if len(parts) < 2:
        return False
    # Mỗi phần phải có khả năng là biểu thức toán học (có số + toán tử hoặc số đơn trong ngữ cảnh đa)
    # Kiểm tra mỗi phần normalize được thành biểu thức có toán tử hoặc là số
    math_parts = 0
    for p in parts:
        # Thử normalize, nếu ra biểu thức có toán tử thì tính
        expr = normalize_expression(p, last_result=None)
        if expr and any(ch in expr for ch in "+-*/%"):
            math_parts += 1
        elif re.search(r"\d", p) and any(op in p for op in ["+", "-", "*", "/", "×", "÷", "^", "%"]):
            math_parts += 1
        elif re.search(r"\d", p):
            # Số đơn trong đa biểu thức cũng tính (ví dụ "1 và 2 số nào lớn hơn")
            math_parts += 1
    # Cần ít nhất 2 phần toán học
    if math_parts >= 2:
        # Loại trừ RAG: nếu chứa hint chuyên môn thì không phải
        rag_hints = ["quy chế","quy định","quyết định","thông báo","tài liệu","học viện"]
        if any(h in q_low for h in rag_hints):
            return False
        return True
    return False

def calculate_multi(question: str, last_result=None, style=None, want_steps: bool = False) -> dict:
    """Xử lý đa biểu thức: liệt kê / cộng gộp / so sánh.
    Returns dict: {success: bool, answer: str, results: list, expression: str, error: str}
    """
    try:
        parts = _split_multi_expressions(question)
        if len(parts) < 2:
            return {"success": False, "error": "không tách được đa biểu thức", "expression": ""}
        # Tính từng phần
        eval_results = []
        exprs = []
        for p in parts:
            # Loại bỏ đuôi so sánh/tổng như " cộng lại", " số nào lớn hơn" trước khi tính
            # Trích biểu thức toán học đầu tiên trong phần để tránh lẫn chữ
            p_candidate = p
            # Tìm biểu thức có toán tử trước
            m_expr = re.search(r"-?\d+(?:[.,]\d+)?(?:\s*[\+\-\*/%×÷\^]\s*-?\d+(?:[.,]\d+)?)+", p)
            if m_expr:
                p_candidate = m_expr.group(0)
            else:
                m_num = re.search(r"-?\d+(?:[.,]\d+)?", p)
                if m_num:
                    p_candidate = m_num.group(0)
            expr = normalize_expression(p_candidate, last_result=None)
            # Nếu normalize không ra (ví dụ "2" đơn), thử số thuần
            if not expr:
                # Thử lấy số thuần từ p_candidate
                m = re.search(r"-?\d+(?:[.,]\d+)?", p_candidate)
                if m:
                    num_tok = m.group(0).replace(",", ".")
                    try:
                        dec = Decimal(num_tok)
                        eval_results.append(dec)
                        exprs.append(num_tok)
                        continue
                    except Exception:
                        pass
                return {"success": False, "error": f"không tính được phần '{p}'", "expression": p}
            try:
                dec = safe_eval(expr)
                eval_results.append(dec)
                exprs.append(expr)
            except Exception as e:
                return {"success": False, "error": f"lỗi phần '{p}': {e}", "expression": expr}
        # Xác định kiểu: cộng gộp / so sánh / liệt kê
        q_low = question.lower()
        # Cộng gộp: "cộng lại", "tổng", "gộp", "cộng gộp", "cộng vào"
        is_sum = any(kw in q_low for kw in ["cộng lại","cong lai","tổng","tong","gộp","gop","cộng gộp","cong gop","cộng vào","cong vao"])
        # So sánh: chứa từ so sánh hoặc ký hiệu
        is_cmp = any(kw in q_low for kw in ["lớn hơn","lon hon","nhỏ hơn","nho hon","bé hơn","be hon","lớn","lon","nhỏ","nho","bé","be","bằng","bang","so sánh","so sanh","so với","so voi", ">", "<", "=", "≥", "≤", "≠"]) and len(eval_results) >= 2
        # Nếu có cả cộng và so sánh, ưu tiên so sánh nếu có từ so sánh rõ ràng, ngược lại cộng
        has_cmp_word = any(kw in q_low for kw in ["lớn hơn","lon hon","nhỏ hơn","nho hon","bé hơn","be hon","so sánh","so sanh","lớn","lon","nhỏ","nho","bé","be"])
        if has_cmp_word and len(eval_results) >= 2:
            is_cmp = True
            is_sum = False
        if is_cmp and len(eval_results) >= 2:
            # So sánh 2 kết quả đầu (mở rộng: nếu >2 thì so sánh lần lượt, nhưng hiện lấy 2 đầu)
            left_dec = eval_results[0]
            right_dec = eval_results[1]
            left_str = exprs[0]
            right_str = exprs[1]
            # Dùng _format_comparison_answer để tạo câu tự nhiên
            # Tạo op giả: "compare" nếu không có ký hiệu rõ
            op = "compare"
            # Tìm op thực tế trong câu
            if any(kw in q_low for kw in ["lớn hơn","lon hon","lớn","lon"]):
                op = ">"
            elif any(kw in q_low for kw in ["nhỏ hơn","nho hon","nhỏ","nho","bé hơn","be hon","bé","be"]):
                op = "<"
            elif any(kw in q_low for kw in ["bằng nhau","bang nhau","bằng","bang"]):
                op = "=="
            # Nếu có ký hiệu
            if ">=" in q_low or "≥" in q_low:
                op = ">="
            elif "<=" in q_low or "≤" in q_low:
                op = "<="
            elif "!=" in q_low or "≠" in q_low or "khác" in q_low or "khac" in q_low:
                op = "!="
            answer = _format_comparison_answer(left_str, op, right_str, left_dec, right_dec, style, question)
            # Nếu là liệt kê so sánh nhiều hơn 2, thêm chi tiết
            expr_display = f"{format_decimal(left_dec)} {op} {format_decimal(right_dec)}" if op != "compare" else f"{format_decimal(left_dec)} so với {format_decimal(right_dec)}"
            return {"success": True, "answer": answer, "results": eval_results, "exprs": exprs, "expression": expr_display, "left": left_dec, "right": right_dec, "op": op}
        elif is_sum:
            total = sum(eval_results, Decimal(0))
            # Hiển thị: "1+1 = 2, 2+2 = 4, tổng = 6" hoặc "2 + 4 = 6"
            parts_str = ", ".join([f"{exprs[i]} = {format_decimal(eval_results[i])}" for i in range(len(exprs))])
            total_str = format_decimal(total)
            sum_expr = " + ".join([format_decimal(d) for d in eval_results])
            answer = f"{parts_str}, tổng là {total_str} ({sum_expr} = {total_str})."
            if style == "plain":
                answer = f"Kết quả là {total_str}."
            return {"success": True, "answer": answer, "results": eval_results, "exprs": exprs, "expression": sum_expr + f" = {total_str}", "result": total_str}
        else:
            # Liệt kê mặc định
            parts_str = ", ".join([f"{exprs[i]} = {format_decimal(eval_results[i])}" for i in range(len(exprs))])
            answer = f"{parts_str}."
            if style == "plain":
                # plain ngắn gọn
                answer = f"Kết quả là {', '.join([format_decimal(d) for d in eval_results])}."
            return {"success": True, "answer": answer, "results": eval_results, "exprs": exprs, "expression": ", ".join(exprs), "result": ", ".join([format_decimal(d) for d in eval_results])}
    except Exception as e:
        return {"success": False, "error": str(e), "expression": ""}

def format_decimal(d: Decimal) -> str:
    # Highest precision, remove trailing zeros, plain fixed
    # Use normalize but handle exponent
    try:
        # Use 'f' formatting
        s = format(d, 'f')
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
            if s == "-0":
                s = "0"
            if s == "" or s == "-":
                s = "0"
        return s
    except Exception:
        return str(d.normalize())

def safe_eval(expr: str) -> Decimal:
    if not expr or not expr.strip():
        raise ValueError("biểu thức rỗng")
    # Basic safety: check allowed chars
    # Allow digits, ., + - * / % ( ) , and letters for functions
    if re.search(r"[^0-9\.\+\-\*/%\(\)\, a-zA-Z_]", expr):
        # Check if contains invalid characters after filtering, but expr already filtered
        pass
    # Parse AST
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"cú pháp không hợp lệ: {e.msg}")

    _AstNum = getattr(ast, "Num", None)
    _AstConstant = getattr(ast, "Constant", None)

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif _AstConstant is not None and isinstance(node, _AstConstant):
            if isinstance(node.value, (int, float, Decimal)):
                return Decimal(str(node.value))
            elif isinstance(node.value, str):
                # should not happen
                raise ValueError(f"hằng không hợp lệ: {node.value}")
            else:
                raise ValueError(f"hằng không hỗ trợ: {node.value}")
        elif _AstNum is not None and isinstance(node, _AstNum):  # Python <3.8 compat
            return Decimal(str(node.n))
        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                if right == 0:
                    raise ZeroDivisionError("không thể chia cho 0")
                return left / right
            elif isinstance(node.op, ast.Mod):
                if right == 0:
                    raise ZeroDivisionError("không thể chia cho 0")
                return left % right
            elif isinstance(node.op, ast.Pow):
                return _pow_dec(left, right)
            elif isinstance(node.op, ast.FloorDiv):
                if right == 0:
                    raise ZeroDivisionError("không thể chia cho 0")
                return left // right
            else:
                raise ValueError(f"toán tử không hỗ trợ: {type(node.op).__name__}")
        elif isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return operand
            elif isinstance(node.op, ast.USub):
                return -operand
            else:
                raise ValueError(f"toán tử đơn không hỗ trợ: {type(node.op).__name__}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name not in ALLOWED_FUNCS:
                    raise ValueError(f"hàm không hỗ trợ: {func_name}")
                args = [_eval(arg) for arg in node.args]
                # handle log with optional base
                try:
                    return ALLOWED_FUNCS[func_name](*args)
                except ZeroDivisionError:
                    raise
                except Exception as e:
                    raise ValueError(f"lỗi hàm {func_name}: {e}")
            else:
                raise ValueError("chỉ hỗ trợ gọi hàm dạng tên")
        elif isinstance(node, ast.Name):
            if node.id in ALLOWED_NAMES:
                return ALLOWED_NAMES[node.id]
            else:
                raise ValueError(f"tên không hỗ trợ: {node.id}")
        elif isinstance(node, ast.Tuple):
            # for function args? ast will handle each arg separately
            raise ValueError("tuple không hỗ trợ")
        else:
            raise ValueError(f"nút không hỗ trợ: {type(node).__name__}")

    result = _eval(tree)
    if not isinstance(result, Decimal):
        result = Decimal(str(result))
    return result

def calculate(question: str, last_result: Decimal | str | None = None, want_steps: bool = False, style: str | None = None) -> dict:
    """Tính toán câu hỏi toán học.
    Returns dict: {success: bool, expression: str, result: str|Decimal, result_dec: Decimal, steps: str|None, error: str|None}
    """
    try:
        expr = normalize_expression(question, last_result)
        if not expr:
            return {"success": False, "error": "không trích được biểu thức toán học", "expression": "", "result": None}
        # Additional safety: limit length
        if len(expr) > 200:
            return {"success": False, "error": "biểu thức quá dài", "expression": expr, "result": None}
        result_dec = safe_eval(expr)
        result_str = format_decimal(result_dec)
        steps = None
        if want_steps:
            # Generate steps explanation
            # Simple steps: show expression + result, and mention order if needed
            # For high precision, steps are generic unless style requests detail
            if style in ("detailed", "step_by_step"):
                steps = f"Biểu thức: {expr}\nKết quả: {result_str}\nGiải thích: thực hiện theo thứ tự ưu tiên ngoặc → lũy thừa/căn → nhân/chia/lấy dư → cộng/trừ, độ chính xác Decimal prec=50."
            else:
                steps = f"{expr} = {result_str}"
        return {
            "success": True,
            "expression": expr,
            "result": result_str,
            "result_dec": result_dec,
            "steps": steps,
            "error": None,
        }
    except ZeroDivisionError as e:
        return {"success": False, "error": str(e), "expression": expr if 'expr' in locals() else "", "result": None}
    except Exception as e:
        return {"success": False, "error": str(e), "expression": expr if 'expr' in locals() else "", "result": None}
