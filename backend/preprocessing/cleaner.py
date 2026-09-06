"""
backend/preprocessing/cleaner.py - Làm sạch văn bản sau khi trích xuất
====================================================================
Nhiệm vụ:
- Loại bỏ header/footer, letterhead, số trang, chữ ký, quốc huy và các cụm nhiễu.
- Chuẩn hóa khoảng trắng và định dạng dòng.
- Hoạt động trên chuỗi text (không sửa file gốc).
"""
import re

NOISE_RES = [
    re.compile(r"^ban c[oơ] y[eế]u ch[ií]nh ph[uủ]$", re.I),
    re.compile(r"^h[oọ]c vi[eệ]n k[yỹ] thu[aậ]t m[aậ]t m[aã]$", re.I),
    re.compile(r"^c[oọ]ng h[oò]a x[aã] h[oộ]i ch[uủ] ngh[iĩ]a vi[eệ]t nam$", re.I),
    re.compile(r"^đ[oộ]c l[aâ]p\s*[-–]\s*t[uư] do\s*[-–]\s*h[aạ]nh ph[uú]c$", re.I),
    re.compile(r"^s[oố]\s*:", re.I),
    re.compile(r"^h[aà] n[oộ]i, ngày", re.I),
    re.compile(r"^trang\s*\d+", re.I),
    re.compile(r"^tm\.", re.I),
    re.compile(r"^kt\.", re.I),
    re.compile(r"^tl\.", re.I),
    re.compile(r"^k[ýy] tên", re.I),
    re.compile(r"^th[oô]ng b[aá]o$", re.I),
    re.compile(r"^[\s\u2500-\u257F\u2010-\u2015¯_\-=–—.]{3,}$"),
    re.compile(r"^\d{1,3}$"),
]

NOISE_PHRASES = [
    "BAN CƠ YẾU CHÍNH PHỦ",
    "HỌC VIỆN KỸ THUẬT MẬT MÃ",
    "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
    "ĐỘC LẬP – TỰ DO – HẠNH PHÚC",
    "SỐ:",
    "HÀ NỘI, NGÀY",
    "THÔNG BÁO",
    "TM.",
    "KT.",
    "TL.",
    "TRANG",
    "¯¯¯¯¯",
    "/TB-HVM",
    "/QĐ-HVM",
]
NOISE_PHRASE_RES = [re.compile(r"^" + re.escape(p) + r"$", re.I) for p in NOISE_PHRASES]

ENACT_RE = re.compile(r"\(ban hành kèm theo[^)]*\)", re.I)
CODE_RE = re.compile(r"/(TB|QĐ|QD)-HVM", re.I)


def _is_noise(line: str) -> bool:
    t = line.strip().lower()
    if not t:
        return False
    for r in NOISE_RES:
        if r.match(t):
            return True
    for r in NOISE_PHRASE_RES:
        if r.match(t):
            return True
    return False


def clean_text(text: str) -> str:
    if not text:
        return ""
    out = []
    for ln in text.splitlines():
        if _is_noise(ln):
            continue
        ln = ENACT_RE.sub("", ln)
        ln = CODE_RE.sub("", ln)
        ln = ln.replace("–", "-").replace("—", "-")
        ln = re.sub(r"\s+", " ", ln).strip()
        if ln:
            out.append(ln)
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()