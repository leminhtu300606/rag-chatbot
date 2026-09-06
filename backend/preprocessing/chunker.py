"""
backend/preprocessing/chunker.py - Chia văn bản thành chunk theo ngữ nghĩa và ngữ cảnh
=======================================================================================
Nhiệm vụ:
- Chia văn bản tiếng Việt thành các đoạn nhỏ phù hợp cho embedding và retrieval.
- Hỗ trợ hai chế độ:
  * Semantic: tách câu, embed, đo cosine liền kề, ngắt khi dưới ngưỡng, gộp theo max_chars.
  * Contextual: giữ tiêu đề Chương/Điều/Mục/Khoản làm tiền tố, thêm overlap câu giữa các chunk,
    tùy chọn làm giàu ngữ cảnh bằng LLM (kiểu Anthropic).
- Cấu hình chi tiết trong backend/config.py -> CHUNK.
"""

import re

import numpy as np

import backend.config as cfg
CHUNK = cfg.CHUNK

# Regex nhận diện tiêu đề văn bản Việt Nam (quy chế, quy định, quyết định)
SECTION_RE = re.compile(
    r"^\s*(Chương\s+[IVXLCDM\d]+|Phần\s+[IVXLCDM\d]+|Mục\s+\d+|Điều\s+\d+[\.\:]?|Khoản\s+\d+[\.\:]?|"
    r"CHƯƠNG\s+[IVXLCDM\d]+|ĐIỀU\s+\d+|MỤC\s+\d+).*",
    re.IGNORECASE | re.MULTILINE,
)

HEADING_LINE_RE = re.compile(
    r"^\s*(Chương|Điều|Mục|Khoản|Phần)\s+[IVXLCDM\d\.\:]?",
    re.IGNORECASE,
)


def split_sentences(text: str) -> list[str]:
    try:
        from underthesea import sent_tokenize
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def semantic_chunk(
    text: str,
    embed_fn,
    min_chars: int | None = None,
    max_chars: int | None = None,
    threshold: float | None = None,
) -> list[str]:
    if min_chars is None:
        min_chars = cfg.CHUNK["min_chars"]
    if max_chars is None:
        max_chars = cfg.CHUNK["max_chars"]
    if threshold is None:
        threshold = cfg.CHUNK["break_threshold"]
    """Chia chunk cơ bản theo ngữ nghĩa (tính tương tự cosine giữa các câu liền kề)."""
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [sentences[0]] if len(sentences[0]) >= 20 else []

    embeddings = [np.asarray(v) for v in embed_fn(sentences)]
    sims = [_cosine(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]

    breaks = [0]
    for i, s in enumerate(sims):
        if s < threshold:
            breaks.append(i + 1)
    breaks.append(len(sentences))

    raw_chunks = []
    for j in range(len(breaks) - 1):
        seg = sentences[breaks[j]:breaks[j + 1]]
        if seg:
            raw_chunks.append(" ".join(seg))

    merged = []
    buf = ""
    for c in raw_chunks:
        if not buf or len(buf) + len(c) + 1 <= max_chars:
            buf = (buf + " " + c).strip() if buf else c
        else:
            merged.append(buf)
            buf = c
    if buf:
        merged.append(buf)

    final = []
    for c in merged:
        if len(c) <= max_chars:
            final.append(c)
        else:
            cur = ""
            for s in split_sentences(c):
                if len(cur) + len(s) + 1 <= max_chars:
                    cur = (cur + " " + s).strip() if cur else s
                else:
                    if cur:
                        final.append(cur)
                    cur = s
            if cur:
                final.append(cur)

    return [c for c in final if len(c) >= 20]


# ── Contextual chunking ──

def split_by_sections(text: str) -> list[tuple[str, str]]:
    """
    Tách văn bản thành các (heading, section_text) theo tiêu đề Chương/Điều.

    Nếu không tìm thấy heading, trả về [("", text)].
    """
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    cur_heading = ""
    cur_lines: list[str] = []
    heading_found = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cur_lines.append(line)
            continue
        # Kiểm tra có phải dòng tiêu đề không (ngắn <120 ký tự và match regex)
        if len(stripped) < 150 and SECTION_RE.match(stripped):
            # Lưu section trước
            if cur_lines or heading_found:
                section_text = "\n".join(cur_lines).strip()
                if section_text or cur_heading:
                    sections.append((cur_heading, section_text))
                cur_lines = []
            cur_heading = stripped
            heading_found = True
        else:
            # Nếu là heading dạng inline "Điều 5. Nội dung..." thì tách heading ra
            m = re.match(r"^\s*(Điều\s+\d+[\.\:]?\s*)(.+)", stripped, re.IGNORECASE)
            if m and len(m.group(1)) < 30:
                if cur_lines:
                    section_text = "\n".join(cur_lines).strip()
                    if section_text:
                        sections.append((cur_heading, section_text))
                    cur_lines = []
                cur_heading = m.group(1).strip()
                cur_lines.append(m.group(2))
                heading_found = True
            else:
                cur_lines.append(line)

    # Section cuối
    section_text = "\n".join(cur_lines).strip()
    if section_text or cur_heading:
        sections.append((cur_heading, section_text))

    if not heading_found:
        return [("", text.strip())]

    # Lọc section rỗng
    sections = [(h, t) for h, t in sections if t.strip()]
    return sections if sections else [("", text.strip())]


def _add_overlap(chunks: list[str], overlap_sentences: int = 1) -> list[str]:
    """Thêm overlap câu giữa các chunk liên tiếp để giữ ngữ cảnh."""
    if overlap_sentences <= 0 or len(chunks) <= 1:
        return chunks
    overlapped = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            overlapped.append(chunk)
            continue
        # Lấy N câu cuối của chunk trước làm tiền tố
        prev_sents = split_sentences(chunks[i - 1])
        overlap = " ".join(prev_sents[-overlap_sentences:]) if len(prev_sents) >= overlap_sentences else chunks[i - 1][:200]
        # Tránh trùng lặp hoàn toàn
        if overlap and overlap not in chunk[:300]:
            enriched = f"{overlap} {chunk}"
            # Cắt nếu quá max_chars thì giữ chunk gốc
            if len(enriched) <= CHUNK["max_chars"] + 300:
                overlapped.append(enriched)
            else:
                overlapped.append(chunk)
        else:
            overlapped.append(chunk)
    return overlapped


def _enrich_with_heading(chunk: str, heading: str, metadata: dict | None = None) -> str:
    """Tiền tố heading + metadata vào chunk để chunk tự chứa ngữ cảnh."""
    prefixes = []
    if metadata:
        # Thêm category/filename nhẹ để phân biệt nguồn khi retrieve
        cat = metadata.get("category", "")
        if cat and cat != "unknown":
            prefixes.append(f"[{cat}]")
    if heading:
        prefixes.append(heading.strip())
    if prefixes:
        prefix = " | ".join(prefixes) + ":\n"
        # Tránh lặp nếu chunk đã bắt đầu bằng heading
        if chunk.strip().lower().startswith(heading.strip().lower()[:20]):
            return chunk
        return f"{prefix}{chunk}"
    return chunk


def _generate_llm_context(chunk: str, heading: str, doc_snippet: str, metadata: dict | None = None) -> str:
    """
    Kiểu Anthropic contextual retrieval: gọi LLM tạo 1-2 câu mô tả ngữ cảnh.
    Chỉ chạy nếu CHUNK["use_llm_context"]=True và Ollama sẵn sàng.
    Fallback về heading nếu LLM lỗi.
    """
    try:
        from backend.config import LLM_MODEL, OLLAMA_BASE
        import requests

        doc_info = ""
        if metadata:
            doc_info = f"Tài liệu: {metadata.get('filename','')} ({metadata.get('category','')})"
        prompt = (
            "Bạn hãy viết 1-2 câu ngắn gọn mô tả ngữ cảnh của đoạn trích dưới đây trong toàn bộ tài liệu, "
            "để đoạn trích có thể hiểu được khi đứng riêng.\n"
            f"{doc_info}\n"
            f"Tiêu đề gần nhất: {heading or 'Không có'}\n"
            f"Đoạn trích:\n{chunk[:800]}\n"
            f"Ngữ cảnh toàn văn (rút gọn):\n{doc_snippet[:1200]}\n"
            "Chỉ trả về 1-2 câu mô tả, không lặp lại đoạn trích."
        )
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False, "options": {"num_predict": 80}},
            timeout=30,
        )
        resp.raise_for_status()
        ctx = resp.json().get("response", "").strip()
        if ctx and len(ctx) > 20:
            return f"{ctx}\n{chunk}"
    except Exception:
        pass
    return chunk


def contextual_chunk(
    text: str,
    embed_fn,
    metadata: dict | None = None,
    min_chars: int | None = None,
    max_chars: int | None = None,
    threshold: float | None = None,
    context_window: int | None = None,
    heading_enrich: bool | None = None,
    use_llm_context: bool | None = None,
) -> list[str]:
    """
    Chunk theo ngữ cảnh.

    Luồng:
    1. Tách văn bản thành sections theo Chương/Điều (split_by_sections)
    2. Mỗi section -> semantic_chunk
    3. Mỗi chunk được làm giàu:
       - Tiền tố heading (nếu heading_enrich)
       - Overlap N câu trước (context_window)
       - (Tùy chọn) LLM sinh mô tả ngữ cảnh

    Args:
        text: văn bản đã clean
        embed_fn: hàm embed
        metadata: dict {filename, category, ...} để gắn vào context
        context_window: số câu overlap (mặc định CHUNK["context_window"])
        heading_enrich: có tiền tố heading không
        use_llm_context: có gọi LLM tạo context không

    Returns:
        list[str]: các chunk đã giàu ngữ cảnh
    """
    if min_chars is None:
        min_chars = cfg.CHUNK["min_chars"]
    if max_chars is None:
        max_chars = cfg.CHUNK["max_chars"]
    if threshold is None:
        threshold = cfg.CHUNK["break_threshold"]
    if context_window is None:
        context_window = cfg.CHUNK.get("context_window", 1)
    if heading_enrich is None:
        heading_enrich = cfg.CHUNK.get("heading_enrich", True)
    if use_llm_context is None:
        use_llm_context = cfg.CHUNK.get("use_llm_context", False)

    text = text.strip()
    if not text:
        return []

    # Nếu tắt contextual -> fallback semantic thuần
    if not cfg.CHUNK.get("contextual", True):
        return semantic_chunk(text, embed_fn, min_chars, max_chars, threshold)

    sections = split_by_sections(text)
    all_chunks: list[str] = []
    # Để LLM có snippet toàn văn rút gọn
    doc_snippet = text[:2000] if use_llm_context else ""

    for heading, sec_text in sections:
        if not sec_text.strip():
            continue
        # Semantic chunk trong từng section (giữ coherence theo heading)
        sec_chunks = semantic_chunk(sec_text, embed_fn, min_chars, max_chars, threshold)
        # Nếu section ngắn (< min_chars) mà semantic trả về rỗng, giữ nguyên section
        if not sec_chunks and len(sec_text) >= 20:
            sec_chunks = [sec_text] if len(sec_text) <= max_chars else [
                sec_text[i:i+max_chars] for i in range(0, len(sec_text), max_chars)
            ]

        for chunk in sec_chunks:
            enriched = chunk
            if heading_enrich and heading:
                enriched = _enrich_with_heading(enriched, heading, metadata)
            if use_llm_context:
                enriched = _generate_llm_context(enriched, heading, doc_snippet, metadata)
            all_chunks.append(enriched)

    # Overlap giữa các chunk liên tiếp (toàn cục, không phân biệt section)
    if context_window and context_window > 0:
        # Chỉ overlap nếu chunk chưa có LLM context (đã đủ dài)
        if not use_llm_context:
            all_chunks = _add_overlap(all_chunks, overlap_sentences=context_window)

    # Lọc lại
    return [c for c in all_chunks if len(c.strip()) >= 20]


# Alias để pipeline có thể import linh hoạt
chunk_text = contextual_chunk

