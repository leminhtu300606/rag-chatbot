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

# Ngưỡng sàn an toàn - đoạn ngắn hơn sẽ cho véc-tơ nhiễu, không có giá trị truy hồi
_MIN_FLOOR = 20
_MAX_OVERLAP_EXTRA = 300


def _effective_min_chars(requested: int | None) -> int:
    """Tính ngưỡng hiệu dụng = max(ngưỡng cấu hình, ngưỡng sàn 20), kẹp không vượt max_chars."""
    try:
        raw = int(requested) if requested is not None else int(cfg.CHUNK.get("min_chars", _MIN_FLOOR))
    except Exception:
        raw = _MIN_FLOOR
    eff = max(raw, _MIN_FLOOR)
    # Không để ngưỡng hiệu dụng vượt quá max_chars (tránh lọc sạch mọi đoạn khi cấu hình sai)
    try:
        max_c = int(cfg.CHUNK.get("max_chars", 1200))
        if eff > max_c:
            eff = max(_MIN_FLOOR, max_c)
    except Exception:
        pass
    return eff


def _effective_max_chars(requested: int | None) -> int:
    try:
        return int(requested) if requested is not None else int(cfg.CHUNK.get("max_chars", 1200))
    except Exception:
        return int(cfg.CHUNK.get("max_chars", 1200))

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


def fast_chunk(
    text: str,
    min_chars: int | None = None,
    max_chars: int | None = None,
) -> list[str]:
    """
    Chunk nhanh không dùng embedding - chỉ tách câu và gộp theo max_chars.
    CPU-friendly, dùng cho file lớn để tránh chậm.
    """
    if min_chars is None:
        min_chars = cfg.CHUNK["min_chars"]
    if max_chars is None:
        max_chars = cfg.CHUNK["max_chars"]
    effective_min = _effective_min_chars(min_chars)
    effective_max = _effective_max_chars(max_chars)
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        s = sentences[0].strip()
        if len(s) >= _MIN_FLOOR:
            return [s[:effective_max]]
        return []
    # Gộp câu liên tiếp đến khi đạt max_chars
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        if not buf:
            buf = s
        elif len(buf) + len(s) + 1 <= effective_max:
            buf = f"{buf} {s}"
        else:
            chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)
    # Gộp vụn ngắn
    if len(chunks) > 1:
        coalesced: list[str] = []
        cur = ""
        for c in chunks:
            if not cur:
                cur = c
            elif len(cur) < effective_min and len(cur) + len(c) + 1 <= effective_max:
                cur = f"{cur} {c}"
            elif len(c) < effective_min and len(cur) + len(c) + 1 <= effective_max:
                cur = f"{cur} {c}"
            else:
                coalesced.append(cur)
                cur = c
        if cur:
            coalesced.append(cur)
        if len(coalesced) > 1 and len(coalesced[-1]) < effective_min:
            if len(coalesced[-2]) + len(coalesced[-1]) + 1 <= effective_max:
                coalesced[-2] = f"{coalesced[-2]} {coalesced[-1]}"
                coalesced.pop()
        chunks = coalesced
    # Lọc và xử lý quá dài
    final: list[str] = []
    for c in chunks:
        if len(c) <= effective_max:
            final.append(c)
        else:
            # Cắt theo câu
            cur = ""
            for s in split_sentences(c):
                if len(cur) + len(s) + 1 <= effective_max:
                    cur = (cur + " " + s).strip() if cur else s
                else:
                    if cur:
                        final.append(cur)
                    cur = s
            if cur:
                final.append(cur)
    filtered = [c for c in final if len(c) >= effective_min]
    if not filtered:
        fallback = [c for c in final if len(c) >= _MIN_FLOOR]
        if fallback:
            try:
                from backend.utils.dedup import deduplicate_chunks
                fallback = deduplicate_chunks(fallback, threshold=cfg.CHUNK.get("dedup_threshold", 0.92), exact_only=cfg.CHUNK.get("dedup_exact_only", False))
            except Exception:
                pass
            return fallback
        stripped = text.strip()
        if len(stripped) >= _MIN_FLOOR:
            return [stripped[:effective_max]]
    try:
        from backend.utils.dedup import deduplicate_chunks
        filtered = deduplicate_chunks(filtered, threshold=cfg.CHUNK.get("dedup_threshold", 0.92), exact_only=cfg.CHUNK.get("dedup_exact_only", False))
    except Exception:
        pass
    return filtered


def _should_use_semantic(text: str) -> bool:
    """Quyết định có dùng semantic (embedding) hay fast cho CPU."""
    # Nếu tắt semantic toàn cục thì dùng fast
    if not cfg.CHUNK.get("use_semantic", True):
        return False
    # CPU-friendly: mặc định dùng fast trên CPU để rebuild nhanh, chỉ dùng semantic khi có GPU
    try:
        if cfg.DEVICE == "cpu" and not cfg.CHUNK.get("force_semantic_on_cpu", False):
            return False
    except Exception:
        pass
    # Nếu text quá dài hoặc quá nhiều câu thì dùng fast để tiết kiệm CPU
    max_chars = cfg.CHUNK.get("semantic_max_chars", 3000)
    max_sents = cfg.CHUNK.get("semantic_max_sents", 30)
    if len(text) > max_chars:
        return False
    try:
        sents = split_sentences(text)
        if len(sents) > max_sents:
            return False
    except Exception:
        pass
    return True


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
    # Fast path cho CPU khi text dài để tiết kiệm thời gian
    if not _should_use_semantic(text):
        return fast_chunk(text, min_chars=min_chars, max_chars=max_chars)
    # Ngưỡng hiệu dụng: tôn trọng cấu hình nhưng không bao giờ dưới sàn 20
    effective_min = _effective_min_chars(min_chars)
    effective_max = _effective_max_chars(max_chars)
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        # Đoạn chỉ một câu: giữ nếu đạt sàn an toàn, kể cả khi chưa đủ ngưỡng mong muốn (tránh mất dữ liệu điều ngắn)
        if len(sentences[0]) >= _MIN_FLOOR:
            # Nếu câu ngắn hơn ngưỡng hiệu dụng nhưng là duy nhất, vẫn giữ để không mất điều khoản ngắn có giá trị
            return [sentences[0]] if len(sentences[0]) >= effective_min else [sentences[0]]
        return []

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
        if not buf or len(buf) + len(c) + 1 <= effective_max:
            buf = (buf + " " + c).strip() if buf else c
        else:
            merged.append(buf)
            buf = c
    if buf:
        merged.append(buf)

    # Tinh chỉnh: gộp đoạn vụn ngắn hơn ngưỡng hiệu dụng nếu vẫn trong giới hạn tối đa
    # Giúp tránh đoạn quá ngắn khi ngữ nghĩa tách vụn
    if len(merged) > 1:
        coalesced: list[str] = []
        cur_buf = ""
        for c in merged:
            if not cur_buf:
                cur_buf = c
            elif len(cur_buf) < effective_min and len(cur_buf) + len(c) + 1 <= effective_max:
                cur_buf = f"{cur_buf} {c}"
            elif len(c) < effective_min and len(cur_buf) + len(c) + 1 <= effective_max:
                cur_buf = f"{cur_buf} {c}"
            else:
                coalesced.append(cur_buf)
                cur_buf = c
        if cur_buf:
            coalesced.append(cur_buf)
        # Nếu đoạn cuối vẫn quá ngắn, gộp ngược vào đoạn trước nếu vừa
        if len(coalesced) > 1 and len(coalesced[-1]) < effective_min:
            if len(coalesced[-2]) + len(coalesced[-1]) + 1 <= effective_max:
                coalesced[-2] = f"{coalesced[-2]} {coalesced[-1]}"
                coalesced.pop()
        merged = coalesced

    final = []
    for c in merged:
        if len(c) <= effective_max:
            final.append(c)
        else:
            cur = ""
            for s in split_sentences(c):
                if len(cur) + len(s) + 1 <= effective_max:
                    cur = (cur + " " + s).strip() if cur else s
                else:
                    if cur:
                        final.append(cur)
                    cur = s
            if cur:
                final.append(cur)

    filtered = [c for c in final if len(c) >= effective_min]
    # Bảo toàn dữ liệu: nếu mọi đoạn đều ngắn hơn ngưỡng mong muốn nhưng vẫn đạt sàn an toàn,
    # giữ lại các đoạn đạt sàn thay vì trả về rỗng (tránh mất văn bản ngắn hợp lệ)
    if not filtered:
        fallback = [c for c in final if len(c) >= _MIN_FLOOR]
        if fallback:
            try:
                from backend.utils.dedup import deduplicate_chunks
                fallback = deduplicate_chunks(fallback, threshold=cfg.CHUNK.get("dedup_threshold", 0.92), exact_only=cfg.CHUNK.get("dedup_exact_only", False))
            except Exception:
                pass
            return fallback
        # Trường hợp văn bản gốc ngắn hơn ngưỡng hiệu dụng nhưng vẫn có nghĩa
        stripped = text.strip()
        if len(stripped) >= _MIN_FLOOR:
            return [stripped[:effective_max]]
    # Dedup cho semantic_chunk cũng cần để tránh lặp
    try:
        from backend.utils.dedup import deduplicate_chunks
        filtered = deduplicate_chunks(filtered, threshold=cfg.CHUNK.get("dedup_threshold", 0.92), exact_only=cfg.CHUNK.get("dedup_exact_only", False))
    except Exception:
        pass
    return filtered


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


LEVEL1_RE = re.compile(r"^\s*\d+\.\s+.*", re.MULTILINE)
LEVEL2_RE = re.compile(r"^\s*\d+\.\d+\.?\s+.*", re.MULTILINE)

def _split_numbered_hierarchical(parent_heading: str, sec_text: str) -> list[tuple[str, str]] | None:
    """
    Tách phân cấp cho Điều có dạng:
      1. Đối với sinh viên không thuộc...
      1.1. Học viên phải đeo thẻ...
      1.2. Khi đến trường...
    Trả về list (combined_heading, chunk_text) hoặc None nếu không có đánh số phân cấp.
    Mỗi điểm 1.x là một chunk riêng với heading đủ ngữ cảnh.
    """
    # Kiểm tra có đánh số không
    if not LEVEL1_RE.search(sec_text) and not LEVEL2_RE.search(sec_text):
        return None
    # Tìm tất cả vị trí Level1
    l1_matches = list(LEVEL1_RE.finditer(sec_text))
    if not l1_matches:
        # Chỉ có Level2 không có Level1 -> tách trực tiếp Level2
        l2_matches = list(LEVEL2_RE.finditer(sec_text))
        if not l2_matches:
            return None
        # Intro trước L2 đầu tiên
        chunks: list[tuple[str, str]] = []
        first_start = l2_matches[0].start()
        intro = sec_text[:first_start].strip()
        if intro and len(intro) >= _MIN_FLOOR:
            chunks.append((parent_heading, intro))
        for idx, m in enumerate(l2_matches):
            start = m.start()
            end = l2_matches[idx + 1].start() if idx + 1 < len(l2_matches) else len(sec_text)
            block = sec_text[start:end].strip()
            if not block:
                continue
            # Tách số và nội dung còn lại của dòng đầu
            first_line = block.splitlines()[0] if block else ""
            mm = re.match(r"^\s*(\d+\.\d+)\.?\s+(.*)", first_line)
            num = mm.group(1) if mm else ""
            remainder = mm.group(2) if mm and mm.group(2) else ""
            # Phần còn lại sau dòng đầu
            rest_lines = block.splitlines()[1:]
            rest = "\n".join(rest_lines).strip()
            # Ghép remainder + rest thành chunk text
            chunk_text = remainder
            if rest:
                chunk_text = f"{chunk_text}\n{rest}" if chunk_text else rest
            chunk_text = chunk_text.strip()
            if len(chunk_text) < _MIN_FLOOR:
                continue
            combined = parent_heading
            if num:
                combined = f"{combined} | {num}." if combined else f"{num}."
            chunks.append((combined, chunk_text))
        return chunks if chunks else None

    chunks: list[tuple[str, str]] = []
    # Intro trước Level1 đầu
    first_l1_start = l1_matches[0].start()
    intro = sec_text[:first_l1_start].strip()
    if intro and len(intro) >= _MIN_FLOOR:
        chunks.append((parent_heading, intro))

    for i, m1 in enumerate(l1_matches):
        l1_start = m1.start()
        l1_end = l1_matches[i + 1].start() if i + 1 < len(l1_matches) else len(sec_text)
        l1_block = sec_text[l1_start:l1_end].strip()
        if not l1_block:
            continue
        l1_line = m1.group(0).strip()
        # Nội dung sau dòng L1
        # Lấy phần sau dòng đầu tiên của block
        lines_in_block = l1_block.splitlines()
        first_line = lines_in_block[0] if lines_in_block else ""
        # Phần còn lại sau dòng L1
        rest_after_l1 = "\n".join(lines_in_block[1:]).strip() if len(lines_in_block) > 1 else ""
        # Kiểm tra block này có chứa Level2 không
        l2_in_block = list(LEVEL2_RE.finditer(l1_block))
        # Nhưng tìm trong rest_after_l1 thôi, không tính dòng L1
        l2_in_rest = list(LEVEL2_RE.finditer(rest_after_l1)) if rest_after_l1 else []
        if l2_in_rest:
            # Thêm chunk tổng hợp cho cả nhóm 1. để truy vấn rộng lấy đủ 1.1-1.4
            combined_group_text = rest_after_l1.strip()
            if len(combined_group_text) >= _MIN_FLOOR:
                combined_group_heading = parent_heading
                if l1_line:
                    combined_group_heading = f"{combined_group_heading} | {l1_line}" if combined_group_heading else l1_line
                # Nếu quá dài, sẽ được cắt ở bước sau, nhưng vẫn thêm để đảm bảo recall
                chunks.append((combined_group_heading, combined_group_text))
            # Có phân cấp con -> tách từng 1.x
            # Tìm vị trí L2 trong rest_after_l1
            # Cần offset vì rest_after_l1 là substring
            # Dùng finditer trên rest_after_l1
            l2_matches_rest = list(LEVEL2_RE.finditer(rest_after_l1))
            for j, m2 in enumerate(l2_matches_rest):
                l2_start = m2.start()
                l2_end = l2_matches_rest[j + 1].start() if j + 1 < len(l2_matches_rest) else len(rest_after_l1)
                l2_block = rest_after_l1[l2_start:l2_end].strip()
                if not l2_block:
                    continue
                l2_first = l2_block.splitlines()[0] if l2_block else ""
                mm2 = re.match(r"^\s*(\d+\.\d+)\.?\s+(.*)", l2_first)
                l2_num = mm2.group(1) if mm2 else ""
                l2_rem = mm2.group(2) if mm2 and mm2.group(2) else ""
                l2_rest = "\n".join(l2_block.splitlines()[1:]).strip() if len(l2_block.splitlines()) > 1 else ""
                # Thêm ngữ cảnh l1 để chunk 1.1 (về Thẻ) vẫn chứa "không thuộc" và "trang phục" cho truy vấn rộng
                chunk_text = f"{l1_line} {l2_rem}".strip() if l1_line else l2_rem
                if l2_rest:
                    chunk_text = f"{chunk_text}\n{l2_rest}" if chunk_text else l2_rest
                chunk_text = chunk_text.strip()
                if len(chunk_text) < _MIN_FLOOR:
                    continue
                # Heading kết hợp: parent + l1_line
                combined = parent_heading
                if l1_line:
                    combined = f"{combined} | {l1_line}" if combined else l1_line
                if l2_num:
                    combined = f"{combined} | {l2_num}." if combined else f"{l2_num}."
                chunks.append((combined, chunk_text))
        else:
            # Không có Level2 -> mỗi Level1 là một chunk riêng
            # Nội dung là phần sau số + rest_after_l1 (nếu có)
            m1_inner = re.match(r"^\s*(\d+)\.\s+(.*)", first_line)
            l1_rem = m1_inner.group(2) if m1_inner and m1_inner.group(2) else ""
            chunk_text = l1_rem
            if rest_after_l1:
                chunk_text = f"{chunk_text}\n{rest_after_l1}" if chunk_text else rest_after_l1
            chunk_text = chunk_text.strip()
            if len(chunk_text) < _MIN_FLOOR:
                continue
            combined = parent_heading
            # Với flat, không cần thêm l1 vào heading vì chunk đã là l1, heading chỉ cần parent
            # Nhưng để giữ phân biệt, có thể thêm số
            # Giữ parent làm heading, text đã có số
            # Để retrieval tốt, thêm l1_line vào heading cũng được, nhưng sẽ dài
            # Chọn chỉ dùng parent
            chunks.append((combined, chunk_text if len(chunk_text) >= _MIN_FLOOR else l1_line))
    return chunks if chunks else None


def _add_overlap(chunks: list[str], overlap_sentences: int = 1) -> list[str]:
    """
    Thêm overlap câu giữa các chunk liên tiếp.
    Lưu ý: chỉ dùng nội bộ trong cùng một Điều/Chương, không dùng toàn cục.
    Không cho phép vượt max_chars.
    """
    if overlap_sentences <= 0 or len(chunks) <= 1:
        return chunks
    overlapped = []
    effective_max = _effective_max_chars(None)
    for i, chunk in enumerate(chunks):
        if i == 0:
            overlapped.append(chunk)
            continue
        # Lấy N câu cuối của chunk trước làm tiền tố (chỉ trong cùng section)
        prev_sents = split_sentences(chunks[i - 1])
        overlap = " ".join(prev_sents[-overlap_sentences:]) if len(prev_sents) >= overlap_sentences else chunks[i - 1][:200]
        # Tránh trùng lặp hoàn toàn - kiểm tra overlap đã có trong chunk chưa
        if overlap and overlap not in chunk[:350]:
            enriched = f"{overlap} {chunk}"
            # Cho phép vượt nhẹ 40 ký tự để giữ ngữ cảnh, sẽ được cắt gọn sau khi gắn heading nếu cần
            if len(enriched) <= effective_max + 40:
                overlapped.append(enriched)
            else:
                # Nếu vượt quá nhiều, giữ nguyên chunk gốc
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

    effective_min = _effective_min_chars(min_chars)
    effective_max = _effective_max_chars(max_chars)

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
        # ── Cách 1: Tách phân cấp số 1./1.1 trước khi semantic ──
        hierarchical = _split_numbered_hierarchical(heading, sec_text)
        if hierarchical is not None:
            enriched_hier: list[str] = []
            for comb_heading, chunk_text in hierarchical:
                if not chunk_text or len(chunk_text.strip()) < _MIN_FLOOR:
                    continue
                # Nếu chunk_text quá dài, tách tiếp bằng semantic
                if len(chunk_text) > effective_max:
                    sub_pieces = semantic_chunk(chunk_text, embed_fn, min_chars, max_chars, threshold)
                    if not sub_pieces:
                        sub_pieces = [chunk_text[i:i+effective_max] for i in range(0, len(chunk_text), effective_max)]
                    for piece in sub_pieces:
                        if len(piece.strip()) < _MIN_FLOOR:
                            continue
                        enriched = piece
                        if heading_enrich and comb_heading:
                            enriched = _enrich_with_heading(enriched, comb_heading, metadata)
                        if use_llm_context:
                            enriched = _generate_llm_context(enriched, comb_heading, doc_snippet, metadata)
                        if len(enriched) > effective_max:
                            if ":\n" in enriched and heading_enrich and comb_heading:
                                pref, body = enriched.split(":\n", 1)
                                pref += ":\n"
                                body = body[: max(0, effective_max - len(pref))]
                                enriched = (pref + body).strip()
                            else:
                                enriched = enriched[:effective_max].strip()
                        enriched_hier.append(enriched)
                else:
                    enriched = chunk_text
                    if heading_enrich and comb_heading:
                        enriched = _enrich_with_heading(enriched, comb_heading, metadata)
                    if use_llm_context:
                        enriched = _generate_llm_context(enriched, comb_heading, doc_snippet, metadata)
                    if len(enriched) > effective_max:
                        if ":\n" in enriched and heading_enrich and comb_heading:
                            pref, body = enriched.split(":\n", 1)
                            pref += ":\n"
                            body = body[: max(0, effective_max - len(pref))]
                            enriched = (pref + body).strip()
                        else:
                            enriched = enriched[:effective_max].strip()
                    enriched_hier.append(enriched)
            all_chunks.extend(enriched_hier)
            continue
        # Fallback: Semantic chunk trong từng section (giữ coherence theo heading)
        sec_chunks = semantic_chunk(sec_text, embed_fn, min_chars, max_chars, threshold)
        # Nếu section ngắn mà semantic trả về rỗng, giữ nguyên section nếu đạt sàn an toàn
        # (tránh mất điều khoản ngắn có giá trị khi ngưỡng mong muốn cao)
        if not sec_chunks and len(sec_text.strip()) >= _MIN_FLOOR:
            sec_chunks = [sec_text] if len(sec_text) <= effective_max else [
                sec_text[i:i+effective_max] for i in range(0, len(sec_text), effective_max)
            ]

        if not sec_chunks:
            continue

        # --- Gộp đoạn vụn thô trong cùng section trước khi xử lý tiếp ---
        # Tránh mất đoạn ngắn trong cùng Điều, cho phép vượt nhẹ 150 ký tự để cứu đoạn ngắn
        # Thực hiện trước heading/overlap để không nhân đôi heading
        _COALESCE_EXTRA = 150
        if len(sec_chunks) > 1:
            coalesced_raw: list[str] = []
            buf = ""
            for c in sec_chunks:
                if not buf:
                    buf = c
                elif len(buf.strip()) < effective_min and len(buf) + len(c) + 1 <= effective_max + _COALESCE_EXTRA:
                    buf = f"{buf} {c}"
                elif len(c.strip()) < effective_min and len(buf) + len(c) + 1 <= effective_max + _COALESCE_EXTRA:
                    buf = f"{buf} {c}"
                else:
                    coalesced_raw.append(buf)
                    buf = c
            if buf:
                coalesced_raw.append(buf)
            if len(coalesced_raw) > 1 and len(coalesced_raw[-1].strip()) < effective_min:
                if len(coalesced_raw[-2]) + len(coalesced_raw[-1]) + 1 <= effective_max + _COALESCE_EXTRA:
                    coalesced_raw[-2] = f"{coalesced_raw[-2]} {coalesced_raw[-1]}"
                    coalesced_raw.pop()
            sec_chunks = coalesced_raw

        # --- Overlap chỉ trong cùng Điều/Chương, trước khi gắn heading để tránh lặp heading ---
        # Đây là điểm sửa chính cho vấn đề xuyên ranh giới Điều/Chương
        if context_window and context_window > 0 and not use_llm_context and len(sec_chunks) > 1:
            sec_chunks = _add_overlap(sec_chunks, overlap_sentences=context_window)

        # Gắn heading / LLM context cho từng chunk trong section
        enriched_sec: list[str] = []
        for chunk in sec_chunks:
            enriched = chunk
            if heading_enrich and heading:
                enriched = _enrich_with_heading(enriched, heading, metadata)
            if use_llm_context:
                enriched = _generate_llm_context(enriched, heading, doc_snippet, metadata)
            enriched_sec.append(enriched)

        # Đảm bảo không vượt max_chars sau khi gắn heading (cắt thân nếu cần, giữ heading)
        # Cắt chính xác để len(strip) vẫn đạt ngưỡng, tránh bị lọc do thừa khoảng trắng
        for idx, c in enumerate(enriched_sec):
            if len(c) > effective_max:
                if ":\n" in c and heading_enrich and heading:
                    pref, body = c.split(":\n", 1)
                    pref += ":\n"
                    body = body.strip()
                    body = body[: max(0, effective_max - len(pref))]
                    enriched_sec[idx] = (pref + body).strip()
                    # Đảm bảo không ngắn hơn ngưỡng do cắt quá tay - nếu vẫn ngắn, giữ nguyên đã cắt
                    if len(enriched_sec[idx].strip()) < effective_min and len(c.strip()) >= effective_min:
                        # Trường hợp hiếm: heading dài khiến thân bị cắt quá ngắn, giữ lại tối đa có thể
                        enriched_sec[idx] = c[:effective_max].strip()
                else:
                    enriched_sec[idx] = c[:effective_max].strip()

        all_chunks.extend(enriched_sec)

    # Lọc lại theo ngưỡng hiệu dụng (tôn trọng cấu hình + sàn 20)
    # Lưu ý: sau khi enrich heading, độ dài tăng nên kiểm tra sau enrich mới chính xác
    # Giữ tất cả đoạn đạt sàn an toàn (>=20) để không mất Điều ngắn; việc gộp đoạn vụn đã xử lý per-section ở trên
    filtered = [c for c in all_chunks if len(c.strip()) >= _MIN_FLOOR]
    if not filtered:
        # Bảo toàn dữ liệu phức tạp: nếu không có đoạn nào đạt ngưỡng mong muốn nhưng có đoạn đạt sàn, giữ lại
        fallback = [c for c in all_chunks if len(c.strip()) >= _MIN_FLOOR]
        if fallback:
            # Dedup trước khi trả về để tránh lặp 20 bullet học bổng
            try:
                from backend.utils.dedup import deduplicate_chunks
                fallback = deduplicate_chunks(fallback, threshold=cfg.CHUNK.get("dedup_threshold", 0.92), exact_only=cfg.CHUNK.get("dedup_exact_only", False))
            except Exception:
                pass
            return fallback
        if text.strip() and len(text.strip()) >= _MIN_FLOOR:
            return [text.strip()[:effective_max]]
    # Dedup cuối cùng để loại chunk trùng do overlap + heading
    try:
        from backend.utils.dedup import deduplicate_chunks
        filtered = deduplicate_chunks(filtered, threshold=cfg.CHUNK.get("dedup_threshold", 0.92), exact_only=cfg.CHUNK.get("dedup_exact_only", False))
    except Exception:
        pass
    return filtered


# Alias để pipeline có thể import linh hoạt
chunk_text = contextual_chunk

