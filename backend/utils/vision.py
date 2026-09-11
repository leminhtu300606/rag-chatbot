"""
backend/utils/vision.py - Vision LLM cho multimodal (CPU-friendly, conditional)
================================================================================
Chỉ chạy khi VISION.enabled=True và query chứa trigger_keywords hoặc chunk là figure/chart.
Mặc định tắt để chạy CPU nhẹ (không yêu cầu GPU).
Khi bật, dùng Ollama vision model nhỏ (qwen2-vl:2b ~1.6GB) gọi qua /api/chat với image base64.

CPU-friendly: model 2b chạy được trên CPU 16GB RAM (chậm nhưng hoạt động), có timeout và fallback.
"""

import base64
import io
from pathlib import Path
from typing import Optional

try:
    import backend.config as cfg
    VISION_CFG = getattr(cfg, "VISION", {})
except Exception:
    VISION_CFG = {}

def _is_vision_enabled() -> bool:
    try:
        return bool(VISION_CFG.get("enabled", False))
    except Exception:
        return False

def _should_trigger_vision(query: str, context: list[dict] | None = None) -> bool:
    if not _is_vision_enabled():
        return False
    if not query:
        return False
    q_low = query.lower()
    triggers = VISION_CFG.get("trigger_keywords", []) if isinstance(VISION_CFG, dict) else []
    for kw in triggers:
        if kw.lower() in q_low:
            return True
    # Nếu context có figure/chart/table image thì cũng trigger
    if context:
        for c in context:
            md = c.get("metadata", {}) if isinstance(c, dict) else {}
            if md.get("chunk_type") in ("figure", "chart") or md.get("has_image"):
                return True
    return False

def _encode_image_base64(image_path: str | Path) -> Optional[str]:
    try:
        p = Path(image_path)
        if not p.exists():
            return None
        data = p.read_bytes()
        return base64.b64encode(data).decode("utf-8")
    except Exception:
        return None

def describe_image(image_path: str | Path, prompt: str = "Mô tả chi tiết hình ảnh này bằng tiếng Việt, tập trung vào nội dung chính, quan hệ không gian và dữ liệu nếu là biểu đồ/bảng.") -> Optional[str]:
    """
    Gọi Ollama vision model để mô tả ảnh.
    Returns: description string hoặc None nếu lỗi/disabled.
    """
    if not _is_vision_enabled():
        return None
    b64 = _encode_image_base64(image_path)
    if not b64:
        return None
    import requests
    model = VISION_CFG.get("model", "qwen2-vl:2b") if isinstance(VISION_CFG, dict) else "qwen2-vl:2b"
    base = VISION_CFG.get("base", getattr(cfg, "OLLAMA_BASE", "http://localhost:11434")) if isinstance(VISION_CFG, dict) else "http://localhost:11434"
    max_tokens = VISION_CFG.get("max_tokens", 256) if isinstance(VISION_CFG, dict) else 256
    # Xác định mime
    ext = Path(image_path).suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    try:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [b64],
                }
            ],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        resp = requests.post(f"{base}/api/chat", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"[vision] lỗi describe_image {image_path}: {e}")
        return None

def enrich_chunk_with_vision(chunk: dict) -> dict:
    """
    Enrich chunk figure/chart bằng vision description nếu có image_path.
    Trả về chunk mới (copy) với field vision_description nếu thành công.
    """
    if not _is_vision_enabled():
        return chunk
    img_path = chunk.get("image_path") or chunk.get("metadata", {}).get("image_path")
    if not img_path:
        return chunk
    desc = describe_image(img_path)
    if desc:
        new_chunk = chunk.copy()
        new_chunk["vision_description"] = desc
        # Thêm vào text để embedding có context
        if "text" in new_chunk:
            new_chunk["text"] = f"{new_chunk['text']}\n[Mô tả hình ảnh AI]: {desc}"
        return new_chunk
    return chunk

def batch_describe_for_query(query: str, context: list[dict]) -> list[dict]:
    """
    Nếu query trigger vision, enrich các figure/chart chunks trong context.
    """
    if not _should_trigger_vision(query, context):
        return context
    enriched = []
    for c in context:
        md = c.get("metadata", {})
        if md.get("chunk_type") in ("figure", "chart") or md.get("has_image"):
            # Tìm image_path trong metadata
            img_path = md.get("image_path") or md.get("bbox")  # bbox không phải path
            # Nếu không có file ảnh thực tế, bỏ qua
            if img_path and Path(str(img_path)).exists():
                enriched.append(enrich_chunk_with_vision(c))
            else:
                enriched.append(c)
        else:
            enriched.append(c)
    return enriched
