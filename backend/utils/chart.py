"""
backend/utils/chart.py - Chart data extraction (placeholder, CPU-friendly)
=========================================================================
Nguyên tắc: Nếu có thể chuyển visual -> structured data thì nên làm, giữ ảnh gốc.

Với tài liệu hiện tại (không có chart), module này là placeholder để mở rộng khi có biểu đồ.
Khi phát hiện chart (image có nhiều bar/line), sẽ dùng Vision LLM để trích dữ liệu.

Hiện tại: detect_chart heuristic đơn giản (dựa trên drawings hoặc image size), và extract_chart_data stub.
"""

from typing import List, Dict, Optional
from pathlib import Path

def is_chart_image(image_info: dict, page_drawings: int = 0) -> bool:
    """
    Heuristic: chart thường là hình có nhiều drawings (bars/lines) hoặc tỷ lệ width/height đặc trưng.
    Với kho hiện tại: luôn return False để không trigger xử lý nặng.
    """
    # Nếu trang có >50 drawings và image width~800-1200 thì nghi chart
    if page_drawings > 50:
        w = image_info.get("width", 0)
        h = image_info.get("height", 0)
        if 300 < w < 1500 and 100 < h < 800:
            return True
    return False

def extract_chart_data_stub(image_path: str | Path) -> Optional[Dict]:
    """
    Placeholder: sau này sẽ gọi Vision LLM để trích {title, data: [{x,y}]}.
    Hiện trả None để không ảnh hưởng pipeline CPU.
    """
    try:
        import backend.config as cfg
        vision_enabled = getattr(cfg, "VISION", {}).get("enabled", False) if hasattr(cfg, "VISION") else False
        if not vision_enabled:
            return None
        # Nếu bật vision, gọi describe và parse
        from backend.utils.vision import describe_image
        prompt = (
            "Đây là biểu đồ. Hãy trích dữ liệu thành JSON dạng {\"title\": \"...\", \"type\": \"bar|line|pie\", \"data\": [{\"label\": \"2024\", \"value\": 60}]} "
            "Chỉ trả về JSON, không thêm giải thích."
        )
        desc = describe_image(image_path, prompt=prompt)
        if desc:
            import json, re
            # Thử parse JSON trong desc
            m = re.search(r"\{.*\}", desc, flags=re.DOTALL)
            if m:
                return json.loads(m.group(0))
    except Exception as e:
        print(f"[chart] extract fail {image_path}: {e}")
    return None

def chart_to_text(chart_data: Dict) -> str:
    """Chuyển structured chart data thành text cho embedding."""
    if not chart_data:
        return ""
    title = chart_data.get("title", "Biểu đồ")
    data = chart_data.get("data", [])
    if not data:
        return title
    lines = [f"Biểu đồ: {title}"]
    for d in data:
        # linh hoạt key
        label = d.get("label") or d.get("year") or d.get("x") or str(d)
        value = d.get("value") or d.get("sales") or d.get("y") or ""
        lines.append(f"{label}: {value}")
    return "\n".join(lines)
