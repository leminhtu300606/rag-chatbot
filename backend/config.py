"""
Cấu hình chung của hệ thống RAG chatbot Học viện Kỹ thuật Mật mã.

Cấu trúc 3 phần chính:
- PHẦN 1 (preprocessing): DATA_DIR (raw PDF/DOCX) -> PROCESSED_DIR (artifact .parquet)
- PHẦN 2 (indexing):      PROCESSED_DIR -> CHROMA_DIR (vector store) + reranking
- PHẦN 3 (generation):    truy xuất + prompt + LLM (Ollama/Transformers)

Tham khảo: backend/preprocessing/, backend/indexing/, backend/generation/
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ── PHẦN 1: Tiền xử lý dữ liệu (raw -> processed) ──
DATA_DIR = BASE_DIR / "data" / "classified"  # dữ liệu thô: PDF, DOCX, TXT
PROCESSED_DIR = BASE_DIR / "data" / "processed"  # artifact sau chunking: *.parquet (parquet)
PROCESSED_EXT = ".parquet"  # dinh dang du lieu huan luyen: parquet (thay jsonl)

# ── PHẦN 2: Indexing / Vector Store ──
CHROMA_DIR = BASE_DIR / "chroma_db"
INDEXED_MANIFEST = PROCESSED_DIR / ".indexed.json"

EMBED_MODEL = "BAAI/bge-m3"

LLM_BACKEND = "ollama"
# qwen2.5:7b (~4.7GB) can ~8GB RAM/VRAM - can bang tot giua chat luong va tai nguyen
# Cac tag khac: qwen2.5:0.5b (0.4GB), qwen2.5:1.5b (1GB), qwen2.5:3b (1.9GB), qwen2.5:14b (9GB), qwen2.5:32b (20GB)
# Vi du: LLM_MODEL = "qwen2.5:1.5b"  hoac "qwen2.5:14b"  hoac "qwen2.5:32b"
LLM_MODEL = "qwen2.5"
OLLAMA_BASE = "http://localhost:11434"
LLM_THINK = False

# ── PHẦN 1: Chunking (semantic + contextual) ──
CHUNK = {
    "min_chars": 200,
    "max_chars": 1200,
    "overlap_sentences": 1,
    "break_threshold": 0.5,
    # ── Chunk theo ngữ cảnh ──
    "contextual": True,  # bật contextual chunking
    "context_mode": "heading+overlap",  # heading+overlap | llm | overlap
    "context_window": 1,  # số câu chồng lấn giữa các chunk
    "use_llm_context": False,  # True = gọi LLM tạo mô tả ngữ cảnh cho mỗi chunk (Anthropic contextual retrieval)
    "heading_enrich": True,  # True = tiền tố tiêu đề Chương/Điều vào mỗi chunk
}

# ── PHẦN 2: Retrieval & Reranking ──
RETRIEVE_TOP_K = 3
COLLECTION_NAME = "rag_hvm"

# ── PHẦN 3: Generation ──
GEN_MAX_TOKENS = 512
# Tự động nhận phần cứng: ưu tiên GPU nếu có, cho phép ghi đè qua biến môi trường DEVICE
def _detect_device() -> str:
    import os
    env = os.getenv("DEVICE", "").strip().lower()
    if env in ("cpu", "cuda", "cuda:0", "mps"):
        return env
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # Apple Silicon
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

DEVICE = _detect_device()
# Có dùng GPU không (để các module khác tự quyết)
USE_GPU = DEVICE in ("cuda", "cuda:0", "mps")
# Batch size mặc định cho embedding/rerank (tăng khi có GPU)
EMBED_BATCH_SIZE = 64 if USE_GPU else 32
RERANK_BATCH_SIZE = 32 if USE_GPU else 16
# OCR có dùng GPU không
OCR_USE_GPU = USE_GPU

# ── Hội thoại: quản lý ngữ cảnh ──
# Cân bằng: nhớ từ đầu phiên, tốc độ + chính xác, nói thẳng khi không đủ nguồn
# Giữ 8 turns gần nhất nguyên văn (16 messages) + tóm tắt phần cũ từ đầu phiên
CONVERSATION_WINDOW = 8  # giu 8 cap hoi-dap gan nhat (~16 messages) để nhớ ngữ cảnh dài
CONVERSATION_SUMMARY_THRESHOLD = 10  # vuot 10 turns thì tóm tắt phần cũ từ đầu phiên
CONVERSATION_MAX_HISTORY_CHARS = 600  # tăng từ 300 -> 600 để giữ chi tiết, tránh cắt cụt
# Model dung cho rewrite/summarize: None = dung LLM_MODEL chinh, hoac chi dinh model nho hon
REWRITE_MODEL = None  # vi du "qwen2.5:1.5b" de rewrite nhanh
SUMMARY_MODEL = None
REWRITE_MAX_TOKENS = 256
SUMMARY_MAX_TOKENS = 320  # tăng nhẹ để tóm tắt giữ được ý chính từ đầu phiên

CATEGORY_ORDER = [
    "quyet_dinh",
    "quy_che",
    "quy_dinh",
    "tai_lieu_huong_dan",
    "thong_bao",
]

# ── Phong cách trả lời: đổi được trong hội thoại bằng câu tự nhiên ──
# Người dùng có thể nói "đổi sang giọng thân thiện", "giải thích đơn giản thôi", "không dùng kiểu gọi hàm", v.v.
AVAILABLE_STYLES = [
    "formal",       # hành chính, trang trọng
    "friendly",     # thân thiện
    "concise",      # ngắn gọn
    "detailed",     # chi tiết
    "simple",       # đơn giản dễ hiểu, không dùng thuật ngữ
    "academic",     # học thuật
    "casual",       # tự nhiên đời thường
    "humorous",     # hài hước nhẹ
    "empathetic",   # đồng cảm
    "creative",     # sáng tạo
    "bullet",       # gạch đầu dòng
    "step_by_step", # từng bước
    "plain",        # thuần túy, không gọi hàm/không kỹ thuật
]
DEFAULT_STYLE = "casual"
# Gợi ý từ khóa để nhận diện đổi phong cách qua câu tự nhiên
STYLE_KEYWORDS = {
    "formal": ["hành chính", "trang trọng", "formal", "nghiêm túc", "chuẩn mực"],
    "friendly": ["thân thiện", "friendly", "gần gũi", "ấm áp"],
    "concise": ["ngắn gọn", "concise", "tóm tắt", "súc tích", "ngắn thôi"],
    "detailed": ["chi tiết", "detailed", "đầy đủ", "cặn kẽ"],
    "simple": ["đơn giản", "dễ hiểu", "simple", "dễ đọc", "phổ thông"],
    "academic": ["học thuật", "academic", "nghiên cứu"],
    "casual": ["tự nhiên", "đời thường", "casual", "thoải mái"],
    "humorous": ["hài hước", "vui", "hài", "humorous", "vui nhộn"],
    "empathetic": ["đồng cảm", "empathetic", "chia sẻ"],
    "creative": ["sáng tạo", "creative", "mới mẻ"],
    "bullet": ["gạch đầu dòng", "bullet", "dạng liệt kê", "liệt kê"],
    "step_by_step": ["từng bước", "step by step", "theo bước", "quy trình"],
    "plain": ["không gọi hàm", "không dùng hàm", "không kỹ thuật", "plain", "đừng dùng kiểu gọi hàm", "không giải thích bằng gọi hàm", "không dùng kiểu gọi hàm"],
}
# Nhiệt độ gợi ý theo phong cách (dùng cho Ollama options)
STYLE_TEMPERATURE = {
    "formal": 0.2,
    "friendly": 0.6,
    "concise": 0.3,
    "detailed": 0.4,
    "simple": 0.5,
    "academic": 0.2,
    "casual": 0.7,
    "humorous": 0.8,
    "empathetic": 0.6,
    "creative": 0.8,
    "bullet": 0.3,
    "step_by_step": 0.3,
    "plain": 0.5,
}

# ── Xã giao mở rộng: chỉ cung cấp thông tin ngoài lề khi có nguồn chính xác hoặc kiến thức huấn luyện ──
# Mặc định trả lời theo giọng đời thường (casual)
ENABLE_SOCIAL = True
SOCIAL_ALLOWLIST = [
    "xin chào", "chào bạn", "chào", "hello", "hi", "hey",
    "bạn khỏe không", "khỏe không", "khoe khong", "bạn khỏe", "khỏe chứ",
    "cảm ơn", "cam on", "cám ơn", "thank",
    "tạm biệt", "bye", "goodbye",
    "bạn là ai", "ban la ai", "giới thiệu", "gioi thieu",
    "hôm nay thế nào", "hom nay the nao", "thời tiết", "thoi tiet",
    "kể chuyện", "ke chuyen", "đùa", "dua", "vui", "haha", "hihi",
    "giúp tôi", "giup toi", "bạn có thể", "ban co the",
    "tên bạn", "ten ban", "bạn tên gì",
    "chúc", "chuc", "buổi sáng", "buoi sang", "buổi tối", "buoi toi",
]
# Các chủ đề xã giao được phép trả lời bằng kiến thức chung, ngoài ra phải có nguồn
SOCIAL_TOPICS = ["chào hỏi", "sức khỏe", "cảm ơn", "tạm biệt", "giới thiệu", "thời tiết cơ bản", "kể chuyện ngắn", "đùa nhẹ", "hỏi han đời thường"]
