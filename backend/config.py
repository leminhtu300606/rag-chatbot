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

EMBED_MODEL = "dangvantuan/vietnamese-embedding"

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
DEVICE = "cpu"

# ── Hội thoại: quản lý ngữ cảnh ──
# Số lượng turn gần nhất giữ nguyên trong prompt (mỗi turn = 1 user + 1 assistant)
CONVERSATION_WINDOW = 6  # giu 6 cap hoi-dap gan nhat (~12 messages)
CONVERSATION_SUMMARY_THRESHOLD = 12  # khi vuot 12 turns thi tom tat
CONVERSATION_MAX_HISTORY_CHARS = 300  # cat moi message cu de tiet kiem token
# Model dung cho rewrite/summarize: None = dung LLM_MODEL chinh, hoac chi dinh model nho hon
REWRITE_MODEL = None  # vi du "qwen2.5:1.5b" de rewrite nhanh
SUMMARY_MODEL = None
REWRITE_MAX_TOKENS = 256
SUMMARY_MAX_TOKENS = 256

CATEGORY_ORDER = [
    "quyet_dinh",
    "quy_che",
    "quy_dinh",
    "tai_lieu_huong_dan",
    "thong_bao",
]
