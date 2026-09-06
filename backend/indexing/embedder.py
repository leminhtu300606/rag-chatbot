"""
backend/indexing/embedder.py - Tạo embedding tiếng Việt
========================================================
Nhiệm vụ:
- Bao quanh SentenceTransformer để tạo embedding chuẩn hóa cho văn bản tiếng Việt.
- Cung cấp hàm get_model_info(), get_dimension() để kiểm tra model, dim, device
  phục vụ chẩn đoán và ghi metadata vào ChromaDB.
"""
import os
import warnings
import logging
# Chi hien thi tien trinh, an warning CPU/GPU/HF
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_dummy_token_for_silence")
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HUGGING_FACE_HUB_TOKEN", "hf_dummy_token_for_silence")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.disable(logging.WARNING)

from sentence_transformers import SentenceTransformer

from backend.config import DEVICE, EMBED_MODEL

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
        # Fix: model dangvantuan/vietnamese-embedding co max_position_embeddings=258
        # nhung tokenizer/max_seq_length mac dinh 512 -> encode >258 tokens se loi
        # "index 258 is out of bounds". Can gioi han lai de tu dong truncate.
        try:
            max_pos = getattr(_model[0].auto_model.config, "max_position_embeddings", None)
            if max_pos is not None:
                # Tru 2 cho [CLS]/[SEP] de an toan, va gioi han theo tokenizer
                safe_len = int(max_pos) - 2
                # SentenceTransformer dung max_seq_length de truncate truoc khi dua vao model
                if _model.max_seq_length is None or _model.max_seq_length > safe_len:
                    _model.max_seq_length = safe_len
                # Dong bo tokenizer
                try:
                    if _model.tokenizer.model_max_length > safe_len:
                        _model.tokenizer.model_max_length = safe_len
                except Exception:
                    pass
        except Exception:
            pass
    return _model


def embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    return _get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)


def get_model_info() -> dict:
    """Tra ve thong tin model embedding hien tai (de log khi luu vao ChromaDB)."""
    try:
        m = _get_model()
        # thu lay dim bang cach embed 1 cau mau
        sample = m.encode(["test"], normalize_embeddings=True, show_progress_bar=False)
        dim = len(sample[0]) if len(sample) else 0
        return {"model": EMBED_MODEL, "device": DEVICE, "dim": int(dim)}
    except Exception as e:
        return {"model": EMBED_MODEL, "device": DEVICE, "error": str(e)}


def get_dimension() -> int:
    """Lay dimension cua embedding (dung de kiem tra co khop voi ChromaDB khong)."""
    try:
        info = get_model_info()
        return int(info.get("dim", 0))
    except Exception:
        return 0

