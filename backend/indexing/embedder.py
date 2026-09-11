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

import backend.config as cfg
from backend.config import DEVICE, EMBED_MODEL

_model = None
# Lấy batch size tối ưu theo phần cứng
def _get_batch_size() -> int:
    try:
        return int(getattr(cfg, "EMBED_BATCH_SIZE", 32))
    except Exception:
        return 32


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
        # CPU optimization: giới hạn max_seq_length xuống 512 cho BGE-M3 (8192 quá dài, chậm trên CPU)
        # Chunk của ta max 1200 chars ~ 300 tokens, 512 là đủ và nhanh hơn 16x
        try:
            if DEVICE == "cpu" and _model.max_seq_length is not None and _model.max_seq_length > 512:
                _model.max_seq_length = 512
                try:
                    if _model.tokenizer.model_max_length > 512:
                        _model.tokenizer.model_max_length = 512
                except Exception:
                    pass
                print(f"[embedder] CPU mode: giới hạn max_seq_length -> 512 để tăng tốc (gốc {_model.max_seq_length})")
        except Exception:
            pass
    return _model


def embed(texts, batch_size: int | None = None, normalize: bool = True):
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        import numpy as np
        return np.array([])
    if batch_size is None:
        batch_size = _get_batch_size()
    # Tự động chia nhỏ nếu quá nhiều để tránh OOM
    if len(texts) > batch_size * 4:
        import numpy as np
        parts = []
        for i in range(0, len(texts), batch_size * 4):
            chunk = texts[i:i + batch_size * 4]
            emb = _get_model().encode(chunk, normalize_embeddings=normalize, show_progress_bar=False, batch_size=batch_size, convert_to_numpy=True)
            parts.append(emb)
        return np.concatenate(parts, axis=0) if parts else np.array([])
    return _get_model().encode(texts, normalize_embeddings=normalize, show_progress_bar=False, batch_size=batch_size, convert_to_numpy=True)


def warmup():
    """Làm nóng model để request đầu không bị chậm."""
    try:
        _get_model().encode(["khởi động"], normalize_embeddings=True, show_progress_bar=False, batch_size=1)
    except Exception:
        pass


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

