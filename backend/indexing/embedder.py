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

import backend.config as cfg
from backend.config import DEVICE, EMBED_MODEL

_model = None
_runtime_device = None
# Lấy batch size tối ưu theo phần cứng
def _get_batch_size() -> int:
    try:
        return int(getattr(cfg, "EMBED_BATCH_SIZE", 32))
    except Exception:
        return 32


def _get_model():
    """Load the local Transformer directly; SentenceTransformer.encode hangs in this environment."""
    global _model
    global _runtime_device
    if _model is None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        _runtime_device = "cuda" if DEVICE.startswith("cuda") and torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, local_files_only=True)
        model = AutoModel.from_pretrained(EMBED_MODEL, local_files_only=True).to(_runtime_device)
        model.eval()
        _model = (tokenizer, model)
    return _model


def embed(texts, batch_size: int | None = None, normalize: bool = True):
    import numpy as np
    import torch

    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return np.array([])
    if batch_size is None:
        batch_size = _get_batch_size()
    tokenizer, model = _get_model()
    outputs = []
    max_length = 256
    for start in range(0, len(texts), max(1, batch_size)):
        batch = texts[start:start + max(1, batch_size)]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        if normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        outputs.append(pooled.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def warmup():
    """Làm nóng model để request đầu không bị chậm."""
    try:
        embed(["khởi động"], batch_size=1)
    except Exception:
        pass


def get_model_info() -> dict:
    """Tra ve thong tin model embedding hien tai (de log khi luu vao ChromaDB)."""
    try:
        sample = embed(["test"], batch_size=1)
        dim = len(sample[0]) if len(sample) else 0
        return {"model": EMBED_MODEL, "device": _runtime_device or DEVICE, "dim": int(dim)}
    except Exception as e:
        return {"model": EMBED_MODEL, "device": DEVICE, "error": str(e)}


def get_dimension() -> int:
    """Lay dimension cua embedding (dung de kiem tra co khop voi ChromaDB khong)."""
    try:
        info = get_model_info()
        return int(info.get("dim", 0))
    except Exception:
        return 0

