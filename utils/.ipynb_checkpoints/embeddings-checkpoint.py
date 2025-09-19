# utils/embeddings.py  —— 覆盖整个文件
import os
import numpy as np
from typing import List

_DEVICE = None
_MODEL = None

def _pick_device():
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    try:
        import torch
        if torch.cuda.is_available():
            _DEVICE = "cuda"
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        else:
            _DEVICE = "cpu"
    except Exception:
        _DEVICE = "cpu"
    return _DEVICE

def _get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    from sentence_transformers import SentenceTransformer
    dev = _pick_device()
    # 明确指定 device，避免默认检测异常
    _MODEL = SentenceTransformer("sentence-transformers/gtr-t5-base", device=dev)
    return _MODEL

def float_or(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default

def embed_texts(texts: List[str]) -> np.ndarray:
    model = _get_model()
    # encode 会自动用 model.device；此处不显式传入 device 以避免重复控制
    vecs = model.encode(
        texts,
        batch_size=int(float_or("MEM_EMB_BATCH", 384)),
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    return vecs.astype(np.float32)

def embed_texts_unit(texts: List[str]) -> np.ndarray:
    v = embed_texts(texts)
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    return (v / n).astype(np.float32)
