import os
import numpy as np
from typing import List, Dict, Any
from .base import BaseMemoryAdapter
from .faiss_helpers import make_flat_ip

try:
    # 尝试native（比如 qdrant / mem0 原生 SDK）
    import qdrant_client  # 占位示例
    _HAS_NATIVE = bool(os.environ.get("MEM0_USE_NATIVE", "0") == "1")
except Exception:
    _HAS_NATIVE = False

# fallback: 统一用faiss/numpy
import faiss

class Mem0Adapter(BaseMemoryAdapter):
    """
    独特性：
      - 若 MEM0_USE_NATIVE=1 且可用，则走 native-qdrant；
      - 否则fallback到 FAISS（IndexFlatIP），向量单位化后余弦≈内积。
    """
    def __init__(self, namespace: str, indices_dir: str, caches_dir: str, dim: int = 768):
        super().__init__(namespace, indices_dir, caches_dir)
        self.dim = dim
        self.ids: List[str] = []
        self.vecs = None
        self.id2idx = {}
        self.index = None
        self.blocker_prefix = "blocker_"

    def _ensure_index(self):
        if self.index is None:
            self.index = make_flat_ip(self.dim)

    def build(self, docs: List[Dict[str, Any]]):
        from utils.embeddings import embed_texts_unit
        texts = [d["text"] for d in docs]
        ids = [d["id"] for d in docs]
        vecs = embed_texts_unit(texts)
        self._ensure_index()
        self.index.add(vecs)
        self.vecs = vecs if self.vecs is None else np.vstack([self.vecs, vecs])
        base = len(self.ids)
        for i, did in enumerate(ids):
            self.id2idx[did] = base + i
        self.ids.extend(ids)

    def add_blockers(self, docs: List[Dict[str, Any]]):
        for d in docs:
            if not d["id"].startswith(self.blocker_prefix):
                d["id"] = self.blocker_prefix + d["id"]
        return self.build(docs)

    def purge_prefix(self, prefix: str):
        if not self.ids:
            return
        keep_mask = np.array([not _id.startswith(prefix) for _id in self.ids])
        if keep_mask.all():
            return
        kept_ids = [i for i, m in zip(self.ids, keep_mask) if m]
        kept_vecs = self.vecs[keep_mask]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(kept_vecs.astype(np.float32))
        self.vecs = kept_vecs
        self.ids = kept_ids
        self.id2idx = {did: i for i, did in enumerate(self.ids)}

    def search(self, queries: List[str], k: int) -> List[Dict[str, Any]]:
        from utils.embeddings import embed_texts_unit
        if not self.ids:
            return [{"ids": [], "scores": []} for _ in queries]
        q = embed_texts_unit(queries)
        scores, idxs = self.index.search(q, k)
        results = []
        for row_scores, row_idxs in zip(scores, idxs):
            out_ids, out_scores = [], []
            for s, j in zip(row_scores, row_idxs):
                if j == -1:
                    continue
                out_ids.append(self.ids[j])
                out_scores.append(float(s))
            results.append({"ids": out_ids, "scores": out_scores})
        return results

    def save(self):
        os.makedirs(self.indices_dir, exist_ok=True)
        # 简化：仅持久化ids与向量
        np.save(os.path.join(self.indices_dir, f"{self.namespace}_ids.npy"), np.array(self.ids, dtype=object))
        if self.vecs is not None:
            np.save(os.path.join(self.indices_dir, f"{self.namespace}_vecs.npy"), self.vecs)

    def load(self):
        ids_p = os.path.join(self.indices_dir, f"{self.namespace}_ids.npy")
        vecs_p = os.path.join(self.indices_dir, f"{self.namespace}_vecs.npy")
        if not (os.path.exists(ids_p) and os.path.exists(vecs_p)):
            return
        self.ids = np.load(ids_p, allow_pickle=True).tolist()
        self.vecs = np.load(vecs_p).astype(np.float32)
        self.index = make_flat_ip(self.vecs.shape[1])
        self.index.add(self.vecs)
        self.id2idx = {did: i for i, did in enumerate(self.ids)}
