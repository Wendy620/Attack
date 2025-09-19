import os
import numpy as np
from typing import List, Dict, Any
from .base import BaseMemoryAdapter
from .faiss_helpers import make_hnsw_flat

# 尝试memoryos-pro / memoryos 的原生后端（hnsw等）
_HAS_NATIVE = False
try:
    import memoryos  # 占位
    _HAS_NATIVE = True
except Exception:
    _HAS_NATIVE = False

import faiss

class MemoryOSAdapter(BaseMemoryAdapter):
    """
    独特性：
      - 优先使用“degraded-hnsw+cache（若native可用）”
      - 否则fallback到FAISS（IndexHNSWFlat或IndexFlatIP）以模拟近似检索特点
    """
    def __init__(self, namespace: str, indices_dir: str, caches_dir: str, dim: int = 768):
        super().__init__(namespace, indices_dir, caches_dir)
        self.dim = dim
        self.ids: List[str] = []
        self.vecs = None
        self.blocker_prefix = "blocker_"
        self.index = None

    def _ensure(self):
        if self.index is None:
            # 用HNSW模拟近似检索风格
            self.index = make_hnsw_flat(self.dim, 32, 64, 80)
            self.index.hnsw.efSearch = 64
            self.index.hnsw.efConstruction = 80

    def build(self, docs: List[Dict[str, Any]]):
        from utils.embeddings import embed_texts_unit
        texts = [d["text"] for d in docs]
        ids = [d["id"] for d in docs]
        vecs = embed_texts_unit(texts)
        self._ensure()
        self.index.add(vecs)
        self.vecs = vecs if self.vecs is None else np.vstack([self.vecs, vecs])
        self.ids.extend(ids)

    def add_blockers(self, docs: List[Dict[str, Any]]):
        for d in docs:
            if not d["id"].startswith(self.blocker_prefix):
                d["id"] = self.blocker_prefix + d["id"]
        return self.build(docs)

    def purge_prefix(self, prefix: str):
        if not self.ids:
            return
        keep = [i for i in range(len(self.ids)) if not self.ids[i].startswith(prefix)]
        if len(keep) == len(self.ids):
            return
        self.ids = [self.ids[i] for i in keep]
        self.vecs = self.vecs[keep]
        self.index = faiss.IndexHNSWFlat(self.vecs.shape[1], 32)
        self.index.add(self.vecs)

    def search(self, queries: List[str], k: int) -> List[Dict[str, Any]]:
        from utils.embeddings import embed_texts_unit
        if not self.ids:
            return [{"ids": [], "scores": []} for _ in queries]
        q = embed_texts_unit(queries)
        scores, idxs = self.index.search(q, k)
        out = []
        for s_row, i_row in zip(scores, idxs):
            ids = [self.ids[j] for j in i_row if j != -1]
            out.append({"ids": ids, "scores": [float(s) for s in s_row if s != -1]})
        return out

    def save(self):
        os.makedirs(self.indices_dir, exist_ok=True)
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
        self.index = make_hnsw_flat(self.vecs.shape[1], 32, 64, 80)
        self.index.add(self.vecs)
