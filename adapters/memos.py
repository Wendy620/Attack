import os
import numpy as np
from typing import List, Dict, Any
from .base import BaseMemoryAdapter
import faiss
from .faiss_helpers import make_flat_ip

class MemosAdapter(BaseMemoryAdapter):
    """
    独特性：
      - 默认“engine=fallback-numpy”，统一FAISS内积检索；
      - 打开/关闭采样对最终JSR无影响（由检索决定），但保留接口一致性。
    """
    def __init__(self, namespace: str, indices_dir: str, caches_dir: str, dim: int = 768):
        super().__init__(namespace, indices_dir, caches_dir)
        self.dim = dim
        self.ids: List[str] = []
        self.vecs = None
        self.index = None
        self.blocker_prefix = "blocker_"

    def _ensure(self):
        if self.index is None:
            self.index = make_flat_ip(self.dim)

    def build(self, docs: List[Dict[str, Any]]):
        from utils.embeddings import embed_texts_unit
        texts = [d["text"] for d in docs]
        ids = [d["id"] for d in docs]
        vecs = embed_texts_unit(texts)
        self._ensure()
        self.index.add(vecs)
        self.vecs = vecs if self.vecs is None else np.vstack([self.vecs, vecs])
        base = len(self.ids)
        self.ids.extend(ids)
        # 省略id2idx，当前不需要

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
        self.ids = [i for i, m in zip(self.ids, keep_mask) if m]
        self.vecs = self.vecs[keep_mask]
        self.index = faiss.IndexFlatIP(self.vecs.shape[1])
        self.index.add(self.vecs)

    def search(self, queries: List[str], k: int) -> List[Dict[str, Any]]:
        from utils.embeddings import embed_texts_unit
        if not self.ids:
            return [{"ids": [], "scores": []} for _ in queries]
        q = embed_texts_unit(queries)
        scores, idxs = self.index.search(q, k)
        rs = []
        for s_row, i_row in zip(scores, idxs):
            ids = [self.ids[j] for j in i_row if j != -1]
            rs.append({"ids": ids, "scores": [float(s) for s in s_row if s != -1]})
        return rs

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
        self.index = make_flat_ip(self.vecs.shape[1])

        self.index.add(self.vecs)
