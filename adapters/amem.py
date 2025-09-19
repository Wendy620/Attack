import os
import numpy as np
from typing import List, Dict, Any
from .base import BaseMemoryAdapter
import faiss
from .faiss_helpers import make_flat_ip

class AMemAdapter(BaseMemoryAdapter):
    """
    独特性：
      - 假定A-mem更像“稀疏+稠密混合”，但最小可跑版用FAISS扁平+轻量重排模拟：
        1) 用内积取前2k
        2) 用简单BM25-like打分（词频）对前2k重排，取k
    """
    def __init__(self, namespace: str, indices_dir: str, caches_dir: str, dim: int = 768):
        super().__init__(namespace, indices_dir, caches_dir)
        self.dim = dim
        self.ids: List[str] = []
        self.texts: List[str] = []
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
        self.ids.extend(ids)
        self.texts.extend(texts)

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
        self.texts = [self.texts[i] for i in keep]
        self.vecs = self.vecs[keep]
        self.index = faiss.IndexFlatIP(self.vecs.shape[1])
        self.index.add(self.vecs)

    def _bm25lite(self, q: str, docs: List[str], k1=1.2, b=0.75):
        # 极简TF打分（只为体现“独特性”）
        q_terms = q.lower().split()
        scores = []
        avg_len = sum(len(d.split()) for d in docs) / max(1, len(docs))
        for d in docs:
            d_terms = d.lower().split()
            score = 0.0
            for t in q_terms:
                tf = d_terms.count(t)
                dl = len(d_terms)
                score += (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_len, 1e-6)) + 1e-6)
            scores.append(score)
        return np.array(scores, dtype=np.float32)

    def search(self, queries: List[str], k: int) -> List[Dict[str, Any]]:
        from utils.embeddings import embed_texts_unit
        if not self.ids:
            return [{"ids": [], "scores": []} for _ in queries]
        qv = embed_texts_unit(queries)
        top2k = max(k * 2, 10)
        scores, idxs = self.index.search(qv, min(top2k, len(self.ids)))
        outs = []
        for qi, (s_row, i_row) in enumerate(zip(scores, idxs)):
            cand_ids = [self.ids[j] for j in i_row if j != -1]
            cand_txt = [self.texts[j] for j in i_row if j != -1]
            if not cand_ids:
                outs.append({"ids": [], "scores": []})
                continue
            rerank = self._bm25lite(queries[qi], cand_txt)
            order = np.argsort(-rerank)[:k]
            outs.append({"ids": [cand_ids[o] for o in order],
                         "scores": [float(rerank[o]) for o in order]})
        return outs

    def save(self):
        os.makedirs(self.indices_dir, exist_ok=True)
        np.save(os.path.join(self.indices_dir, f"{self.namespace}_ids.npy"), np.array(self.ids, dtype=object))
        np.save(os.path.join(self.indices_dir, f"{self.namespace}_texts.npy"), np.array(self.texts, dtype=object))
        if self.vecs is not None:
            np.save(os.path.join(self.indices_dir, f"{self.namespace}_vecs.npy"), self.vecs)

    def load(self):
        ids_p = os.path.join(self.indices_dir, f"{self.namespace}_ids.npy")
        txt_p = os.path.join(self.indices_dir, f"{self.namespace}_texts.npy")
        vecs_p = os.path.join(self.indices_dir, f"{self.namespace}_vecs.npy")
        if not (os.path.exists(ids_p) and os.path.exists(vecs_p) and os.path.exists(txt_p)):
            return
        self.ids = np.load(ids_p, allow_pickle=True).tolist()
        self.texts = np.load(txt_p, allow_pickle=True).tolist()
        self.vecs = np.load(vecs_p).astype(np.float32)
        self.index = make_flat_ip(self.vecs.shape[1])
        self.index.add(self.vecs)
