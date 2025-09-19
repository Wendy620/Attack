# utils/io.py  —— 覆盖整个文件
import os
import ujson as json
from typing import List, Dict, Any, Optional

DOC_ID_KEYS = ["id", "docid", "doc_id", "pid", "passage_id", "_id", "document_id"]
DOC_TEXT_KEYS = ["text", "contents", "content", "passage", "document", "body"]

Q_ID_KEYS = ["qid", "id", "question_id", "_id"]
Q_TEXT_KEYS = ["query", "question", "text", "prompt"]

def _pick_first(d: Dict[str, Any], keys: List[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None
    
def _normalize_doc(d: Dict[str, Any], fallback_idx: int) -> Optional[Dict[str, Any]]:
    did = _pick_first(d, DOC_ID_KEYS)
    if did is None:
        did = f"doc{fallback_idx}"

    # 1) 先拿 text；为空再用 title
    txt = _pick_first(d, DOC_TEXT_KEYS)
    if (txt is None or len(str(txt).strip()) == 0) and "title" in d:
        if d["title"] is not None and len(str(d["title"]).strip()) > 0:
            # 用 title 兜底
            txt = d["title"]

    # 2) 再看 metadata 兜底
    if (txt is None or len(str(txt).strip()) == 0) and isinstance(d.get("metadata"), dict):
        meta = d["metadata"]
        txt = _pick_first(meta, ["text", "body", "contents", "title"])
        # 如果既有 title 又有 body，可以拼一下
        if txt is None and ("title" in meta and "body" in meta):
            txt = f"{meta.get('title','')} {meta.get('body','')}".strip()

    # 3) 最后再从原字典里找任意一个非空字符串字段兜底
    if txt is None or len(str(txt).strip()) == 0:
        for v in d.values():
            if isinstance(v, str) and len(v.strip()) > 0:
                txt = v
                break

    # 4) 实在没有就返回 None（上层会过滤并计数）
    if txt is None or len(str(txt).strip()) == 0:
        return None

    return {"id": str(did), "text": str(txt).strip()}


def load_corpus(dataset_dir: str, sample: int = None, seed: int = 42) -> List[Dict[str, Any]]:
    """
    期望 {dataset_dir}/corpus.jsonl 每行一个文档，但字段名可多样。
    规范化为 {"id": "...", "text": "..."}，并自动跳过空文本文档。
    """
    import random, sys
    path = os.path.join(dataset_dir, "corpus.jsonl")
    assert os.path.exists(path), f"Not found: {path}"
    raw = [json.loads(l) for l in open(path, "r", encoding="utf-8")]

    # 规范化并过滤 None（空文本/不可用）
    docs = []
    dropped = 0
    for i, d in enumerate(raw):
        nd = _normalize_doc(d, i)
        if nd is None:
            dropped += 1
        else:
            docs.append(nd)

    if dropped > 0:
        print(f"[WARN] load_corpus: dropped {dropped} empty-text docs from {len(raw)} total", file=sys.stderr)

    if sample and sample > 0:
        random.Random(seed).shuffle(docs)
        docs = docs[:sample]
    return docs

def _normalize_query(q: Dict[str, Any], fallback_idx: int) -> Dict[str, Any]:
    qid = _pick_first(q, Q_ID_KEYS)
    if qid is None:
        qid = f"q{fallback_idx}"
    qtxt = _pick_first(q, Q_TEXT_KEYS)
    if qtxt is None and isinstance(q.get("metadata"), dict):
        qtxt = _pick_first(q["metadata"], ["query", "question", "text", "prompt"])
    assert qtxt is not None and len(str(qtxt).strip()) > 0, f"Query has no query-like field: {q.keys()}"
    return {"qid": str(qid), "query": str(qtxt)}

def load_queries(dataset_dir: str, num: int = 100, seed: int = 0) -> List[Dict[str, Any]]:
    """
    期望 {dataset_dir}/queries.jsonl 每行一个查询，字段名可多样。
    该函数会规范化为 {"qid": "...", "query": "..."}。
    """
    import random
    path = os.path.join(dataset_dir, "queries.jsonl")
    assert os.path.exists(path), f"Not found: {path}"
    raw = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
    qs = [_normalize_query(q, i) for i, q in enumerate(raw)]
    random.Random(seed).shuffle(qs)
    return qs[:num]

def write_jsonl(path: str, records: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_jsonl(path):
    if not os.path.exists(path):
        print(f"[WARN] load_jsonl: file not found {path}")
        return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data
