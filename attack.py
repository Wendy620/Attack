import argparse, os, csv
from typing import List, Dict
from utils.seed import set_all_seeds
from utils.io import load_queries, write_jsonl, load_jsonl
from adapters import Mem0Adapter, MemosAdapter, MemoryOSAdapter, AMemAdapter
from metrics import jamming_rate_topk, refusal_rate_from_responses, asr_legacy

import random, re


def synth_blocker_text(q: str, num_tokens: int, style: str = "concat", ratio: float = 0.4) -> str:
    """
    style:
      - concat  : 最强攻击（高 JSR/B@1）
      - dropout : 只保留部分 query 词，更克制
      - random  : 不含 query 词，最弱
    ratio: dropout 风格下，保留 query token 比例（0~1）
    """
    q = q.strip()
    if style == "concat":
        base = ("[MASK] " + q + " ") * max(1, num_tokens // max(3, len(q.split())))
        return (base.strip() + " " + q).strip()

    toks = re.findall(r"\w+|\S", q)
    if style == "dropout":
        keep = [t.lower() for t in toks if t.isalnum() and random.random() < ratio]
        keep = list(dict.fromkeys(keep))
        random.shuffle(keep)
        noise = ["[MASK]"] * max(0, num_tokens - len(keep))
        return (" ".join(keep + noise)).strip()

    if style == "random":
        return ("[MASK] " * max(1, num_tokens)).strip()

    return ("[MASK] " * max(1, num_tokens)).strip()


def get_adapter(backend: str, namespace: str):
    indices_dir = f"./store/indices/{backend}/{namespace}"
    caches_dir  = f"./store/caches/{backend}/{namespace}"
    if backend == "mem0":
        return Mem0Adapter(namespace, indices_dir, caches_dir)
    if backend == "memos":
        return MemosAdapter(namespace, indices_dir, caches_dir)
    if backend == "memoryos":
        return MemoryOSAdapter(namespace, indices_dir, caches_dir)
    if backend == "a-mem":
        return AMemAdapter(namespace, indices_dir, caches_dir)
    raise ValueError(f"Unknown backend: {backend}")

def _to_text_list(items):
    # 统一把 JSONL 里的条目转成字符串列表，兼容多种字段名
    out = []
    for it in (items or []):
        if isinstance(it, str):
            out.append(it)
        elif isinstance(it, dict):
            for k in ("text", "response", "output", "answer", "completion", "content"):
                if k in it and isinstance(it[k], str):
                    out.append(it[k])
                    break
            else:
                # 兜底：整个 dict 序列化
                try:
                    import json as _json
                    out.append(_json.dumps(it, ensure_ascii=False))
                except Exception:
                    out.append(str(it))
        else:
            out.append(str(it))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--purge_blockers", action="store_true")
    ap.add_argument("--backend", required=True, choices=["mem0","memos","memoryos","a-mem"])
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--dataset", required=True, choices=["nq","msmarco"])
    ap.add_argument("--llm_model", required=True, choices=["Llama-2-7b-chat-hf","vicuna-7b-v1.5"])
    ap.add_argument("--emb_model", default="gtr-base")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--num_queries", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--attack", default="jamming", choices=["jamming"])
    ap.add_argument("--doc_init", default="mask", choices=["mask"])
    ap.add_argument("--num_tokens", type=int, default=50)
    ap.add_argument("--n_iters", type=int, default=100)
    ap.add_argument("--early_stop", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)

    ap.add_argument("--oracle_llm", default=None)
    ap.add_argument("--oracle_emb", default="gtr-base")

    ap.add_argument("--resp_clean", default=None, help="clean 阶段回答 JSONL")
    ap.add_argument("--resp_poisoned", default=None, help="攻击后回答 JSONL")

    ap.add_argument("--blocker_style", default="concat", choices=["concat","dropout","random"])
    ap.add_argument("--blocker_ratio", type=float, default=0.4)
    ap.add_argument("--resp_tar", type=str, default=None,
                help="Target string for ASR(legacy) metric; if provided, will compute ASR based on responses containing this string.")


    args = ap.parse_args()
    set_all_seeds(args.seed)

    ds_dir = f"./corpus_poisoning/datasets/{args.dataset}"
    queries = load_queries(ds_dir, num=args.num_queries, seed=args.seed)
    q_texts = [q["query"] for q in queries]

    adapter = get_adapter(args.backend, args.namespace)
    adapter.load()

    if args.purge_blockers:
        adapter.purge_prefix("blocker_")

    # 1) 生成 blockers
    blockers: List[Dict[str,str]] = []
    for i, q in enumerate(q_texts):
        bid = f"blocker_{args.dataset}_{args.seed}_{i}"
        btxt = synth_blocker_text(q, args.num_tokens, style=args.blocker_style, ratio=args.blocker_ratio)
        blockers.append({"id": bid, "text": btxt})
    adapter.add_blockers(blockers)
    adapter.save()

    # 2) 检索 Top-k
    retrieved = adapter.search(q_texts, args.k)
    retrieved_ids = [r["ids"] for r in retrieved]

    jamming_rate = jamming_rate_topk(retrieved_ids, blocker_prefix="blocker_")
    blocker1 = sum(1 for ids in retrieved_ids if ids and str(ids[0]).startswith("blocker_")) / len(retrieved_ids)
    
    # 3) Refusal Rate
    refusal_rate_clean, refusal_rate_poisoned = None, None
    if args.resp_clean:
        responses_clean = load_jsonl(args.resp_clean)
        refusal_rate_clean = refusal_rate_from_responses(responses_clean)
    if args.resp_poisoned:
        responses_poisoned = load_jsonl(args.resp_poisoned)
        refusal_rate_poisoned = refusal_rate_from_responses(responses_poisoned)
    asr_score = None
    if args.resp_poisoned and args.resp_tar:
        poisoned_texts = _to_text_list(responses_poisoned)
        asr_score = asr_legacy(poisoned_texts, args.resp_tar)

    # 4) 保存结果
    out_dir = f"./store/results/rag_gtr-base_x_{args.llm_model}/{args.dataset}/{args.backend}/attack/jamming/seed_{args.seed}"
    os.makedirs(out_dir, exist_ok=True)

    detail = []
    for q, ids in zip(queries, retrieved_ids):
        detail.append({"qid": q["qid"], "query": q["query"], "retrieved_ids": ids})
    write_jsonl(os.path.join(out_dir, "detail.jsonl"), detail)

    sum_csv = os.path.join(out_dir, f"summary_r{args.blocker_ratio}_{args.llm_model}.csv")
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "backend","dataset","llm_model","k","N",
            "refusal_rate_clean","refusal_rate_poisoned",
            "jamming_rate", "blocker@1", "ASR(legacy)",
            "blocker_style","blocker_ratio"
        ])
        w.writeheader()
        w.writerow({
            "backend": args.backend,
            "dataset": args.dataset,
            "llm_model": args.llm_model,
            "k": args.k,
            "N": args.num_queries,
            "refusal_rate_clean": "" if refusal_rate_clean is None else f"{refusal_rate_clean:.3f}",
            "refusal_rate_poisoned": "" if refusal_rate_poisoned is None else f"{refusal_rate_poisoned:.3f}",
            "jamming_rate": f"{jamming_rate:.3f}",
            "blocker@1": f"{blocker1:.3f}",
            "ASR(legacy)": f"{asr_score:.3f}" if asr_score is not None else "None",
            "blocker_style": args.blocker_style,
            "blocker_ratio": f"{args.blocker_ratio:.2f}"
        })
    print(f"[METRICS] JSR(blocker@k)={jamming_rate:.3f}  blocker@1={blocker1:.3f}  ASR(legacy)={asr_score if asr_score is not None else 'None'}")
    print(f"[SAVED] {sum_csv}")

if __name__ == "__main__":
    main()
