import argparse, os, json
from utils.seed import set_all_seeds
from utils.io import load_queries, write_jsonl, load_jsonl
from adapters import Mem0Adapter, MemosAdapter, MemoryOSAdapter, AMemAdapter

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["mem0","memos","memoryos","a-mem"])
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--dataset", required=True, choices=["nq","msmarco"])
    ap.add_argument("--llm_model", required=True, choices=["Llama-2-7b-chat-hf","vicuna-7b-v1.5"])
    ap.add_argument("--emb_model", default="gtr-base")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--num_queries", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oracle_llm", default=None)
    args = ap.parse_args()

    set_all_seeds(args.seed)
    ds_dir = f"./corpus_poisoning/datasets/{args.dataset}"
    queries = load_queries(ds_dir, num=args.num_queries, seed=args.seed)

    adapter = get_adapter(args.backend, args.namespace)
    adapter.load()

    out_dir = f"./store/results/rag_gtr-base_x_{args.llm_model}/{args.dataset}/{args.backend}/clean/seed_{args.seed}"
    os.makedirs(out_dir, exist_ok=True)

    # 只检索一次
    q_texts = [q["query"] for q in queries]
    results = adapter.search(q_texts, args.k)
    retrieved_ids = [r["ids"] for r in results]

    # === 保存回答 JSONL，供 Refusal Rate 使用 ===
    responses = []
    for q in queries:
        # 这里可以接真实 LLM，或先用占位符
        resp = "No answer"  
        responses.append({"qid": q["qid"], "response": resp})

    out_jsonl = os.path.join(out_dir, "clean_responses.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in responses:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 保存检索结果
    records = []
    for q, r in zip(queries, results):
        records.append({
            "qid": q["qid"],
            "query": q["query"],
            "retrieved_ids": r["ids"],
            "scores": r["scores"]
        })
    out_clean = os.path.join(out_dir, f"clean_results_k{args.k}_{args.dataset}{args.num_queries}.jsonl")
    write_jsonl(out_clean, records)

    print(f"[SAVED] {out_jsonl}")
    print(f"[SAVED] {out_clean}")

if __name__ == "__main__":
    main()
