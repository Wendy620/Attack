import argparse, os
from utils.seed import set_all_seeds
from utils.io import load_corpus
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
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--emb_model", default="gtr-base")
    ap.add_argument("--sample", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_all_seeds(args.seed)
    corpus = load_corpus(args.dataset_dir, sample=args.sample, seed=args.seed)
    adapter = get_adapter(args.backend, args.namespace)
    print(f"[INFO] corpus loaded: {len(corpus)} docs")
    adapter.build(corpus)
    adapter.save()
    print(f"[OK] indexed {len(corpus)} docs into {args.backend}@{args.namespace}")

if __name__ == "__main__":
    main()
