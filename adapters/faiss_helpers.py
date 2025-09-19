# adapters/faiss_helpers.py
import os

def want_gpu() -> bool:
    return os.environ.get("FAISS_GPU", "0") == "1"

def has_faiss_gpu() -> bool:
    try:
        import faiss
        return hasattr(faiss, "StandardGpuResources")
    except Exception:
        return False

def make_flat_ip(dim: int):
    import faiss
    if want_gpu() and has_faiss_gpu():
        try:
            res = faiss.StandardGpuResources()
            return faiss.GpuIndexFlatIP(res, dim)
        except Exception:
            pass
    return faiss.IndexFlatIP(dim)

def make_hnsw_flat(dim: int, M: int = 32, ef_s: int = 64, ef_c: int = 80):
    import faiss
    # HNSW 只有 CPU 版本稳定；GPU 这块容易踩坑，保持 CPU，保证可跑性
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efSearch = ef_s
    index.hnsw.efConstruction = ef_c
    return index
