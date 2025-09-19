import abc
from typing import List, Dict, Any, Optional

class BaseMemoryAdapter(abc.ABC):
    """
    标准接口：
      - build(corpus)          : 建索引
      - add_blockers(blockers) : 批量加入blocker文档
      - purge_prefix(prefix)   : 删除以prefix开头的文档（用于清理旧blocker）
      - search(queries, k)     : 检索top-k，返回 [ { "ids": [...], "scores": [...] }, ... ]
      - save() / load()        : 索引持久化
    """
    def __init__(self, namespace: str, indices_dir: str, caches_dir: str):
        self.namespace = namespace
        self.indices_dir = indices_dir
        self.caches_dir = caches_dir

    @abc.abstractmethod
    def build(self, docs: List[Dict[str, Any]]): ...
    @abc.abstractmethod
    def add_blockers(self, docs: List[Dict[str, Any]]): ...
    @abc.abstractmethod
    def purge_prefix(self, prefix: str): ...
    @abc.abstractmethod
    def search(self, queries: List[str], k: int) -> List[Dict[str, Any]]: ...
    @abc.abstractmethod
    def save(self): ...
    @abc.abstractmethod
    def load(self): ...
