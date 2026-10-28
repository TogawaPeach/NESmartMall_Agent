import os
import threading
from functools import lru_cache

from dotenv import load_dotenv
from pymilvus import MilvusClient

from app.knowledge.documents import PROJECT_ROOT

_LOAD_LOCK = threading.Lock()
_LOADED_COLLECTIONS: set[str] = set()


@lru_cache(maxsize=1)
def get_milvus_client() -> MilvusClient:
    """初始化并复用 Milvus 客户端。"""
    load_dotenv(PROJECT_ROOT / ".env")

    return MilvusClient(
        uri=os.environ["MILVUS_URI"],
        db_name=os.environ["MILVUS_DB_NAME"],
        token=os.getenv("MILVUS_TOKEN", ""),
        timeout=10,
    )


def ensure_collection_loaded(
    client: MilvusClient,
    collection_name: str,
    *,
    force: bool = False,
) -> None:
    """确保集合处于 Loaded 状态，避免 Released 状态下检索直接报错。"""
    if not force and collection_name in _LOADED_COLLECTIONS:
        return

    with _LOAD_LOCK:
        # 双重检查，避免并发请求重复 load
        if not force and collection_name in _LOADED_COLLECTIONS:
            return
        client.load_collection(collection_name=collection_name)
        _LOADED_COLLECTIONS.add(collection_name)


if __name__ == "__main__":
    client = get_milvus_client()

    try:
        collections = client.list_collections()
        print("Milvus 连接成功")
        print(f"当前集合：{collections}")
    finally:
        client.close()
