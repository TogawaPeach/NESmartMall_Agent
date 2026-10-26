import os
from functools import lru_cache

from dotenv import load_dotenv
from pymilvus import MilvusClient

from app.knowledge.documents import PROJECT_ROOT


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


if __name__ == "__main__":
    client = get_milvus_client()

    try:
        collections = client.list_collections()
        print("Milvus 连接成功")
        print(f"当前集合：{collections}")
    finally:
        client.close()
