import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from pymilvus.exceptions import MilvusException

from app.knowledge.embeddings import get_embedding_model
from app.knowledge.milvus import ensure_collection_loaded, get_milvus_client

logger = logging.getLogger(__name__)

# 同步 Milvus 调用固定放在一个工作线程中
_MILVUS_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="knowledge-search",
)

_NOT_LOADED_HINTS = ("not loaded", "released", "call load()")


def _is_not_loaded_error(exc: MilvusException) -> bool:
    message = str(exc).lower()
    return any(hint in message for hint in _NOT_LOADED_HINTS)


async def search_knowledge(query: str, top_k: int = 4) -> str:
    """将问题向量化，返回检索到的候选知识文本。"""
    embeddings = get_embedding_model()
    query_vector = await embeddings.aembed_query(query)

    def search():
        client = get_milvus_client()
        collection_name = os.environ["MILVUS_COLLECTION"]

        def run_search():
            return client.search(
                collection_name=collection_name,
                data=[query_vector],
                anns_field="vector",
                limit=top_k,
                output_fields=["text"],
                search_params={"metric_type": "COSINE"},
                consistency_level="Strong",
            )

        # Milvus 重启或资源回收后集合会变成 Released，检索前先确保已加载
        ensure_collection_loaded(client, collection_name)

        try:
            return run_search()
        except MilvusException as exc:
            # 只针对“未加载”类错误重试，其它故障照常抛出
            if not _is_not_loaded_error(exc):
                raise
            logger.warning(
                "Milvus 集合 %s 未处于加载状态，重新加载后重试：%s",
                collection_name,
                exc,
            )
            ensure_collection_loaded(client, collection_name, force=True)
            return run_search()

    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(_MILVUS_EXECUTOR, search)

    if not results or not results[0]:
        return "知识库没有查到相关资料。"

    texts = [
        hit["entity"]["text"]
        for hit in results[0]
        if hit["entity"].get("text")
    ]

    return "\n\n".join(texts) or "知识库没有查到相关资料。"
