import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

from app.knowledge.embeddings import get_embedding_model
from app.knowledge.milvus import get_milvus_client

# 同步 Milvus 调用固定放在一个工作线程中
_MILVUS_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="knowledge-search",
)


async def search_knowledge(query: str, top_k: int = 4) -> str:
    """将问题向量化，返回检索到的候选知识文本。"""
    embeddings = get_embedding_model()
    query_vector = await embeddings.aembed_query(query)

    def search():
        client = get_milvus_client()

        return client.search(
            collection_name=os.environ["MILVUS_COLLECTION"],
            data=[query_vector],
            anns_field="vector",
            limit=top_k,
            output_fields=["text"],
            search_params={"metric_type": "COSINE"},
            consistency_level="Strong",
        )

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
