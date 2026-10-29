import asyncio
import os

from app.knowledge.embeddings import get_embedding_model
from app.knowledge.milvus import get_milvus_client


async def upsert_product_document(product_id, markdown, price):
    collection_name = os.environ["MILVUS_PRODUCT_COLLECTION"]

    if collection_name == os.environ["MILVUS_COLLECTION"]:
        raise ValueError("商品集合不能与商城规则集合相同")

    vectors = await get_embedding_model().aembed_documents([markdown])

    def write():
        client = get_milvus_client()

        if not client.has_collection(collection_name=collection_name):
            client.create_collection(
                collection_name=collection_name,
                dimension=1024,
                metric_type="COSINE",
                auto_id=False,
                enable_dynamic_field=True,
                consistency_level="Strong",
            )

        client.upsert(
            collection_name=collection_name,
            data=[{
                "id": int(product_id),
                "vector": vectors[0],
                "text": markdown,
                "price": price,
            }],
        )

    await asyncio.to_thread(write)
