import os

from app.knowledge.milvus import get_milvus_client
from app.knowledge.vectorize import vectorize_policy_documents

DOCUMENT_ID = "JW-POLICY-001"


def ingest_policy_documents() -> int:
    """将当前商城政策写入专用集合，并核对有效记录数。"""
    client = get_milvus_client()
    collection_name = os.environ["MILVUS_COLLECTION"]

    # 首次运行时创建集合
    if not client.has_collection(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            dimension=1024,
            metric_type="COSINE",
            auto_id=False,
            enable_dynamic_field=True,
            consistency_level="Strong",
        )

    client.load_collection(collection_name=collection_name)

    # 加载、切块、生成向量
    chunks, vectors = vectorize_policy_documents()

    data = [
        {
            "id": chunk.metadata["chunk_id"],
            "vector": vector,
            "text": chunk.page_content,
            "document_id": DOCUMENT_ID,
            "source": chunk.metadata["source"],
            "document_title": chunk.metadata["document_title"],
            "section_title": chunk.metadata.get("section_title", "文档说明"),
            "chunk_id": chunk.metadata["chunk_id"],
        }
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]

    # 相同主键覆盖，重复运行不会新增重复记录
    result = client.upsert(
        collection_name=collection_name,
        data=data,
    )
    print(f"本次写入数量：{result['upsert_count']}")

    # 如果政策删减导致块数减少，清理本政策遗留的旧块
    client.delete(
        collection_name=collection_name,
        filter=f'document_id == "{DOCUMENT_ID}" and id >= {len(data)}',
    )

    # 查询当前有效记录数
    counts = client.query(
        collection_name=collection_name,
        filter=f'document_id == "{DOCUMENT_ID}"',
        output_fields=["count(*)"],
        consistency_level="Strong",
    )
    count = counts[0]["count(*)"]

    if count != len(data):
        raise RuntimeError(f"入库数量不一致：预期 {len(data)}，实际 {count}")

    return count


if __name__ == "__main__":
    count = ingest_policy_documents()
    print(f"政策入库完成，有效记录数：{count}")
