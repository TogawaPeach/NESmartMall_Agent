from langchain_core.documents import Document

from app.knowledge.chunking import split_policy_documents
from app.knowledge.documents import load_policy_documents
from app.knowledge.embeddings import get_embedding_model


def vectorize_policy_documents() -> tuple[list[Document], list[list[float]]]:
    """加载政策、切块，并生成与各块对应的向量。"""
    documents = load_policy_documents()
    chunks = split_policy_documents(documents)

    embeddings = get_embedding_model()
    vectors = embeddings.embed_documents(
        [chunk.page_content for chunk in chunks]
    )

    return chunks, vectors


if __name__ == "__main__":
    chunks, vectors = vectorize_policy_documents()

    print(f"文档块数量：{len(chunks)}")
    print(f"向量数量：{len(vectors)}")

    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk_id = chunk.metadata["chunk_id"]
        title = chunk.metadata.get("section_title", "文档说明")

        print(f"块编号：{chunk_id}，章节：{title}，向量维度：{len(vector)}")
