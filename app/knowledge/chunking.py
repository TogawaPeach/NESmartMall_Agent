from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from app.knowledge.documents import load_policy_documents


def split_policy_documents(documents: list[Document]) -> list[Document]:
    """按 Markdown 标题切块，保留来源、标题和块编号。"""
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "document_title"),
            ("##", "section_title"),
        ],
        strip_headers=False,
    )

    chunks: list[Document] = []

    for document in documents:
        for chunk in splitter.split_text(document.page_content):
            chunk.metadata = {
                **document.metadata,
                **chunk.metadata,
                "chunk_id": len(chunks),
            }
            chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    documents = load_policy_documents()
    chunks = split_policy_documents(documents)

    print(f"切块数量：{len(chunks)}")

    for chunk in chunks:
        print(f"\n元数据：{chunk.metadata}")
        print(f"正文：\n{chunk.page_content}")
