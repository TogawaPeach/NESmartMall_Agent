from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_FILE = PROJECT_ROOT / "mall-policies.md"


def load_policy_documents() -> list[Document]:
    """读取商城服务政策，保留完整正文和来源路径。"""
    loader = TextLoader(
        file_path=str(POLICY_FILE),
        encoding="utf-8",
    )
    return loader.load()


# 手动执行时检查加载结果
if __name__ == "__main__":
    documents = load_policy_documents()

    print(f"加载文档数量：{len(documents)}")

    for document in documents:
        print(f"元数据：{document.metadata}")
        print(f"正文字符数：{len(document.page_content)}")
        print(f"正文前 300 字：\n{document.page_content[:300]}")
