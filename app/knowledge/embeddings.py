import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from app.knowledge.documents import PROJECT_ROOT


@lru_cache(maxsize=1)
def get_embedding_model() -> OpenAIEmbeddings:
    """初始化并复用百炼向量模型客户端。"""
    load_dotenv(PROJECT_ROOT / ".env")

    return OpenAIEmbeddings(
        model=os.environ["EMBEDDING_MODEL"],
        api_key=os.environ["EMBEDDING_API_KEY"],
        base_url=os.environ["EMBEDDING_BASE_URL"],
        dimensions=1024,
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"},
        chunk_size=10,
        request_timeout=30,
        max_retries=0,
    )


if __name__ == "__main__":
    embeddings = get_embedding_model()

    # 先用两条短文本检查接口和向量维度
    vectors = embeddings.embed_documents(["邮费是多少？", "运费怎么算？"])

    print(f"向量数量：{len(vectors)}")
    print(f"每条向量的维度：{[len(vector) for vector in vectors]}")
