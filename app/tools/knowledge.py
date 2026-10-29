import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, tool

from app.knowledge.retrieval import search_knowledge, search_knowledge_candidates

logger = logging.getLogger(__name__)


def create_query_rewriter(model: BaseChatModel):
    """导购工具与客服 Workflow 共用问题改写。"""
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你负责将用户问题改写成适合知识库检索的完整问题。"
                "去掉寒暄、重复和情绪表达，保留原意。"
                "必须保留商品型号、地区、金额、期限、否定和限制条件。"
                "不得添加原问题没有的事实，不得回答问题。"
                "只输出改写后的问题；已经清楚的问题可以保持原样。",
            ),
            ("human", "{query}"),
        ]
    )
    rewrite_chain = rewrite_prompt | model | StrOutputParser()

    async def rewrite_query(query: str) -> str:
        try:
            rewritten = await rewrite_chain.ainvoke(
                {"query": query}, config={"tags": ["query_rewrite"]},
            )
            return rewritten.strip() or query
        except Exception:
            logger.warning("检索问题改写失败，使用原问题继续检索。")
            return query

    return rewrite_query


def create_knowledge_retriever(model: BaseChatModel):
    rewrite_query = create_query_rewriter(model)

    async def retrieve(query: str):
        return await search_knowledge_candidates(await rewrite_query(query))

    return retrieve


def create_knowledge_tool(model: BaseChatModel) -> BaseTool:
    """使用现有聊天模型创建知识检索工具。"""
    rewrite_query = create_query_rewriter(model)

    @tool
    async def query_knowledge(query: str) -> str:
        """检索商城知识库中的政策、规则和商品说明等已入库内容。

        Args:
            query: 结合对话历史补全后的完整问题，保留具体条件。
        """
        query = query.strip()
        if not query:
            return "请提供需要查询的完整问题。"

        return await search_knowledge(await rewrite_query(query))

    return query_knowledge
