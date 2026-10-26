import asyncio
import threading
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel


def test_retrieval_embeds_question_and_returns_text_without_metadata(monkeypatch):
    from app.knowledge import retrieval

    observed = {}

    async def embed_query(query):
        observed["query"] = query
        return [0.25, 0.75]

    def get_client():
        observed["client_thread"] = threading.get_ident()

        def search(**kwargs):
            assert kwargs["data"] == [[0.25, 0.75]]
            assert kwargs["anns_field"] == "vector"
            assert kwargs["collection_name"] == "test_knowledge"
            return [[
                {
                    "id": 5,
                    "distance": 0.88,
                    "entity": {"text": "新疆每单首程运费为25元。", "source": "/internal/policy.md"},
                },
            ]]

        return SimpleNamespace(search=search)

    monkeypatch.setenv("MILVUS_COLLECTION", "test_knowledge")
    monkeypatch.setattr(retrieval, "get_embedding_model", lambda: SimpleNamespace(
        aembed_query=embed_query,
    ))
    monkeypatch.setattr(retrieval, "get_milvus_client", get_client)

    result = asyncio.run(retrieval.search_knowledge("新疆运费怎么算？"))

    assert result == "新疆每单首程运费为25元。"
    assert observed["query"] == "新疆运费怎么算？"
    assert observed["client_thread"] != threading.get_ident()


def test_retrieval_does_not_disguise_database_failure_as_no_results(monkeypatch):
    from app.knowledge import retrieval

    async def embed_query(_query):
        return [0.25, 0.75]

    def search(**_kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setenv("MILVUS_COLLECTION", "test_knowledge")
    monkeypatch.setattr(retrieval, "get_embedding_model", lambda: SimpleNamespace(
        aembed_query=embed_query,
    ))
    monkeypatch.setattr(retrieval, "get_milvus_client", lambda: SimpleNamespace(search=search))

    with pytest.raises(RuntimeError, match="database unavailable"):
        asyncio.run(retrieval.search_knowledge("运费怎么算？"))


@pytest.mark.parametrize(
    ("response", "expected_query"),
    [
        ("  新疆首程配送运费如何收取？  ", "新疆首程配送运费如何收取？"),
        ("", "新疆运费咋算啊？"),
    ],
)
def test_knowledge_tool_uses_rewritten_question_or_original_when_empty(
    monkeypatch, response, expected_query
):
    from app.tools import knowledge

    observed = []

    async def search(query):
        observed.append(query)
        return "新疆每单首程运费为25元。"

    monkeypatch.setattr(knowledge, "search_knowledge", search)
    tool = knowledge.create_knowledge_tool(FakeListChatModel(responses=[response]))

    result = asyncio.run(tool.ainvoke({"query": "新疆运费咋算啊？"}))

    assert result == "新疆每单首程运费为25元。"
    assert observed == [expected_query]


def test_knowledge_tool_still_retrieves_when_rewrite_model_fails(monkeypatch):
    from app.tools import knowledge

    class UnavailableModel(FakeListChatModel):
        def _call(self, *args, **kwargs):
            raise RuntimeError("rewrite unavailable")

    observed = []

    async def search(query):
        observed.append(query)
        return "无理由退货的退回运费由用户承担。"

    monkeypatch.setattr(knowledge, "search_knowledge", search)
    tool = knowledge.create_knowledge_tool(UnavailableModel(responses=[""]))

    result = asyncio.run(tool.ainvoke({"query": "退货运费谁出？"}))

    assert result == "无理由退货的退回运费由用户承担。"
    assert observed == ["退货运费谁出？"]
