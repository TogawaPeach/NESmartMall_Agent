import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from app.repositories.context_repository import ContextRepository


class MemoryRedis:
    def __init__(self):
        self.values = {}
        self.expirations = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, *, ex):
        self.values[key] = value
        self.expirations[key] = ex


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-api-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *_args, **_kwargs: False)
    from app.services import agent_service

    return agent_service


@pytest.mark.parametrize("agent_type", ["guide", "support"])
def test_stream_endpoint_remembers_turns_and_saves_complete_answers(
    service, monkeypatch, agent_type, patch_stream
):
    from app.api.agent import router

    redis = MemoryRedis()
    app = FastAPI()
    app.state.context_repository = ContextRepository(redis)
    app.include_router(router)
    requests = []

    def make_agent(scene):
        async def stream(payload, *, stream_mode, context=None):
            assert stream_mode == "messages"
            requests.append((scene, [dict(message) for message in payload["messages"]]))
            yield AIMessageChunk(content="改写后的内部检索问题"), {
                "langgraph_node": "tools",
                "tags": ["query_rewrite"],
            }
            yield AIMessageChunk(content=[{"type": "reasoning", "reasoning": "内部推理"}]), {
                "langgraph_node": "model",
            }
            yield AIMessageChunk(content="你好"), {"langgraph_node": "model"}
            assert f"context:{scene}:userid:1" not in redis.values or len(requests) > 1
            yield AIMessageChunk(content="，请说"), {"langgraph_node": "model"}

        return stream

    patch_stream(service, agent_type, make_agent(agent_type))

    with TestClient(app) as client:
        response = client.post(
            f"/internal/{agent_type}/turns/stream", json={"userId": 1, "text": "选测距仪"}
        )
        assert response.status_code == 200
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line]
        assert events == [
            {"type": "delta", "payload": {"text": "你好"}},
            {"type": "delta", "payload": {"text": "，请说"}},
            {"type": "done", "payload": {}},
        ]
        key = f"context:{agent_type}:userid:1"
        first_turn = [
            {"role": "user", "content": "选测距仪"},
            {"role": "assistant", "content": "你好，请说"},
        ]
        assert json.loads(redis.values[key]) == first_turn
        assert redis.expirations[key] == 21600

        response = client.post(
            f"/internal/{agent_type}/turns/stream", json={"userId": 1, "text": "用于装修"}
        )
        assert response.status_code == 200

    assert requests == [
        (agent_type, [{"role": "user", "content": "选测距仪"}]),
        (agent_type, [*first_turn, {"role": "user", "content": "用于装修"}]),
    ]
    assert json.loads(redis.values[key]) == [
        *first_turn,
        {"role": "user", "content": "用于装修"},
        {"role": "assistant", "content": "你好，请说"},
    ]
    assert set(redis.values) == {key}


@pytest.mark.parametrize("agent_type", ["guide", "support"])
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_interrupted_stream_preserves_previous_history(service, agent_type, failure, patch_stream):
    redis = MemoryRedis()
    key = f"context:{agent_type}:userid:1"
    original = json.dumps(
        [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "您好"},
        ]
    )
    redis.values[key] = original

    async def stream(_payload, *, stream_mode, context=None):
        yield AIMessageChunk(content="未完成的回答"), {"langgraph_node": "model"}
        raise failure("stream interrupted")

    patch_stream(service, agent_type, stream)

    async def consume():
        answer = getattr(service, f"stream_{agent_type}_answer")
        chunks = []
        with pytest.raises(failure, match="stream interrupted"):
            async for text in answer("1", "新的问题", ContextRepository(redis)):
                chunks.append(text)
        assert chunks == ["未完成的回答"]

    asyncio.run(consume())
    assert redis.values == {key: original}
    assert redis.expirations == {}


@pytest.mark.parametrize("agent_type", ["guide"])
def test_real_agent_graph_keeps_rewrite_and_evidence_out_of_reply_and_history(
    service, monkeypatch, agent_type
):
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    from app.tools import knowledge

    class ToolCallingModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    model = ToolCallingModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "query_knowledge",
            "args": {"query": "新疆运费怎么算？"},
            "id": "test-knowledge-call",
            "type": "tool_call",
        }]),
        AIMessage(content="新疆地区首程配送运费如何收取？"),
        AIMessage(content="新疆每单收取25元首程配送运费。"),
    ])
    queries = []

    async def search(query):
        queries.append(query)
        return "【内部检索原文】新疆、西藏地区每单收取25元首程配送运费。"

    monkeypatch.setattr(knowledge, "search_knowledge", search)
    agent = create_agent(model=model, tools=[knowledge.create_knowledge_tool(model)])
    monkeypatch.setattr(service, f"{agent_type}_agent", agent)
    redis = MemoryRedis()

    async def consume():
        stream = getattr(service, f"stream_{agent_type}_answer")
        return "".join([
            text async for text in stream(
                "1", "新疆运费怎么算？", ContextRepository(redis)
            )
        ])

    answer = asyncio.run(consume())

    assert answer == "新疆每单收取25元首程配送运费。"
    assert queries == ["新疆地区首程配送运费如何收取？"]
    assert json.loads(redis.values[f"context:{agent_type}:userid:1"]) == [
        {"role": "user", "content": "新疆运费怎么算？"},
        {"role": "assistant", "content": "新疆每单收取25元首程配送运费。"},
    ]


def test_support_endpoint_passes_current_jwt_without_saving_it(service, patch_stream):
    from app.api.agent import router

    redis = MemoryRedis()
    app = FastAPI()
    app.state.context_repository = ContextRepository(redis)
    app.include_router(router)
    authorizations = []

    async def stream(payload, *, stream_mode, context=None):
        assert context is not None
        authorizations.append(context.authorization)
        assert "Bearer" not in str(payload)
        yield AIMessageChunk(content="已查询订单。"), {"langgraph_node": "model"}

    patch_stream(service, "support", stream)

    with TestClient(app) as client:
        for token in ("Bearer initial-jwt", "Bearer refreshed-jwt"):
            response = client.post(
                "/internal/support/turns/stream",
                json={"userId": "1", "text": "查询订单 JW1"},
                headers={"Authorization": token},
            )
            assert response.status_code == 200
            assert "已查询订单。" in response.text
            assert token not in response.text

    assert authorizations == ["Bearer initial-jwt", "Bearer refreshed-jwt"]
    assert "Bearer" not in str(redis.values)
