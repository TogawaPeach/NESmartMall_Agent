import asyncio
import json
from types import SimpleNamespace

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
    service, monkeypatch, agent_type
):
    from app.api.agent import router

    redis = MemoryRedis()
    app = FastAPI()
    app.state.context_repository = ContextRepository(redis)
    app.include_router(router)
    requests = []

    def make_agent(scene):
        async def stream(payload, *, stream_mode):
            assert stream_mode == "messages"
            requests.append((scene, [dict(message) for message in payload["messages"]]))
            yield AIMessageChunk(content=[{"type": "reasoning", "reasoning": "内部推理"}]), {}
            yield AIMessageChunk(content="你好"), {}
            assert f"context:{scene}:userid:1" not in redis.values or len(requests) > 1
            yield AIMessageChunk(content="，请说"), {}

        return SimpleNamespace(astream=stream)

    monkeypatch.setattr(service, "guide_agent", make_agent("guide"))
    monkeypatch.setattr(service, "support_agent", make_agent("support"))

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
def test_interrupted_stream_preserves_previous_history(service, monkeypatch, agent_type, failure):
    redis = MemoryRedis()
    key = f"context:{agent_type}:userid:1"
    original = json.dumps(
        [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "您好"},
        ]
    )
    redis.values[key] = original

    async def stream(_payload, *, stream_mode):
        yield AIMessageChunk(content="未完成的回答"), {}
        raise failure("stream interrupted")

    monkeypatch.setattr(service, f"{agent_type}_agent", SimpleNamespace(astream=stream))

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
