import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest


@pytest.fixture
def patch_stream(monkeypatch):
    """历史/SSE测试替换生成边界，实际图的分支另用真实LangGraph测试。"""
    def install(service, agent_type, stream, *, show_handoff=False):
        if agent_type == "guide":
            monkeypatch.setattr(service, "guide_agent", SimpleNamespace(astream=stream))
            return

        async def graph_stream(payload, *, context, stream_mode):
            assert stream_mode == ["messages", "updates"]
            yield "updates", {"route": {"show_handoff": show_handoff}}
            parts = []
            async for chunk, metadata in stream({"messages": [
                *payload["history"], {"role": "user", "content": payload["text"]},
            ]}, context=context, stream_mode="messages"):
                if metadata.get("langgraph_node") == "model":
                    metadata = {"langgraph_node": "answer", "tags": ["support_answer"]}
                    parts.extend(b["text"] for b in chunk.content_blocks if b["type"] == "text")
                yield "messages", (chunk, metadata)
            yield "updates", {"answer": {"answer": "".join(parts)}}

        monkeypatch.setattr(service, "support_workflow", SimpleNamespace(astream=graph_stream))

    return install


@pytest.fixture(autouse=True)
def offline_configuration(monkeypatch):
    """测试仅使用临时配置，禁止加载真实 .env 或连接 MySQL。"""
    from app.db import mysql
    from app.settings import Settings

    for key, value in {
        "DEEPSEEK_API_KEY": "test-api-key",
        "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
        "MALL_GATEWAY_BASE_URL": "http://127.0.0.1:8080",
        "AGENT_MYSQL_HOST": "127.0.0.1",
        "AGENT_MYSQL_USER": "test-user",
        "AGENT_MYSQL_PASSWORD": "test-password",
        "AGENT_MYSQL_DATABASE": "test-agent",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *_args, **_kwargs: False)
    for module_name in ("app.services.agent_service", "app.services.nacos_service"):
        if module_name in sys.modules:
            monkeypatch.setattr(sys.modules[module_name], "load_dotenv", lambda *a, **kw: False)

    @asynccontextmanager
    async def mysql_connection(_settings):
        yield object()

    monkeypatch.setattr(mysql, "mysql_connection", mysql_connection)
    if "app.main" in sys.modules:
        monkeypatch.setattr(sys.modules["app.main"], "mysql_connection", mysql_connection)
