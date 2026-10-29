import asyncio
import json
from contextlib import asynccontextmanager

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from app.schemas.agent_context import AgentContext
from app.schemas.support_workflow import RouteResult
from app.services import support_workflow as workflow


class WorkflowModel(FakeListChatModel):
    decision: dict = {}
    parts: list[dict] = []

    def with_structured_output(self, schema, **kwargs):
        def respond(_prompt):
            data = self.decision if schema is RouteResult else {"parts": self.parts}
            return schema(**data)
        return FakeListChatModel(responses=["内部分类与证据检查文本"]) | RunnableLambda(respond)


def make_graph(model, retrieve, base_url="http://gateway"):
    return workflow.create_support_workflow(
        model,
        workflow.create_knowledge_node(retrieve, model),
        workflow.create_order_node(base_url),
    )


def test_composite_complaint_retrieves_and_preserves_partial_evidence():
    model = WorkflowModel(
        responses=["新疆首程运费25元，校准资料暂缺；可以登记问题。"],
        decision={
            "route": "other", "secondary_routes": ["knowledge"],
            "question": "我要投诉，问新疆运费和A1校准方法", "show_handoff": True,
        },
        parts=[
            {"question": "新疆运费", "status": "supported", "facts": "新疆首程运费25元"},
            {"question": "A1校准", "status": "missing", "facts": "没有该型号校准资料"},
        ],
    )

    async def retrieve(query):
        assert "A1" in query
        return [{"id": 5, "score": 0.88, "score_type": "COSINE", "source": "policy",
                 "document_id": "policy", "text": "新疆首程运费25元"}]

    result = asyncio.run(make_graph(model, retrieve).ainvoke({"text": "我要投诉", "history": []}))
    assert result["evidence"].parts[0].status == "supported"
    assert result["evidence"].parts[1].status == "missing"
    assert result["show_handoff"] is True
    assert result["other_text"]
    assert "校准资料暂缺" in result["answer"]


def test_return_question_queries_current_order_before_retrieval(monkeypatch):
    calls = []

    @asynccontextmanager
    async def client(base_url, authorization):
        assert authorization == "Bearer current-user"
        assert base_url == "http://gateway"

        class Response:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def json(self):
                return {"data": {
                    "orderNo": "JW123", "status": "COMPLETED", "paidAt": None,
                    "payableAmountFen": 100, "paidAmountFen": 100,
                    "items": [{"name": "A1", "model": "A1", "quantity": 1}],
                    "recipient": {"phone": "private"},
                }}

        class Client:
            async def get_order_detail(self, order_no):
                calls.append("order")
                assert order_no == "JW123"
                return Response()

        yield Client()

    monkeypatch.setattr(workflow, "create_java_client", client, raising=False)
    model = WorkflowModel(
        responses=["订单已完成；请补充签收时间与退货原因。"],
        decision={"route": "order", "order_no": "JW123", "order_task": "return",
                  "question": "订单JW123能退吗"},
        parts=[{"question": "退货条件", "status": "clarify", "facts": "缺签收时间和退货原因"}],
    )

    async def retrieve(_query):
        calls.append("knowledge")
        return [{"id": 1, "text": "退货需要核实时间与原因"}]

    result = asyncio.run(make_graph(model, retrieve).ainvoke(
        {"text": "能退吗", "history": [
            {"role": "assistant", "content": "请提供订单号"},
            {"role": "user", "content": "JW123"},
        ]}, context=AgentContext(authorization="Bearer current-user"),
    ))
    assert calls == ["order", "knowledge"]
    assert result["order_result"]["data"]["status"] == "COMPLETED"
    assert "private" not in json.dumps(result, default=str)
    assert "Bearer" not in json.dumps(result, default=str)
    assert result["evidence"].parts[0].status == "clarify"


@pytest.mark.parametrize("failure", [False, True])
def test_return_without_order_facts_does_not_retrieve_or_judge(failure, monkeypatch):
    @asynccontextmanager
    async def failed_client(*args):
        raise RuntimeError("private upstream error")
        yield

    monkeypatch.setattr(workflow, "create_java_client", failed_client, raising=False)
    model = WorkflowModel(
        responses=["暂时无法核实，请提供或核对订单。"],
        decision={"route": "order", "order_task": "return", "question": "这单能退吗",
                  "order_no": "JW123" if failure else None},
    )

    async def retrieve(_query):
        pytest.fail("没有订单事实，不能进入退货规则核实")

    result = asyncio.run(make_graph(model, retrieve).ainvoke(
        {"text": "这单能退吗", "history": []}, context=AgentContext(authorization="jwt"),
    ))
    assert result["order_result"]["kind"] == ("error" if failure else "clarify")
    assert "evidence" not in result
    assert "private" not in json.dumps(result, default=str)


@pytest.mark.parametrize("failure", [False, True])
def test_empty_evidence_and_service_failure_are_distinct(failure):
    model = WorkflowModel(
        responses=["暂时无法提供依据。"],
        decision={"route": "knowledge", "question": "A1如何校准"},
        parts=[{"question": "A1校准", "status": "missing", "facts": "没有A1校准资料"}],
    )

    async def retrieve(_query):
        if failure:
            raise RuntimeError("private Milvus error")
        return []

    result = asyncio.run(make_graph(model, retrieve).ainvoke({"text": "校准", "history": []}))
    if failure:
        assert result["knowledge_error"]
        assert result["show_handoff"] is False
        assert "evidence" not in result
    else:
        assert result["evidence"].parts[0].status == "missing"
        assert result["show_handoff"] is True


def test_stream_outputs_only_final_answer_model():
    model = WorkflowModel(
        responses=["正式回答"],
        decision={"route": "knowledge", "question": "运费"},
        parts=[{"question": "运费", "status": "supported", "facts": "25元"}],
    )

    async def retrieve(_query):
        return [{"id": 1, "text": "25元"}]

    async def consume():
        chunks = []
        async for kind, event in make_graph(model, retrieve).astream(
            {"text": "运费", "history": []}, stream_mode=["messages", "updates"],
        ):
            if kind == "messages":
                chunk, metadata = event
                if "support_answer" in metadata.get("tags", []):
                    chunks.append(chunk.content)
        return "".join(chunks)

    assert asyncio.run(consume()) == "正式回答"


def test_support_endpoint_runs_graph_filters_internal_models_and_saves_answer(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.agent import router
    from app.repositories.context_repository import ContextRepository
    from app.services import agent_service

    class MemoryRedis:
        def __init__(self):
            self.values = {}

        async def get(self, key):
            return self.values.get(key)

        async def set(self, key, value, **kwargs):
            self.values[key] = value

    async def old_stream(*args, **kwargs):
        raise AssertionError("不应继续调用旧客服Agent入口")
        yield

    async def old_route(state):
        return {"decision": RouteResult(route="other", question=state["text"])}

    model = WorkflowModel(
        responses=["有依据的正式回答。"],
        decision={"route": "other", "question": "我要投诉", "show_handoff": True},
    )

    async def retrieve(_query):
        pytest.fail("投诉不应触发知识检索")

    monkeypatch.setattr(
        agent_service, "support_workflow", make_graph(model, retrieve), raising=False,
    )
    old_agent = type("Agent", (), {"astream": old_stream})()
    monkeypatch.setattr(agent_service, "support_agent", old_agent, raising=False)
    monkeypatch.setattr(agent_service, "support_route_node", old_route, raising=False)
    redis = MemoryRedis()
    contexts = ContextRepository(redis)
    app = FastAPI()
    app.state.context_repository = contexts
    app.include_router(router)
    with TestClient(app) as client:
        response = client.post("/internal/support/turns/stream", json={
            "userId": "42", "text": "我要投诉",
        })
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line]
    answer = "".join(e["payload"]["text"] for e in events if e["type"] == "delta")
    assert answer == "有依据的正式回答。"
    assert [e["type"] for e in events if e["type"] != "delta"] == ["action", "done"]
    assert json.loads(redis.values["context:support:userid:42"])[-1] == {
        "role": "assistant", "content": "有依据的正式回答。",
    }
    action_id = next(e["payload"]["actionId"] for e in events if e["type"] == "action")
    assert asyncio.run(contexts.get_handoff_action("42", action_id)) == {
        "history": [], "text": "我要投诉",
    }


def test_return_task_cannot_bypass_order_branch_when_route_is_knowledge():
    model = WorkflowModel(
        responses=["请先明确订单。"],
        decision={"route": "knowledge", "order_task": "return", "question": "这单能退吗"},
    )

    async def retrieve(_query):
        return []

    result = asyncio.run(make_graph(model, retrieve).ainvoke({"text": "能退吗", "history": []}))
    assert result["order_result"]["kind"] == "clarify"


def test_order_list_keeps_candidates_without_selecting_a_target(monkeypatch):
    @asynccontextmanager
    async def client(_base_url, authorization):
        assert authorization == "Bearer list-user"

        class Response:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def json(self):
                return {"data": [
                    {"orderNo": "JW1", "status": "COMPLETED", "items": []},
                    {"orderNo": "JW2", "status": "PENDING_RECEIPT", "items": []},
                ]}

        class Client:
            async def get_user_orders(self):
                return Response()

        yield Client()

    monkeypatch.setattr(workflow, "create_java_client", client)
    model = WorkflowModel(
        responses=["查到两笔订单，请选择目标订单。"],
        decision={"route": "order", "order_task": "list", "question": "帮我找订单号"},
    )

    async def retrieve(_query):
        pytest.fail("寻找订单号不需要查询政策")

    result = asyncio.run(make_graph(model, retrieve).ainvoke(
        {"text": "找订单", "history": []},
        context=AgentContext(authorization="Bearer list-user"),
    ))
    assert result["order_result"]["kind"] == "list"
    assert [o["orderNo"] for o in result["order_result"]["data"]] == ["JW1", "JW2"]
    assert result["decision"].order_no is None


def test_evidence_model_failure_is_not_a_knowledge_gap():
    class UnavailableEvidenceModel(WorkflowModel):
        def with_structured_output(self, schema, **kwargs):
            if schema is RouteResult:
                return super().with_structured_output(schema, **kwargs)

            def fail(_prompt):
                raise RuntimeError("private provider failure")

            return RunnableLambda(fail)

    model = UnavailableEvidenceModel(
        responses=["暂时无法核实。"], decision={"route": "knowledge", "question": "运费？"},
    )

    async def retrieve(_query):
        return [{"id": 1, "text": "运费25元"}]

    result = asyncio.run(make_graph(model, retrieve).ainvoke({"text": "运费", "history": []}))
    assert result["knowledge_error"]
    assert result["show_handoff"] is False
    assert "evidence" not in result


@pytest.mark.parametrize("independent", [None, "新疆运费多少"])
def test_failed_return_only_retrieves_independent_knowledge(independent):
    queries = []
    model = WorkflowModel(
        responses=["请先明确订单；独立政策问题另行核实。"],
        decision={
            "route": "knowledge", "order_task": "return", "question": "这单能退吗？新疆运费多少",
            "knowledge_question": independent,
        },
        parts=[{"question": "新疆运费", "status": "missing", "facts": "没有运费资料"}],
    )

    async def retrieve(query):
        queries.append(query)
        return []

    result = asyncio.run(make_graph(model, retrieve).ainvoke({"text": "能退吗", "history": []}))
    assert result["order_result"]["kind"] == "clarify"
    assert queries == (["新疆运费多少"] if independent else [])
    assert result["show_handoff"] is bool(independent)


def test_missing_user_conditions_still_clarified_when_retrieval_is_empty():
    model = WorkflowModel(
        responses=["请提供测距仪型号。"],
        decision={"route": "knowledge", "question": "测距仪怎么校准"},
        parts=[{"question": "测距仪校准", "status": "clarify", "facts": "请提供型号"}],
    )

    async def retrieve(_query):
        return []

    result = asyncio.run(make_graph(model, retrieve).ainvoke({"text": "怎么校准", "history": []}))
    assert result["evidence"].parts[0].status == "clarify"
    assert result["show_handoff"] is False


def test_multiple_orders_require_clarification_without_silently_choosing(monkeypatch):
    @asynccontextmanager
    async def no_query(*args):
        pytest.fail("明确提及多笔订单时不能只选其中一笔")
        yield

    monkeypatch.setattr(workflow, "create_java_client", no_query)
    model = WorkflowModel(
        responses=["请明确先查询哪一笔订单。"],
        decision={"route": "order", "order_no": "JW1", "question": "查JW1和JW2的状态"},
    )

    async def retrieve(_query):
        pytest.fail("订单澄清不应检索知识")

    result = asyncio.run(make_graph(model, retrieve).ainvoke({
        "text": "查JW1和JW2的状态", "history": [],
    }, context=AgentContext(authorization="current")))
    assert result["order_result"]["kind"] == "clarify"


def test_deepseek_stream_disables_thinking_only_for_forced_structured_calls():
    import httpx
    from langchain_deepseek import ChatDeepSeek

    requests = []

    def response_chunk(delta, finish_reason=None):
        return {
            "id": "test-completion", "object": "chat.completion.chunk", "created": 0,
            "model": "deepseek-flash", "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason},
            ],
        }

    async def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        name = body["tools"][0]["function"]["name"] if body.get("tools") else None
        if name:
            if body.get("thinking", {}).get("type", "enabled") != "disabled":
                return httpx.Response(400, json={"error": {
                    "message": "Thinking mode does not support this tool_choice",
                    "type": "invalid_request_error", "code": "invalid_request_error",
                }})
            arguments = (
                {"route": "knowledge", "question": "新疆运费多少"}
                if name == "RouteResult" else {"parts": [
                    {"question": "新疆运费", "status": "supported", "facts": "运费25元"},
                ]}
            )
            chunks = [response_chunk({"tool_calls": [{
                "index": 0, "id": "test-call", "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }]}, "tool_calls")]
        else:
            chunks = [
                response_chunk({"reasoning_content": "内部思考"}),
                response_chunk({"content": "新疆运费25元。"}, "stop"),
            ]
        content = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            content=content + "data: [DONE]\n\n",
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            model = ChatDeepSeek(
                model="deepseek-flash", api_key="test-key", api_base="https://deepseek.invalid",
                http_async_client=client, max_retries=0,
                extra_body={"thinking": {"type": "enabled"}},
            )

            async def retrieve(_query):
                return [{"id": 1, "text": "新疆运费25元"}]

            answer = []
            async for kind, event in make_graph(model, retrieve).astream(
                {"text": "新疆运费多少", "history": []},
                stream_mode=["messages", "updates"],
            ):
                if kind == "messages":
                    chunk, metadata = event
                    if metadata.get("langgraph_node") == "answer":
                        answer.extend(
                            b["text"] for b in chunk.content_blocks if b["type"] == "text"
                        )
            return "".join(answer)

    assert asyncio.run(run()) == "新疆运费25元。"
    assert [body["thinking"]["type"] for body in requests] == ["disabled", "disabled", "enabled"]
