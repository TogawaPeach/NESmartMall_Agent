import asyncio
import json
from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage


@asynccontextmanager
async def gateway(handler):
    application = web.Application()
    application.router.add_get("/api/orders/{order_no}", handler)
    runner = web.AppRunner(application)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


def make_agent(tool, order_no="JW2112134513910874112"):
    class ToolCallingModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    model = ToolCallingModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "query_order",
            "args": {"order_no": order_no},
            "id": "order-call",
            "type": "tool_call",
        }]),
        AIMessage(content="订单查询结束。"),
    ])
    return create_agent(model=model, tools=[tool])


async def query(tool, authorization="Bearer user-a", order_no="JW2112134513910874112"):
    from app.schemas.agent_context import AgentContext

    result = await make_agent(tool, order_no).ainvoke(
        {"messages": [{"role": "user", "content": "查询订单"}]},
        context=AgentContext(authorization=authorization),
    )
    message = next(message for message in result["messages"] if isinstance(message, ToolMessage))
    assert authorization not in str(result["messages"]) if authorization else True
    return json.loads(message.content)


@pytest.mark.parametrize(
    ("paid_at", "paid_amount", "status"),
    [(None, None, "CANCELLED"), ("2026-10-19T10:55:00Z", 14700, "PENDING_SHIPMENT")],
)
def test_order_tool_forwards_jwt_and_returns_only_status_payment_time_and_amounts(
    paid_at, paid_amount, status
):
    from app.tools.order import create_order_tool

    async def run():
        async def handler(request):
            assert request.headers["Authorization"] == "Bearer user-a"
            assert request.match_info["order_no"] == "JW2112134513910874112"
            return web.json_response({"code": 200, "message": "OK", "data": {
                "orderNo": "JW2112134513910874112",
                "status": status,
                "paidAt": paid_at,
                "payableAmountFen": 14700,
                "paidAmountFen": paid_amount,
                "recipient": {"name": "private", "address": "private", "phone": "123****1234"},
                "items": [{"name": "private product"}],
                "logistics": None,
            }})

        async with gateway(handler) as base_url:
            tool = create_order_tool(base_url)
            assert set(tool.tool_call_schema.model_json_schema()["properties"]) == {"order_no"}
            assert await query(tool) == {
                "paidAt": paid_at,
                "status": status,
                "payableAmountFen": 14700,
                "paidAmountFen": paid_amount,
            }

    asyncio.run(run())


def test_concurrent_order_queries_keep_each_requests_jwt():
    from app.tools.order import create_order_tool

    async def run():
        arrived = asyncio.Event()
        tokens = []

        async def handler(request):
            token = request.headers["Authorization"]
            tokens.append(token)
            if len(tokens) == 2:
                arrived.set()
            await asyncio.wait_for(arrived.wait(), timeout=3)
            return web.json_response({"code": 200, "data": {
                "status": "COMPLETED" if token == "Bearer user-a" else "CANCELLED",
                "paidAt": None,
                "payableAmountFen": 14700,
                "paidAmountFen": None,
            }})

        async with gateway(handler) as base_url:
            tool = create_order_tool(base_url)
            first, second = await asyncio.gather(
                query(tool, "Bearer user-a"), query(tool, "Bearer user-b")
            )
            assert first["status"] == "COMPLETED"
            assert second["status"] == "CANCELLED"
            assert sorted(tokens) == ["Bearer user-a", "Bearer user-b"]

    asyncio.run(run())


@pytest.mark.parametrize(
    ("status", "error"),
    [(401, "UNAUTHORIZED"), (403, "UNAUTHORIZED"), (404, "ORDER_NOT_ACCESSIBLE"),
     (500, "SERVICE_ERROR"), (302, "SERVICE_ERROR")],
)
def test_http_failure_does_not_return_order_data(status, error):
    from app.tools.order import create_order_tool

    async def run():
        async def handler(_request):
            return web.Response(status=status, text="private upstream error")

        async with gateway(handler) as base_url:
            result = await query(create_order_tool(base_url))
            assert result["error"] == error
            assert "status" not in result
            assert "private" not in str(result)

    asyncio.run(run())


@pytest.mark.parametrize("payload", [{"data": None}, {"data": {}}, [], "invalid"])
def test_malformed_response_does_not_crash_the_agent(payload):
    from app.tools.order import create_order_tool

    async def run():
        async def handler(_request):
            return web.json_response(payload)

        async with gateway(handler) as base_url:
            assert (await query(create_order_tool(base_url)))["error"] == "SERVICE_ERROR"

    asyncio.run(run())


@pytest.mark.parametrize(
    ("authorization", "order_no", "error"),
    [("", "JW1", "UNAUTHORIZED"), ("Bearer user-a", "../orders/JW1", "INVALID_ORDER_NO")],
)
def test_invalid_input_does_not_send_a_request(authorization, order_no, error):
    from app.tools.order import create_order_tool

    async def run():
        async def handler(_request):
            pytest.fail("invalid input must not reach the gateway")

        async with gateway(handler) as base_url:
            result = await query(create_order_tool(base_url), authorization, order_no)
            assert result["error"] == error

    asyncio.run(run())


def test_java_client_automatically_adds_jwt_to_each_request():
    from app.clients.java import create_java_client

    async def run():
        received = []

        async def handler(request):
            received.append((request.match_info["order_no"], request.headers["Authorization"]))
            return web.json_response({"code": 200, "message": "OK", "data": {
                "paidAt": None,
                "status": "CANCELLED",
                "payableAmountFen": 14700,
                "paidAmountFen": None,
            }})

        async with gateway(handler) as base_url:
            async with create_java_client(base_url, "Bearer client-jwt") as client:
                for order_no in ("JW1", "JW2"):
                    async with await client.get_order_detail(order_no) as response:
                        assert response.status == 200
                        assert (await response.json())["data"]["status"] == "CANCELLED"

        assert received == [("JW1", "Bearer client-jwt"), ("JW2", "Bearer client-jwt")]

    asyncio.run(run())


def test_timeout_returns_a_query_failure(monkeypatch):
    from app.clients.java import JavaClient
    from app.tools import order

    async def timeout(*args, **kwargs):
        raise TimeoutError

    monkeypatch.setattr(JavaClient, "get_order_detail", timeout)
    result = asyncio.run(query(order.create_order_tool("http://127.0.0.1:8080")))
    assert result["error"] == "TIMEOUT"
