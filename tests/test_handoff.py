import asyncio
import json

from asyncmy.errors import IntegrityError
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessageChunk

from app.repositories.context_repository import ContextRepository
from app.services.ticket_service import TicketService


class MemoryRedis:
    def __init__(self):
        self.values = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, *, ex):
        self.values[key] = value
        return True


class MemoryTickets:
    def __init__(self):
        self.rows = {}

    async def create_ticket(self, ticket_no, question):
        if ticket_no in self.rows:
            raise IntegrityError(1062, "duplicate")
        self.rows[ticket_no] = {
            "ticket_no": str(ticket_no), "question": question,
            "answer": None, "is_deleted": 0,
        }
        return 1


def services():
    contexts = ContextRepository(MemoryRedis())
    tickets = MemoryTickets()
    service = TicketService(
        FakeListChatModel(responses=["A1 测距仪校准方法尚未解决。"]), tickets, 1,
    )
    return contexts, tickets, service


def test_each_submission_creates_a_new_ticket():
    async def run():
        contexts, tickets, service = services()
        action = await contexts.create_handoff_action("42", [], "我要转人工")
        first, second = await asyncio.gather(
            service.create_from_action("42", action, contexts),
            service.create_from_action("42", action, contexts),
        )
        assert first["ticket_no"] != second["ticket_no"]
        assert len(tickets.rows) == 2

    asyncio.run(run())


def test_ticket_endpoint_uses_owner_snapshot_and_returns_string_ticket_numbers():
    from app.api.agent import router

    contexts, tickets, service = services()
    action = asyncio.run(contexts.create_handoff_action("42", [], "我要投诉"))
    app = FastAPI()
    app.state.context_repository = contexts
    app.state.ticket_service = service
    app.include_router(router)
    with TestClient(app) as client:
        assert client.post("/internal/support/tickets", json={
            "userId": "43", "actionId": action,
        }).status_code == 410
        assert tickets.rows == {}
        first = client.post("/internal/support/tickets", json={
            "userId": "42", "actionId": action,
        })
        second = client.post("/internal/support/tickets", json={
            "userId": "42", "actionId": action,
        })
        assert first.status_code == second.status_code == 200
        assert first.json()["ticketNo"] != second.json()["ticketNo"]
        assert isinstance(first.json()["ticketNo"], str)
        assert len(tickets.rows) == 2
        assert "暂无真人接待" in first.json()["message"]


def test_stream_handoff_uses_fixed_snapshot_and_does_not_create_ticket(patch_stream):
    from app.api.agent import router
    from app.services import agent_service

    contexts, tickets, _service = services()

    async def stream(*args, **kwargs):
        yield AIMessageChunk(content="可以登记你的问题。"), {"langgraph_node": "model"}

    patch_stream(agent_service, "support", stream, show_handoff=True)
    app = FastAPI()
    app.state.context_repository = contexts
    app.include_router(router)
    with TestClient(app) as client:
        response = client.post("/internal/support/turns/stream", json={
            "userId": "42", "text": "我要投诉，也想问校准方法",
        })
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line]
    assert [event["type"] for event in events] == ["delta", "action", "done"]
    assert events[1]["payload"]["kind"] == "transfer_human"
    snapshot = asyncio.run(contexts.get_handoff_action("42", events[1]["payload"]["actionId"]))
    assert snapshot == {"history": [], "text": "我要投诉，也想问校准方法"}
    assert tickets.rows == {}


def test_expired_action_returns_gone_without_creating_ticket():
    from app.api.agent import router

    contexts, tickets, service = services()
    app = FastAPI()
    app.state.context_repository = contexts
    app.state.ticket_service = service
    app.include_router(router)
    with TestClient(app) as client:
        response = client.post("/internal/support/tickets", json={
            "userId": "42", "actionId": "expired-action",
        })
    assert response.status_code == 410
    assert tickets.rows == {}


def test_summary_failure_is_retryable_and_does_not_report_success():
    from app.api.agent import router

    contexts, tickets, service = services()
    action = asyncio.run(contexts.create_handoff_action("42", [], "我要转人工"))
    original_chain = service.summary_chain

    class FailedSummary:
        async def ainvoke(self, *args, **kwargs):
            raise RuntimeError("private provider failure")

    service.summary_chain = FailedSummary()
    app = FastAPI()
    app.state.context_repository = contexts
    app.state.ticket_service = service
    app.include_router(router)
    with TestClient(app) as client:
        response = client.post("/internal/support/tickets", json={
            "userId": "42", "actionId": action,
        })
        assert response.status_code == 503
        assert "private" not in response.text
        assert "ticketNo" not in response.json()
        assert tickets.rows == {}
        service.summary_chain = original_chain
        response = client.post("/internal/support/tickets", json={
            "userId": "42", "actionId": action,
        })
        assert response.status_code == 200
        assert len(tickets.rows) == 1
