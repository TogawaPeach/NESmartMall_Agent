import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.services.agent_service import (
    stream_guide_answer,
    stream_support_answer,
)

router = APIRouter(
    prefix="/internal",
    tags=["Agent"],
)


# 两个接口共用 SSE 包装，各自传入对应 Agent 的文本流
def stream_response(chunks: AsyncIterator[str]) -> StreamingResponse:
    async def generate_events():
        async for text in chunks:
            event = {
                "type": "delta",
                "payload": {"text": text},
            }
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        yield 'data: {"type":"done","payload":{}}\n\n'

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
    )


# 导购接口
@router.post("/guide/turns/stream")
async def stream_guide_turn(request: Request):
    data = await request.json()
    repository = request.app.state.context_repository

    return stream_response(
        stream_guide_answer(
            str(data["userId"]),
            data["text"],
            repository,
        )
    )


# 客服接口
@router.post("/support/turns/stream")
async def stream_support_turn(request: Request):
    data = await request.json()
    repository = request.app.state.context_repository

    return stream_response(
        stream_support_answer(
            str(data["userId"]),
            data["text"],
            repository,
            authorization=request.headers.get("Authorization", ""),
        )
    )
