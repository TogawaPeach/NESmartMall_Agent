import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.services.agent_service import (
    stream_guide_answer,
    stream_support_answer,
)
from app.services.ticket_service import HandoffExpiredError

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/internal",
    tags=["Agent"],
)


# 两个接口共用 SSE 包装，支持文本块和结构化事件
def stream_response(
    chunks: AsyncIterator[str | dict[str, Any]],
) -> StreamingResponse:
    async def generate_events():
        async for chunk in chunks:
            if isinstance(chunk, str):
                event = {
                    "type": "delta",
                    "payload": {"text": chunk},
                }
            else:
                event = chunk

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


@router.post("/support/tickets")
async def create_support_ticket(request: Request):
    data = await request.json()
    try:
        result = await request.app.state.ticket_service.create_from_action(
            str(data["userId"]),
            data["actionId"],
            request.app.state.context_repository,
        )
    except HandoffExpiredError as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from None
    except Exception as exc:
        logger.error("人工服务申请提交失败：%s", type(exc).__name__)
        raise HTTPException(
            status_code=503, detail="提交失败，请稍后重试。",
        ) from None

    return {
        "ticketNo": result["ticket_no"],
        "message": "人工服务申请已记录。本项目为演示系统，暂无真人接待。",
    }
