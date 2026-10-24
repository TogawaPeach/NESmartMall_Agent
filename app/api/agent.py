import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.services.agent_service import stream_answer

router = APIRouter(
    prefix="/internal/agent",
    tags=["Agent"],
)


@router.post("/turns/stream")
async def stream_turn(request: Request):
    data = await request.json()

    async def generate_events():
        # 调用 Service，每收到一段回答就发送一次
        async for text in stream_answer(data["text"]):
            event = {
                "type": "delta",
                "payload": {"text": text},
            }
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        # 正常生成结束
        yield 'data: {"type":"done","payload":{}}\n\n'

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
    )
