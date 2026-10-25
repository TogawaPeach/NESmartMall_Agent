import os
from collections.abc import AsyncIterator
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from app.repositories.context_repository import AgentType, ContextRepository

# 加载项目根目录的 .env
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path, override=True)

# 初始化模型
model = init_chat_model(
    model="deepseek-flash",
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    max_retries=0,
)

# 导购 Agent：负责了解需求、辅助选购
guide_agent = create_agent(
    model=model,
    tools=[],
    system_prompt=(
        "你是经纬智选商城的导购助手，请用中文回答。严禁输出MarkDown格式文本"
        "我们的商城是一个测绘工具B2C平台，只有一个商家，专门卖关于测绘的用品"
        "你的职责是了解用户的用途、预算和偏好，帮助用户梳理选购需求，"
        "并根据用户提供的信息解释商品差异。"
        "信息不足时先询问，不编造具体商品信息。"
        "当前没有商品、库存、价格或订单查询工具，也没有接入商城知识库，"
        "不要声称已经查询商城数据，不编造实时价格、库存、优惠或商城政策。"
    ),
)

# 客服 Agent：负责商品使用及售前售后咨询
support_agent = create_agent(
    model=model,
    tools=[],
    system_prompt=(
        "你是经纬智选商城的客服助手，请用中文回答。严禁输出MarkDown格式文本"
        "我们的商城是一个测绘工具B2C平台，只有一个商家，专门卖关于测绘的用品"
        "你的职责是围绕用户提供的商品和问题，处理商品使用及售前售后咨询。"
        "缺少商品信息或问题细节时先询问，不假定已经收到商品页上下文。"
        "当前没有商品、订单、物流或售后操作工具，也没有接入商城知识库，"
        "不要声称已经查询或修改业务数据，不编造商城政策、订单状态或物流轨迹，"
        "不承诺退款到账、审批通过或人工已接通。"
    ),
)


# 两个 Agent 共用历史读写和流式回答逻辑
async def _stream_answer(
    user_id: str,
    text: str,
    agent_type: AgentType,
    repository: ContextRepository,
) -> AsyncIterator[str]:
    # 读取历史，并追加本轮用户消息
    history = await repository.get_history(user_id, agent_type)
    history.append({"role": "user", "content": text})

    agent = guide_agent if agent_type == "guide" else support_agent
    answer_parts: list[str] = []

    async for chunk, _metadata in agent.astream(
        {"messages": history},
        stream_mode="messages",
    ):
        for block in chunk.content_blocks:
            if block["type"] == "text":
                delta = block["text"]
                answer_parts.append(delta)
                yield delta

    # 流式生成正常结束后，保存完整回复和历史
    history.append(
        {
            "role": "assistant",
            "content": "".join(answer_parts),
        }
    )

    await repository.save_history(user_id, agent_type, history)


# 导购 Service
async def stream_guide_answer(
    user_id: str,
    text: str,
    repository: ContextRepository,
) -> AsyncIterator[str]:
    async for delta in _stream_answer(user_id, text, "guide", repository):
        yield delta


# 客服 Service
async def stream_support_answer(
    user_id: str,
    text: str,
    repository: ContextRepository,
) -> AsyncIterator[str]:
    async for delta in _stream_answer(user_id, text, "support", repository):
        yield delta
