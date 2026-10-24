import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

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

# 当前只创建一个 Agent
agent = create_agent(
    model=model,
    tools=[],
    system_prompt="你是经纬智选商城的智能助手，请用中文回答用户问题。",
)


# 接收问题，逐段返回回答文本
async def stream_answer(text: str):
    messages = {"messages": [{"role": "user", "content": text}]}

    async for chunk, _metadata in agent.astream(
        messages,
        stream_mode="messages",
    ):
        for block in chunk.content_blocks:
            if block["type"] == "text":
                yield block["text"]
