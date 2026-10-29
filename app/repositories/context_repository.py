import json
from typing import Any, Literal
from uuid import uuid4

from redis.asyncio import Redis

AgentType = Literal["guide", "support"]

# 六小时，单位：秒
CONTEXT_TTL = 6 * 60 * 60


class ContextRepository:
    def __init__(self, redis: Redis):
        self.redis = redis

    def _key(self, user_id: str, agent_type: AgentType) -> str:
        # 导购和客服分别保存历史
        return f"context:{agent_type}:userid:{user_id}"

    async def get_history(
        self,
        user_id: str,
        agent_type: AgentType,
    ) -> list[dict[str, str]]:
        """读取历史，没有记录或已过期时返回空列表。"""
        value = await self.redis.get(self._key(user_id, agent_type))

        if value is None:
            return []

        return json.loads(value)

    async def save_history(
        self,
        user_id: str,
        agent_type: AgentType,
        messages: list[dict[str, str]],
    ) -> None:
        """只保存用户消息和助手回复，每次保存续期六小时。"""
        history = [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in messages
            if message["role"] in {"user", "assistant"}
        ]

        await self.redis.set(
            self._key(user_id, agent_type),
            json.dumps(history, ensure_ascii=False),
            ex=CONTEXT_TTL,
        )

    async def create_handoff_action(
        self,
        user_id: str,
        history: list[dict[str, str]],
        text: str,
    ) -> str:
        """保存本轮之前的历史和本轮消息，返回按钮动作标识。"""
        action_id = uuid4().hex
        key = f"handoff:support:userid:{user_id}:action:{action_id}"

        await self.redis.set(
            key,
            json.dumps(
                {
                    "history": history,
                    "text": text,
                },
                ensure_ascii=False,
            ),
            ex=CONTEXT_TTL,
        )

        return action_id

    async def get_handoff_action(
        self,
        user_id: str,
        action_id: str,
    ) -> dict[str, Any] | None:
        """点击按钮时，读取当前用户对应的聊天快照。"""
        key = f"handoff:support:userid:{user_id}:action:{action_id}"
        value = await self.redis.get(key)

        if value is None:
            return None

        return json.loads(value)
