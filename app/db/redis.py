from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from redis.asyncio import Redis


@asynccontextmanager
async def redis_connection(url: str) -> AsyncIterator[Redis]:
    async with Redis.from_url(
        url,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    ) as client:
        # 启动时检查 Redis 是否可用
        await client.ping()

        # 将连接交给调用方使用
        yield client

    # 退出上下文时，自动关闭客户端和内部连接池
