from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiohttp
from uplink import AiohttpClient, Consumer, Path, get


class JavaClient(Consumer):
    @get("api/orders/{order_no}")
    def get_order_detail(self, order_no: Path):
        """查询当前用户的订单详情，请求头由客户端统一设置。"""


@asynccontextmanager
async def create_java_client(
    base_url: str,
    authorization: str,
) -> AsyncIterator[JavaClient]:
    """创建独立会话，所有 Java 接口调用自动携带本轮用户的 JWT。"""
    async with aiohttp.ClientSession(
        headers={"Authorization": authorization},
        timeout=aiohttp.ClientTimeout(total=10),
    ) as session:
        yield JavaClient(
            base_url=base_url.rstrip("/") + "/",
            client=AiohttpClient(session=session),
        )
