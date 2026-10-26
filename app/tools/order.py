import re
from typing import Any

import aiohttp
from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool

from app.clients.java import create_java_client
from app.schemas.agent_context import AgentContext


def failure(error: str, message: str) -> dict[str, Any]:
    return {"error": error, "message": message}


def create_order_tool(gateway_base_url: str) -> BaseTool:
    @tool
    async def query_order(
        order_no: str,
        runtime: ToolRuntime[AgentContext],
    ) -> dict[str, Any]:
        """查询当前用户订单的付款时间、状态、应付金额和实付金额。

        金额单位为分，未付款的付款时间和实付金额为 null。
        需要用户提供明确的订单号，不能猜测；本工具只查询，不修改订单。

        Args:
            order_no: 用户提供的完整订单号，例如 JW2112134513910874112。
        """
        order_no = order_no.strip()
        if not re.fullmatch(r"JW[1-9][0-9]{0,61}", order_no):
            return failure("INVALID_ORDER_NO", "订单号格式不正确，请核对。")

        authorization = runtime.context.authorization if runtime.context else ""
        if not authorization:
            return failure("UNAUTHORIZED", "缺少登录凭证，请重新登录。")

        try:
            async with create_java_client(gateway_base_url, authorization) as client:
                async with await client.get_order_detail(order_no=order_no) as response:
                    if response.status in (401, 403):
                        return failure("UNAUTHORIZED", "登录已失效或无权访问，请重新登录。")
                    if response.status == 404:
                        return failure("ORDER_NOT_ACCESSIBLE", "订单不存在或无法访问。")
                    if response.status != 200:
                        return failure("SERVICE_ERROR", "订单服务暂时不可用，请稍后重试。")

                    payload = await response.json()

            data = payload["data"]
            return {
                "paidAt": data["paidAt"],
                "status": data["status"],
                "payableAmountFen": data["payableAmountFen"],
                "paidAmountFen": data["paidAmountFen"],
            }
        except TimeoutError:
            return failure("TIMEOUT", "订单查询超时，请稍后重试。")
        except (aiohttp.ClientError, ValueError, KeyError, TypeError):
            # 不将上游异常原文、请求头或其他订单信息返回给模型。
            return failure("SERVICE_ERROR", "订单查询失败，请稍后重试。")

    return query_order
