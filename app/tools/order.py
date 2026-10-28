import logging
import re
from typing import Any

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool

from app.clients.java import create_java_client
from app.schemas.agent_context import AgentContext

logger = logging.getLogger(__name__)

ORDER_QUERY_FAILED = "订单查询失败，请稍后重试。"


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
        # 订单号会直接拼进请求路径，必须先挡住非法字符
        if not re.fullmatch(r"JW[1-9][0-9]{0,61}", order_no):
            return {"error": "订单号格式不正确，请核对。"}

        try:
            async with create_java_client(
                gateway_base_url,
                runtime.context.authorization,
            ) as client:
                async with await client.get_order_detail(order_no=order_no) as response:
                    data = (await response.json())["data"]

            return {
                "paidAt": data["paidAt"],
                "status": data["status"],
                "payableAmountFen": data["payableAmountFen"],
                "paidAmountFen": data["paidAmountFen"],
            }
        except Exception:
            # 不将上游异常原文、请求头或其他订单信息返回给模型。
            logger.exception("订单查询失败")
            return {"error": ORDER_QUERY_FAILED}

    return query_order


def create_order_products_tool(gateway_base_url: str) -> BaseTool:
    @tool
    async def query_order_products(
        order_no: str,
        runtime: ToolRuntime[AgentContext],
    ) -> dict[str, Any]:
        """按订单号查询当前用户购买的全部商品快照。

        返回下单时的商品信息，成交单价单位为分，不代表当前售价。

        Args:
            order_no: 需要查询的完整订单号。
        """
        try:
            async with create_java_client(
                gateway_base_url,
                runtime.context.authorization,
            ) as client:
                async with await client.get_order_detail(order_no=order_no) as response:
                    payload = await response.json()

            return {
                "orderNo": order_no,
                "items": [
                    {
                        "id": item["id"],
                        "productId": item["productId"],
                        "name": item["name"],
                        "model": item["model"],
                        "coverUrl": item["coverUrl"],
                        "unitPriceFen": item["unitPriceFen"],
                        "quantity": item["quantity"],
                    }
                    for item in payload["data"]["items"]
                ],
            }
        except Exception:
            raise RuntimeError("查询订单商品快照失败") from None

    return query_order_products


def create_user_orders_tool(gateway_base_url: str) -> BaseTool:
    @tool
    async def query_user_orders(
        runtime: ToolRuntime[AgentContext],
    ) -> dict[str, Any]:
        """查询当前登录用户的订单列表，帮助用户寻找订单号。

        返回订单号、状态、下单时间、应付金额和购买商品摘要。
        金额单位为分，商品信息为成交快照。
        多笔订单可能匹配时，应请用户明确目标，不能默认选择最新一单。
        本工具只查询，不修改订单。
        """
        try:
            async with create_java_client(
                gateway_base_url,
                runtime.context.authorization,
            ) as client:
                async with await client.get_user_orders() as response:
                    payload = await response.json()

            return {
                "orders": [
                    {
                        "orderNo": order["orderNo"],
                        "status": order["status"],
                        "createdAt": order["createdAt"],
                        "payableAmountFen": order["payableAmountFen"],
                        "items": [
                            {
                                "productId": item["productId"],
                                "name": item["name"],
                                "model": item["model"],
                                "quantity": item["quantity"],
                            }
                            for item in order["items"]
                        ],
                    }
                    for order in payload["data"]
                ],
            }
        except Exception:
            raise RuntimeError("查询本人订单列表失败") from None

    return query_user_orders
