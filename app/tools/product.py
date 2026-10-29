import asyncio
import logging
import os
from decimal import Decimal, InvalidOperation
from typing import Any

from langchain_core.tools import BaseTool, tool

from app.knowledge.embeddings import get_embedding_model
from app.knowledge.milvus import ensure_collection_loaded, get_milvus_client

logger = logging.getLogger(__name__)


def _price_to_fen(value: str | None) -> int | None:
    if value is None:
        return None
    amount = Decimal(value) * 100
    if not amount.is_finite() or amount < 0 or amount != amount.to_integral_value():
        raise ValueError("预算必须是非负金额，最多保留两位小数")
    return int(amount)


def create_product_tool() -> BaseTool:
    @tool
    async def query_products(
        query: str,
        min_price_yuan: str | None = None,
        max_price_yuan: str | None = None,
        exclude_product_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """按选购需求和预算检索商城商品，最多返回三件商品。

        价格为商家保存商品时同步的价格，不包含实时库存或上下架信息。
        资料中未提供的性能不能视为商品满足该要求。

        Args:
            query: 结合历史补全的选购需求，保留商品类型、名称、型号和用途。
            min_price_yuan: 用户预算下限，单位元，使用字符串，如 "100.50"。
                未提供下限时传 null，不自行设定。
            max_price_yuan: 用户预算上限，单位元，使用字符串，如 "500"。
                未提供上限或不限预算时传 null，不自行设定。
            exclude_product_ids: 本次检索需要排除的真实商品 ID 字符串列表。
        """
        try:
            minimum = _price_to_fen(min_price_yuan)
            maximum = _price_to_fen(max_price_yuan)
            excluded = sorted({int(value) for value in (exclude_product_ids or [])})
        except (InvalidOperation, ValueError):
            return {"error": "预算需为非负金额且最多两位小数，排除商品需使用真实商品 ID。"}

        if minimum is not None and maximum is not None and minimum > maximum:
            return {"error": "预算下限高于上限，请澄清用户的预算范围。"}

        query = query.strip()
        if not query:
            return {"error": "请先了解用户需要选购什么商品。"}

        collection_name = os.environ["MILVUS_PRODUCT_COLLECTION"]
        if collection_name == os.environ["MILVUS_COLLECTION"]:
            raise ValueError("商品集合不能与商城规则集合相同")

        conditions = []
        if minimum is not None:
            conditions.append(f"price >= {minimum}")
        if maximum is not None:
            conditions.append(f"price <= {maximum}")
        if excluded:
            conditions.append(f"id not in {excluded}")

        try:
            vector = await get_embedding_model().aembed_query(query)

            def search():
                client = get_milvus_client()
                if not client.has_collection(collection_name=collection_name):
                    return []
                ensure_collection_loaded(client, collection_name)
                return client.search(
                    collection_name=collection_name,
                    data=[vector],
                    anns_field="vector",
                    filter=" and ".join(conditions),
                    limit=3,
                    output_fields=["text", "price"],
                    search_params={"metric_type": "COSINE"},
                    consistency_level="Strong",
                )

            results = await asyncio.to_thread(search)
        except Exception:
            logger.exception("商品检索失败")
            return {"error": "商品检索暂时失败，请稍后重试，不能据此判断商城没有商品。"}

        products = []
        seen = set()
        for hit in results[0] if results else []:
            product_id = str(hit["id"])
            if product_id in seen or not hit["entity"].get("text"):
                continue
            seen.add(product_id)
            price = hit["entity"].get("price")
            products.append({
                "productId": product_id,
                "markdown": hit["entity"]["text"],
                "priceFen": price,
                "priceYuan": (
                    format(Decimal(price) / 100, ".2f") if price is not None else None
                ),
            })

        return {
            "products": products,
            "message": "" if products else "本次未检索到符合条件的商品，请勿自行放宽预算。",
        }

    return query_products
