from asyncmy.pool import Pool


class ProductKnowledgeRepository:
    def __init__(self, pool: Pool):
        self.pool = pool

    async def save_markdown(
        self,
        product_id: int,
        markdown: str,
        price: int | None,
    ) -> int:
        async with self.pool.acquire() as connection:
            async with connection.cursor() as cursor:
                return await cursor.execute(
                    """
                    INSERT INTO product_knowledge (
                        product_id, markdown, price
                    )
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE markdown = %s, price = %s
                    """,
                    (product_id, markdown, price, markdown, price),
                )
