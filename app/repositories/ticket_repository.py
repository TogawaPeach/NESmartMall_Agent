from typing import Any

from asyncmy.cursors import DictCursor
from asyncmy.pool import Pool


class TicketRepository:
    def __init__(self, pool: Pool):
        self.pool = pool

    async def create_ticket(self, ticket_no: int, question: str) -> int:
        """保存工单，工单号由调用方使用雪花算法生成。"""
        async with self.pool.acquire() as connection:
            async with connection.cursor() as cursor:
                return await cursor.execute(
                    """
                    INSERT INTO support_ticket (ticket_no, question)
                    VALUES (%s, %s)
                    """,
                    (ticket_no, question),
                )

    async def list_pending(self) -> list[dict[str, Any]]:
        """查询尚未逻辑删除的工单，包含已填答案但尚未发布的记录。"""
        async with self.pool.acquire() as connection:
            async with connection.cursor(DictCursor) as cursor:
                await cursor.execute(
                    """
                    SELECT
                        CAST(ticket_no AS CHAR) AS ticket_no,
                        question,
                        answer,
                        is_deleted
                    FROM support_ticket
                    WHERE is_deleted = 0
                    ORDER BY ticket_no DESC
                    """
                )
                return await cursor.fetchall()

    async def get_ticket(self, ticket_no: int) -> dict[str, Any] | None:
        """按工单号查询未删除的工单。"""
        async with self.pool.acquire() as connection:
            async with connection.cursor(DictCursor) as cursor:
                await cursor.execute(
                    """
                    SELECT
                        CAST(ticket_no AS CHAR) AS ticket_no,
                        question,
                        answer,
                        is_deleted
                    FROM support_ticket
                    WHERE ticket_no = %s AND is_deleted = 0
                    """,
                    (ticket_no,),
                )
                return await cursor.fetchone()

    async def save_answer(self, ticket_no: int, answer: str) -> int:
        """保存后台填写的答案，不触发发布或逻辑删除。"""
        async with self.pool.acquire() as connection:
            async with connection.cursor() as cursor:
                return await cursor.execute(
                    """
                    UPDATE support_ticket
                    SET answer = %s
                    WHERE ticket_no = %s AND is_deleted = 0
                    """,
                    (answer, ticket_no),
                )

    async def soft_delete(self, ticket_no: int) -> int:
        """知识成功发布到向量数据库后，由 Service 调用。"""
        async with self.pool.acquire() as connection:
            async with connection.cursor() as cursor:
                return await cursor.execute(
                    """
                    UPDATE support_ticket
                    SET is_deleted = 1
                    WHERE ticket_no = %s AND is_deleted = 0
                    """,
                    (ticket_no,),
                )
