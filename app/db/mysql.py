from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncmy
from asyncmy.pool import Pool

from app.settings import Settings


@asynccontextmanager
async def mysql_connection(settings: Settings) -> AsyncIterator[Pool]:
    pool = await asyncmy.create_pool(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password.get_secret_value(),
        db=settings.mysql_database,
        charset="utf8mb4",
        autocommit=True,
        minsize=1,
        maxsize=5,
        connect_timeout=5,
    )

    try:
        yield pool
    finally:
        pool.close()
        await pool.wait_closed()
