"""FastAPI 应用入口；用 Uvicorn 加载 app.main:app。"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.agent import router as agent_router
from app.db.redis import redis_connection
from app.repositories.context_repository import ContextRepository
from app.services.nacos_service import nacos_lifespan
from app.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings if settings is not None else Settings()
    publish_docs = settings.app_env != "production"

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # 先连接 Redis，再注册 Nacos
        async with redis_connection(settings.redis_url) as redis:
            app.state.context_repository = ContextRepository(redis)

            async with nacos_lifespan(app):
                yield

        # 关闭时先注销 Nacos，再释放 Redis 连接

    application = FastAPI(
        lifespan=lifespan,
        title=settings.app_name,
        version=__version__,
        description="经纬智选商城 Agent 客服服务。",
        docs_url="/docs" if publish_docs else None,
        redoc_url="/redoc" if publish_docs else None,
        openapi_url="/openapi.json" if publish_docs else None,
    )
    application.include_router(agent_router)
    return application


app = create_app()
