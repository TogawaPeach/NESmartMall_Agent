"""FastAPI 应用入口；用 Uvicorn 加载 app.main:app。"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.agent import router as agent_router
from app.api.product import router as product_router
from app.db.mysql import mysql_connection
from app.db.redis import redis_connection
from app.repositories.context_repository import ContextRepository
from app.repositories.product_knowledge_repository import ProductKnowledgeRepository
from app.repositories.ticket_repository import TicketRepository
from app.services.agent_service import model
from app.services.nacos_service import nacos_lifespan
from app.services.product_service import ProductService
from app.services.ticket_service import TicketService
from app.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings if settings is not None else Settings()
    publish_docs = settings.app_env != "production"

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with redis_connection(settings.redis_url) as redis:
            app.state.context_repository = ContextRepository(redis)

            async with mysql_connection(settings) as mysql_pool:
                app.state.mysql_pool = mysql_pool
                app.state.product_service = ProductService(
                    chat_model=model,
                    repository=ProductKnowledgeRepository(mysql_pool),
                )
                app.state.ticket_repository = TicketRepository(mysql_pool)
                app.state.ticket_service = TicketService(
                    model=model,
                    repository=app.state.ticket_repository,
                    instance=settings.snowflake_instance,
                )

                async with nacos_lifespan(app):
                    yield

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
    application.include_router(product_router)
    return application


app = create_app()
