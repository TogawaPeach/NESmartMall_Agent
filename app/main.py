"""FastAPI 应用入口；用 Uvicorn 加载 app.main:app。"""

from fastapi import FastAPI

from app import __version__
from app.api.agent import router as agent_router
from app.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings if settings is not None else Settings()
    publish_docs = settings.app_env != "production"

    application = FastAPI(
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
