from contextlib import nullcontext

import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_production_does_not_publish_api_documentation(path):
    from app.main import create_app
    from app.settings import Settings

    application = create_app(Settings(_env_file=None, app_env="production"))
    # 文档开关测试不连接注册中心。
    application.router.lifespan_context = lambda _app: nullcontext()

    with TestClient(application) as client:
        assert client.get(path).status_code == 404
