import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from redis.asyncio import Redis
from v2.nacos import ClientConfigBuilder, NacosNamingService
from v2.nacos.common.nacos_exception import NacosException


@pytest.fixture
def redis_client(monkeypatch):
    client = AsyncMock(spec=Redis)
    client.ping = AsyncMock(return_value=True)
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    monkeypatch.setattr(Redis, "from_url", Mock(return_value=client))
    return client


@pytest.fixture
def naming_client(monkeypatch, redis_client, tmp_path):
    # 单独运行本文件时，模型初始化也使用测试配置，不读取真实 .env。
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-api-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    config = {
        "NACOS_SERVER_ADDR": "127.0.0.1:8848",
        "NACOS_NAMESPACE_ID": "test-namespace-id",
        "NACOS_USERNAME": "test-user",
        "NACOS_PASSWORD": "test-password",
        "NACOS_SERVICE_NAME": "mall-agent-service",
        "NACOS_GROUP": "DEFAULT_GROUP",
        "NACOS_REGISTER_IP": "192.0.2.10",
        "NACOS_REGISTER_PORT": "8090",
    }
    for key, value in config.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *_args, **_kwargs: False)
    module = sys.modules.get("app.services.nacos_service")
    if module is not None:
        monkeypatch.setattr(module, "load_dotenv", lambda *_args, **_kwargs: False)

    sdk_config = ClientConfigBuilder().server_address("127.0.0.1:8848").build()
    sdk_config.log_dir = str(tmp_path)
    sdk_config.cache_dir = str(tmp_path)
    sdk_config.load_cache_at_start = False

    async def create_sdk_client():
        return NacosNamingService(sdk_config)

    client = asyncio.run(create_sdk_client())
    client.register_instance = AsyncMock(return_value=True)
    client.deregister_instance = AsyncMock(return_value=True)
    # 保留真实 SDK 的 updater 和 shutdown，只隔离网络关闭操作。
    client.grpc_client_proxy.close_client = AsyncMock()
    factory = AsyncMock(return_value=client)
    monkeypatch.setattr(NacosNamingService, "create_naming_service", factory)
    yield client, factory
    for handler in list(client.logger.handlers):
        if Path(getattr(handler, "baseFilename", "")).parent == tmp_path:
            client.logger.removeHandler(handler)
            handler.close()


def application():
    from app.main import create_app
    from app.settings import Settings

    return create_app(Settings(_env_file=None, app_env="test"))


def test_startup_registers_configured_instance_and_shutdown_removes_it(naming_client, redis_client):
    client, factory = naming_client
    app = application()
    with TestClient(app):
        assert app.state.context_repository.redis is redis_client
        redis_client.ping.assert_awaited_once()
        factory.assert_awaited_once()
        config = factory.await_args.args[0]
        assert config.namespace_id == "test-namespace-id"
        assert config.server_list == ["127.0.0.1:8848"]
        assert config.username == "test-user"
        client.register_instance.assert_awaited_once()
        instance = client.register_instance.await_args.args[0]
        assert instance.service_name == "mall-agent-service"
        assert instance.group_name == "DEFAULT_GROUP"
        assert instance.ip == "192.0.2.10"
        assert instance.port == 8090
        client.deregister_instance.assert_not_awaited()
        client.grpc_client_proxy.close_client.assert_not_awaited()
        assert not client.service_info_updater.stop_event.is_set()

    removed = client.deregister_instance.await_args.args[0]
    assert (removed.service_name, removed.group_name, removed.ip, removed.port) == (
        "mall-agent-service",
        "DEFAULT_GROUP",
        "192.0.2.10",
        8090,
    )
    client.grpc_client_proxy.close_client.assert_awaited_once()
    assert client.service_info_updater.stop_event.is_set()
    redis_client.__aexit__.assert_awaited_once()


@pytest.mark.parametrize(
    "error",
    [None, RuntimeError("registration unavailable"), NacosException(500, "DEADLINE_EXCEEDED")],
)
def test_registration_failure_stops_startup_and_closes_client(naming_client, redis_client, error):
    client, _factory = naming_client
    if error is not None:
        client.register_instance.side_effect = error
    else:
        client.register_instance.return_value = False

    with pytest.raises(type(error) if error is not None else RuntimeError) as caught:
        with TestClient(application()):
            pytest.fail("Registration failure must prevent startup")

    if error is not None:
        assert caught.value is error
    client.deregister_instance.assert_not_awaited()
    client.grpc_client_proxy.close_client.assert_awaited_once()
    assert client.service_info_updater.stop_event.is_set()
    redis_client.__aexit__.assert_awaited_once()


def test_redis_failure_prevents_nacos_registration(naming_client, redis_client):
    _client, factory = naming_client
    redis_client.ping.side_effect = ConnectionError("redis unavailable")

    with pytest.raises(ConnectionError, match="redis unavailable"):
        with TestClient(application()):
            pytest.fail("Redis failure must prevent startup")

    factory.assert_not_awaited()
    redis_client.__aexit__.assert_awaited_once()


def test_deregistration_failure_still_closes_client(naming_client):
    client, _factory = naming_client
    client.deregister_instance.side_effect = RuntimeError("deregistration unavailable")
    with pytest.raises(RuntimeError, match="deregistration unavailable"):
        with TestClient(application()):
            pass
    client.grpc_client_proxy.close_client.assert_awaited_once()
    assert client.service_info_updater.stop_event.is_set()


def test_connection_close_failure_still_stops_updater(naming_client):
    client, _factory = naming_client
    client.grpc_client_proxy.close_client.side_effect = RuntimeError("close unavailable")

    with pytest.raises(RuntimeError, match="close unavailable"):
        with TestClient(application()):
            pass

    assert client.service_info_updater.stop_event.is_set()
