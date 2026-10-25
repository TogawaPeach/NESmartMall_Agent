import sys
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from v2.nacos import NacosNamingService


@pytest.fixture
def naming_client(monkeypatch):
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

    client = Mock(spec=NacosNamingService)
    client.register_instance = AsyncMock(return_value=True)
    client.deregister_instance = AsyncMock(return_value=True)
    client.shutdown = AsyncMock()
    factory = AsyncMock(return_value=client)
    monkeypatch.setattr(NacosNamingService, "create_naming_service", factory)
    return client, factory


def application():
    from app.main import create_app
    from app.settings import Settings

    return create_app(Settings(_env_file=None, app_env="test"))


def test_startup_registers_configured_instance_and_shutdown_removes_it(naming_client):
    client, factory = naming_client
    with TestClient(application()):
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
        client.shutdown.assert_not_awaited()

    removed = client.deregister_instance.await_args.args[0]
    assert (removed.service_name, removed.group_name, removed.ip, removed.port) == (
        "mall-agent-service",
        "DEFAULT_GROUP",
        "192.0.2.10",
        8090,
    )
    assert [call[0] for call in client.mock_calls] == [
        "register_instance",
        "deregister_instance",
        "shutdown",
    ]


@pytest.mark.parametrize("raises", [False, True])
def test_registration_failure_stops_startup_and_closes_client(naming_client, raises):
    client, _factory = naming_client
    if raises:
        client.register_instance.side_effect = RuntimeError("registration unavailable")
    else:
        client.register_instance.return_value = False

    with pytest.raises(RuntimeError):
        with TestClient(application()):
            pytest.fail("Registration failure must prevent startup")

    client.deregister_instance.assert_not_awaited()
    client.shutdown.assert_awaited_once()


def test_deregistration_failure_still_closes_client(naming_client):
    client, _factory = naming_client
    client.deregister_instance.side_effect = RuntimeError("deregistration unavailable")
    with pytest.raises(RuntimeError, match="deregistration unavailable"):
        with TestClient(application()):
            pass
    client.shutdown.assert_awaited_once()
