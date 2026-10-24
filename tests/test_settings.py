import pytest
from pydantic import ValidationError


def test_environment_variables_override_dotenv(monkeypatch, tmp_path):
    from app.settings import Settings

    env_file = tmp_path / ".env"
    env_file.write_text("AGENT_APP_NAME=file-service\nAGENT_APP_ENV=local\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_APP_NAME", "environment-service")
    monkeypatch.setenv("AGENT_APP_ENV", "test")

    settings = Settings(_env_file=env_file)

    assert settings.app_name == "environment-service"
    assert settings.app_env == "test"


def test_dotenv_loads_without_shell_variables(monkeypatch, tmp_path):
    from app.settings import Settings

    monkeypatch.delenv("AGENT_APP_NAME", raising=False)
    monkeypatch.delenv("AGENT_APP_ENV", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("AGENT_APP_NAME=dotenv-service\nAGENT_APP_ENV=test\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)

    assert settings.app_name == "dotenv-service"
    assert settings.app_env == "test"


@pytest.mark.parametrize("app_env", ["prodution", "staging", ""])
def test_invalid_environment_is_rejected(app_env):
    from app.settings import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env=app_env)
