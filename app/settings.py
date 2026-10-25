"""配置入口：仅声明当前阶段实际使用的环境变量。"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        str_strip_whitespace=True,
    )

    app_name: str = Field(default="mall-agent-service", min_length=1)
    app_env: Literal["local", "test", "production"] = "local"
    redis_url: str = Field(
        default="redis://localhost:6379/1",
        min_length=1,
    )
