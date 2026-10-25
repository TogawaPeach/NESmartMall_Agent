import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from v2.nacos import (
    ClientConfigBuilder,
    DeregisterInstanceParam,
    NacosNamingService,
    RegisterInstanceParam,
)


@asynccontextmanager
async def nacos_lifespan(_app: FastAPI):
    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(env_path, override=True)

    config = (
        ClientConfigBuilder()
        .server_address(os.environ["NACOS_SERVER_ADDR"])
        .namespace_id(os.environ["NACOS_NAMESPACE_ID"])
        .username(os.environ["NACOS_USERNAME"])
        .password(os.environ["NACOS_PASSWORD"])
        .build()
    )

    instance = {
        "service_name": os.environ["NACOS_SERVICE_NAME"],
        "group_name": os.getenv("NACOS_GROUP", "DEFAULT_GROUP"),
        "ip": os.environ["NACOS_REGISTER_IP"],
        "port": int(os.environ["NACOS_REGISTER_PORT"]),
    }

    client = await NacosNamingService.create_naming_service(config)
    registered = False

    try:
        registered = await client.register_instance(RegisterInstanceParam(**instance))
        if not registered:
            raise RuntimeError("Python 服务注册到 Nacos 失败")

        # 应用在这里运行，SDK 维护注册连接
        yield
    finally:
        try:
            if registered:
                await client.deregister_instance(DeregisterInstanceParam(**instance))
        finally:
            await client.shutdown()
