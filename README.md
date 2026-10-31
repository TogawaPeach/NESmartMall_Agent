# NESmartMall · 经纬智选 Agent

![Python](https://img.shields.io/badge/Python-3.12-3776AB)
![FastAPI](https://img.shields.io/badge/FastAPI-009688)
![LangChain](https://img.shields.io/badge/LangChain-1-1C3C3C)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C)
![MySQL](https://img.shields.io/badge/MySQL-4479A1)
![Redis](https://img.shields.io/badge/Redis-DC382D)
![Milvus](https://img.shields.io/badge/Milvus-00A1EA)
![Nacos](https://img.shields.io/badge/Nacos-2C9DD5)

面向测绘工具商城的 Python Agent 服务，基于 FastAPI、LangChain 与 LangGraph 构建，提供智能导购、售前售后咨询、订单查询与演示工单留存，通过 SSE 向 Java 后端流式返回回答。

## 核心功能

- **智能导购**：结合用户用途、预算与历史对话检索商品，每次最多推荐三件，并说明有资料支持的推荐理由。
- **客服 Workflow**：结合历史将问题路由到知识咨询、订单查询或其他分支，支持一轮中的多个诉求，只向前端输出正式回答与交互动作。
- **商城知识检索**：商城政策 Markdown 经切块和百炼向量模型编码后写入 Milvus；知识咨询先检索、再判断证据是否足够，资料不足时说明限制。
- **订单只读查询**：携带本轮用户的 JWT 调用 Java 网关，查询本人订单列表、订单状态、金额与成交商品快照。
- **商品知识建档**：接收 Java 同步的商品 ID、名称、型号、头图和价格，调用模型整理 Markdown，保存至 MySQL 并写入独立商品向量集合。
- **上下文与工单**：Redis 按用户隔离导购和客服历史，保存后续期六小时；客服可提供转人工按钮，用户点击后汇总问题并生成雪花编号工单。
- **服务发现与流式通信**：启动时向 Nacos 注册 `mall-agent-service`，通过独立的导购、客服接口返回 SSE 事件。

> 项目持续开发中。转人工仅记录演示工单，暂无真人接待；后台补答与知识发布闭环尚未接通。商品推荐使用已同步的资料和价格，不代表实时库存、上下架状态或实时价格；跨轮“换一批”的自动排除尚未完成。

## 项目结构

```text
NESmartMall_Agent
├── app/
│   ├── api/             # 内部接口与 SSE 事件包装
│   ├── clients/         # 携带用户凭证的 Java HTTP 客户端
│   ├── db/              # MySQL、Redis 连接管理
│   ├── knowledge/       # 文档加载、切块、向量入库与检索
│   ├── repositories/    # 聊天上下文、商品知识与工单存储
│   ├── schemas/         # Agent 上下文与 Workflow 状态结构
│   ├── services/        # 导购 Agent、客服 Workflow 与业务服务
│   ├── tools/           # 知识、商品与订单查询工具
│   ├── main.py          # FastAPI 应用与生命周期
│   └── settings.py      # 应用与数据库配置
├── tests/               # 离线测试
├── mall-policies.md     # 商城政策知识源
├── pyproject.toml       # 项目依赖与开发工具配置
├── uv.lock              # 依赖锁文件
└── run.py               # 本地启动入口，端口 8090
```

## 本地运行

以下说明对应仓库已提交的 `.env` 配置版本。

1. 准备 Python 3.12、uv、MySQL、Redis、Milvus 与 Nacos，以及 DeepSeek 和百炼向量模型的 API 凭据。订单查询还需要启动 Java 网关及对应业务服务。
2. 在 MySQL 中准备 Agent 自有数据库和已有业务表：`support_ticket` 使用 `ticket_no`、`question`、`answer`、`is_deleted`；`product_knowledge` 使用 `product_id`、`markdown`、`price`，其中 `product_id` 为主键或唯一键，价格单位为分。应用不会自动建表，当前仓库未附建表脚本。
3. 在项目根目录安装锁定依赖：

   ```bash
   uv sync --locked
   ```

4. 在根目录创建 `.env`，按实际环境填写以下配置。该文件已被 Git 忽略，密钥与密码请勿提交。

   ```dotenv
   AGENT_APP_ENV=local
   AGENT_REDIS_URL=redis://127.0.0.1:6379/1
   AGENT_MYSQL_HOST=127.0.0.1
   AGENT_MYSQL_PORT=3306
   AGENT_MYSQL_USER=your_mysql_user
   AGENT_MYSQL_PASSWORD=your_mysql_password
   AGENT_MYSQL_DATABASE=nesmartmall_agent

   DEEPSEEK_API_KEY=your_deepseek_api_key
   DEEPSEEK_BASE_URL=https://api.deepseek.com
   EMBEDDING_MODEL=text-embedding-v4
   EMBEDDING_API_KEY=your_embedding_api_key
   EMBEDDING_BASE_URL=https://your-embedding-endpoint.example/v1

   MILVUS_URI=http://127.0.0.1:19530
   MILVUS_DB_NAME=default
   MILVUS_TOKEN=
   MILVUS_COLLECTION=mall_policies
   MILVUS_PRODUCT_COLLECTION=mall_products

   NACOS_SERVER_ADDR=127.0.0.1:8848
   NACOS_NAMESPACE_ID=your_namespace_id
   NACOS_USERNAME=your_nacos_username
   NACOS_PASSWORD=your_nacos_password
   NACOS_SERVICE_NAME=mall-agent-service
   NACOS_GROUP=DEFAULT_GROUP
   NACOS_REGISTER_IP=127.0.0.1
   NACOS_REGISTER_PORT=8090

   MALL_GATEWAY_BASE_URL=http://127.0.0.1:8080
   ```

   将 `EMBEDDING_BASE_URL` 替换为实际的百炼 OpenAI 兼容接口地址；向量维度固定为 1024。政策集合与商品集合必须使用不同名称。Nacos 的 namespace 和 group 需与 Java 客服服务一致，注册 IP 和端口必须能被 Java 服务访问；远程部署时不能使用回环地址。商品建档会将头图传给聊天模型，需确认所用模型支持该输入。

5. 首次使用政策检索，或修改根目录 `mall-policies.md` 后，手动执行入库：

   ```bash
   uv run python -m app.knowledge.ingest
   ```

   该命令调用真实向量模型并更新 Milvus，重复运行会覆盖当前政策块、清理本政策已删减的旧块。商品集合在首次商品同步时创建。

6. 启动服务：

   ```bash
   uv run python run.py
   ```

默认监听 `127.0.0.1:8090`，本地接口文档为 `http://127.0.0.1:8090/docs`。设置 `AGENT_APP_ENV=production` 后关闭 Swagger、ReDoc 和 OpenAPI 文档。需要其他监听地址时可直接使用 Uvicorn，例如 `uv run uvicorn app.main:app --host 0.0.0.0 --port 8090`，并同步调整 Nacos 注册地址。

## 导购与客服接入

```text
Vue 前端 → Gateway 登录校验 → Java 客服服务 → Python Agent
                                               ├── Redis：聊天历史与按钮快照
                                               ├── MySQL：商品知识与工单
                                               ├── Milvus：政策与商品检索
                                               └── Java Gateway：本人订单查询
```

Python 提供内部接口，用户身份与会话归属由 Java 校验；订单查询继续携带当前用户的 `Authorization`。前端通过 Gateway 的 `/api` 接口访问，不直接传用户 ID 或访问 Python 内部接口。

| 方法 | Python 内部路径 | 用途 |
| --- | --- | --- |
| `POST` | `/internal/guide/turns/stream` | 导购对话，接收 `userId`、`text` |
| `POST` | `/internal/support/turns/stream` | 客服对话，接收 `userId`、`text`，订单查询需透传 `Authorization` |
| `POST` | `/internal/support/tickets` | 点击转人工动作后建单，接收 `userId`、`actionId` |
| `POST` | `/internal/guide/products` | 同步商品资料，接收 `productId`、`name`、`model`、`coverUrl`、`price`，成功返回 `204` |

对话接口使用 `text/event-stream`，基本事件格式如下，每个事件以空行分隔：

```text
data: {"type":"delta","payload":{"text":"回答片段"}}

data: {"type":"done","payload":{}}

```

客服在需要时还会发出 `action` 事件，动作类型为 `transfer_human`。用户点击后，Java 将 `actionId` 与已校验的用户身份转发至工单接口；快照有效期为六小时，过期返回 `410`，成功返回 `ticketNo` 和演示状态说明。生成按钮本身不会创建工单，重复提交有效动作会分别建单。

部署时应将内部接口限制在可信服务网络内；Gateway 与反向代理需透传 SSE、关闭响应缓冲，并允许足够的流式响应时长。

## 常用命令

| 命令 | 说明 |
| --- | --- |
| `uv sync --locked` | 按锁文件安装运行与开发依赖 |
| `uv run python run.py` | 启动 Agent 服务 |
| `uv run python -m app.knowledge.ingest` | 将商城政策向量化并写入 Milvus |
| `uv run pytest` | 运行离线测试 |
| `uv run ruff check .` | 检查代码规范 |
| `uv run ruff format --check .` | 检查代码格式 |

离线测试使用模拟模型与存储，不代表真实模型、中间件或前后端联调通过。当前 pytest 缓存配置写入相邻的 `NESmartMall_Require` 目录；仅克隆本仓库时可使用 `uv run pytest -o cache_dir=.pytest_cache`。

## 关联项目

- [商城后端 NESmartMall_Backend](https://github.com/TogawaPeach/NESmartMall_Backend)
- [商城前端 NESmartMall_Front](https://github.com/TogawaPeach/NESmartMall_Front)
