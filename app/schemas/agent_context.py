from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentContext:
    """本轮请求的运行上下文，不写入对话历史。"""

    authorization: str = field(repr=False)
