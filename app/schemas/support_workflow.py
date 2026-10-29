from typing import Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field

from app.knowledge.retrieval import KnowledgeCandidate

Route = Literal["knowledge", "order", "other"]


class RouteResult(BaseModel):
    route: Route = Field(description="本轮主要处理分支")
    question: str = Field(description="结合历史补全后的完整问题")
    order_no: str | None = Field(
        default=None,
        description="用户已明确的订单号，没有则为空，不能猜测",
    )
    secondary_routes: list[Route] = Field(
        default_factory=list,
        description="同一轮还涉及的其他分支，不重复主要分支",
    )
    show_handoff: bool = Field(
        default=False,
        description="是否显示转人工按钮",
    )
    order_task: Literal["detail", "list", "return"] = Field(
        default="detail",
        description="订单明细查询、寻找订单号，或判断具体订单的退货条件",
    )
    knowledge_question: str | None = Field(
        default=None,
        description="独立通用知识子问题，不含具体订单退货资格；没有则为空",
    )


class EvidencePart(BaseModel):
    question: str = Field(description="可独立处理的知识子问题")
    status: Literal["supported", "missing", "clarify"]
    facts: str = Field(description="有依据的事实，或缺少的资料、需要用户补充的条件")


class KnowledgeEvidence(BaseModel):
    parts: list[EvidencePart] = Field(description="覆盖所有知识子问题，分别检查依据")


class SupportState(TypedDict):
    text: str
    history: list[dict[str, str]]  # 本轮之前的客服历史
    decision: NotRequired[RouteResult]
    candidates: NotRequired[list[KnowledgeCandidate]]
    evidence: NotRequired[KnowledgeEvidence]
    knowledge_error: NotRequired[str]
    order_result: NotRequired[dict[str, Any]]
    other_text: NotRequired[str]
    show_handoff: NotRequired[bool]
    answer: NotRequired[str]
