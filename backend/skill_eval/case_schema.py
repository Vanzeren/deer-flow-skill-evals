from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ALLOWED_TAGS = frozenset(
    {
        "explicit",  # 明确表达：请求中直接包含 survey/review/analyze 等触发词
        "implicit",  # 隐式表达：请求未直接使用触发词，但意图属于该类别
        "multi-paper",  # 多篇论文：需要跨多篇论文综合
        "sibling-collision",  # 相邻碰撞：两个 skill 的触发边界接近，容易误判
        "keyword-collision",  # 关键词碰撞：请求中包含学术关键词（如 BibTeX），但不属于学术任务
        "near-domain",  # 近领域：与学术相关，但不需要任何 skill
        "near-boundary",  # 边界附近：接近分类边界，容易触发误判
        "direct-answer",  # 直接回答：不需要工具调用即可完成
        "quality",  # 质量评测：需进入完整 Agent + Judge 质量评测流程
        "research",  # 研究类：需要搜索或检索，但不属于学术论文
        "non-academic",  # 非学术：与研究论文无关
        "unrelated",  # 完全无关
        "translation",  # 翻译任务
        "coding",  # 编程任务
        "citation-format",  # 引用格式：有特定引用格式要求
        "time-window",  # 时间窗口：有时间范围限制
        "synthesis",  # 综合分析：要求跨论文主题综合
        "comparison",  # 论文比较：要求跨论文方法比较
        "single-result",  # 单结果：只需找一个结果
    }
)
type RouteLabel = Literal[
    "systematic-literature-review",
    "academic-paper-review",
    "none",
]

CANDIDATE_SKILLS = (
    "systematic-literature-review",
    "academic-paper-review",
)


class RoutingCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    input: str
    expected_route: RouteLabel
    rationale: str
    tags: list[str] = Field(default_factory=list)
    expected_output: str | None = Field(default=None, description="Reference answer for quality judge comparison")

    @field_validator("id", "input", "rationale")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: list[str]) -> list[str]:
        normalized = [tag.strip() for tag in value]
        if any(not tag for tag in normalized):
            raise ValueError("tags must not contain blank values")
        if len(normalized) != len(set(normalized)):
            raise ValueError("tags must be unique")
        unknown = [tag for tag in normalized if tag not in ALLOWED_TAGS]
        if unknown:
            raise ValueError(f"unknown tag(s): {', '.join(unknown)}. Allowed: {', '.join(sorted(ALLOWED_TAGS))}")
        return normalized
