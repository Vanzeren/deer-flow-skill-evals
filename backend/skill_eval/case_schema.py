from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

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

    @field_validator("id", "input", "rationale")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str]) -> list[str]:
        normalized = [tag.strip() for tag in value]
        if any(not tag for tag in normalized):
            raise ValueError("tags must not contain blank values")
        if len(normalized) != len(set(normalized)):
            raise ValueError("tags must be unique")
        return normalized
