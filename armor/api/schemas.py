from typing import Literal, Optional

from pydantic import BaseModel, PrivateAttr


class Finding(BaseModel):
    label: str
    score: float
    evidence: Optional[str] = None


class InspectRequest(BaseModel):
    direction: Literal["prompt", "response"]
    content: str
    context: Optional[dict] = None


class InspectResponse(BaseModel):
    decision: Literal["allow", "flag", "block", "review"]
    risk_score: float
    findings: list[Finding]
    route: Literal["fast", "base", "escalated"]
    latency_ms: float
    model_version: str
    _log_id: int | None = PrivateAttr(default=None)
