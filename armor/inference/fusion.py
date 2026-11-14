"""Conservative request-level fusion for selectively inspected windows."""

from dataclasses import dataclass, field


@dataclass
class WindowResult:
    index: int
    start_token: int
    end_token: int
    max_depth: int
    resolved: bool
    decision: str | None = None
    confidence: float | None = None
    findings: list[dict] = field(default_factory=list)
    escalation_reason: str | None = None


def fuse_request_evidence(
    windows: list[WindowResult],
    *,
    deterministic_block: bool = False,
    inspection_budget_exceeded: bool = False,
) -> tuple[str, str | None]:
    """Uninspected or conflicting regions can never contribute an ALLOW."""
    if deterministic_block:
        return "block", "deterministic_critical_evidence"
    if inspection_budget_exceeded:
        return "review", "inspection_budget_exceeded"
    if not windows:
        return "review", "no_window_evidence"
    if any(w.resolved and w.decision == "block" and w.max_depth >= 22 for w in windows):
        return "block", "validated_full_depth_window_block"
    if any(not w.resolved or w.decision in (None, "review") for w in windows):
        return "review", "unresolved_or_uninspected_window"
    if any(w.decision == "flag" for w in windows):
        return "flag", "nonblocking_window_finding"
    if all(w.resolved and w.decision == "allow" for w in windows):
        return "allow", None
    return "review", "conflicting_window_evidence"
