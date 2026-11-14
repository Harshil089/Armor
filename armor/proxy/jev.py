"""Development-only Jev adjudication for requests Armor routed to review."""

import os
from datetime import datetime, timezone

from armor.observability import log_store

async def adjudicate_review(client, content: str, review_reason: str | None, findings: list[dict]) -> tuple[str, str, bool]:
    """Return Jev's final binary decision; every non-explicit allow fails closed."""
    url = os.environ.get("ARMOR_JEV_REVIEW_URL")
    if not url:
        return "block", "jev_reviewer_not_configured", False
    try:
        response = await client.post(
            url,
            json={"content": content, "review_reason": review_reason, "findings": findings},
            timeout=15.0,
        )
        response.raise_for_status()
        action = response.json().get("action")
        if action in {"allow", "block"}:
            return action, f"jev_{action}", True
        return "block", "jev_invalid_action", False
    except Exception:
        return "block", "jev_unavailable_or_invalid_response", False


async def resolve_review(client, result, content: str) -> str:
    """Apply Jev's binary verdict and persist its pseudo-label on the inspect row."""
    action, reason, has_verdict = await adjudicate_review(
        client, content, getattr(result, "review_reason", None),
        [finding.model_dump() for finding in result.findings],
    )
    log_id = getattr(result, "_log_id", None)
    if log_id is None and has_verdict:
        action, reason = "block", "jev_source_log_missing"
        has_verdict = False
    if has_verdict:
        try:
            log_store.record_feedback(
                log_store.DB_PATH, log_id, action, "decision", "reviewer_ai_b",
                datetime.now(timezone.utc).isoformat(),
            )
        except Exception:
            action, reason = "block", "jev_label_persistence_failed"
            has_verdict = False
    result.decision = action
    result.model_version += "+jev-review"
    return reason
