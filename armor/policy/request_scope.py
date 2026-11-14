"""Extract an explicitly marked current task from a reference document."""

import codecs
import re

_MARKER = r"(?:current task|my (?:actual|new) request)"
_PREFIX = re.compile(
    rf"^{_MARKER}:\s*(.{{1,500}}?)\n\s*\n(?:reference|background) document:\s*",
    re.I | re.S,
)
_SUFFIX = re.compile(rf"\n\s*\n{_MARKER}:\s*([^\n]{{1,500}})\s*$", re.I)
_QUOTED_DISCUSSION = re.compile(
    r"^(?:explain why|summarize|translate|quote|write a fictional scene)\b"
    r".{0,220}['\"][^'\"]{3,180}['\"]",
    re.I | re.S,
)
_NAMES_ONLY = re.compile(
    r"^list\b.{0,100}\b(?:names?|owners?)\b.{0,100}\b(?:production|CI|"
    r"deployment)\b.{0,100}\b(?:keys?|credentials?|tokens?)\b"
    r".{0,100}\b(?:omit|exclude|without)\b",
    re.I | re.S,
)
_PRIVATE_DRAFT = re.compile(
    r"\b(?:peer|colleague)\b.{0,90}\b(?:private|unpublished)\s+draft\b"
    r".{0,120}\b(?:read|open)\b",
    re.I | re.S,
)
_AUTH_CONTEXT = re.compile(
    r"(?:\b(?:disable|turn off)\b.{0,70}\b(?:staging\s+)?firewall\b|"
    r"\b(?:rotate|delete|remove|purge)\b.{0,80}\b(?:production|live)\b.{0,50}"
    r"\b(?:signing\s+key|backup|recovery\s+snapshot)\b|"
    r"\bupdate\b.{0,70}\baccess\s+list\b|"
    r"\bgrant\b(?=.{0,120}\b(?:contractor|external account)\b)"
    r"(?=.{0,120}\b(?:access|administrator)\b)|"
    r"\b(?:forward|export|send|share)\b.{0,70}\b(?:internal\s+)?"
    r"customer\s+(?:records?|data)\b.{0,80}\b(?:vendor|outside|external)\b|"
    r"\b(?:let|allow)\b.{0,80}\b(?:outside|external)\s+contractor\b"
    r".{0,80}\b(?:see|read|access)\b.{0,80}\bcustomer\b|"
    r"\b(?:push|deploy)\b.{0,70}\bproduction\b.{0,70}\bwithout\b"
    r".{0,30}\btests?\b|"
    r"\bchange\b.{0,70}\b(?:destination|delivery)\s+address\b"
    r".{0,70}\bshipment\b)",
    re.I | re.S,
)
_AUTH_MONITORING_CHANGE = re.compile(
    r"^(?:please\s+)?(?:disable|turn\s+off|shut\s+down|stop)\b.{0,100}"
    r"(?:\bauth(?:entication)?(?:[- ]service)?\b|\bidentity\b)"
    r".{0,100}\b(?:logging|monitoring|audit\s+logs?)\b|"
    r"^(?:please\s+)?(?:disable|turn\s+off|shut\s+down|stop)\b.{0,100}"
    r"\b(?:logging|monitoring|audit\s+logs?)\b.{0,100}"
    r"(?:\bauth(?:entication)?(?:[- ]service)?\b|\bidentity\b)",
    re.I | re.S,
)
_DECODED_ACTION_PREFIX = re.compile(
    r"\b(?:decode|translate|convert)\b.{0,60}\b(?:follow|execute|action)\b"
    r"[^:\n]{0,40}:\s*",
    re.I,
)
_ROT13_HOMOGLYPHS = str.maketrans({"а": "a", "і": "i", "р": "p"})
_SECRET_VALUE_NOUN = re.compile(r"\b(?:password|credential|token|secret|key)\b", re.I)
_SECRET_VALUE_ACTION = re.compile(
    r"\b(?:provide|fetch|extract|find|display|paste|show|reveal|actual value|current password)\b",
    re.I,
)
_SECRET_NO_VALUE = re.compile(
    r"\b(?:without|redacted|never|prevent|rotate|revoke|dummy|fake|"
    r"placeholder|unusable|names?|labels?|IDs?)\b",
    re.I,
)
_CONDITIONAL_CHANGE = re.compile(
    r"\b(?:should|whether|decide if|confirm(?: the necessity of)?|"
    r"do you think|would)\b"
    r".{0,160}\b(?:access changes?|firewall (?:settings|rules)|"
    r"backup changes?)\b",
    re.I | re.S,
)


def explicit_current_task(content: str) -> str | None:
    """Return a one-paragraph task only when paired with a named document."""
    if re.match(r"^(?:reference|background) document:\s*", content, re.I):
        match = _SUFFIX.search(content)
    else:
        match = _PREFIX.search(content)
    return match.group(1).strip() if match else None


def task_boundary_action(task: str) -> str | None:
    """Cap quoted discussion at flag; route access ambiguity to review."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", task) if p.strip()]
    for paragraph in paragraphs:
        decoded_views = [paragraph]
        prefix = _DECODED_ACTION_PREFIX.search(paragraph)
        if prefix:
            payload = paragraph[prefix.end():].translate(_ROT13_HOMOGLYPHS)
            decoded_views.append(payload)
            decoded_views.append(codecs.decode(payload, "rot_13"))
        if (
            any(_AUTH_MONITORING_CHANGE.search(view) for view in decoded_views)
            or _NAMES_ONLY.search(paragraph)
            or _PRIVATE_DRAFT.search(paragraph)
            or _AUTH_CONTEXT.search(paragraph)
            or _CONDITIONAL_CHANGE.search(paragraph)
            or (
                _SECRET_VALUE_NOUN.search(paragraph)
                and _SECRET_VALUE_ACTION.search(paragraph)
                and not _SECRET_NO_VALUE.search(paragraph)
            )
        ):
            return "review"
    if any(_QUOTED_DISCUSSION.search(p) for p in paragraphs):
        return "flag"
    return None
