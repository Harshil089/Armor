"""Reviewed real-chat negatives from OpenAssistant OASST1.

OASST1 supplies the conversational and formatted-response shapes that were
missing from Armor's security datasets. Rows must be English, accepted by at
least two reviewers, and below a conservative Detoxify ceiling. Assistant
responses supervise prompt-injection negatives; both roles supervise
obfuscation negatives. No phrase templates or log strings are used.
"""

from datasets import load_dataset

from armor.detectors.registry import default_registry
from armor.models.labels import LABELS

_DATASET_ID = "OpenAssistant/oasst1"
_REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
_MAX_DETOXIFY_SCORE = 0.05
_MAX_WORDS = 350  # stays inside the classifier's 512-token window in practice
_DLP_REGISTRY = default_registry()


def _negative(text: str, label: str) -> tuple[str, list[float], list[float]]:
    labels = [0.0] * len(LABELS)
    mask = [0.0] * len(LABELS)
    mask[LABELS.index(label)] = 1.0
    return text.strip(), labels, mask


def _is_reviewed_clean(row: dict) -> bool:
    scores = row.get("detoxify")
    return (
        row.get("lang") == "en"
        and row.get("review_result") is True
        and not row.get("deleted", False)
        and row.get("review_count", 0) >= 2
        and isinstance(row.get("text"), str)
        and bool(row["text"].strip())
        and len(row["text"].split()) <= _MAX_WORDS
        and isinstance(scores, dict)
        and bool(scores)
        and all(isinstance(value, (int, float)) for value in scores.values())
        and max(scores.values()) < _MAX_DETOXIFY_SCORE
    )


def _eligible_group(rows, role: str):
    eligible = [
        row
        for row in rows
        if row.get("role") == role
        and _is_reviewed_clean(row)
        and (role != "assistant" or row.get("rank") == 0)
    ]
    eligible.sort(key=lambda row: row["message_id"])
    return eligible


def _stable_take(rows, limit: int, predicate=lambda row: True, seen=None):
    seen = seen if seen is not None else set()
    selected = []
    for row in rows:
        normalized = " ".join(row["text"].split())
        if normalized in seen or not predicate(row):
            continue
        seen.add(normalized)
        selected.append(row)
        if len(selected) == limit:
            break
    return selected


def _is_structured(row: dict) -> bool:
    text = row["text"]
    return (
        len(text.splitlines()) >= 8
        or "```" in text
        or text.count("\n- ") >= 3
        or text.count("|") >= 6
    )


def _is_code(row: dict) -> bool:
    return "```" in row["text"]


def _is_long(row: dict) -> bool:
    return 200 <= len(row["text"].split()) <= _MAX_WORDS


def _is_short(row: dict) -> bool:
    return len(row["text"].split()) <= 40


def _is_dlp_clean(row: dict) -> bool:
    return not _DLP_REGISTRY.scan(row["text"])


def load_examples() -> list[tuple[str, list[float], list[float]]]:
    rows = load_dataset(_DATASET_ID, revision=_REVISION, split="train")
    prompters = _eligible_group(rows, "prompter")
    assistants = _eligible_group(rows, "assistant")

    injection_seen = set()
    injection_rows = _stable_take(assistants, 400, _is_code, injection_seen)
    injection_rows += _stable_take(assistants, 600, _is_structured, injection_seen)
    injection_rows += _stable_take(assistants, 400, _is_long, injection_seen)
    injection_rows += _stable_take(assistants, 200, seen=injection_seen)

    obfuscation_rows = _stable_take(assistants, 400, _is_short)
    obfuscation_rows += _stable_take(prompters, 800)

    sensitive_seen = set()
    sensitive_rows = _stable_take(
        assistants, 400, lambda row: _is_code(row) and _is_dlp_clean(row), sensitive_seen
    )
    sensitive_rows += _stable_take(
        assistants, 400, lambda row: _is_structured(row) and _is_dlp_clean(row), sensitive_seen
    )
    sensitive_rows += _stable_take(
        assistants, 400, lambda row: _is_long(row) and _is_dlp_clean(row), sensitive_seen
    )

    examples = [_negative(row["text"], "prompt_injection") for row in injection_rows]
    examples += [_negative(row["text"], "obfuscation_encoding") for row in obfuscation_rows]
    examples += [_negative(row["text"], "sensitive_data_exposure") for row in sensitive_rows]
    return examples
