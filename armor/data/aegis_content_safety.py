"""Human-annotated safe dialogue negatives from NVIDIA Aegis v1.

Only prompt-only rows with unanimous ``Safe`` annotations are used. Aegis
``llm_response``, combined, and multi-turn text can include more than one
dialogue leg, while Armor classifies one proxy leg at a time, so those rows'
grain does not match inference.
"""

from datasets import load_dataset

from armor.models.labels import LABELS

_DATASET_ID = "nvidia/Aegis-AI-Content-Safety-Dataset-1.0"
_REVISION = "bd96d862068e47630197de64eb91f8d1481ff3e0"
_LABELS = ("harmful_content", "sensitive_data_exposure", "obfuscation_encoding")
_MAX_ROWS = 800


def _negative(text: str) -> tuple[str, list[float], list[float]]:
    labels = [0.0] * len(LABELS)
    mask = [0.0] * len(LABELS)
    for label in _LABELS:
        mask[LABELS.index(label)] = 1.0
    return text.strip(), labels, mask


def _is_unanimously_safe(row: dict) -> bool:
    count = row.get("num_annotations", 0)
    votes = [row.get(f"labels_{index}") for index in range(count)]
    return bool(votes) and all(vote == "Safe" for vote in votes)


def load_examples(max_rows: int = _MAX_ROWS) -> list[tuple[str, list[float], list[float]]]:
    rows = load_dataset(_DATASET_ID, revision=_REVISION, split="train")
    eligible = [
        row
        for row in rows
        if row.get("text_type") == "user_message"
        and isinstance(row.get("text"), str)
        and row["text"].strip()
        and _is_unanimously_safe(row)
    ]
    # Aegis IDs are random UUIDs. Sorting by ID gives a stable sample from
    # the whole source instead of a topic-biased head slice.
    eligible.sort(key=lambda row: row["id"])
    selected = []
    seen_text = set()
    for row in eligible:
        normalized = " ".join(row["text"].split())
        if normalized in seen_text:
            continue
        seen_text.add(normalized)
        selected.append(row)
        if len(selected) == max_rows:
            break
    return [_negative(row["text"]) for row in selected]
