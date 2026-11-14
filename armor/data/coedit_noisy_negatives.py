"""Natural English error/correction pairs used as obfuscation negatives.

CoEdIT's GEC rows contain a noisy sentence and its human-curated correction.
Both sides are benign and supervise only ``obfuscation_encoding`` so spelling
and grammar noise cannot act as a shortcut for concealment intent.
"""

import re

from datasets import load_dataset

from armor.models.labels import LABELS

_DATASET_ID = "grammarly/coedit"
_REVISION = "e9a255c33ef910bc33a9d2b522653fa87521583e"
_MAX_PAIRS = 400
_MIN_WORDS = 3
_MAX_WORDS = 40


def _negative(text: str) -> tuple[str, list[float], list[float]]:
    labels = [0.0] * len(LABELS)
    mask = [0.0] * len(LABELS)
    mask[LABELS.index("obfuscation_encoding")] = 1.0
    return text.strip(), labels, mask


def _source_text(row: dict) -> str | None:
    source = row.get("src")
    target = row.get("tgt")
    if row.get("task") != "gec" or not isinstance(source, str) or not isinstance(target, str):
        return None
    _, separator, text = source.partition(": ")
    text = text.strip() if separator else source.strip()
    target = target.strip()
    if (
        text == target
        or not _MIN_WORDS <= len(text.split()) <= _MAX_WORDS
        or not _is_local_edit_pair(text, target)
    ):
        return None
    return text


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for column, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _is_local_edit_pair(source: str, target: str) -> bool:
    source_words = re.findall(r"[A-Za-z']+", source.lower())
    target_words = re.findall(r"[A-Za-z']+", target.lower())
    if len(source_words) != len(target_words):
        return False
    changes = [(left, right) for left, right in zip(source_words, target_words) if left != right]
    return 1 <= len(changes) <= 2 and all(
        max(len(left), len(right)) >= 4 and _edit_distance(left, right) <= 2
        for left, right in changes
    )


def load_examples(max_pairs: int = _MAX_PAIRS) -> list[tuple[str, list[float], list[float]]]:
    rows = load_dataset(_DATASET_ID, revision=_REVISION, split="train")
    eligible = []
    for row in rows:
        source = _source_text(row)
        if source:
            eligible.append((str(row["_id"]), source, row["tgt"].strip()))
    eligible.sort(key=lambda item: item[0])

    examples = []
    seen = set()
    for _, source, target in eligible:
        pair = []
        for text in (source, target):
            normalized = " ".join(text.split())
            if normalized not in seen:
                seen.add(normalized)
                pair.append(_negative(text))
        if len(pair) == 2:
            examples.extend(pair)
        if len(examples) == max_pairs * 2:
            break
    return examples
