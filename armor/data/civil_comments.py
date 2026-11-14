"""google/civil_comments training examples for harmful_content.

ponytail: replaces lmsys/toxic-chat entirely (see README.md licensing
notice history / QUEUE items) — that source was cc-by-nc-4.0, which a
checkpoint trained on arguably inherits, blocking an unrestricted
open-source release. civil_comments is CC0-1.0 (public domain, no
conditions at all), 1.8M real organic public comments with a continuous
toxicity score plus sub-scores (severe_toxicity, obscene, threat, insult,
identity_attack, sexual_explicit), not a synthetic or red-team-only set —
so it supplies real hard negatives (ordinary benign comments in the same
domain) rather than needing them sourced separately.

Not the full 1.8M rows: that would dwarf every other label's volume and
distort the per-label batch composition again (the exact failure mode the
per-label loss fix addressed). Takes a fixed first-N deterministic slice
(reproducible, no shuffling) and caps positive/negative counts to roughly
the same order of magnitude as the source it replaces.
"""

from datasets import load_dataset

from armor.models.labels import LABELS

_LABEL_INDEX = LABELS.index("harmful_content")
_SCAN_ROWS = 200_000
_MAX_POSITIVE = 2500
_MAX_NEGATIVE = 2500
_POSITIVE_THRESHOLD = 0.5
_NEGATIVE_THRESHOLD = 0.1  # clearly benign, avoids ambiguous mid-range rows


def load_examples() -> list[tuple[str, list[float], list[float]]]:
    # ponytail: split="train[:N]" slice syntax — datasets only materializes
    # the requested 200k-row slice, not the full 1.8M-row dataset. Loading
    # the full dataset here (as an earlier version did) OOM-killed a retrain
    # that also had the ModernBERT encoder loaded in the same process; this
    # is a real memory fix, not just style.
    dataset = load_dataset("google/civil_comments", split=f"train[:{_SCAN_ROWS}]")

    examples = []
    n_pos = 0
    n_neg = 0
    for row in dataset:
        if n_pos >= _MAX_POSITIVE and n_neg >= _MAX_NEGATIVE:
            break

        toxicity = row["toxicity"]
        is_positive = toxicity >= _POSITIVE_THRESHOLD
        is_negative = toxicity < _NEGATIVE_THRESHOLD

        if is_positive and n_pos < _MAX_POSITIVE:
            n_pos += 1
        elif is_negative and n_neg < _MAX_NEGATIVE:
            n_neg += 1
        else:
            continue

        label_vec = [0.0] * len(LABELS)
        mask_vec = [0.0] * len(LABELS)
        label_vec[_LABEL_INDEX] = 1.0 if is_positive else 0.0
        mask_vec[_LABEL_INDEX] = 1.0
        examples.append((row["text"], label_vec, mask_vec))

    return examples
