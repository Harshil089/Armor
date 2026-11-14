"""The frozen set is only meaningful if it stays independent of what the
model trains on. A case that also appears verbatim in a synthetic generator
measures memorisation, not generalisation — one did ("Our team finished the
sprint ahead of schedule this week.", shared with synthetic_pii's negative
pool) and went unnoticed until checked by hand.
"""

from collections import Counter

from armor.data.synthetic_dangerous_tool_action import generate_examples as generate_dta
from armor.data.synthetic_long_negatives import generate_examples as generate_long_negatives
from armor.data.synthetic_obfuscation import generate_examples as generate_obfuscation
from armor.data.synthetic_pii import generate_examples as generate_pii
from armor.data.synthetic_short_negatives import generate_examples as generate_short_negatives
from armor.data.synthetic_topic_adjacent_negatives import generate_examples as generate_topic_adjacent
from armor.eval.regression_set import REGRESSION_CASES
from armor.models.labels import LABELS

_MIN_CASES_PER_SIDE = 4


def test_no_case_appears_in_synthetic_training_data():
    generated = set()
    for generate in (generate_obfuscation, generate_dta, generate_pii):
        for text, _, _ in generate(3000, 3000):
            generated.add(text.strip())
    # Single-arg generators: negative-only boosters whose pools the
    # corresponding regression cases must stay disjoint from (see
    # regression_set.py's own notes on each guard section).
    for generate in (generate_short_negatives, generate_long_negatives, generate_topic_adjacent):
        for text, _, _ in generate(3000):
            generated.add(text.strip())
    overlap = sorted({text for text, _, _ in REGRESSION_CASES if text.strip() in generated})
    assert overlap == [], f"regression cases leaked into synthetic training data: {overlap}"


def test_cases_are_unique():
    texts = [text for text, _, _ in REGRESSION_CASES]
    duplicates = sorted({t for t, count in Counter(texts).items() if count > 1})
    assert duplicates == [], f"duplicate regression cases: {duplicates}"


def test_every_label_has_enough_cases_on_both_sides():
    counts = Counter((label, is_positive) for _, label, is_positive in REGRESSION_CASES)
    for label in LABELS:
        positives = counts[(label, True)]
        negatives = counts[(label, False)]
        # Guards the original 2-and-2 sizing that made per-label scores
        # swing (dangerous_tool_action read 0.045 to 0.213 across runs).
        assert positives >= _MIN_CASES_PER_SIDE, f"{label} has only {positives} positive cases"
        assert negatives >= _MIN_CASES_PER_SIDE, f"{label} has only {negatives} negative cases"


def test_labels_are_known():
    unknown = sorted({label for _, label, _ in REGRESSION_CASES if label not in LABELS})
    assert unknown == [], f"regression cases reference unknown labels: {unknown}"
