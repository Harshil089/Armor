"""Tests for matched short-request harmful_content pairs."""

from armor.data.synthetic_harmful_content_pairs import generate_examples
from armor.eval.regression_set import REGRESSION_CASES
from armor.models.labels import LABELS

_HARM = LABELS.index("harmful_content")


def test_generate_examples_balanced_and_masked():
    examples = generate_examples()
    assert len(examples) == 464  # 192 pairs * 2 + 80 benign build negatives
    positives = [ex for ex in examples if ex[1][_HARM] == 1.0]
    negatives = [ex for ex in examples if ex[1][_HARM] == 0.0]
    assert len(positives) == 192
    assert len(negatives) == 272
    for _text, y, m in examples:
        assert m[_HARM] == 1.0
        assert sum(m) == 1.0
        assert sum(y) in (0.0, 1.0)


def test_no_leakage_into_regression_set():
    generated = {text.lower() for text, _y, _m in generate_examples()}
    for case in REGRESSION_CASES:
        assert case[0].lower() not in generated
