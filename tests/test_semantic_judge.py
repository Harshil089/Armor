"""Fast, mocked tests -- no real model weights loaded. Tests JSON
extraction/validation and the fail-open path in isolation. The real model
is exercised only by the slow test at the bottom (ARMOR_RUN_SLOW_TESTS=1),
matching tests/test_calibration_integration.py's pattern.
"""

import os

import pytest

from armor.models.semantic_judge import SEMANTIC_FEATURES, SemanticJudge


def test_parse_valid_json():
    generated = (
        '{"authorization_confidence": 0.2, "target_sensitivity": 0.8, '
        '"harm_potential": 0.7, "actionability": 0.6, "ambiguity": 0.9}'
    )
    result = SemanticJudge._parse(generated)
    assert result == [0.2, 0.8, 0.7, 0.6, 0.9]


def test_parse_extracts_json_from_surrounding_text():
    generated = (
        'Sure, here is my analysis:\n'
        '{"authorization_confidence": 0.1, "target_sensitivity": 0.5, '
        '"harm_potential": 0.5, "actionability": 0.5, "ambiguity": 0.5}\n'
        "Let me know if you need more."
    )
    result = SemanticJudge._parse(generated)
    assert result == [0.1, 0.5, 0.5, 0.5, 0.5]


def test_parse_returns_none_on_malformed_json():
    assert SemanticJudge._parse("not json at all") is None
    assert SemanticJudge._parse("{broken: json,,, }") is None


def test_parse_returns_none_on_missing_key():
    generated = '{"authorization_confidence": 0.2, "target_sensitivity": 0.8}'
    assert SemanticJudge._parse(generated) is None


def test_parse_returns_none_on_out_of_range_value():
    generated = (
        '{"authorization_confidence": 1.5, "target_sensitivity": 0.8, '
        '"harm_potential": 0.7, "actionability": 0.6, "ambiguity": 0.9}'
    )
    assert SemanticJudge._parse(generated) is None


def test_parse_returns_none_on_non_numeric_value():
    generated = (
        '{"authorization_confidence": "high", "target_sensitivity": 0.8, '
        '"harm_potential": 0.7, "actionability": 0.6, "ambiguity": 0.9}'
    )
    assert SemanticJudge._parse(generated) is None


def test_analyze_returns_none_on_generation_error(monkeypatch):
    judge = SemanticJudge.__new__(SemanticJudge)  # skip __init__, no model load

    class _BoomTokenizer:
        def apply_chat_template(self, *a, **k):
            raise RuntimeError("boom")

    judge.tokenizer = _BoomTokenizer()
    judge.model = None
    judge.device = "cpu"
    assert judge.analyze("anything") is None


@pytest.mark.skipif(
    os.environ.get("ARMOR_RUN_SLOW_TESTS") != "1",
    reason="downloads and loads the real semantic judge model; set ARMOR_RUN_SLOW_TESTS=1 to run",
)
def test_real_model_produces_valid_shaped_output():
    from armor.models.semantic_judge import get_semantic_judge

    judge = get_semantic_judge()
    result = judge.analyze("My roommate moved out owing me money and left their phone.")
    assert result is not None
    assert len(result) == len(SEMANTIC_FEATURES)
    assert all(0.0 <= v <= 1.0 for v in result)
