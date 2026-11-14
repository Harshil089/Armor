from armor.eval.regression_gate import check_regression


def test_clean_pass_when_new_matches_or_beats_baseline():
    baseline = {"prompt_injection": 0.4, "jailbreak": 0.3}
    new = {"prompt_injection": 0.45, "jailbreak": 0.5}
    assert check_regression(new, baseline) == []


def test_flags_clear_regression_on_one_label():
    baseline = {"prompt_injection": 0.4, "jailbreak": 0.3}
    new = {"prompt_injection": 0.01, "jailbreak": 0.35}  # prompt_injection collapsed
    assert check_regression(new, baseline) == ["prompt_injection"]


def test_flags_direction_flip():
    baseline = {"dangerous_tool_action": 0.2}
    new = {"dangerous_tool_action": -0.3}  # now scores negatives higher than positives
    assert check_regression(new, baseline) == ["dangerous_tool_action"]


def test_flags_material_drop_even_when_new_score_stays_positive():
    baseline = {"obfuscation_encoding": 0.84}
    new = {"obfuscation_encoding": 0.61}
    assert check_regression(new, baseline) == ["obfuscation_encoding"]


def test_label_never_separated_at_baseline_is_not_flagged():
    # baseline never learned this label (e.g. obfuscation_encoding, no training
    # rows) — can't regress on a signal that was never there
    baseline = {"obfuscation_encoding": 0.02}
    new = {"obfuscation_encoding": -0.01}
    assert check_regression(new, baseline) == []


def test_missing_label_in_new_metrics_treated_as_zero_and_flagged():
    baseline = {"harmful_content": 0.3}
    new = {}  # label absent entirely from new run
    assert check_regression(new, baseline) == ["harmful_content"]
