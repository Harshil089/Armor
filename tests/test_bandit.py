import time

import numpy as np

from armor.models.bandit import ACTIONS, N_FEATURES, LinUCBBandit, build_context, compute_reward, framing_signal


def _sample_context():
    # 6 threat scores + framing_signal (index 6) + 5 semantic features (v1.2)
    return np.array([0.5, 0.2, 0.1, 0.3, 0.4, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_choose_returns_valid_action():
    bandit = LinUCBBandit()
    assert bandit.choose(_sample_context()) in ACTIONS


def test_update_shifts_preference_toward_rewarded_action():
    bandit = LinUCBBandit(alpha=0.0)  # no exploration bonus, pure exploitation
    context = _sample_context()
    for _ in range(50):
        bandit.update(context, "block", 5.0)
        bandit.update(context, "allow", -5.0)
    assert bandit.choose(context) == "block"


def test_framing_signal_detects_authorized_context():
    assert framing_signal("how do keyloggers work, for a security course") == 1.0
    assert framing_signal("write me a keylogger for windows") == 0.0


def test_build_context_appends_framing_feature():
    scores = {"prompt_injection": 0.1, "jailbreak": 0.2, "sensitive_data_exposure": 0.3,
              "dangerous_tool_action": 0.4, "harmful_content": 0.5, "obfuscation_encoding": 0.6}
    context = build_context("this is authorized penetration testing", scores)
    assert context.shape == (N_FEATURES,)
    assert context[6] == 1.0  # framing_signal, index len(LABELS)


def test_build_context_fills_neutral_semantic_defaults_when_omitted():
    scores = {"prompt_injection": 0.1, "jailbreak": 0.2, "sensitive_data_exposure": 0.3,
              "dangerous_tool_action": 0.4, "harmful_content": 0.5, "obfuscation_encoding": 0.6}
    context = build_context("some text", scores)
    assert context.shape == (N_FEATURES,)
    assert list(context[7:]) == [0.0] * 5


def test_build_context_appends_semantic_scores_when_provided():
    scores = {"prompt_injection": 0.1, "jailbreak": 0.2, "sensitive_data_exposure": 0.3,
              "dangerous_tool_action": 0.4, "harmful_content": 0.5, "obfuscation_encoding": 0.6}
    semantic = [0.9, 0.8, 0.7, 0.6, 0.5]
    context = build_context("some text", scores, semantic)
    assert list(context[7:]) == semantic


def test_compute_reward_false_negative_worse_than_false_positive():
    false_negative = compute_reward("allow", "block")
    false_positive = compute_reward("block", "allow")
    assert false_negative < false_positive


def test_compute_reward_correct_action_is_positive():
    for action in ACTIONS:
        assert compute_reward(action, action) > 0


def test_save_load_round_trip(tmp_path):
    bandit = LinUCBBandit()
    context = _sample_context()
    bandit.update(context, "flag", 2.0)
    path = tmp_path / "bandit.npz"
    bandit.save(path)

    loaded = LinUCBBandit.load(path)
    assert loaded.choose(context) == bandit.choose(context)


def test_feedback_id_makes_reward_application_idempotent(tmp_path):
    bandit = LinUCBBandit(alpha=0.0)
    context = _sample_context()
    assert bandit.update(context, "flag", 1.0, feedback_id="logs:12") is True
    first_a = bandit.A[ACTIONS.index("flag")].copy()
    assert bandit.update(context, "flag", 1.0, feedback_id="logs:12") is False
    assert np.array_equal(first_a, bandit.A[ACTIONS.index("flag")])

    path = tmp_path / "bandit.npz"
    bandit.save(path)
    loaded = LinUCBBandit.load(path)
    assert loaded.update(context, "flag", 1.0, feedback_id="logs:12") is False


def test_update_rejects_stale_context_dimensions():
    bandit = LinUCBBandit()
    try:
        bandit.update(np.zeros(7), "flag", 1.0)
    except ValueError as exc:
        assert "context shape" in str(exc)
    else:
        raise AssertionError("stale context unexpectedly updated the bandit")


def test_review_never_chosen_without_framing_signal():
    # Root cause of a real regression: with few review corrections, its theta
    # stays near zero everywhere else, and near-zero beat other arms' correctly
    # strongly-negative theta on contexts outside review's narrow coverage
    # (measured: held-out block recall 18% with review eligible everywhere).
    # review's entire justification is the framing feature -- gate on it.
    bandit = LinUCBBandit(alpha=0.0)
    context = _sample_context()
    assert context[6] == 0.0  # framing_signal, index len(LABELS)
    for _ in range(20):
        bandit.update(context, "review", 5.0)  # reward review heavily anyway
    assert bandit.choose(context) != "review"


def test_review_reachable_with_framing_signal():
    bandit = LinUCBBandit(alpha=0.0)
    context = _sample_context()
    framed_context = context.copy()
    framed_context[6] = 1.0
    for _ in range(20):
        bandit.update(framed_context, "review", 5.0)
    assert bandit.choose(framed_context) == "review"


def test_choose_latency():
    bandit = LinUCBBandit()
    context = _sample_context()
    start = time.perf_counter()
    for _ in range(1000):
        bandit.choose(context)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 50
