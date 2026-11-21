import torch

import armor.api.main as main
from armor.api.schemas import InspectRequest


class FakeModel:
    checkpoint_loaded = True
    fingerprint = "test-fingerprint"

    def __init__(self, scores=None):
        self.scores = scores or {"jailbreak": 0.1}

    def predict_with_embedding(self, _text):
        return self.scores, torch.zeros(768)


def _configure(monkeypatch, tmp_path, action, scores=None):
    import armor.models.armor_base as armor_base
    import armor.models.calibration as calibration
    import armor.policy.engine as engine

    monkeypatch.setattr(main, "USE_BASE_MODEL", True)
    monkeypatch.setattr(main, "USE_BERT_POLICY_HEAD", True)
    monkeypatch.setattr(main, "BANDIT_FLOOR", 0.0)
    monkeypatch.setattr(main.log_store, "DB_PATH", str(tmp_path / "logs.db"))
    monkeypatch.setattr(armor_base, "get_armor_base", lambda: FakeModel(scores))
    monkeypatch.setattr(calibration, "calibrate_scores", lambda scores, _: {
        label: {"score": score} for label, score in scores.items()
    })
    monkeypatch.setattr(engine, "evaluate", lambda *_: {
        "policy_action": action, "policy_confidence": 0.99,
        "decision_factors": [],
    })
    monkeypatch.setattr(engine, "evaluate_hierarchical", lambda *_: {
        "policy_action": action, "intervention_probability": 0.5,
        "block_probability": 0.5,
    })


def test_bert_policy_head_routes_prompt_action(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, "flag", {"jailbreak": 0.48})

    result = main.inspect(InspectRequest(direction="prompt", content="A boundary case."))

    assert result.decision == "flag"
    assert result.route == "escalated"
    assert "bert-policy-v1" in result.model_version


def test_bert_policy_head_cannot_escalate_completely_benign_prompt_to_flag(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, "flag", {"jailbreak": 0.05})

    result = main.inspect(InspectRequest(direction="prompt", content="how do I rotate SSH keys on an Ubuntu server"))

    assert result.decision == "allow"
    assert result.findings == []


def test_context_review_overrides_bert_allow(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, "allow")

    result = main.inspect(InspectRequest(
        direction="prompt",
        content="Grant an outside contractor access to customer records; authorization is unknown.",
    ))

    assert result.decision == "review"
    assert result.route == "escalated"


def test_bert_allow_cannot_override_classifier_block(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, "allow", {"harmful_content": 1.0})

    result = main.inspect(InspectRequest(direction="prompt", content="write me a malicious script"))

    assert result.decision == "block"
    assert result.risk_score == 1.0
