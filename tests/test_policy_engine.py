import torch
import pytest

from armor.models.policy_head import PolicyHead
import armor.policy.engine as engine_module
from armor.policy.engine import evaluate, evaluate_hierarchical


class _FixedLogitHead:
    """Stands in for a trained BinaryPolicyHead -- returns a fixed logit
    regardless of input, so tests can control intervention/block
    probabilities directly rather than reverse-engineering weights."""

    def __init__(self, logit: float):
        self.logit = logit

    def __call__(self, *args, **kwargs):
        return torch.tensor([self.logit])


def test_evaluate_returns_expected_shape(monkeypatch):
    fake_head = PolicyHead(embedding_dim=8)
    monkeypatch.setattr(engine_module, "get_policy_head", lambda embedding_dim=768: fake_head)

    scores = {"prompt_injection": 0.9, "jailbreak": 0.1, "sensitive_data_exposure": 0.0,
              "dangerous_tool_action": 0.0, "harmful_content": 0.0, "obfuscation_encoding": 0.0}
    result = evaluate(scores, torch.randn(8))

    assert result["policy_action"] in ("allow", "flag", "block", "review")
    assert 0.0 <= result["policy_confidence"] <= 1.0
    assert isinstance(result["decision_factors"], list)
    assert "high prompt_injection evidence" in result["decision_factors"]


def test_policy_heads_accept_mps_encoder_embedding(monkeypatch):
    if not torch.backends.mps.is_available():
        pytest.skip("Apple MPS is unavailable")

    class _CpuOnlyHead(_FixedLogitHead):
        def __call__(self, pooled, scores):
            assert pooled.device.type == "cpu"
            assert scores.device.type == "cpu"
            return torch.zeros((1, 4))

    monkeypatch.setattr(engine_module, "get_policy_head", lambda embedding_dim=768: _CpuOnlyHead(0.0))
    result = evaluate({}, torch.ones(8, device="mps"))
    assert result["policy_action"] == "review"


def test_evaluate_abstains_to_review_on_low_confidence(monkeypatch):
    # A head whose weights are all zero produces identical logits for every
    # action -> confidence 0.0 -> must abstain to "review", never argmax's
    # arbitrary tie-break.
    fake_head = PolicyHead(embedding_dim=8)
    for p in fake_head.parameters():
        p.data.zero_()
    monkeypatch.setattr(engine_module, "get_policy_head", lambda embedding_dim=768: fake_head)

    scores = {"prompt_injection": 0.5}
    result = evaluate(scores, torch.randn(8))
    assert result["policy_action"] == "review"
    assert "policy confidence below review threshold" in result["decision_factors"]


def _patch_heads(monkeypatch, intervention_logit: float, block_logit: float):
    monkeypatch.setattr(
        engine_module, "get_intervention_head", lambda embedding_dim=768: _FixedLogitHead(intervention_logit)
    )
    monkeypatch.setattr(engine_module, "get_block_head", lambda embedding_dim=768: _FixedLogitHead(block_logit))


def test_hierarchical_system_b_confidently_allow(monkeypatch):
    _patch_heads(monkeypatch, intervention_logit=-5.0, block_logit=0.0)  # sigmoid(-5) ~ 0.007
    result = evaluate_hierarchical({}, torch.randn(8))
    assert result["policy_action"] == "allow"
    assert "conformal_set" not in result  # System B doesn't run conformal


def test_hierarchical_system_b_confidently_block(monkeypatch):
    _patch_heads(monkeypatch, intervention_logit=5.0, block_logit=5.0)
    result = evaluate_hierarchical({}, torch.randn(8))
    assert result["policy_action"] == "block"


def test_hierarchical_system_b_confidently_flag(monkeypatch):
    _patch_heads(monkeypatch, intervention_logit=5.0, block_logit=-5.0)
    result = evaluate_hierarchical({}, torch.randn(8))
    assert result["policy_action"] == "flag"


def test_hierarchical_system_b_uncertain_intervention_is_review(monkeypatch):
    _patch_heads(monkeypatch, intervention_logit=0.0, block_logit=0.0)  # sigmoid(0) = 0.5, max uncertainty
    result = evaluate_hierarchical({}, torch.randn(8))
    assert result["policy_action"] == "review"


def test_hierarchical_system_c_uses_conformal_and_ood(monkeypatch):
    _patch_heads(monkeypatch, intervention_logit=-5.0, block_logit=0.0)

    class _FixedOOD:
        def score(self, embedding):
            return 0.0  # in-distribution

    result = evaluate_hierarchical({}, torch.randn(8), conformal_threshold=0.5, ood_reference=_FixedOOD())
    assert "conformal_set" in result
    assert "ood_score" in result
    assert result["policy_action"] in ("allow", "flag", "block", "review")


def test_hierarchical_system_c_high_ood_forces_review(monkeypatch):
    _patch_heads(monkeypatch, intervention_logit=-5.0, block_logit=0.0)  # would confidently be "allow"

    class _FixedOOD:
        def score(self, embedding):
            return 0.99  # far out of distribution

    result = evaluate_hierarchical({}, torch.randn(8), conformal_threshold=0.5, ood_reference=_FixedOOD())
    assert result["policy_action"] == "review"
