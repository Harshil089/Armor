import torch

from armor.models.uncertainty import (
    hierarchical_decision,
    intervention_block_to_class_probs,
    policy_confidence,
    should_abstain,
)


def test_confidence_high_for_clear_winner():
    logits = torch.tensor([10.0, -10.0, -10.0, -10.0])
    assert policy_confidence(logits) > 0.9


def test_confidence_low_for_near_tie():
    logits = torch.tensor([1.0, 1.01, -10.0, -10.0])
    assert policy_confidence(logits) < 0.1


def test_should_abstain_respects_floor():
    assert should_abstain(0.1, floor=0.3) is True
    assert should_abstain(0.5, floor=0.3) is False


def test_intervention_block_to_class_probs_sums_to_one():
    probs = intervention_block_to_class_probs(0.8, 0.3)
    assert abs(sum(probs) - 1.0) < 1e-9


def test_intervention_block_to_class_probs_low_intervention_favors_allow():
    allow_p, flag_p, block_p = intervention_block_to_class_probs(0.05, 0.9)
    assert allow_p > flag_p
    assert allow_p > block_p


def test_hierarchical_decision_singleton_set_is_that_action():
    result = hierarchical_decision([0.9, 0.05, 0.05], conformal_threshold=0.2, ood_score=0.1)
    assert result["final_action"] == "allow"
    assert result["conformal_set"] == ["allow"]


def test_hierarchical_decision_ambiguous_set_is_review():
    result = hierarchical_decision([0.4, 0.35, 0.25], conformal_threshold=0.7, ood_score=0.1)
    assert result["final_action"] == "review"
    assert len(result["conformal_set"]) > 1


def test_hierarchical_decision_high_ood_forces_review_even_if_confident():
    result = hierarchical_decision([0.95, 0.03, 0.02], conformal_threshold=0.2, ood_score=0.9, ood_threshold=0.6)
    assert result["final_action"] == "review"
