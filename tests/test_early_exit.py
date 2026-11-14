import pytest
import torch

from armor.models.early_exit import (
    EarlyExitHeads,
    candidate_from_logits,
    distillation_loss,
    load_exit_heads,
    save_exit_heads,
)


def test_exit_candidate_only_returns_conservative_allow():
    logits = torch.tensor([[-10.0] * 6 + [12.0]])
    candidate = candidate_from_logits(
        logits, depth=6, safe_threshold=0.99, max_threat_threshold=0.02
    )
    assert candidate is not None
    assert candidate.decision == "allow"
    assert candidate.depth == 6
    assert candidate_from_logits(
        torch.tensor([[0.0] * 6 + [12.0]]), depth=6,
        safe_threshold=0.99, max_threat_threshold=0.02,
    ) is None
    assert candidate_from_logits(
        logits, depth=6, safe_threshold=0.99, max_threat_threshold=0.02,
        structural_warning=True,
    ) is None
    assert candidate_from_logits(logits, depth=22) is None


def test_exit_distillation_penalizes_false_safe_targets_asymmetrically():
    student = torch.zeros((2, 7), requires_grad=True)
    teacher = torch.zeros((2, 6))
    safe_targets = torch.tensor([1.0, 0.0])
    weighted = distillation_loss(student, teacher, safe_targets, unsafe_weight=8.0)
    unweighted = distillation_loss(student, teacher, safe_targets, unsafe_weight=1.0)
    assert weighted > unweighted
    weighted.backward()
    assert student.grad is not None
    assert student.grad[1, -1] > student.grad[0, -1].abs()


def test_exit_heads_checkpoint_is_bound_to_encoder_fingerprint(tmp_path):
    heads = EarlyExitHeads(16)
    checkpoint = tmp_path / "exit.pt"
    save_exit_heads(heads, str(checkpoint), encoder_fingerprint="abc")
    loaded = load_exit_heads(16, str(checkpoint), expected_encoder_fingerprint="abc")
    assert loaded is not None
    with pytest.raises(ValueError, match="different ArmorBase encoder"):
        load_exit_heads(16, str(checkpoint), expected_encoder_fingerprint="other")
