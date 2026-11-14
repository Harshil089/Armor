import os

import pytest
import torch

from armor.models.policy_head import POLICY_ACTIONS, PolicyHead, build_input
from armor.models.policy_head import BinaryPolicyHead, save_intervention_head, save_block_head


def test_forward_returns_one_logit_per_action():
    head = PolicyHead(embedding_dim=8)
    pooled = torch.randn(1, 8)
    scores = torch.rand(1, 6)
    logits = head(pooled, scores)
    assert logits.shape == (1, len(POLICY_ACTIONS))


def test_build_input_fills_missing_labels_with_zero():
    pooled = torch.randn(8)
    embedding, threat_scores = build_input(pooled, {"prompt_injection": 0.9})
    assert torch.equal(embedding, pooled)
    assert threat_scores.shape == (6,)
    assert threat_scores[0] == 0.9
    assert threat_scores[1:].sum() == 0.0


def test_save_load_round_trip(tmp_path):
    from armor.models.policy_head import save_policy_head

    head = PolicyHead(embedding_dim=8)
    path = tmp_path / "policy_head.pt"
    save_policy_head(head, path=str(path))

    loaded = PolicyHead(embedding_dim=8)
    loaded.load_state_dict(torch.load(path, weights_only=True))

    pooled = torch.randn(1, 8)
    scores = torch.rand(1, 6)
    assert torch.equal(head(pooled, scores), loaded(pooled, scores))


def test_binary_head_returns_scalar_logit_per_example():
    head = BinaryPolicyHead(embedding_dim=8)
    pooled = torch.randn(4, 8)
    scores = torch.rand(4, 6)
    logits = head(pooled, scores)
    assert logits.shape == (4,)


def test_binary_head_save_load_round_trip(tmp_path):
    head = BinaryPolicyHead(embedding_dim=8)
    path = str(tmp_path / "intervention_head.pt")
    save_intervention_head(head, path=path)

    loaded = BinaryPolicyHead(embedding_dim=8)
    loaded.load_state_dict(torch.load(path, weights_only=True))

    pooled = torch.randn(1, 8)
    scores = torch.rand(1, 6)
    assert torch.equal(head(pooled, scores), loaded(pooled, scores))


@pytest.mark.skipif(
    os.environ.get("ARMOR_RUN_SLOW_TESTS") != "1",
    reason="loads the real ArmorBase model; set ARMOR_RUN_SLOW_TESTS=1 to run",
)
def test_predict_with_embedding_calls_encoder_exactly_once():
    """The whole point of predict_with_embedding() is reusing the same
    forward pass predict() already does -- not a second one for the policy
    head. Spies on the real encoder's forward to prove it, not just infer
    it from reading the code."""
    from armor.models.armor_base import get_armor_base

    model = get_armor_base()
    call_count = 0
    original_forward = model.encoder.forward

    def spy_forward(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_forward(*args, **kwargs)

    model.encoder.forward = spy_forward
    try:
        scores, pooled = model.predict_with_embedding("test input")
        assert call_count == 1
        assert pooled.shape[-1] == model.encoder.config.hidden_size
    finally:
        model.encoder.forward = original_forward
