"""Tests for window score fusion in armor.api.main."""

import torch

from armor.api.main import _fuse_window_scores
from armor.models.labels import LABELS


def test_fuse_window_scores_uses_max_for_pi_and_second_max_for_others():
    pi_idx = LABELS.index("prompt_injection")
    harm_idx = LABELS.index("harmful_content")
    window_raw = torch.zeros((4, len(LABELS)))
    window_raw[0, pi_idx] = 0.92
    window_raw[1, harm_idx] = 0.85  # single-window spike on benign doc
    window_raw[2, harm_idx] = 0.21

    fused = _fuse_window_scores(window_raw)
    assert abs(fused["prompt_injection"] - 0.92) < 1e-6
    assert abs(fused["harmful_content"] - 0.21) < 1e-6


def test_window_scores_early_stops_on_pi_hit():
    from types import SimpleNamespace
    from armor.models.armor_base import ArmorBase, TokenContext

    calls = []

    class FakeEncoder:
        def __call__(self, input_ids, attention_mask):
            calls.append(input_ids.shape[0])
            b = input_ids.shape[0]
            return SimpleNamespace(last_hidden_state=torch.zeros((b, input_ids.shape[1], 4)))

    class FakeHead:
        def __init__(self):
            self.weight = torch.zeros((len(LABELS), 4))

        def __call__(self, pooled):
            logits = torch.full((pooled.shape[0], len(LABELS)), -4.0)
            logits[0, LABELS.index("prompt_injection")] = 4.0  # sigmoid ~ 0.982
            return logits

    dummy = SimpleNamespace(
        tokenizer=SimpleNamespace(cls_token_id=101, sep_token_id=102, pad_token_id=0),
        device="cpu",
        encoder=FakeEncoder(),
        head=FakeHead(),
    )
    # 120 tokens with window=32, overlap=8 -> 5 windows; batch_size=2 -> 3 batches without early stop
    ids = torch.arange(1, 121).unsqueeze(0)
    ctx = TokenContext(
        inputs={"input_ids": ids, "attention_mask": torch.ones_like(ids)},
        token_count=120,
        original_length=120,
        timings_ms={},
    )
    scores, pooled = ArmorBase.window_scores(
        dummy, ctx, window=32, overlap=8, batch_size=2, early_stop_threshold=0.85
    )
    assert len(calls) == 1
    assert scores.shape[0] == 2
    assert pooled.shape == (2, 4)
