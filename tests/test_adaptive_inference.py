import pytest
import torch
from types import SimpleNamespace

from armor.inference.fusion import WindowResult, fuse_request_evidence
from armor.inference.windowing import (
    InspectionBudgetExceeded,
    expand_window_context,
    plan_windows,
)
from armor.models.armor_base import TokenContext
import armor.inference.adaptive as adaptive


class FakeTokenizer:
    def __init__(self):
        self.encode_calls = 0

    def __call__(self, *_args, **_kwargs):
        self.encode_calls += 1
        raise AssertionError("window planning must slice token IDs, not retokenize")

    def build_inputs_with_special_tokens(self, ids):
        return [101, *ids, 102]

    def get_special_tokens_mask(self, ids, already_has_special_tokens=False):
        if already_has_special_tokens:
            return [int(token in (101, 102)) for token in ids]
        return [1, *([0] * len(ids)), 1]


def _context(size):
    ids = [101, *range(1000, 1000 + size), 102]
    offsets = [(0, 0), *[(i, i + 1) for i in range(size)], (0, 0)]
    return TokenContext(
        inputs={"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)))},
        token_count=len(ids), original_length=size,
        offsets=torch.tensor([offsets]),
    )


def test_window_planner_slices_ids_with_overlap_and_no_retokenization():
    tokenizer = FakeTokenizer()
    context = _context(1000)
    windows = plan_windows(context, tokenizer, window_size=512, overlap=96)
    assert len(windows) == 3
    assert windows[0].start_token == 0
    assert windows[1].start_token == 414
    assert windows[-1].end_token == 1000
    assert windows[0].context.inputs["input_ids"].shape[-1] <= 512
    assert windows[0].context.inputs["input_ids"][0, 0] == 101
    assert windows[-1].context.inputs["input_ids"][0, -1] == 102
    assert windows[0].end_token - windows[1].start_token == 96
    assert tokenizer.encode_calls == 0


def test_window_budget_and_context_expansion_are_explicit():
    tokenizer = FakeTokenizer()
    context = _context(1000)
    with pytest.raises(InspectionBudgetExceeded, match="inspection_budget_exceeded"):
        plan_windows(context, tokenizer, window_size=512, overlap=96, max_windows=2)
    windows = plan_windows(context, tokenizer, window_size=512, overlap=96)
    expanded = expand_window_context(context, windows[1], tokenizer, margin=64, max_tokens=600)
    assert expanded.token_count <= 600
    assert expanded.offsets[0, 1, 0] == windows[1].start_token - 64
    assert tokenizer.encode_calls == 0


def test_request_fusion_never_treats_uninspected_content_as_allow():
    allow = WindowResult(0, 0, 512, 6, True, "allow", 0.999)
    unresolved = WindowResult(1, 416, 928, 6, False, None, None)
    assert fuse_request_evidence([allow, unresolved]) == (
        "review", "unresolved_or_uninspected_window"
    )
    assert fuse_request_evidence([allow], inspection_budget_exceeded=True) == (
        "review", "inspection_budget_exceeded"
    )
    block = WindowResult(1, 416, 928, 22, True, "block", 0.99)
    assert fuse_request_evidence([allow, block]) == (
        "block", "validated_full_depth_window_block"
    )
    flag = WindowResult(1, 416, 928, 12, True, "flag", 0.8)
    assert fuse_request_evidence([allow, flag]) == (
        "flag", "nonblocking_window_finding"
    )
    assert fuse_request_evidence([allow]) == ("allow", None)
    assert fuse_request_evidence([], deterministic_block=True) == (
        "block", "deterministic_critical_evidence"
    )


def test_selective_window_execution_keeps_unsafe_window_for_full_depth(monkeypatch):
    class FakeModel:
        fingerprint = None
        encoder_fingerprint = "fixture"

        def __init__(self):
            self.encoder = SimpleNamespace(
                config=SimpleNamespace(hidden_size=16, pad_token_id=0),
                layers=[object() for _ in range(22)],
            )
            self.head = FakeThreatHead()

        def begin_encoder(self, context):
            batch, seq = context.inputs["input_ids"].shape
            hidden = torch.zeros((batch, seq, 16))
            hidden[:, 0, 0] = context.inputs["input_ids"][:, 1].float()
            return SimpleNamespace(
                hidden_states=hidden,
                attention_mask=torch.ones((batch, 1, 1, seq)),
                sliding_window_mask=torch.ones((batch, 1, seq, seq)),
                position_ids=torch.arange(seq).unsqueeze(0),
                next_layer=1, context=context, timings_ms={},
            )

        def continue_encoder(self, state, end_layer):
            state.next_layer = end_layer + 1
            return state

    class FakeThreatHead(torch.nn.Module):
        def forward(self, hidden):
            marker = hidden[:, 0]
            logits = torch.full((marker.shape[0], 6), -10.0)
            logits[:, 0] = torch.where(marker == 1000, 10.0, -10.0)
            return logits

    class FakeExit(torch.nn.Module):
        def forward(self, hidden):
            marker = hidden[:, 0, 0]
            safe = torch.where(marker == 1000, -12.0, 12.0)
            return torch.cat([torch.full((len(marker), 6), -10.0), safe[:, None]], dim=1)

    monkeypatch.setattr(adaptive, "load_exit_heads", lambda *a, **k: SimpleNamespace(
        layer6=FakeExit(), layer12=FakeExit()
    ))
    context = _context(1000)
    tokenizer = FakeTokenizer()
    result = adaptive.run_adaptive_windows(
        FakeModel(), context, tokenizer, expand_suspicious=False,
        max_full_depth_windows=4, exit_policy="patience",
    )
    assert result.decision == "block"
    assert result.trace["window_count"] == 3
    assert result.trace["windows_exited_at_12"] == 2
    assert result.trace["windows_full_depth"] == 1
    assert result.trace["deep_window_count"] == 1
    assert next(w for w in result.windows if w.index == 0).max_depth == 22


def test_single_request_can_exit_at_six_and_patience_at_twelve(monkeypatch):
    class FakeModel:
        fingerprint = None
        encoder_fingerprint = "fixture"

        def __init__(self):
            self.encoder = SimpleNamespace(
                config=SimpleNamespace(hidden_size=16), layers=[object() for _ in range(22)]
            )
            self.head = torch.nn.Linear(16, 6)
            self.depths = []

        def begin_encoder(self, context):
            hidden = torch.zeros((1, context.token_count, 16))
            return SimpleNamespace(
                hidden_states=hidden, attention_mask=torch.ones((1, 1, 1, context.token_count)),
                sliding_window_mask=torch.ones((1, 1, context.token_count, context.token_count)),
                position_ids=torch.arange(context.token_count).unsqueeze(0), next_layer=1,
                context=context, timings_ms={},
            )

        def continue_encoder(self, state, depth):
            self.depths.append(depth)
            state.next_layer = depth + 1
            return state

    class SafeExit(torch.nn.Module):
        def forward(self, hidden):
            return torch.tensor([[-10.0] * 6 + [12.0]]).expand(hidden.shape[0], -1)

    monkeypatch.setattr(adaptive, "load_exit_heads", lambda *a, **k: SimpleNamespace(
        layer6=SafeExit(), layer12=SafeExit()
    ))
    context = TokenContext(
        inputs={"input_ids": torch.tensor([[101, 3, 102]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.long)},
        token_count=3, original_length=1,
    )
    model = FakeModel()
    result = adaptive.run_adaptive_single(model, context, exit_policy="single")
    assert result.decision == "allow"
    assert result.trace["windows_exited_at_6"] == 1
    assert model.depths == [6]

    model = FakeModel()
    result = adaptive.run_adaptive_single(model, context, exit_policy="patience")
    assert result.decision == "allow"
    assert result.trace["windows_exited_at_12"] == 1
    assert model.depths == [6, 12]

    model = FakeModel()
    result = adaptive.run_adaptive_single(
        model, context, exit_policy="patience", max_inspection_time_ms=0.0001
    )
    assert result.decision == "review"
    assert result.review_reason == "inspection_budget_exceeded"
    assert result.windows[0].resolved is False
