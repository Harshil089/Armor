"""Same-ModernBERT selective-depth execution for long-input shadow tests."""

import time
from dataclasses import dataclass

import torch

from armor.inference.fusion import WindowResult, fuse_request_evidence
from armor.inference.windowing import (
    InspectionBudgetExceeded,
    TokenWindow,
    expand_window_context,
    plan_windows,
)
from armor.models.armor_base import EncoderState, TokenContext
from armor.models.early_exit import candidate_from_logits, load_exit_heads
from armor.models.labels import LABELS


@dataclass
class AdaptiveWindowResult:
    decision: str
    review_reason: str | None
    windows: list[WindowResult]
    trace: dict[str, object]


def _batch_context(windows: list[TokenWindow], pad_token_id: int, original: TokenContext):
    max_len = max(w.context.inputs["input_ids"].shape[-1] for w in windows)
    ids = torch.full((len(windows), max_len), pad_token_id, dtype=torch.long)
    masks = torch.zeros((len(windows), max_len), dtype=torch.long)
    for i, window in enumerate(windows):
        row = window.context.inputs["input_ids"][0]
        ids[i, : row.numel()] = row
        masks[i, : row.numel()] = 1
    return TokenContext(
        inputs={"input_ids": ids, "attention_mask": masks},
        token_count=max_len,
        original_length=original.original_length,
    )


def _subset(state: EncoderState, indices: list[int]) -> EncoderState:
    selected = torch.tensor(indices, dtype=torch.long, device=state.hidden_states.device)
    inputs = {
        name: value.index_select(0, selected.to(value.device))
        for name, value in state.context.inputs.items()
    }
    context = TokenContext(
        inputs=inputs,
        token_count=state.context.token_count,
        original_length=state.context.original_length,
    )
    return EncoderState(
        hidden_states=state.hidden_states.index_select(0, selected),
        attention_mask=state.attention_mask.index_select(0, selected.to(state.attention_mask.device)),
        sliding_window_mask=state.sliding_window_mask.index_select(
            0, selected.to(state.sliding_window_mask.device)
        ),
        position_ids=state.position_ids,
        next_layer=state.next_layer,
        context=context,
        timings_ms={},
    )


def _score_action(scores: dict[str, float], finding_threshold: float, block_threshold: float):
    risk = max(scores.values(), default=0.0)
    return "block" if risk >= block_threshold else "flag" if risk >= finding_threshold else "allow", risk


def _budget_exceeded(started: float, budget_ms: float) -> bool:
    return budget_ms > 0 and (time.perf_counter() - started) * 1000 >= budget_ms


def _calibrate(model, raw: dict[str, float]) -> dict[str, float]:
    if getattr(model, "fingerprint", None) is None:
        return raw
    from armor.models.calibration import calibrate_scores

    return {label: item["score"] for label, item in calibrate_scores(raw, model.fingerprint).items()}


def _candidate(model, logits: torch.Tensor, depth: int):
    raw = dict(zip(LABELS, torch.sigmoid(logits.detach()).cpu().tolist()))
    return candidate_from_logits(
        logits,
        depth=depth,
        calibrated_threat_scores=_calibrate(model, raw),
    )


def _base_result(window: TokenWindow, depth: int, decision: str, confidence: float):
    return WindowResult(
        window.index, window.start_token, window.end_token, depth, True,
        decision, confidence,
    )


def run_adaptive_single(
    model,
    context: TokenContext,
    *,
    finding_threshold: float = 0.5,
    block_threshold: float = 0.8,
    exit_policy: str = "patience",
    max_inspection_time_ms: float = 3000.0,
) -> AdaptiveWindowResult:
    """Evaluate the whole request with optional same-stack early ALLOW exits."""
    started = time.perf_counter()
    if exit_policy not in {"single", "patience"}:
        raise ValueError("exit_policy must be 'single' or 'patience'")
    if max_inspection_time_ms < 0:
        raise ValueError("max_inspection_time_ms cannot be negative")
    heads = load_exit_heads(
        model.encoder.config.hidden_size,
        expected_encoder_fingerprint=getattr(model, "encoder_fingerprint", None),
        device=getattr(model, "device", "cpu"),
    )
    if heads is None:
        return AdaptiveWindowResult("review", "exit_heads_unavailable", [], {
            "approximate_compute_path": "heads_unavailable_review",
            "total_ms": (time.perf_counter() - started) * 1000,
        })
    state = model.begin_encoder(context)
    model.continue_encoder(state, 6)
    out6 = heads.layer6(state.hidden_states)
    candidate6 = _candidate(model, out6[0], 6)
    stage_depth = 6
    selected = None
    if exit_policy == "single" and candidate6 is not None:
        selected = candidate6
    if selected is None and _budget_exceeded(started, max_inspection_time_ms):
        result = WindowResult(0, 0, context.token_count or 0, 6, False,
                              escalation_reason="inspection_budget_exceeded")
        return AdaptiveWindowResult("review", "inspection_budget_exceeded", [result], {
            "window_count": 1, "approximate_compute_path": "time_budget_review",
            "windows_exited_at_6": 0, "windows_exited_at_12": 0,
            "windows_full_depth": 0, "inspection_budget_exceeded": True,
            "total_ms": (time.perf_counter() - started) * 1000,
        })
    if selected is None:
        model.continue_encoder(state, 12)
        out12 = heads.layer12(state.hidden_states)
        candidate12 = _candidate(model, out12[0], 12)
        stage_depth = 12
        if candidate12 is not None and (exit_policy == "single" or candidate6 is not None):
            selected = candidate12
        if selected is None and _budget_exceeded(started, max_inspection_time_ms):
            result = WindowResult(0, 0, context.token_count or 0, 12, False,
                                  escalation_reason="inspection_budget_exceeded")
            return AdaptiveWindowResult("review", "inspection_budget_exceeded", [result], {
                "window_count": 1, "approximate_compute_path": "time_budget_review",
                "windows_exited_at_6": 0, "windows_exited_at_12": 0,
                "windows_full_depth": 0, "inspection_budget_exceeded": True,
                "total_ms": (time.perf_counter() - started) * 1000,
            })
    if selected is not None:
        result = WindowResult(0, 0, context.token_count or 0, selected.depth,
                              True, "allow", selected.confidence)
        decision, reason = "allow", None
    else:
        if stage_depth < len(model.encoder.layers):
            model.continue_encoder(state, len(model.encoder.layers))
        with torch.no_grad():
            logits = model.head(state.hidden_states[:, 0, :])
            raw = dict(zip(LABELS, torch.sigmoid(logits).squeeze(0).cpu().tolist()))
        scores = _calibrate(model, raw)
        decision, risk = _score_action(scores, finding_threshold, block_threshold)
        result = WindowResult(0, 0, context.token_count or 0, len(model.encoder.layers),
                              True, decision, risk)
        reason = None
    trace = {
        "window_count": 1,
        "windows_exited_at_6": int(result.max_depth == 6),
        "windows_exited_at_12": int(result.max_depth == 12),
        "windows_full_depth": int(result.max_depth >= len(model.encoder.layers)),
        "approximate_compute_path": f"single_request_{result.max_depth}_layers",
        **state.timings_ms,
        "total_ms": (time.perf_counter() - started) * 1000,
    }
    return AdaptiveWindowResult(decision, reason, [result], trace)


def run_adaptive_windows(
    model,
    context: TokenContext,
    tokenizer,
    *,
    window_size: int = 512,
    overlap: int = 96,
    max_windows: int | None = None,
    max_full_depth_windows: int = 4,
    max_context_expansions: int = 2,
    max_expanded_tokens: int = 768,
    finding_threshold: float = 0.5,
    block_threshold: float = 0.8,
    exit_policy: str = "patience",
    max_inspection_time_ms: float = 3000.0,
    expand_suspicious: bool = True,
    deterministic_block: bool = False,
) -> AdaptiveWindowResult:
    """Shallow-batch windows, continue only unresolved windows, fuse safely.

    This is a candidate path: callers must shadow it against full-depth Armor
    before using its result to answer a request. Model depth is shared and
    continuation uses each selected window's retained layer output.
    """
    started = time.perf_counter()
    if exit_policy not in {"single", "patience"}:
        raise ValueError("exit_policy must be 'single' or 'patience'")
    if max_windows is not None and max_windows < 1:
        raise ValueError("max_windows must be positive")
    if max_full_depth_windows < 0 or max_context_expansions < 0:
        raise ValueError("deep-window and context-expansion budgets cannot be negative")
    if max_inspection_time_ms < 0:
        raise ValueError("max_inspection_time_ms cannot be negative")
    try:
        windows = plan_windows(
            context, tokenizer, window_size=window_size, overlap=overlap,
            max_windows=max_windows,
        )
    except InspectionBudgetExceeded as exc:
        decision, reason = fuse_request_evidence([], inspection_budget_exceeded=True)
        return AdaptiveWindowResult(decision, reason, [], {
            "window_count": 0, "approximate_compute_path": "budget_review",
            "reason": str(exc), "total_ms": (time.perf_counter() - started) * 1000,
        })
    heads = load_exit_heads(
        model.encoder.config.hidden_size,
        expected_encoder_fingerprint=getattr(model, "encoder_fingerprint", None),
        device=getattr(model, "device", "cpu"),
    )
    if heads is None:
        unresolved = [WindowResult(w.index, w.start_token, w.end_token, 0, False,
                                   escalation_reason="exit_heads_unavailable") for w in windows]
        decision, reason = fuse_request_evidence(unresolved)
        return AdaptiveWindowResult(decision, reason, unresolved, {
            "window_count": len(windows), "approximate_compute_path": "heads_unavailable_review",
            "total_ms": (time.perf_counter() - started) * 1000,
        })

    pad_id = getattr(model.encoder.config, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
    batch = _batch_context(windows, int(pad_id), context)
    state = model.begin_encoder(batch)
    stage_times: dict[str, float] = {}
    t = time.perf_counter()
    model.continue_encoder(state, 6)
    stage_times["layer6_encoder_ms"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    out6 = heads.layer6(state.hidden_states)
    stage_times["layer6_head_ms"] = (time.perf_counter() - t) * 1000
    candidate6 = [_candidate(model, row, 6) for row in out6]
    results: list[WindowResult | None] = [None] * len(windows)

    layer12_indices = []
    for i, candidate in enumerate(candidate6):
        if exit_policy == "single" and candidate is not None:
            results[i] = _base_result(windows[i], 6, "allow", candidate.confidence)
        else:
            layer12_indices.append(i)
    if layer12_indices and _budget_exceeded(started, max_inspection_time_ms):
        for i in layer12_indices:
            results[i] = WindowResult(
                windows[i].index, windows[i].start_token, windows[i].end_token,
                6, False, escalation_reason="inspection_budget_exceeded",
            )
        concrete = [r for r in results if r is not None]
        decision, reason = fuse_request_evidence(
            concrete, deterministic_block=deterministic_block,
            inspection_budget_exceeded=True,
        )
        return AdaptiveWindowResult(decision, reason, concrete, {
            "window_count": len(windows), "windows_exited_at_6": sum(
                r.max_depth == 6 and r.resolved for r in concrete
            ), "windows_exited_at_12": 0, "windows_full_depth": 0,
            "approximate_compute_path": "time_budget_review",
            "inspection_budget_exceeded": True,
            "total_ms": (time.perf_counter() - started) * 1000,
        })
    state12 = _subset(state, layer12_indices) if layer12_indices else None
    del state

    layer22_indices: list[int] = []
    if state12 is not None:
        t = time.perf_counter()
        model.continue_encoder(state12, 12)
        stage_times["layer12_encoder_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        out12 = heads.layer12(state12.hidden_states)
        stage_times["layer12_head_ms"] = (time.perf_counter() - t) * 1000
        candidate12 = [_candidate(model, row, 12) for row in out12]
        for local_i, original_i in enumerate(layer12_indices):
            prior = candidate6[original_i]
            current = candidate12[local_i]
            if current is not None and (exit_policy == "single" or prior is not None):
                results[original_i] = _base_result(
                    windows[original_i], 12, "allow",
                    min(current.confidence, prior.confidence) if prior is not None else current.confidence,
                )
            else:
                layer22_indices.append(local_i)

    if layer22_indices and (
        len(layer22_indices) > max_full_depth_windows
        or _budget_exceeded(started, max_inspection_time_ms)
    ):
        reason = "inspection_budget_exceeded"
        for local_i in layer22_indices:
            original_i = layer12_indices[local_i]
            results[original_i] = WindowResult(
                windows[original_i].index, windows[original_i].start_token,
                windows[original_i].end_token, 12, False,
                escalation_reason=reason,
            )
    elif layer22_indices and state12 is not None:
        state22 = _subset(state12, layer22_indices)
        del state12
        t = time.perf_counter()
        model.continue_encoder(state22, len(model.encoder.layers))
        stage_times["layer22_encoder_ms"] = (time.perf_counter() - t) * 1000
        with torch.no_grad():
            batch_logits = model.head(state22.hidden_states[:, 0, :])
            raw_batch_scores = torch.sigmoid(batch_logits).cpu().tolist()
        for local_row, state_index in enumerate(layer22_indices):
            original_i = layer12_indices[state_index]
            raw = dict(zip(LABELS, raw_batch_scores[local_row]))
            scores = _calibrate(model, raw)
            decision, risk = _score_action(scores, finding_threshold, block_threshold)
            result = _base_result(windows[original_i], len(model.encoder.layers), decision, risk)
            results[original_i] = result
    elif layer22_indices:
        raise AssertionError("deep windows selected without retained layer-12 state")

    concrete = [r for r in results if r is not None]
    expansion_count = 0
    if expand_suspicious and max_context_expansions > 0:
        for result in list(concrete):
            if result.max_depth < len(model.encoder.layers) or result.decision not in {"flag", "block"}:
                continue
            if expansion_count >= max_context_expansions:
                result.resolved = False
                result.escalation_reason = "inspection_budget_exceeded"
                continue
            if _budget_exceeded(started, max_inspection_time_ms):
                result.resolved = False
                result.escalation_reason = "inspection_budget_exceeded"
                continue
            window = windows[result.index]
            expanded = expand_window_context(
                context, window, tokenizer, max_tokens=max_expanded_tokens
            )
            expanded_state = model.begin_encoder(expanded)
            model.continue_encoder(expanded_state, len(model.encoder.layers))
            with torch.no_grad():
                logits = model.head(expanded_state.hidden_states[:, 0, :])
                raw = dict(zip(LABELS, torch.sigmoid(logits).squeeze(0).cpu().tolist()))
            scores = _calibrate(model, raw)
            action, risk = _score_action(scores, finding_threshold, block_threshold)
            concrete.append(WindowResult(
                index=window.index, start_token=max(0, window.start_token - 64),
                end_token=min(context.token_count or 0, window.end_token + 64),
                max_depth=len(model.encoder.layers), resolved=True,
                decision=action, confidence=risk,
                escalation_reason="context_expanded",
            ))
            expansion_count += 1

    decision, reason = fuse_request_evidence(
        concrete, deterministic_block=deterministic_block,
        inspection_budget_exceeded=any(not r.resolved for r in concrete),
    )
    trace = {
        "window_count": len(windows),
        "windows_exited_at_6": sum(r.max_depth == 6 for r in concrete if r.resolved),
        "windows_exited_at_12": sum(r.max_depth == 12 for r in concrete if r.resolved),
        "windows_full_depth": sum(r.max_depth >= len(model.encoder.layers) for r in concrete if r.resolved),
        "context_expansion_count": expansion_count,
        "deep_window_count": len(layer22_indices),
        "inspection_budget_exceeded": any(not r.resolved for r in concrete),
        "approximate_compute_path": "windowed_shallow_then_selective_deep",
        **stage_times,
        "total_ms": (time.perf_counter() - started) * 1000,
    }
    return AdaptiveWindowResult(decision, reason, concrete, trace)
