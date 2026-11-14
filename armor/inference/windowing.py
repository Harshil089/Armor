"""Token-ID window planning without per-window retokenization."""

from dataclasses import dataclass

import torch

from armor.models.armor_base import TokenContext


class InspectionBudgetExceeded(RuntimeError):
    pass


@dataclass
class TokenWindow:
    index: int
    start_token: int
    end_token: int
    context: TokenContext


def _content_positions(input_ids: list[int], tokenizer) -> list[int]:
    try:
        special = tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
        if len(special) == len(input_ids):
            positions = [i for i, bit in enumerate(special) if not bit]
            if positions:
                return positions
    except (AttributeError, NotImplementedError, TypeError):
        pass
    # ModernBERT uses a leading CLS and trailing SEP. This fallback is for
    # simple compatible tokenizer doubles and makes no text/tokenization call.
    return list(range(1, max(1, len(input_ids) - 1)))


def _make_context(body_ids: list[int], body_offsets, tokenizer, original_length: int) -> TokenContext:
    ids = tokenizer.build_inputs_with_special_tokens(body_ids)
    try:
        # Fast tokenizers reject already_has_special_tokens=False. Build the
        # sequence first, then identify the inserted boundary tokens.
        special_mask = tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
    except (AttributeError, AssertionError, NotImplementedError, TypeError):
        special_count = max(0, len(ids) - len(body_ids))
        prefix_count = special_count // 2
        special_mask = [1] * prefix_count + [0] * len(body_ids) + [1] * (len(ids) - prefix_count - len(body_ids))
    offsets = []
    body_offset_iter = iter(body_offsets or [(0, 0)] * len(body_ids))
    for special in special_mask:
        offsets.append((0, 0) if special else next(body_offset_iter, (0, 0)))
    return TokenContext(
        inputs={
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        },
        token_count=len(ids),
        original_length=original_length,
        offsets=torch.tensor([offsets], dtype=torch.long),
    )


def plan_windows(
    context: TokenContext,
    tokenizer,
    *,
    window_size: int = 512,
    overlap: int = 96,
    max_windows: int | None = None,
) -> list[TokenWindow]:
    """Build overlapping windows by slicing existing token IDs and offsets.

    The 512/96 defaults are initial experimental values near the current
    training context. They remain explicit arguments so frozen gates can
    compare other overlaps before serving promotion.
    """
    if context.oversize:
        raise InspectionBudgetExceeded(context.oversize_reason or "input exceeds model context")
    if window_size < 4 or overlap < 0:
        raise ValueError("window_size must be >= 4 and overlap must be nonnegative")
    inputs = context.inputs
    ids = inputs["input_ids"][0].tolist()
    offsets = context.offsets[0].tolist() if context.offsets is not None else [(0, 0)] * len(ids)
    body_positions = _content_positions(ids, tokenizer)
    body_ids = [ids[i] for i in body_positions]
    body_offsets = [offsets[i] for i in body_positions]
    special_count = len(tokenizer.build_inputs_with_special_tokens([]))
    body_size = window_size - special_count
    if body_size <= 0 or overlap >= body_size:
        raise ValueError("overlap must be smaller than the available content window")
    if len(ids) <= window_size:
        return [TokenWindow(0, 0, len(body_ids), context)]

    step = body_size - overlap
    starts = [0]
    while starts[-1] + body_size < len(body_ids):
        starts.append(starts[-1] + step)
    spans = [(start, min(start + body_size, len(body_ids))) for start in starts]
    spans = [span for span in spans if span[0] < span[1]]
    if max_windows is not None and len(spans) > max_windows:
        raise InspectionBudgetExceeded("inspection_budget_exceeded")
    return [
        TokenWindow(
            index=index,
            start_token=start,
            end_token=end,
            context=_make_context(
                body_ids[start:end], body_offsets[start:end], tokenizer, context.original_length
            ),
        )
        for index, (start, end) in enumerate(spans)
    ]


def expand_window_context(
    context: TokenContext,
    window: TokenWindow,
    tokenizer,
    *,
    margin: int = 64,
    max_tokens: int = 768,
) -> TokenContext:
    """Create a bounded contiguous context around a selected window."""
    if margin < 0 or max_tokens < 4:
        raise ValueError("margin must be nonnegative and max_tokens at least four")
    ids = context.inputs["input_ids"][0].tolist()
    offsets = context.offsets[0].tolist() if context.offsets is not None else [(0, 0)] * len(ids)
    body_positions = _content_positions(ids, tokenizer)
    body_ids = [ids[i] for i in body_positions]
    body_offsets = [offsets[i] for i in body_positions]
    special_count = len(tokenizer.build_inputs_with_special_tokens([]))
    width = max_tokens - special_count
    if width <= 0:
        raise ValueError("max_tokens leaves no room for content")
    start = max(0, window.start_token - margin)
    end = min(len(body_ids), window.end_token + margin)
    if end - start > width:
        start = max(0, min(start, len(body_ids) - width))
        end = min(len(body_ids), start + width)
    return _make_context(body_ids[start:end], body_offsets[start:end], tokenizer, context.original_length)
