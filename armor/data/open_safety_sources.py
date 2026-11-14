"""Commercially licensed, ungated open safety datasets (MIT / Apache-2.0 / CC-BY-4.0).

All sources are revision-pinned and verified against their Hugging Face cards:
- bench-llms/or-bench (CC-BY-4.0, ungated): over-refusal hard benign prompts
  (harmful_content + dangerous_tool_action negatives) and toxic contrasts.
- walledai/SaladBench (Apache-2.0, ungated): hierarchical safety prompts.
- cyberec/Prompt-injection-dataset (Apache-2.0, ungated): prompt injection
  positives paired with security-adjacent hard negatives.

Attribution and terms: THIRD_PARTY_NOTICES.md.
"""

import random

from datasets import load_dataset

from armor.models.labels import LABELS

_PI = LABELS.index("prompt_injection")
_TOOL = LABELS.index("dangerous_tool_action")
_HARM = LABELS.index("harmful_content")


def _row(text: str, positive: bool, labels: tuple[int, ...], pos_label: int = _HARM):
    y, m = [0.0] * len(LABELS), [0.0] * len(LABELS)
    for idx in labels:
        y[idx] = float(positive and idx == pos_label)
        m[idx] = 1.0
    return text.strip(), y, m


def _sample(rows: list[str], cap: int, seed: int = 0) -> list[str]:
    unique = sorted({r.strip() for r in rows if r and r.strip()})
    random.Random(seed).shuffle(unique)
    return unique[:cap]


def load_or_bench_hard(cap: int = 1000):
    """bench-llms/or-bench (or-bench-hard-1k), CC-BY-4.0: hard benign prompts that sound unsafe."""
    rev = "fd6ee135ee63ff6c4f3ff72c0e39627bf0a7f314"
    rows = load_dataset("bench-llms/or-bench", "or-bench-hard-1k", revision=rev, split="train")
    return [_row(t, False, (_HARM, _TOOL)) for t in _sample(rows["prompt"], cap)]


def load_or_bench_toxic(cap: int = 600):
    """bench-llms/or-bench (or-bench-toxic), CC-BY-4.0: harmful/toxic contrast prompts."""
    rev = "fd6ee135ee63ff6c4f3ff72c0e39627bf0a7f314"
    rows = load_dataset("bench-llms/or-bench", "or-bench-toxic", revision=rev, split="train")
    return [_row(t, True, (_HARM,)) for t in _sample(rows["prompt"], cap)]


def load_salad_bench(cap: int = 1000):
    """walledai/SaladBench (prompts/base), Apache-2.0: harmful base prompts."""
    rev = "5d9d078571dd171990162ba6201bc81c41c7c867"
    rows = load_dataset("walledai/SaladBench", "prompts", revision=rev, split="base")
    return [_row(t, True, (_HARM,)) for t in _sample(rows["prompt"], cap)]


def load_cyberec_pi(cap_pos: int = 750, cap_neg: int = 750):
    """cyberec/Prompt-injection-dataset (core/train), Apache-2.0: PI + security-adjacent negatives."""
    rev = "cfe0ce4a4d5eb9528ffaca55cf0db27ac609b8b1"
    rows = load_dataset("cyberec/Prompt-injection-dataset", "core", revision=rev, split="train")
    pos, neg = [], []
    for r in rows:
        text = r.get("text") or ""
        if not text.strip():
            continue
        if int(r["label"]) == 1:
            pos.append(text)
        else:
            neg.append(text)
    out = [_row(t, True, (_PI,), pos_label=_PI) for t in _sample(pos, cap_pos)]
    out += [_row(t, False, (_PI,), pos_label=_PI) for t in _sample(neg, cap_neg)]
    return out


LOADERS = {
    "or_bench_hard": load_or_bench_hard,
    "or_bench_toxic": load_or_bench_toxic,
    "salad_bench": load_salad_bench,
    "cyberec_pi": load_cyberec_pi,
}
