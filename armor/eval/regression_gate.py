"""Regression gate: compare a freshly trained head against the previous one
on the frozen regression set (regression_set.py) before it's allowed to
replace the deployed checkpoint."""

from collections import defaultdict

import torch

from armor.eval.regression_set import REGRESSION_CASES
from armor.models.labels import LABELS


@torch.no_grad()
def evaluate(head: torch.nn.Module, encoder, tokenizer, device: str = "cpu") -> dict[str, float]:
    """Per-label mean(positive scores) - mean(negative scores) over the
    frozen regression set. Positive number = correctly separated in the
    right direction; near zero or negative = not separating, or flipped."""
    scores_by_label: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"pos": [], "neg": []})

    for text, label, is_positive in REGRESSION_CASES:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        pooled = encoder(**inputs).last_hidden_state[:, 0, :]
        scores = torch.sigmoid(head(pooled)).squeeze(0).tolist()
        score = scores[LABELS.index(label)]
        scores_by_label[label]["pos" if is_positive else "neg"].append(score)

    metrics = {}
    for label, buckets in scores_by_label.items():
        pos_mean = sum(buckets["pos"]) / len(buckets["pos"]) if buckets["pos"] else 0.0
        neg_mean = sum(buckets["neg"]) / len(buckets["neg"]) if buckets["neg"] else 0.0
        metrics[label] = pos_mean - neg_mean
    return metrics


def check_regression(
    new_metrics: dict[str, float], baseline_metrics: dict[str, float], tolerance: float = 0.05
) -> list[str]:
    """Labels whose separation fell by more than tolerance.

    Labels without a baseline signal cannot regress. Empty list = pass.
    """
    regressed = []
    for label, baseline_gap in baseline_metrics.items():
        new_gap = new_metrics.get(label, 0.0)
        if baseline_gap > tolerance and baseline_gap - new_gap > tolerance:
            regressed.append(label)
    return regressed
