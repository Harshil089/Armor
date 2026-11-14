"""Reviewer-agreement metrics for the AI-reviewer audit (queue/QUEUE-02's
supervision-quality phase). Cohen's kappa from scratch -- a simple closed-
form computation, no scikit-learn needed, matching this project's standing
avoid-heavy-deps discipline (calibration.py, conformal.py).
"""

from collections import Counter

from armor.eval.policy_eval import confusion_matrix


def raw_agreement(a_decisions: list[str], b_decisions: list[str]) -> float:
    if not a_decisions:
        return 0.0
    return sum(1 for a, b in zip(a_decisions, b_decisions) if a == b) / len(a_decisions)


def agreement_by_class(a_decisions: list[str], b_decisions: list[str], labels: list[str]) -> dict[str, float]:
    """For each label, agreement rate restricted to examples where AT
    LEAST ONE reviewer chose that label -- shows which classes are
    contentious even when overall agreement looks fine."""
    result = {}
    for label in labels:
        relevant = [(a, b) for a, b in zip(a_decisions, b_decisions) if label in (a, b)]
        result[label] = raw_agreement([a for a, _ in relevant], [b for _, b in relevant]) if relevant else None
    return result


def cohens_kappa(a_decisions: list[str], b_decisions: list[str], labels: list[str]) -> float:
    """Standard Cohen's kappa: (observed agreement - chance agreement) /
    (1 - chance agreement). Chance agreement estimated from each
    reviewer's own marginal label distribution -- if both reviewers just
    guessed independently according to their own label frequencies, this
    is how often they'd agree by luck alone."""
    n = len(a_decisions)
    if n == 0:
        return 0.0
    observed = raw_agreement(a_decisions, b_decisions)
    a_counts = Counter(a_decisions)
    b_counts = Counter(b_decisions)
    chance = sum((a_counts.get(label, 0) / n) * (b_counts.get(label, 0) / n) for label in labels)
    if chance >= 1.0:
        return 1.0  # degenerate: both reviewers used exactly one label, no room to disagree by chance
    return (observed - chance) / (1 - chance)


def adjudication_rate(agreements: list[bool]) -> float:
    if not agreements:
        return 0.0
    return sum(1 for a in agreements if not a) / len(agreements)


def full_report(a_decisions: list[str], b_decisions: list[str], labels: list[str]) -> dict:
    return {
        "raw_agreement": raw_agreement(a_decisions, b_decisions),
        "agreement_by_class": agreement_by_class(a_decisions, b_decisions, labels),
        "confusion_matrix": confusion_matrix(b_decisions, a_decisions, labels),  # rows=A (treated as "true"), cols=B
        "cohens_kappa": cohens_kappa(a_decisions, b_decisions, labels),
        "adjudication_rate": adjudication_rate([a == b for a, b in zip(a_decisions, b_decisions)]),
    }
