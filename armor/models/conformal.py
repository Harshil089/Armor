"""Split-conformal prediction over {allow, flag, block} -- turns a point
prediction into a calibrated prediction SET, so "review" (queue/QUEUE-02's
formulation-bottleneck fix) can mean "multiple materially different
actions remain plausible" instead of a fourth thing to classify. Pure
numpy, no new dependency, same constraint as calibration.py's own
from-scratch Venn-Abers implementation.

Standard split-conformal classification (Vovk et al.): calibrate a
nonconformity threshold on a held-out slice, then at inference include
every class whose nonconformity score falls under that threshold. Not
"review" itself -- review is produced (armor/models/uncertainty.py) when
the returned set has more than one member.
"""

import numpy as np

CONFORMAL_CLASSES = ["allow", "flag", "block"]


def nonconformity_scores(class_probs: np.ndarray, true_class_idx: np.ndarray) -> np.ndarray:
    """class_probs: (n, len(CONFORMAL_CLASSES)) softmax-style probabilities.
    true_class_idx: (n,) integer index into CONFORMAL_CLASSES.
    Returns 1 - predicted_prob[true_class] per example -- low score means
    the model was confidently correct."""
    return 1.0 - class_probs[np.arange(len(class_probs)), true_class_idx]


def calibrate(class_probs: np.ndarray, true_class_idx: np.ndarray, alpha: float = 0.1) -> float:
    """Returns the nonconformity threshold for (1 - alpha) target coverage,
    e.g. alpha=0.1 -> 90% of true labels fall under the returned threshold
    on the calibration set, by construction. Needs a genuinely held-out
    calibration slice -- calibrating on the same data trained the heads
    would overstate confidence, the same leakage lesson from Phase 2's
    frozen holdout."""
    scores = nonconformity_scores(class_probs, true_class_idx)
    n = len(scores)
    # Finite-sample correction (Vovk): the (1-alpha) quantile of n+1 values
    # (n calibration scores plus a hypothetical worst case), not a naive
    # quantile of just the n observed scores -- the standard correction so
    # coverage holds at finite n, not just asymptotically.
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, q_level, method="higher"))


def predict_set(class_probs: np.ndarray, threshold: float) -> set[str]:
    """class_probs: (len(CONFORMAL_CLASSES),) for one example. Returns the
    set of class names whose nonconformity score (1 - prob) is at or below
    the calibrated threshold -- may contain 0 classes at the boundary in
    rare cases (numerical edge case at threshold=0), handled by the caller
    (armor/models/uncertainty.py) as maximal uncertainty, not an error."""
    return {
        CONFORMAL_CLASSES[i]
        for i, prob in enumerate(class_probs)
        if (1.0 - prob) <= threshold
    }
