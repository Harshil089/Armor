"""Policy-head confidence / abstention. Distinct from classifier confidence
(armor_base's own sigmoid scores) -- a detector can be certain a dangerous
capability is present while the policy layer is still uncertain about
authorization or context. See queue/QUEUE-02's Phase 1 entry.

Margin-based, not entropy or an OOD score -- simplest measurable method
first, per the approved plan. A small gap between the top-2 predicted
actions means the policy head sees two genuinely competing outcomes as
plausible, which is exactly the case a hard argmax silently discards.
"""

import os

import torch

REVIEW_CONFIDENCE_FLOOR = float(os.environ.get("ARMOR_POLICY_REVIEW_FLOOR", "0.3"))


def policy_confidence(logits: torch.Tensor) -> float:
    """Softmax margin between the top-1 and top-2 predicted actions, in
    [0, 1]. 1.0 = the head is certain; near 0.0 = two actions are nearly
    tied."""
    probs = torch.softmax(logits, dim=-1)
    top2 = torch.topk(probs, k=min(2, probs.shape[-1]))
    if top2.values.shape[-1] < 2:
        return 1.0
    return float(top2.values[0] - top2.values[1])


def should_abstain(confidence: float, floor: float = REVIEW_CONFIDENCE_FLOOR) -> bool:
    return confidence < floor


# --- Hierarchical combiner (queue/QUEUE-02's formulation-bottleneck fix) ---
#
# review is never predicted directly by a head here -- it's what this
# function returns when the conformal prediction set has more than one
# member, or when the input looks nothing like anything in the training
# set (OOD). Both are genuine "the system doesn't know," not a fifth
# semantic class.

OOD_REVIEW_THRESHOLD = float(os.environ.get("ARMOR_OOD_REVIEW_THRESHOLD", "0.6"))


def intervention_block_to_class_probs(intervention_prob: float, block_prob: float) -> "list[float]":
    """Maps the two hierarchical head outputs onto a valid 3-way
    distribution over conformal.CONFORMAL_CLASSES = [allow, flag, block],
    so the existing conformal machinery (built for an ordinary softmax)
    works unchanged on a hierarchical head pair:
      P(allow) = 1 - intervention_prob
      P(flag)  = intervention_prob * (1 - block_prob)
      P(block) = intervention_prob * block_prob
    These three terms sum to 1 by construction."""
    allow_prob = 1.0 - intervention_prob
    flag_prob = intervention_prob * (1.0 - block_prob)
    block_only_prob = intervention_prob * block_prob
    return [allow_prob, flag_prob, block_only_prob]


def hierarchical_decision(
    class_probs: "list[float]",
    conformal_threshold: float,
    ood_score: float,
    ood_threshold: float = OOD_REVIEW_THRESHOLD,
) -> dict:
    """Combines the conformal prediction set with the OOD signal into one
    final shadow-mode action. Never silently collapses uncertainty to the
    more permissive action (allow) -- an empty or multi-member conformal
    set, or a high OOD score, both resolve to 'review', regardless of
    which class happens to have the highest raw probability."""
    from armor.models.conformal import predict_set

    conformal_set = predict_set(class_probs, conformal_threshold)
    if ood_score >= ood_threshold:
        # Force ambiguity for genuinely novel input -- an OOD sample
        # shouldn't be trusted just because the heads produced a
        # confident-looking (but unvalidated-for-this-input) probability.
        final_action = "review"
    elif len(conformal_set) == 1:
        final_action = next(iter(conformal_set))
    else:
        final_action = "review"
    return {
        "conformal_set": sorted(conformal_set),
        "final_action": final_action,
    }
