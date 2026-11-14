"""Composes the policy head + uncertainty gate into one prompt action.
When `ARMOR_BERT_POLICY_HEAD=1`, armor/api/main.py uses this action for
prompt decisions; otherwise it remains a logged recommendation. The
2026-09-24 promotion evidence and remaining limits are in
docs/broad_accuracy_resolution_20260923.md.

The caller in armor/api/main.py preserves Fast Scan and score-threshold
blocks when applying the learned action. The heads may disagree with those
signals; their recommendation must not erase an existing block.
"""

import torch

from armor.models.labels import LABELS
from armor.models.policy_head import (
    POLICY_ACTIONS,
    build_input,
    get_block_head,
    get_intervention_head,
    get_policy_head,
)
from armor.models.uncertainty import (
    hierarchical_decision,
    intervention_block_to_class_probs,
    policy_confidence,
    should_abstain,
)

_HIGH_EVIDENCE_THRESHOLD = 0.5


def _decision_factors(scores: dict[str, float], confidence: float, abstained: bool) -> list[str]:
    factors = [f"high {label} evidence" for label in LABELS if scores.get(label, 0.0) >= _HIGH_EVIDENCE_THRESHOLD]
    if abstained:
        factors.append("policy confidence below review threshold")
    else:
        factors.append("policy confidence above review threshold")
    return factors


def evaluate_hierarchical(
    scores: dict[str, float],
    pooled_embedding: torch.Tensor,
    conformal_threshold: float | None = None,
    ood_reference=None,
    reference_bank=None,
) -> dict:
    """Systems B and C of the A/B/C comparison (queue/QUEUE-02's
    formulation-bottleneck fix), one code path rather than two near-
    duplicates:
    - System B (conformal_threshold=None): intervention_head + block_head,
      simple margin-based decision (no conformal/OOD/similarity).
    - System C (conformal_threshold given): adds conformal abstention,
      OOD, and reference-bank similarity as supplementary evidence.
    Never predicts 'review' directly from either head -- it's always
    produced by the uncertainty layer, per the approved plan."""
    intervention_head = get_intervention_head(embedding_dim=pooled_embedding.shape[-1])
    block_head = get_block_head(embedding_dim=pooled_embedding.shape[-1])
    # The large encoder may run on MPS, while these small policy heads remain
    # on CPU. Moving the pooled vector keeps the decision heads usable without
    # copying their weights or creating a second accelerator model.
    embedding, threat_scores = build_input(pooled_embedding.cpu(), scores)

    with torch.no_grad():
        intervention_prob = float(torch.sigmoid(intervention_head(embedding.unsqueeze(0), threat_scores.unsqueeze(0))))
        block_prob = float(torch.sigmoid(block_head(embedding.unsqueeze(0), threat_scores.unsqueeze(0))))

    class_probs = intervention_block_to_class_probs(intervention_prob, block_prob)

    result = {
        "intervention_probability": intervention_prob,
        "block_probability": block_prob,
    }

    if conformal_threshold is None:
        # System B: simple margin decision, no conformal/OOD/similarity --
        # per the spec's own "Intended policy interpretation": confidently
        # low intervention -> allow; confidently high -> evaluate block
        # head; confidently low block -> flag; confidently high -> block;
        # anything insufficiently confident on either question -> review.
        margin = abs(intervention_prob - 0.5) * 2  # 0 at the boundary, 1 at certainty
        if margin < 0.3:
            action = "review"
        elif intervention_prob < 0.5:
            action = "allow"
        else:
            block_margin = abs(block_prob - 0.5) * 2
            action = "review" if block_margin < 0.3 else ("block" if block_prob >= 0.5 else "flag")
        result["policy_action"] = action
        return result

    # System C: conformal + OOD (+ reference-bank similarity, logged as
    # diagnostic evidence, never overriding the learned decision, per the
    # approved plan's explicit constraint).
    ood_score = ood_reference.score(pooled_embedding) if ood_reference is not None else 0.0
    decision = hierarchical_decision(class_probs, conformal_threshold, ood_score)
    result["conformal_set"] = decision["conformal_set"]
    result["ood_score"] = ood_score
    result["policy_action"] = decision["final_action"]
    if reference_bank is not None:
        result["nearest_reference_similarity"] = reference_bank.nearest_similarity(pooled_embedding)
    return result


def evaluate(scores: dict[str, float], pooled_embedding: torch.Tensor) -> dict:
    """Returns {"policy_action", "policy_confidence", "decision_factors"}.
    Never raises on a well-formed head/embedding -- caller decides how to
    handle any failure the same fail-open way as the bandit/semantic judge
    calls it sits alongside."""
    head = get_policy_head(embedding_dim=pooled_embedding.shape[-1])
    embedding, threat_scores = build_input(pooled_embedding.cpu(), scores)
    with torch.no_grad():
        logits = head(embedding.unsqueeze(0), threat_scores.unsqueeze(0)).squeeze(0)
    confidence = policy_confidence(logits)
    abstained = should_abstain(confidence)
    action = "review" if abstained else POLICY_ACTIONS[int(torch.argmax(logits))]
    return {
        "policy_action": action,
        "policy_confidence": confidence,
        "decision_factors": _decision_factors(scores, confidence, abstained),
    }
