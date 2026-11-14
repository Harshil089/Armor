"""Active-learning candidate miner (queue/QUEUE-02's supervision-quality
phase, Phase 4). Its job is NOT to classify requests -- it's "which
example should a reviewer label next." Reuses everything from Phase 1 and
the hierarchical-head experiment: armor_base's embeddings, the policy
heads (boundary margin), ReferenceBank (rare-class similarity), OODReference
(novelty) -- no new model or embedding infrastructure, per the spec.
"""

from collections import defaultdict

import numpy as np
import torch

from armor.models.ood import OODReference
from armor.models.reference_bank import ReferenceBank


def boundary_margin_score(class_probs: list[float]) -> float:
    """Low score = the top-2 predicted classes are close together, i.e. a
    genuine boundary case worth a reviewer's attention. 0.0 = perfect tie,
    1.0 = one class totally dominant."""
    sorted_probs = sorted(class_probs, reverse=True)
    if len(sorted_probs) < 2:
        return 1.0
    return sorted_probs[0] - sorted_probs[1]


def disagreement_score(policy_prediction: str, reference_bank: ReferenceBank, embedding) -> float:
    """High score = the policy head's prediction disagrees with which
    reference-bank class this embedding is actually nearest to -- a
    concrete, checkable form of "model disagreement" from the spec's
    Phase 4 source #2, using infrastructure already built rather than
    wiring up a second live judge."""
    similarities = reference_bank.nearest_similarity(embedding)
    if not similarities or all(v == 0.0 for v in similarities.values()):
        return 0.0
    nearest_class = max(similarities, key=similarities.get)
    return 1.0 if nearest_class != policy_prediction else 0.0


def neighbor_to_rare_class(embedding, reference_bank: ReferenceBank, rare_classes=("flag",)) -> float:
    """High score = this embedding sits close to known flag/review
    exemplars -- real traffic that resembles a rare policy region, per the
    spec's Phase 4 source #3."""
    similarities = reference_bank.nearest_similarity(embedding)
    if not similarities:
        return 0.0
    return max((similarities.get(c, 0.0) for c in rare_classes), default=0.0)


def novelty_score(embedding, ood_reference: OODReference) -> float:
    """Wraps OODReference -- high score = far from anything already
    labeled, per the spec's Phase 4 source #5. Novelty alone must not
    determine policy decisions (existing OODReference contract already
    enforces this: it only ever feeds uncertainty, never a decision)."""
    return ood_reference.score(embedding)


def cluster_for_diversity(embeddings: list[torch.Tensor], n_clusters: int) -> list[int]:
    """Assigns each embedding to one of n_clusters groups via a minimal
    k-means (pure numpy -- no new dependency), so candidate selection can
    take one/few representatives per cluster instead of near-duplicates.
    Deliberately simple: a fixed small number of Lloyd iterations, not a
    convergence-tracked implementation -- this is for sampling diversity,
    not a production clustering system.
    ponytail: fixed iteration count (10), no convergence check or
    k-means++ init -- fine at the audit's scale (dozens to low hundreds of
    candidates); revisit if this needs to run on thousands of embeddings.
    """
    if not embeddings:
        return []
    stacked = np.stack([e.numpy() if isinstance(e, torch.Tensor) else np.asarray(e) for e in embeddings])
    n_clusters = min(n_clusters, len(stacked))
    rng = np.random.default_rng(0)
    centroids = stacked[rng.choice(len(stacked), n_clusters, replace=False)]

    assignments = np.zeros(len(stacked), dtype=int)
    for _ in range(10):
        dists = np.linalg.norm(stacked[:, None, :] - centroids[None, :, :], axis=2)
        assignments = dists.argmin(axis=1)
        for k in range(n_clusters):
            members = stacked[assignments == k]
            if len(members):
                centroids[k] = members.mean(axis=0)
    return assignments.tolist()


def select_diverse_candidates(candidates: list[dict], embeddings: list, n_clusters: int, per_cluster: int = 1) -> list[dict]:
    """candidates: list of dicts (must align 1:1 with embeddings), each
    already carrying a priority score under some key the caller sorts by
    before calling this -- diversity sampling picks the top `per_cluster`
    candidates *within* each cluster, not a global top-N, so one dominant
    cluster can't crowd out everything else."""
    clusters = cluster_for_diversity(embeddings, n_clusters)
    by_cluster = defaultdict(list)
    for candidate, cluster_id in zip(candidates, clusters):
        by_cluster[cluster_id].append(candidate)
    selected = []
    for members in by_cluster.values():
        selected.extend(members[:per_cluster])
    return selected
