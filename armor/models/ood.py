"""Lightweight out-of-distribution signal, reusing Phase 1's ModernBERT
pooled embedding -- no second encoder pass. Nearest-neighbor cosine
distance to the training set's own embeddings, per queue/QUEUE-02's
hierarchical-policy plan: the simplest measurable OOD method, only add
capacity (e.g. Mahalanobis) if evaluation shows this underfits.

Purpose is narrow, per the approved plan: known-distribution input ->
normal policy path; far-from-training-distribution input -> raise
uncertainty, not directly choose allow or block. Never a policy engine on
its own.
"""

import numpy as np
import torch


class OODReference:
    def __init__(self):
        self._embeddings: np.ndarray | None = None

    def fit(self, embeddings: list[torch.Tensor] | np.ndarray) -> None:
        stacked = np.stack([e.numpy() if isinstance(e, torch.Tensor) else e for e in embeddings])
        # L2-normalize once at fit time so score() is a plain dot product,
        # not a normalize-per-query cosine computation on every call.
        norms = np.linalg.norm(stacked, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._embeddings = stacked / norms

    def score(self, embedding: torch.Tensor | np.ndarray) -> float:
        """Returns cosine DISTANCE (1 - similarity) to the nearest fitted
        embedding -- 0.0 means "identical to something in training data",
        higher means more novel. Returns 1.0 (maximally OOD) if fit() was
        never called, rather than raising -- fail toward more uncertainty,
        not a crash."""
        if self._embeddings is None or len(self._embeddings) == 0:
            return 1.0
        vec = embedding.numpy() if isinstance(embedding, torch.Tensor) else np.asarray(embedding)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return 1.0
        vec = vec / norm
        similarities = self._embeddings @ vec
        return float(1.0 - similarities.max())
