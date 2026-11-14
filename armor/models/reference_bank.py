"""Reference-exemplar similarity, queue/QUEUE-02's hierarchical-policy
plan Step 6: supplementary policy evidence, never the policy engine
itself. Per-class (allow/flag/block) embeddings from the training set,
cosine similarity to the nearest one computed at inference. Reuses Phase
1's pooled embedding -- no second encoder pass.

The bank is built from available labeled data, and its provenance stays
visible via
label_source_distribution(), not silently treated as human-quality.
"""

from collections import defaultdict

import numpy as np
import torch

REFERENCE_CLASSES = ["allow", "flag", "block"]


class ReferenceBank:
    def __init__(self):
        self._banks: dict[str, np.ndarray] = {}
        self._label_sources: dict[str, list[str]] = defaultdict(list)

    def fit(self, examples: list[tuple[torch.Tensor, str, str]]) -> None:
        """examples: (embedding, class_name, label_source) tuples. Only
        REFERENCE_CLASSES are stored -- review examples aren't a semantic
        class to hold exemplars for (see uncertainty.py)."""
        by_class: dict[str, list[np.ndarray]] = defaultdict(list)
        for embedding, class_name, label_source in examples:
            if class_name not in REFERENCE_CLASSES:
                continue
            vec = embedding.numpy() if isinstance(embedding, torch.Tensor) else np.asarray(embedding)
            by_class[class_name].append(vec)
            self._label_sources[class_name].append(label_source)
        for class_name, vecs in by_class.items():
            stacked = np.stack(vecs)
            norms = np.linalg.norm(stacked, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._banks[class_name] = stacked / norms

    def label_source_distribution(self) -> dict[str, dict[str, int]]:
        """Provenance visibility -- never silently treat pseudo_model
        exemplars as human-quality."""
        result = {}
        for class_name, sources in self._label_sources.items():
            counts: dict[str, int] = defaultdict(int)
            for s in sources:
                counts[s] += 1
            result[class_name] = dict(counts)
        return result

    def nearest_similarity(self, embedding: torch.Tensor | np.ndarray) -> dict[str, float]:
        """Returns {class_name: max cosine similarity to that class's
        exemplars}, 0.0 for a class with no fitted exemplars rather than
        raising."""
        vec = embedding.numpy() if isinstance(embedding, torch.Tensor) else np.asarray(embedding)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return {c: 0.0 for c in REFERENCE_CLASSES}
        vec = vec / norm
        result = {}
        for class_name in REFERENCE_CLASSES:
            bank = self._banks.get(class_name)
            result[class_name] = float((bank @ vec).max()) if bank is not None and len(bank) else 0.0
        return result
