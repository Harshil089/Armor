"""Stratified sampler for the 100-example human-label audit (queue/
QUEUE-02's supervision-quality phase). 25 per class, diverse, no near-
duplicates -- reuses the same embedding-cosine-similarity approach already
built for armor/models/reference_bank.py rather than a new dedup method.
"""

import hashlib
import random

import numpy as np


def _sample_id(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def stratified_sample(
    rows: list[tuple[str, str, str]],
    model,
    per_class: int = 25,
    similarity_threshold: float = 0.95,
    seed: int = 0,
) -> list[dict]:
    """rows: (content, correction, label_source) tuples, e.g. from
    armor/models/train_policy.py::load_training_rows(). Returns a list of
    {sample_id, content, existing_label, label_source} dicts, up to
    per_class for each of allow/flag/block/review, diverse (no two
    selected examples within a class have cosine similarity above
    similarity_threshold on their pooled embedding)."""
    random.seed(seed)
    by_class: dict[str, list[tuple[str, str]]] = {"allow": [], "flag": [], "block": [], "review": []}
    for content, correction, label_source in rows:
        if correction in by_class:
            by_class[correction].append((content, label_source))

    selected = []
    for label, candidates in by_class.items():
        random.shuffle(candidates)
        chosen_embeddings = []
        for content, label_source in candidates:
            if len(chosen_embeddings) >= per_class:
                break
            _, pooled = model.predict_with_embedding(content)
            vec = pooled.numpy() if hasattr(pooled, "numpy") else np.asarray(pooled)
            norm = np.linalg.norm(vec)
            if norm == 0:
                continue
            vec = vec / norm
            if chosen_embeddings:
                sims = np.stack(chosen_embeddings) @ vec
                if sims.max() >= similarity_threshold:
                    continue  # too similar to something already selected -- skip, keep diversity
            chosen_embeddings.append(vec)
            selected.append({
                "sample_id": _sample_id(content),
                "content": content,
                "existing_label": label,
                "label_source": label_source,
            })

    return selected
