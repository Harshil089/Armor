"""Small AI-labeled policy reference head over ArmorBase embeddings."""

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = Path(__file__).with_name("checkpoints") / "review_reference_v1.npz"
RUBRIC = ROOT / "docs/policy_rubric_v1_1.md"


@lru_cache(maxsize=1)
def _load():
    data = np.load(CHECKPOINT, allow_pickle=False)
    return {key: data[key] for key in data.files}


def predict_reference_action(pooled, base_fingerprint: str) -> str:
    """Return the highest-scoring v1.1 action; reject stale reference data."""
    data = _load()
    rubric_hash = hashlib.sha256(RUBRIC.read_bytes()).hexdigest()
    if str(data["base_fingerprint"]) != base_fingerprint or str(data["rubric_sha256"]) != rubric_hash:
        raise RuntimeError("review reference checkpoint does not match the base model or rubric")
    vector = pooled.detach().cpu().numpy() if hasattr(pooled, "detach") else np.asarray(pooled)
    vector = vector.astype(np.float64) - data["mean"]
    vector /= max(np.linalg.norm(vector), 1e-8)
    scores = (vector @ data["reference_embeddings"].T) @ data["alpha"]
    return str(data["actions"][scores.argmax()])
