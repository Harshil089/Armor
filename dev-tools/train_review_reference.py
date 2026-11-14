"""Fit the v1.1 review reference from the frozen 80 AI-labeled requests."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from armor.models.armor_base import get_armor_base
from armor.policy.review_reference import CHECKPOINT, RUBRIC


ROOT = Path(__file__).resolve().parents[1]
ACTIONS = np.array(["allow", "flag", "block", "review"])
RIDGE = 0.1


def main():
    torch.set_num_threads(4)
    model = get_armor_base()
    rows = list(map(json.loads, (ROOT / "docs/broad_gate_v1_queue_20260923.jsonl").read_text().splitlines()))
    labels = {row["sample_id"]: row for row in json.loads(
        (ROOT / "docs/broad_gate_v1_jev_labels_20260923.json").read_text())}
    embeddings = []
    for row in rows:
        _, pooled = model.predict_with_embedding(row["content"])
        embeddings.append(pooled.numpy())
    matrix = np.stack(embeddings).astype(np.float64)
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    normalized = centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-8)
    targets = np.stack([
        [labels[row["sample_id"]]["label"] == action for action in ACTIONS]
        for row in rows
    ]).astype(np.float64)
    alpha = np.linalg.solve(normalized @ normalized.T + RIDGE * np.eye(len(rows)), targets)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CHECKPOINT,
        actions=ACTIONS,
        mean=mean,
        reference_embeddings=normalized,
        alpha=alpha,
        base_fingerprint=np.array(model.fingerprint),
        rubric_sha256=np.array(hashlib.sha256(RUBRIC.read_bytes()).hexdigest()),
        reviewer=np.array("reviewer_ai_jev_v1_1"),
        train_ids=np.array([row["sample_id"] for row in rows]),
        ridge=np.array(RIDGE),
    )
    print(CHECKPOINT, len(rows), model.fingerprint)


if __name__ == "__main__":
    main()
