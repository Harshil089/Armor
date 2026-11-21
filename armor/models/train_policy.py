"""Bootstrap trainer for the Phase 1 policy head (queue/QUEUE-02). Separate
from armor/models/train.py deliberately -- armor_base's own 6-label
training is untouched by this. The encoder stays frozen; only the small
PolicyHead trains.

Training data uses explicit production-decision corrections and trusted
legacy corrections. Single-model labels remain available for opt-in
experiments, but do not enter routine retraining by default.
"""

import argparse
import random
import sqlite3

import torch
from torch import nn

from armor.models.armor_base import get_armor_base
from armor.models.calibration import calibrate_scores
from armor.models.policy_head import (
    CHECKPOINT_PATH,
    POLICY_ACTIONS,
    BinaryPolicyHead,
    PolicyHead,
    build_input,
    save_block_head,
    save_intervention_head,
    save_policy_head,
)
from armor.observability.log_store import DB_PATH, init_db

_EXCLUDED_CORRECTIONS = {"contaminated-manual-label", "stale-6dim"}
_VALID_CORRECTIONS = set(POLICY_ACTIONS)


def load_training_rows(
    db_path=DB_PATH, *, include_pseudo_model: bool = False, include_jev_reviews: bool = False
) -> list[tuple[str, str, str]]:
    """Return trusted (content, correction, label_source) rows by default.

    Explicit production-decision feedback takes precedence. The legacy
    correction field remains a provenance-tagged compatibility source.
    """
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT content, COALESCE(decision_correction, analyst_correction) AS correction, "
        "COALESCE(decision_label_source, label_source, 'pseudo_model') AS source FROM logs "
        "WHERE decision_correction IS NOT NULL OR "
        "(analyst_correction IS NOT NULL AND bandit_correction IS NULL)"
    ).fetchall()
    conn.close()
    return [
        (r["content"], r["correction"], r["source"])
        for r in rows
        if r["correction"] in _VALID_CORRECTIONS
        and r["correction"] not in _EXCLUDED_CORRECTIONS
        and (
            include_pseudo_model
            or r["source"] in {"human", "reviewer_consensus"}
            or (include_jev_reviews and r["source"] == "reviewer_ai_b")
        )
    ]


def class_weights(examples) -> torch.Tensor:
    """Inverse-frequency weighting -- block outnumbers flag/review 5-10x in
    every dataset snapshot this project has produced (natural traffic and
    red-teaming both skew toward clear-cut cases). Without this, cross-
    entropy optimizes for the majority class almost for free; adding more
    raw flag/review examples alone plateaued (queue/QUEUE-02 Phase 2's
    second targeted-data round), so the loss itself needs to weight them
    up, not just their count."""
    counts = torch.zeros(len(POLICY_ACTIONS))
    for _, _, label in examples:
        counts[label] += 1
    counts = counts.clamp(min=1)
    weights = counts.sum() / (len(POLICY_ACTIONS) * counts)
    return weights


def train(
    epochs: int = 10, lr: float = 1e-3, seed: int = 0, *,
    include_pseudo_model: bool = False, include_jev_reviews: bool = False,
) -> PolicyHead:
    rows = load_training_rows(
        include_pseudo_model=include_pseudo_model, include_jev_reviews=include_jev_reviews
    )
    if not rows:
        raise ValueError("no trusted policy feedback rows; collect human or reviewer-consensus decision labels first")
    n_human = sum(1 for _, _, src in rows if src == "human")
    n_consensus = sum(1 for _, _, src in rows if src == "reviewer_consensus")
    n_jev = sum(1 for _, _, src in rows if src == "reviewer_ai_b")
    n_provisional = len(rows) - n_human - n_consensus - n_jev
    print(
        f"training rows: {len(rows)} ({n_human} human, {n_consensus} reviewer consensus, "
        f"{n_jev} Jev AI, {n_provisional} other provisional AI)"
    )

    model = get_armor_base()
    fingerprint = getattr(model, "fingerprint", None)
    examples = []
    for content, correction, _ in rows:
        raw_scores, pooled = model.predict_with_embedding(content)
        scores = (
            {label: c["score"] for label, c in calibrate_scores(raw_scores, fingerprint).items()}
            if fingerprint is not None
            else raw_scores
        )
        embedding, threat_scores = build_input(pooled, scores)
        examples.append((embedding.detach(), threat_scores.detach(), POLICY_ACTIONS.index(correction)))

    random.seed(seed)
    random.shuffle(examples)

    head = PolicyHead(embedding_dim=examples[0][0].shape[-1])
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    weights = class_weights(examples)
    print(f"class weights ({POLICY_ACTIONS}): {weights.tolist()}")
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    for epoch in range(epochs):
        total_loss = 0.0
        for embedding, threat_scores, label in examples:
            optimizer.zero_grad()
            logits = head(embedding.unsqueeze(0), threat_scores.unsqueeze(0))
            loss = loss_fn(logits, torch.tensor([label]))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch + 1}/{epochs} loss={total_loss / len(examples):.4f}")

    head.eval()
    return head


# --- Hierarchical training (queue/QUEUE-02's formulation-bottleneck fix) ---


def _hierarchical_labels(correction: str) -> tuple[int, int, bool] | None:
    """Returns (intervention_label, block_label, block_applies) for a
    resolved correction, or None if this row can't supervise either head.
    A logged 'review' correction is Jev's own final call, not an
    intermediate state with a separate resolved_action field (this
    project's schema doesn't track the two separately) -- with no
    resolution to fall back on, these rows are excluded from both heads
    entirely and used instead as abstention/OOD evaluation cases
    (armor/eval/policy_eval.py), per the approved plan's Step 5."""
    if correction == "allow":
        return 0, 0, False  # intervention=no; block label unused, masked out
    if correction == "flag":
        return 1, 0, True
    if correction == "block":
        return 1, 1, True
    return None  # "review" with no resolution


def train_hierarchical(
    epochs: int = 10, lr: float = 1e-3, seed: int = 0, *,
    include_pseudo_model: bool = False, include_jev_reviews: bool = False,
) -> tuple[BinaryPolicyHead, BinaryPolicyHead]:
    rows = load_training_rows(
        include_pseudo_model=include_pseudo_model, include_jev_reviews=include_jev_reviews
    )
    if not rows:
        raise ValueError("no trusted policy feedback rows; collect human or reviewer-consensus decision labels first")
    model = get_armor_base()
    fingerprint = getattr(model, "fingerprint", None)

    examples = []  # (embedding, threat_scores, intervention_label, block_label, block_mask)
    excluded_review = 0
    for content, correction, _ in rows:
        labels = _hierarchical_labels(correction)
        if labels is None:
            excluded_review += 1
            continue
        intervention_label, block_label, block_applies = labels
        raw_scores, pooled = model.predict_with_embedding(content)
        scores = (
            {label: c["score"] for label, c in calibrate_scores(raw_scores, fingerprint).items()}
            if fingerprint is not None
            else raw_scores
        )
        embedding, threat_scores = build_input(pooled, scores)
        examples.append((
            embedding.detach(), threat_scores.detach(),
            float(intervention_label), float(block_label), float(block_applies),
        ))

    print(
        f"hierarchical training rows: {len(examples)} "
        f"({excluded_review} unresolved 'review' rows excluded, not treated as a class)"
    )

    random.seed(seed)
    random.shuffle(examples)

    embedding_dim = examples[0][0].shape[-1]
    intervention_head = BinaryPolicyHead(embedding_dim)
    block_head = BinaryPolicyHead(embedding_dim)
    optimizer = torch.optim.Adam(
        list(intervention_head.parameters()) + list(block_head.parameters()), lr=lr
    )
    intervention_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    block_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    for epoch in range(epochs):
        total_intervention_loss, total_block_loss, n_block_examples = 0.0, 0.0, 0
        for embedding, threat_scores, intervention_label, block_label, block_mask in examples:
            optimizer.zero_grad()

            intervention_logit = intervention_head(embedding.unsqueeze(0), threat_scores.unsqueeze(0))
            intervention_loss = intervention_loss_fn(intervention_logit, torch.tensor([intervention_label]))

            block_logit = block_head(embedding.unsqueeze(0), threat_scores.unsqueeze(0))
            # Masked, per the approved plan: an allow row contributes 0 to
            # the block loss rather than being filtered out of the batch --
            # simpler to get right than reconstructing two different
            # per-example batches from one shared embedding pass.
            block_loss = block_loss_fn(block_logit, torch.tensor([block_label])) * block_mask

            loss = intervention_loss + block_loss
            loss.backward()
            optimizer.step()

            total_intervention_loss += intervention_loss.item()
            if block_mask > 0:
                total_block_loss += block_loss.item()
                n_block_examples += 1

        print(
            f"epoch {epoch + 1}/{epochs} "
            f"intervention_loss={total_intervention_loss / len(examples):.4f} "
            f"block_loss={total_block_loss / max(n_block_examples, 1):.4f} (n={n_block_examples})"
        )

    intervention_head.eval()
    block_head.eval()
    return intervention_head, block_head


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hierarchical", action="store_true", help="train the intervention+block heads instead of the flat 4-class head")
    parser.add_argument("--include-pseudo-model", action="store_true", help="include single-model labels for an explicitly experimental run")
    parser.add_argument("--include-jev-reviews", action="store_true", help="include only Jev's AI review labels as provisional training examples")
    parser.add_argument("--output", default=CHECKPOINT_PATH, help="flat-head checkpoint output path (defaults to the active path)")
    args = parser.parse_args()

    if args.hierarchical:
        if args.output != CHECKPOINT_PATH:
            parser.error("--output applies to the flat head; use the head path environment variables for hierarchical output")
        trained_intervention, trained_block = train_hierarchical(
            epochs=args.epochs, lr=args.lr, include_pseudo_model=args.include_pseudo_model,
            include_jev_reviews=args.include_jev_reviews,
        )
        save_intervention_head(trained_intervention)
        save_block_head(trained_block)
        print("saved intervention + block head checkpoints")
    else:
        trained = train(
            epochs=args.epochs, lr=args.lr, include_pseudo_model=args.include_pseudo_model,
            include_jev_reviews=args.include_jev_reviews,
        )
        save_policy_head(trained, args.output)
        print(f"saved policy head checkpoint to {args.output}")
