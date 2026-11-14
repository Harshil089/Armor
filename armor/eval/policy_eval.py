"""Formalizes queue/QUEUE-02 Phase 2's evaluation methodology (previously
scratchpad-only) into a reusable module -- the A/B/C comparison (flat
4-class head vs. hierarchical heads vs. hierarchical+conformal+OOD) needs
the exact same frozen holdout and ambiguity subset for a fair comparison,
and Step 9's metric suite needs to be reproducible, not one-off.

Two holdout constructions, both proven in Phase 2:
- frozen_holdout(): regression_set.py cases that leaked into the training
  pool (sent through /v1/inspect during round-1 red-teaming) are excluded
  from training and used purely as test data -- independently authored,
  not a random split of the same generation process.
- ambiguity_subset(): pairs from logs.db where the OLD 7-dim context (6
  threat scores + framing_signal) was near-identical (Euclidean distance,
  not cosine -- cosine ignores magnitude and produced nonsense matches,
  see the module docstring history in queue/QUEUE-02) but the correct
  action differed. All-6-scores-exactly-0.0 rows are excluded first -- a
  scores-dict artifact, not genuine "confidently benign" output.
"""

import json
import sqlite3
from collections import Counter, defaultdict

import numpy as np

from armor.eval.regression_set import REGRESSION_CASES
from armor.models.policy_head import POLICY_ACTIONS
from armor.models.train_policy import _EXCLUDED_CORRECTIONS, load_training_rows
from armor.observability.log_store import DB_PATH

AMBIGUITY_DIST_THRESHOLD = 0.15


def frozen_holdout(db_path=DB_PATH) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Returns (train_rows, holdout_rows), both (content, correction,
    label_source) tuples, from load_training_rows()."""
    regression_texts = {text.strip() for text, _, _ in REGRESSION_CASES}
    # Preserve historical evaluation semantics; routine training defaults to
    # trusted labels only, but these named legacy slices were defined over
    # the original mixed-provenance correction pool.
    rows = load_training_rows(db_path, include_pseudo_model=True)
    holdout = [(c, corr, src) for c, corr, src in rows if c.strip() in regression_texts]
    train = [(c, corr, src) for c, corr, src in rows if c.strip() not in regression_texts]
    return train, holdout


def _rows_with_old_context(db_path=DB_PATH) -> list[tuple[str, np.ndarray, str]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT content, bandit_context, COALESCE(decision_correction, analyst_correction) AS correction FROM logs "
        "WHERE bandit_context IS NOT NULL AND "
        "(decision_correction IS NOT NULL OR (analyst_correction IS NOT NULL AND bandit_correction IS NULL))"
    ).fetchall()
    conn.close()

    result = []
    for r in rows:
        if r["correction"] in _EXCLUDED_CORRECTIONS:
            continue
        ctx = json.loads(r["bandit_context"])
        if len(ctx) < 7:
            continue
        old_vec = np.array(ctx[:7])
        if np.all(old_vec[:6] == 0.0):
            continue  # scores-dict artifact, not genuine confident-benign output
        result.append((r["content"], old_vec, r["correction"]))
    return result


def ambiguity_subset(db_path=DB_PATH) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Returns (train_rows, holdout_rows). The holdout is every unique text
    involved in a near-identical-but-different-label pair (Euclidean
    distance < AMBIGUITY_DIST_THRESHOLD on the old 7-dim context), excluded
    entirely from training -- these are the specific cases the
    representation fix exists to address, tested as genuine holdout, not
    training data."""
    data = _rows_with_old_context(db_path)
    text_to_corrections: dict[str, set] = defaultdict(set)
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            c1, v1, corr1 = data[i]
            c2, v2, corr2 = data[j]
            if corr1 == corr2:
                continue
            if np.linalg.norm(v1 - v2) < AMBIGUITY_DIST_THRESHOLD:
                text_to_corrections[c1].add(corr1)
                text_to_corrections[c2].add(corr2)

    # A text appearing with two different corrections across duplicate log
    # rows is itself noise, excluded rather than guessed at.
    ambiguity_texts = {t: next(iter(corrs)) for t, corrs in text_to_corrections.items() if len(corrs) == 1}

    rows = load_training_rows(db_path, include_pseudo_model=True)
    train = [(c, corr, src) for c, corr, src in rows if c.strip() not in ambiguity_texts]
    holdout = [(t, corr, "ambiguity_subset") for t, corr in ambiguity_texts.items()]
    return train, holdout


def confusion_matrix(preds: list[str], trues: list[str], labels: list[str]) -> dict[str, dict[str, int]]:
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for pred, true in zip(preds, trues):
        if true in matrix and pred in matrix[true]:
            matrix[true][pred] += 1
    return matrix


def policy_metrics(preds: list[str], trues: list[str], labels: list[str] = None) -> dict:
    """Step 9's metric suite: confusion matrix, per-class precision/
    recall/F1, macro F1, plus the safety/rare-state breakdowns called out
    specifically (block recall/precision/FN count, flag recall/precision
    and its two misroute directions)."""
    labels = labels or sorted(set(trues) | set(preds))
    matrix = confusion_matrix(preds, trues, labels)

    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        fn = sum(matrix[label][p] for p in labels if p != label)
        fp = sum(matrix[t][label] for t in labels if t != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    macro_f1 = sum(c["f1"] for c in per_class.values()) / len(per_class) if per_class else 0.0
    overall_acc = sum(1 for p, t in zip(preds, trues) if p == t) / len(trues) if trues else 0.0

    result = {
        "confusion_matrix": matrix,
        "per_class": per_class,
        "macro_f1": macro_f1,
        "accuracy": overall_acc,
    }

    if "block" in labels:
        result["block_false_negative_count"] = sum(
            matrix["block"][p] for p in labels if p != "block"
        )
    if "flag" in labels:
        result["flag_to_allow_errors"] = matrix["flag"].get("allow", 0)
        result["flag_to_block_errors"] = matrix["flag"].get("block", 0)
    if "review" in labels:
        review_predicted = sum(matrix[t].get("review", 0) for t in labels)
        review_predicted_correctly = matrix["review"]["review"] if "review" in matrix else 0
        result["review_rate"] = review_predicted / len(preds) if preds else 0.0
        result["review_precision"] = (
            review_predicted_correctly / review_predicted if review_predicted else 0.0
        )
    return result


def logged_feedback_comparison(db_path=DB_PATH, *, include_provisional: bool = False) -> dict:
    """Compare stored production, policy-head, and bandit actions on one set.

    Only explicit production-decision corrections qualify as labels. By
    default, only human and reviewer-consensus sources count; provisional AI
    labels are reported separately unless explicitly requested.
    """
    # Keep this utility usable against pre-feedback databases as well as the
    # running service: init_db applies the additive SQLite column migration.
    from armor.observability.log_store import init_db

    init_db(db_path)
    valid = set(POLICY_ACTIONS)
    trusted_sources = {"human", "reviewer_consensus"}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT decision, policy_action, bandit_action, decision_correction, decision_label_source "
        "FROM logs WHERE decision_correction IS NOT NULL"
    ).fetchall()
    conn.close()

    eligible = []
    provisional = 0
    missing_prediction = 0
    invalid_label = 0
    for row in rows:
        label = row["decision_correction"]
        source = row["decision_label_source"]
        if label not in valid:
            invalid_label += 1
            continue
        if source not in trusted_sources:
            provisional += 1
            if not include_provisional:
                continue
        if any(row[key] not in valid for key in ("decision", "policy_action", "bandit_action")):
            missing_prediction += 1
            continue
        eligible.append(row)

    labels = sorted(valid)
    expected = [row["decision_correction"] for row in eligible]
    return {
        "n": len(eligible),
        "label_sources": dict(Counter(row["decision_label_source"] for row in eligible)),
        "excluded_provisional": provisional if not include_provisional else 0,
        "excluded_missing_prediction": missing_prediction,
        "excluded_invalid_label": invalid_label,
        "production_decision": policy_metrics([row["decision"] for row in eligible], expected, labels),
        "policy_head": policy_metrics([row["policy_action"] for row in eligible], expected, labels),
        "bandit": policy_metrics([row["bandit_action"] for row in eligible], expected, labels),
    }


def coverage_vs_accuracy(
    preds: list[str], trues: list[str], confidences: list[float], coverage_levels=(1.0, 0.95, 0.9, 0.8)
) -> dict[float, dict]:
    """Step 9's coverage-vs-accuracy table: at each coverage level, keep
    only the top-N-by-confidence predictions (the rest treated as if they
    were routed to REVIEW instead) and report accuracy on what's left.
    More informative than review recall alone -- shows whether abstaining
    on the least-confident slice concentrates the errors there."""
    order = sorted(range(len(preds)), key=lambda i: -confidences[i])
    n = len(preds)
    result = {}
    for level in coverage_levels:
        keep = order[: int(n * level)]
        if not keep:
            result[level] = {"accuracy": None, "n": 0}
            continue
        correct = sum(1 for i in keep if preds[i] == trues[i])
        result[level] = {"accuracy": correct / len(keep), "n": len(keep)}
    return result
