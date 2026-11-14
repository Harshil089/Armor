#!/usr/bin/env python3
"""Replay agreed v18 AI-reviewer cases through Armor in an isolated DB.

This is a development diagnostic, not ground-truth collection. The two AI
reviewers' agreement is stored as reviewer_ai_consensus, which is provisional:
it records a prospective bandit reward but cannot change live bandit weights.
"""

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = ROOT / "docs/broad_gate_v18_queue_20260924.jsonl"
DEFAULT_REVIEWER_A = ROOT / "docs/broad_gate_v18_reviewer_ai_a_labels_20260924.json"
DEFAULT_REVIEWER_B = ROOT / "docs/broad_gate_v18_reviewer_ai_b_labels_20260924.json"
DEFAULT_COMPARISON = ROOT / "docs/broad_gate_v18_comparison_20260924.json"
DEFAULT_OUTPUT = ROOT / "docs/broad_gate_v18_feedback_replay_20260924.json"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--reviewer-a", type=Path, default=DEFAULT_REVIEWER_A)
    parser.add_argument("--reviewer-b", type=Path, default=DEFAULT_REVIEWER_B)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    queue = [json.loads(line) for line in args.queue.read_text(encoding="utf-8").splitlines() if line.strip()]
    reviewer_a = {row["sample_id"]: row["label"] for row in read_json(args.reviewer_a)["labels"]}
    reviewer_b = {row["sample_id"]: row["label"] for row in read_json(args.reviewer_b)["labels"]}
    comparison = {row["sample_id"]: row for row in read_json(args.comparison)["records"]}
    if len({row["sample_id"] for row in queue}) != len(queue):
        raise SystemExit("queue contains duplicate sample IDs")
    if set(reviewer_a) != set(reviewer_b) or not set(reviewer_a).issubset(comparison):
        raise SystemExit("reviewer/comparison sample IDs do not align")

    agreed = [row for row in queue if reviewer_a[row["sample_id"]] == reviewer_b[row["sample_id"]]]
    if not agreed:
        raise SystemExit("there are no reviewer-agreed cases to replay")

    # Select the real production serving configuration before importing app
    # modules. All generated logs stay in a disposable database.
    os.environ["ARMOR_USE_BASE_MODEL"] = "1"
    os.environ["ARMOR_BERT_POLICY_HEAD"] = "1"
    with tempfile.TemporaryDirectory(prefix="armor-v18-feedback-") as temp_dir:
        db_path = Path(temp_dir) / "logs.db"

        from fastapi.testclient import TestClient

        from armor.api.main import app
        from armor.eval.policy_eval import logged_feedback_comparison
        from armor.observability import log_store

        log_store.DB_PATH = db_path
        log_store.init_db(db_path)
        client = TestClient(app)
        outcomes = []

        for sample in agreed:
            sample_id = sample["sample_id"]
            label = reviewer_a[sample_id]
            response = client.post(
                "/v1/inspect", json={"direction": "prompt", "content": sample["content"]}
            )
            if response.status_code != 200:
                raise RuntimeError(f"inspect failed for {sample_id}: {response.status_code} {response.text[:300]}")
            live = response.json()

            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            logged = conn.execute(
                "SELECT * FROM logs WHERE content = ? ORDER BY id DESC LIMIT 1", (sample["content"],)
            ).fetchone()
            conn.close()
            if logged is None:
                raise RuntimeError(f"inspect did not log sample {sample_id}")

            has_bandit = bool(logged["bandit_action"] and logged["bandit_context"])
            target = "both" if has_bandit else "decision"
            corrected = client.post(
                f"/admin/logs/{logged['id']}/correct",
                json={
                    "correction": label,
                    "target": target,
                    "label_source": "reviewer_ai_consensus",
                },
            )
            if corrected.status_code != 200:
                raise RuntimeError(
                    f"feedback failed for {sample_id}: {corrected.status_code} {corrected.text[:300]}"
                )
            feedback = corrected.json()
            if feedback["reward_applied"]:
                raise RuntimeError(f"provisional AI feedback unexpectedly changed bandit weights for {sample_id}")

            stored = log_store.get_entry(db_path, logged["id"])
            live_candidate = comparison[sample_id]["candidate"]["decision"]
            outcomes.append({
                "sample_id": sample_id,
                "reviewer_ai_a": reviewer_a[sample_id],
                "reviewer_ai_b": reviewer_b[sample_id],
                "consensus_label": label,
                "target": target,
                "logged_id": logged["id"],
                "production_decision": live["decision"],
                "policy_action": stored["policy_action"],
                "bandit_action": stored["bandit_action"],
                "model_version": live["model_version"],
                "matches_v18_candidate": live["decision"] == live_candidate,
                "prospective_bandit_reward": feedback["reward"],
                "bandit_reward_applied": feedback["reward_applied"],
                "decision_label_source": stored["decision_label_source"],
                "bandit_label_source": stored["bandit_label_source"],
            })

        metrics = logged_feedback_comparison(db_path, include_provisional=True)
        candidate_matches = sum(row["matches_v18_candidate"] for row in outcomes)
        promoted_hash = read_json(ROOT / "docs/broad_gate_v18_promotion_20260924.json")[
            "promoted_checkpoint_sha256"
        ]
        policy_head_hash = hashlib.sha256(
            (ROOT / "armor/models/checkpoints/policy_head.pt").read_bytes()
        ).hexdigest()
        if policy_head_hash != promoted_hash:
            raise RuntimeError(
                "loaded policy-head checkpoint does not match the promoted v18 artifact: "
                f"{policy_head_hash} != {promoted_hash}"
            )

        report = {
            "version": "v18-ai-consensus-feedback-replay",
            "human_ground_truth": False,
            "label_semantics": "provisional agreement between reviewer_ai_a and reviewer_ai_b; not ground truth",
            "label_source": "reviewer_ai_consensus",
            "reviewer_a": read_json(args.reviewer_a)["reviewer_id"],
            "reviewer_b": read_json(args.reviewer_b)["reviewer_id"],
            "queue_n": len(queue),
            "reviewer_agreement_n": len(agreed),
            "reviewer_agreement_rate": len(agreed) / len(queue) if queue else 0.0,
            "labels_by_action": dict(Counter(row["consensus_label"] for row in outcomes)),
            "replayed_n": len(outcomes),
            "target_both_n": sum(row["target"] == "both" for row in outcomes),
            "decision_only_n": sum(row["target"] == "decision" for row in outcomes),
            "v18_candidate_match_n": candidate_matches,
            "v18_candidate_match_rate": candidate_matches / len(outcomes),
            "base_encoder_checkpoint_sha256": hashlib.sha256(
                (ROOT / "armor/models/checkpoints/head.pt").read_bytes()
            ).hexdigest(),
            "policy_head_checkpoint_sha256": policy_head_hash,
            "promoted_policy_head_sha256": promoted_hash,
            "metrics_on_complete_shared_prediction_set": metrics,
            "records": outcomes,
            "database": "isolated temporary SQLite database; deleted after run",
            "bandit_weights_changed": False,
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "output": str(args.output),
            "reviewer_agreement_n": len(agreed),
            "candidate_match_n": candidate_matches,
            "metrics_n": metrics["n"],
            "excluded_missing_predictions": metrics["excluded_missing_prediction"],
            "target_both_n": report["target_both_n"],
            "decision_only_n": report["decision_only_n"],
            "bandit_weights_changed": False,
        }, indent=2))


if __name__ == "__main__":
    main()
