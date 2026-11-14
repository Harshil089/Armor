"""Freeze diverse logged requests for AI-only v1.1 action gates."""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from armor.eval.audit_sampling import stratified_sample
from armor.models.armor_base import get_armor_base


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--final-gate", action="store_true")
    parser.add_argument("--final-check", action="store_true")
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--version", choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7"])
    args = parser.parse_args()
    version = args.version or ("v5" if args.confirmation else "v4" if args.final_check
                               else "v3" if args.final_gate else "v2" if args.holdout else "v1")
    excluded = set()
    conn = sqlite3.connect(ROOT / "armor/observability/review_audit.db")
    excluded.update(row[0] for row in conn.execute("SELECT sample_id FROM review_audit"))
    conn.close()
    for name in ("regression_action_84", "new_blind_ai_v1_1_holdout", "guided_v1_25"):
        path = ROOT / f"docs/long_text_{name}_candidate_final_20260923.json"
        excluded.update(row["id"] for row in json.loads(path.read_text())["rows"])
    path = ROOT / "docs/ai_only_blind_batch_40_20260923.json"
    excluded.update(row["sample_id"] for row in json.loads(path.read_text())["rows"])
    for path in (ROOT / "docs").glob("long_text*20260923.jsonl"):
        excluded.update(row.get("sample_id") for row in map(json.loads, path.read_text().splitlines()))
    if version != "v1":
        excluded.update(row["sample_id"] for row in map(
            json.loads, (ROOT / "docs/broad_gate_v1_queue_20260923.jsonl").read_text().splitlines()))
    if version in ("v3", "v4", "v5", "v6", "v7"):
        excluded.update(row["sample_id"] for row in map(
            json.loads, (ROOT / "docs/broad_gate_v2_queue_20260923.jsonl").read_text().splitlines()))
    if version in ("v4", "v5", "v6", "v7"):
        excluded.update(row["sample_id"] for row in map(
            json.loads, (ROOT / "docs/broad_gate_v3_queue_20260923.jsonl").read_text().splitlines()))
    if version in ("v5", "v6", "v7"):
        excluded.update(row["sample_id"] for row in map(
            json.loads, (ROOT / "docs/broad_gate_v4_queue_20260923.jsonl").read_text().splitlines()))
    if version in ("v6", "v7"):
        excluded.update(row["sample_id"] for row in map(
            json.loads, (ROOT / "docs/broad_gate_v5_queue_20260923.jsonl").read_text().splitlines()))
    if version == "v7":
        excluded.update(row["sample_id"] for row in map(
            json.loads, (ROOT / "docs/broad_gate_v6_queue_20260923.jsonl").read_text().splitlines()))

    conn = sqlite3.connect(ROOT / "armor/observability/logs.db")
    rows = conn.execute(
        "SELECT content, analyst_correction, COALESCE(label_source, 'pseudo_model') "
        "FROM logs WHERE direction = 'prompt' AND analyst_correction IN "
        "('allow', 'flag', 'block', 'review')"
    ).fetchall()
    conn.close()
    seen = set()
    pool = []
    for content, old_action, source in rows:
        sample_id = hashlib.sha256(content.encode()).hexdigest()[:16]
        if (sample_id in excluded or sample_id in seen or not 20 <= len(content) <= 8000):
            continue
        seen.add(sample_id)
        pool.append((content, old_action, source))
    selected = stratified_sample(
        pool, get_armor_base(), per_class=15 if version != "v1" else 30,
        similarity_threshold=0.98 if version != "v1" else 0.95,
        seed={"v1": 20260924, "v2": 20260925, "v3": 20260926,
              "v4": 20260927, "v5": 20260928, "v6": 20260929,
              "v7": 20260930}[version],
    )
    # The old action is only a selection stratum. Hide it and its source
    # from Jev so no past policy output can influence the fresh label.
    public = [{"sample_id": row["sample_id"], "content": row["content"]}
              for row in selected]
    (ROOT / f"docs/broad_gate_{version}_queue_20260923.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in public))
    (ROOT / f"docs/broad_gate_{version}_sampling_20260923.json").write_text(json.dumps({
        "human_ground_truth": False, "prior_label_used_only_for_sampling": True,
        "excluded_previous_audit_and_gates": True, "pool_size": len(pool),
        "selected": [{key: row[key] for key in ("sample_id", "existing_label", "label_source")}
                     for row in selected],
    }, indent=2))
    print(len(pool), len(selected), ROOT / f"docs/broad_gate_{version}_queue_20260923.jsonl")


if __name__ == "__main__":
    main()
