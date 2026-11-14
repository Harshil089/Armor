"""Replay the frozen v1.1 AI gate through Armor's actual inspect path."""

import json
import argparse
from collections import Counter
from pathlib import Path

from armor.api.main import inspect
from armor.api.schemas import InspectRequest


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7"], default="v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    version = args.version
    queue = ROOT / f"docs/broad_gate_{version}_queue_20260923.jsonl"
    labels = {row["sample_id"]: row for row in json.loads(
        (ROOT / f"docs/broad_gate_{version}_jev_labels_20260923.json").read_text())}
    sampling = {row["sample_id"]: row for row in json.loads(
        (ROOT / f"docs/broad_gate_{version}_sampling_20260923.json").read_text())["selected"]}
    rows = []
    for sample in map(json.loads, queue.read_text().splitlines()):
        sample_id = sample["sample_id"]
        result = inspect(InspectRequest(direction="prompt", content=sample["content"]))
        rows.append({"sample_id": sample_id, "label": labels[sample_id]["label"],
                     "label_confidence": labels[sample_id]["confidence"],
                     "reviewer": labels[sample_id]["reviewer"],
                     "sampling_stratum": sampling[sample_id]["existing_label"],
                     "decision": result.decision, "route": result.route,
                     "risk_score": result.risk_score,
                     "findings": [finding.model_dump() for finding in result.findings],
                     "latency_ms": result.latency_ms})
    path = args.output or ROOT / f"docs/broad_gate_{version}_armor_baseline_20260923.json"
    path.write_text(json.dumps({"rubric_version": "v1.1", "human_ground_truth": False,
                                "rows": rows}, indent=2))
    print(path, len(rows), Counter((row["label"], row["decision"]) for row in rows))


if __name__ == "__main__":
    main()
