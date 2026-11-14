"""Replay a frozen AI-reviewed long-input gate through the real inspect path."""

import argparse
import json
from collections import Counter
from pathlib import Path

from armor.api.main import inspect
from armor.api.schemas import InspectRequest


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gate", choices=["development", "holdout", "final"])
    parser.add_argument("--version", choices=["v4", "v5", "v6", "v7"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    version = (args.version or "v4") if args.gate == "final" else "v3"
    gate_path = ROOT / f"docs/long_text_gate_{version}_{args.gate}_20260923.jsonl"
    labels_path = ROOT / f"docs/long_text_gate_{version}_ai_labels_20260923.json"
    output = args.output or ROOT / f"docs/long_text_gate_{version}_{args.gate}_serving_20260923.json"
    labels = {row["sample_id"]: row for row in json.loads(labels_path.read_text())}
    rows = []
    for sample in map(json.loads, gate_path.read_text().splitlines()):
        label = labels[sample["sample_id"]]
        result = inspect(InspectRequest(direction="prompt", content=sample["content"]))
        rows.append({
            "sample_id": sample["sample_id"],
            "variant": sample["variant"],
            "base_id": sample["base_id"],
            "tokens": sample["tokens"],
            "reviewer_label": label["label"],
            "reviewer_confidence": label["confidence"],
            "reviewer": label["reviewer"],
            "decision": result.decision,
            "route": result.route,
            "finding_labels": [finding.label for finding in result.findings],
            "latency_ms": round(result.latency_ms, 2),
        })
    output.write_text(json.dumps({
        "gate": args.gate,
        "human_ground_truth": False,
        "rubric_version": "v1.1",
        "rows": rows,
    }, indent=2))
    print(output)
    for variant in sorted({row["variant"] for row in rows}):
        subset = [row for row in rows if row["variant"] == variant]
        print(variant, len(subset), dict(Counter((row["reviewer_label"], row["decision"]) for row in subset)))


if __name__ == "__main__":
    main()
