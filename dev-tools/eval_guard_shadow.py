"""Compare a local safety guard against frozen Armor action slices.

Development-only: calls a local Ollama model and never changes serving.
"""

import argparse
import hashlib
import json
import re
import sqlite3
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = "sileader/qwen3guard:0.6b"
SAFETY = re.compile(r"Safety:\s*(Safe|Controversial|Unsafe)", re.I)
CATEGORIES = re.compile(r"Categories:\s*([^\n]+)", re.I)


def cases():
    conn = sqlite3.connect(ROOT / "armor/observability/logs.db")
    lookup = {
        hashlib.sha256(content.encode()).hexdigest()[:16]: content
        for (content,) in conn.execute("SELECT DISTINCT content FROM logs")
    }
    conn.close()
    for name in ("regression_action_84", "new_blind_ai_v1_1_holdout", "guided_v1_25"):
        path = ROOT / f"docs/long_text_{name}_candidate_final_20260923.json"
        for row in json.loads(path.read_text())["rows"]:
            content = lookup.get(row["id"])
            if content is None:
                continue
            yield {"id": row["id"], "slice": name, "label": row["label"],
                   "baseline": row["decision"], "content": content}
    for version in ("v6", "v7"):
        labels = {row["sample_id"]: row for row in json.loads(
            (ROOT / f"docs/long_text_gate_{version}_ai_labels_20260923.json").read_text())}
        outcomes = {row["sample_id"]: row for row in json.loads(
            (ROOT / f"docs/long_text_gate_{version}_{'verified' if version == 'v6' else 'candidate'}_20260923.json").read_text())["rows"]}
        path = ROOT / f"docs/long_text_gate_{version}_final_20260923.jsonl"
        for row in map(json.loads, path.read_text().splitlines()):
            sample_id = row["sample_id"]
            yield {"id": sample_id, "slice": f"long_{version}_{row['variant']}",
                   "label": labels[sample_id]["label"],
                   "label_confidence": labels[sample_id]["confidence"],
                   "baseline": outcomes[sample_id]["decision"], "content": row["content"]}
    for version, slice_name in (("v1", "fresh_broad_v1_1"),
                                ("v2", "fresh_broad_v1_1_holdout"),
                                ("v3", "fresh_broad_v1_1_v3"),
                                ("v4", "fresh_broad_v1_1_v4")):
        labels = {row["sample_id"]: row for row in json.loads(
            (ROOT / f"docs/broad_gate_{version}_jev_labels_20260923.json").read_text())}
        outcomes = {row["sample_id"]: row for row in json.loads(
            (ROOT / f"docs/broad_gate_{version}_armor_baseline_20260923.json").read_text())["rows"]}
        for row in map(json.loads, (ROOT / f"docs/broad_gate_{version}_queue_20260923.jsonl").read_text().splitlines()):
            sample_id = row["sample_id"]
            yield {"id": sample_id, "slice": slice_name,
                   "label": labels[sample_id]["label"],
                   "label_confidence": labels[sample_id]["confidence"],
                   "baseline": outcomes[sample_id]["decision"], "content": row["content"]}


def classify(content, model):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": content}],
                       "stream": False, "options": {"temperature": 0, "num_predict": 80}}).encode()
    request = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    output = payload["message"]["content"]
    safety = SAFETY.search(output)
    categories = CATEGORIES.search(output)
    return {"safety": safety.group(1).lower() if safety else None,
            "categories": categories.group(1).strip() if categories else None,
            "raw": output[:300], "duration_ms": round(payload.get("total_duration", 0) / 1e6, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--slice", default="all")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/broad_guard_shadow_20260923.json")
    args = parser.parse_args()
    completed = {row["id"]: row for row in json.loads(args.output.read_text())["rows"]} if args.output.exists() else {}
    selected = [row for row in cases() if args.slice == "all" or row["slice"] == args.slice]
    if args.limit:
        selected = selected[:args.limit]
    for index, case in enumerate(selected, 1):
        if case["id"] in completed:
            continue
        result = classify(case["content"], args.model)
        completed[case["id"]] = {key: value for key, value in case.items() if key != "content"} | result
        args.output.write_text(json.dumps({"model": args.model, "human_ground_truth": False,
                                           "rows": list(completed.values())}, indent=2))
        print(index, case["slice"], case["label"], case["baseline"],
              result["safety"], result["categories"], result["duration_ms"], flush=True)


if __name__ == "__main__":
    main()
