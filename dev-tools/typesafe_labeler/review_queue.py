#!/usr/bin/env python3
"""Label a supplied policy queue with Jev without writing to Armor's DB."""
import argparse
import json
import os
from pathlib import Path

from typesafe_sdk import Choice, TypeSafeClient

V11 = Path("docs/policy_rubric_v1_1.md").read_text()
ENV_PATH = Path("/Users/harshilbuch/typesafe/.env")
for line in ENV_PATH.read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("queue")
    parser.add_argument("output")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rubric-addendum", default="docs/policy_rubric_v1_2_proposal.md")
    parser.add_argument("--rubric-version", default="v1.2-proposal-full")
    args = parser.parse_args()
    addendum = Path(args.rubric_addendum).read_text()
    policy = V11 + "\n\n" + addendum.split("## Validation status", 1)[0]
    rows = [json.loads(line) for line in Path(args.queue).read_text().splitlines() if line.strip()]
    rows = rows[args.start: args.start + args.limit if args.limit else None]
    results = []
    with TypeSafeClient() as client:
        for i, row in enumerate(rows, start=args.start):
            answer = client.system_one(
                state={"rubric": policy, "content": row["content"]},
                questions={"action": Choice(
                    instructions=("Apply the rubric to the content as written. "
                                  "Return exactly the single correct Armor action."),
                    criteria={"allow": None, "flag": None, "block": None, "review": None},
                )},
            ).choices["action"]
            results.append({"sample_id": row["sample_id"], "label": answer.choice,
                "confidence": answer.confidence, "reviewer": "Jev via TypeSafe",
                "reviewer_id": "reviewer_ai_b",
                "rubric_version": args.rubric_version})
            if (i + 1) % 10 == 0:
                Path(args.output).write_text(json.dumps({
                    "reviewer": "Jev via TypeSafe", "reviewer_id": "reviewer_ai_b",
                    "rubric_version": args.rubric_version, "human_ground_truth": False,
                    "labels": results,
                }, indent=2) + "\n")
                print(f"labeled {i + 1} rows", flush=True)
    Path(args.output).write_text(json.dumps({
        "reviewer": "Jev via TypeSafe", "reviewer_id": "reviewer_ai_b",
        "rubric_version": args.rubric_version, "human_ground_truth": False,
        "labels": results,
    }, indent=2) + "\n")
    print(f"complete: {len(results)} labels", flush=True)


if __name__ == "__main__":
    main()
