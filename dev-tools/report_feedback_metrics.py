#!/usr/bin/env python3
"""Compare production, policy-head, and bandit actions on vetted corrections."""

import argparse
import json
from pathlib import Path

from armor.eval.policy_eval import logged_feedback_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-provisional", action="store_true", help="include provisional single-reviewer and AI-consensus labels")
    parser.add_argument("--output", help="optional JSON output path")
    parser.add_argument("--db", help="SQLite log database to evaluate (default: Armor's local logs.db)")
    args = parser.parse_args()
    kwargs = {"include_provisional": args.include_provisional}
    if args.db:
        kwargs["db_path"] = Path(args.db)
    result = logged_feedback_comparison(**kwargs)
    rendered = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
