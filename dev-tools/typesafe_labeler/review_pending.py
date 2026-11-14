#!/usr/bin/env python3
"""Send existing unresolved prompt-review rows to Jev and store AI labels."""

import argparse
import json
import os
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path

DB_PATH = Path(os.environ.get(
    "ARMOR_LOG_DB_PATH", "/Users/harshilbuch/Armor/armor/observability/logs.db"
))
ARMOR_URL = os.environ.get("ARMOR_URL", "http://127.0.0.1:8001")
JEV_URL = os.environ.get("ARMOR_JEV_REVIEW_URL", "http://127.0.0.1:8762/review")


def post_json(url: str, body: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=50) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="maximum cases; 0 means all")
    args = parser.parse_args()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, content, findings, review_reason FROM logs "
        "WHERE source='inspect' AND direction='prompt' AND decision='review' "
        "AND decision_correction IS NULL ORDER BY id"
    ).fetchall()
    conn.close()
    if args.limit:
        rows = rows[:args.limit]

    labeled = skipped = 0
    for row in rows:
        try:
            verdict = post_json(JEV_URL, {
                "content": row["content"],
                "review_reason": row["review_reason"],
                "findings": json.loads(row["findings"] or "[]"),
            })
            action = verdict.get("action")
            if action not in {"allow", "block"}:
                skipped += 1
                print(f"id={row['id']} skipped: Jev returned no binary verdict", flush=True)
                continue
            post_json(f"{ARMOR_URL}/admin/logs/{row['id']}/correct", {
                "correction": action,
                "target": "decision",
                "label_source": "reviewer_ai_b",
            })
            labeled += 1
            print(
                f"id={row['id']} Jev={action} confidence={verdict.get('confidence', 0):.2f}",
                flush=True,
            )
        except (OSError, urllib.error.URLError, ValueError) as exc:
            skipped += 1
            print(f"id={row['id']} skipped: reviewer/API unavailable ({type(exc).__name__})", flush=True)
    print(f"complete: {labeled} labeled, {skipped} skipped")


if __name__ == "__main__":
    main()
