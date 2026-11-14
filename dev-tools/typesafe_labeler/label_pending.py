#!/usr/bin/env python3
"""DEV-ONLY. See README.md — not part of Armor, not a project dependency.

Presses the admin UI's "Correct the bandit's recommendation" buttons
(allow/flag/block/review) automatically via TypeSafe's Jev model, through
the same admin feedback endpoint used by the UI. One content judgment is
stored against both Armor's decision and the band's recommendation.
Single-model labels are provisional: their prospective reward is logged but
does not change online bandit weights until consensus is recorded.

One-shot: labels whatever's currently pending, then exits.
--watch:   polls forever, labeling new pending rows as they appear.
"""

import argparse
import json
import os
import sqlite3
import time
import urllib.request
from pathlib import Path

from typesafe_sdk import Choice, TypeSafeClient
from armor.observability.log_store import init_db

ARMOR_URL = os.environ.get("ARMOR_URL", "http://127.0.0.1:8001")
DB_PATH = Path(
    os.environ.get("ARMOR_LOG_DB_PATH", "/Users/harshilbuch/Armor/armor/observability/logs.db")
)
_TYPESAFE_ENV = Path("/Users/harshilbuch/typesafe/.env")

for line in _TYPESAFE_ENV.read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

_ARMOR_POLICY = """Armor is a content-safety classifier for LLM proxy traffic. It judges a
piece of text (a user prompt or a model response) against six threat labels:
prompt_injection, jailbreak, sensitive_data_exposure, dangerous_tool_action,
harmful_content, obfuscation_encoding. The final action for a piece of
content is one of:
- allow: ordinary, safe content with no real policy concern.
- flag: borderline or mildly concerning; log and let through, don't block.
- block: a genuine violation of one or more labels above; must not pass.
- review: genuinely dual-use content whose safety depends on context not
  visible in the text alone (e.g. legitimate security research framed the
  same way as a real attack) — send to a human rather than auto-decide.
"""


def _post_json(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{ARMOR_URL}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def judge(client: TypeSafeClient, content: str, armor_decision: str, bandit_action: str | None):
    response = client.system_one(
        state={
            "policy": _ARMOR_POLICY,
            "content": content,
            "armor_threshold_decision": armor_decision,
            "armor_bandit_recommendation": bandit_action or "n/a",
        },
        questions={
            "correct_action": Choice(
                instructions=(
                    "Per `policy`, what is the single correct moderation action for `content`? "
                    "Judge `content` on its own merits — armor_threshold_decision and "
                    "armor_bandit_recommendation are shown for context only, not as the answer."
                ),
                criteria={"allow": None, "flag": None, "block": None, "review": None},
            ),
        },
    )
    return response.choices["correct_action"]


def label_one_batch(client: TypeSafeClient) -> int:
    init_db(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, content, decision, bandit_action FROM logs "
        "WHERE bandit_action IS NOT NULL AND analyst_correction IS NULL "
        "AND decision_correction IS NULL AND bandit_correction IS NULL ORDER BY id"
    ).fetchall()
    conn.close()

    for row in rows:
        answer = judge(client, row["content"], row["decision"], row["bandit_action"])
        result = _post_json(
            f"/admin/logs/{row['id']}/correct",
            {"correction": answer.choice, "target": "both", "label_source": "pseudo_model"},
        )
        print(
            f"  id={row['id']} bandit_said={row['bandit_action']:6s} "
            f"jev_says={answer.choice:6s} confidence={answer.confidence:.2f} "
            f"prospective_reward={result['reward']:+.1f} applied={result['reward_applied']}  {row['content'][:60]!r}"
        )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="poll continuously instead of one-shot")
    parser.add_argument("--interval", type=float, default=30.0, help="seconds between polls in --watch mode")
    args = parser.parse_args()

    with TypeSafeClient() as client:
        if not args.watch:
            n = label_one_batch(client)
            print(f"labeled {n} pending rows")
            return

        print(f"watching for pending bandit recommendations every {args.interval}s (ctrl-c to stop)")
        while True:
            n = label_one_batch(client)
            if n:
                print(f"labeled {n} pending rows")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
