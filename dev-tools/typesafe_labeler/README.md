# TypeSafe/Jev bandit auto-labeler (dev-only)

**Development only.** TypeSafe remains outside Armor's dependencies and
Armor's `.venv`. This directory contains the batch bandit labeler and a
localhost Jev bridge used only when `ARMOR_JEV_REVIEW_ENABLED=1`.

## What it does

Armor's admin log UI (`/admin/logs-page`) shows a "Correct the bandit's
recommendation" panel with four buttons (allow / flag / block / review)
for any logged request where the bandit made a recommendation. Clicking
one calls `POST /admin/logs/{id}/correct` — that's the only way the
bandit gets a reward signal and learns.

This tool presses those same buttons automatically: it polls Armor's
`logs.db` for rows with a pending bandit recommendation
(`bandit_action IS NOT NULL AND analyst_correction IS NULL`), asks
TypeSafe's Jev model for an independent judgment of the correct action,
and submits it through the exact same public `/admin/logs/{id}/correct`
endpoint a human would use. Armor never knows the difference — it's just
another correction arriving through its normal API.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Needs `/Users/harshilbuch/typesafe/.env` to contain `TYPESAFE_API_KEY`
(quoted or unquoted, both handled).

## Usage

One-shot (labels whatever's currently pending, then exits):

```bash
.venv/bin/python label_pending.py
```

Continuous (polls every `--interval` seconds, labels new pending rows as
they appear, runs until killed):

```bash
.venv/bin/python label_pending.py --watch --interval 30
```

Armor's server must be running (default `http://127.0.0.1:8001`, override
with `ARMOR_URL`).

## Live review adjudication (development only)

`jev_review_server.py` listens on `127.0.0.1:8762`. The proxy sends prompt or
response content whose classifier decision is `review`. Jev returns a
constrained binary action: only explicit `allow` proceeds; `block`, invalid
output, timeout, and service failure block it. For streaming output Armor
buffers each uninspected batch until inspection and any Jev review finish;
previously approved batches may already have reached the client. Each explicit
Jev verdict is stored against its inspect log row with `decision_label_source`
set to `reviewer_ai_b`; it is never tagged as human ground truth. A fallback
block caused by a Jev outage is not recorded as a Jev label.

The local launch agents in `scripts/com.armor.server.plist` and
`scripts/com.armor.jev-review.plist` enable this development path. Armor
does not depend on the TypeSafe SDK; the bridge uses this directory's venv
and reads the API key from `/Users/harshilbuch/typesafe/.env` without
printing it. If the bridge is unavailable, Armor blocks reviewed content.

Jev labels are eligible for an **offline candidate** policy-head run with
`uv run python -m armor.models.train_policy --include-jev-reviews --output /tmp/jev-policy-candidate.pt`.
This includes Jev labels while leaving other provisional AI labels out, and
keeps the active checkpoint untouched. Review the reported training
provenance and run the normal holdout gates before promoting a candidate.
No per-request or automatic live retraining occurs.

To label already-pending prompt reviews once, run
`.venv/bin/python dev-tools/typesafe_labeler/review_pending.py`. New proxy
reviews are adjudicated synchronously when the development bridge is enabled.
