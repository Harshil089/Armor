# Feedback targets and reward loop

## Why the log fields are separate

`bandit_action` is a shadow recommendation. `decision` is what Armor actually
returned. A reviewer may correct either one, or provide one canonical action
for both. Treating every correction as a bandit correction made the old
`analyst_correction` field ambiguous and could train the policy head from a
bandit-only judgment.

The admin log page now offers three targets: Armor's production decision, the
bandit recommendation, or the same canonical label for both. The API keeps
`target="bandit"` as its default so older TypeSafe clients continue to work.
The Jev labeler explicitly uses `target="both"` because its prompt judges the
correct action for the content, not just whether the bandit should have chosen
something else.

## Reward and provenance

For bandit feedback, Armor records the prospective reward from the existing
asymmetric reward table. A false positive such as `block` corrected to `allow`
gets a negative reward. Feedback tagged `human` or independently adjudicated
`reviewer_consensus` updates the bandit; `pseudo_model`, `reviewer_ai_a`,
`reviewer_ai_b`, and `reviewer_ai_consensus` feedback records its prospective
reward without changing live bandit weights. AI-only agreement is not human
ground truth; use `reviewer_ai_consensus` to keep that limitation visible.
Bandit update event IDs persist in the checkpoint for retry safety.

The SQLite change is additive. Existing `analyst_correction`, `reward`, and
`label_source` columns remain intact; new explicit fields hold production
corrections, bandit corrections, their provenance, and whether the reward was
applied. A null `bandit_reward_applied` on an old row means unknown legacy
status, not “not applied.” No historical rows were rewritten.

## Fine-tuning and evaluation

Routine `armor.models.train_policy` runs use only valid `human` or
`reviewer_consensus` labels. Single-model labels can be included explicitly
with `--include-pseudo-model` for a development experiment. Bandit-only
corrections are excluded from new policy-head training. Numeric reward values
are never BERT targets; the policy head trains on the labeled action.

Run `python dev-tools/report_feedback_metrics.py` to compare production,
policy-head, and bandit actions on the same explicitly corrected cases. It
defaults to trusted sources and requires all three predictions on each row.
`--include-provisional` adds individual AI and AI-consensus labels for
exploratory analysis; `--db PATH` selects an isolated evaluation database.
`dev-tools/replay_v18_ai_consensus.py` demonstrates this path on the v18
two-reviewer agreement set in a disposable SQLite database. Its report is an
AI-only diagnostic and it does not update the production log database or
bandit weights.
This is a retrospective diagnostic over corrected traffic, not an untouched
promotion holdout; promotion still requires a separate frozen confirmation
gate.

### v18 AI-consensus replay (2026-09-24)

The replay used the 148 of 150 held-out cases on which `reviewer_ai_a` and
`reviewer_ai_b` agreed. Current production decisions matched the promoted v18
candidate on all 148. The bandit emitted a recommendation on 108; 40 were
outside its recommendation range. On the same 108 complete cases, AI-label
agreement was 88.0% for the production decision/policy head and 48.1% for the
shadow bandit. Bandit recall was 35.9% for `flag`, 90.5% for `block`, and 0%
for `review` (19 AI-consensus `review` cases). These are conditional,
AI-only diagnostic metrics, not ground-truth accuracy or promotion evidence.
The full per-case record and confusion matrices are in
`docs/broad_gate_v18_feedback_replay_20260924.json`. The replay verified the
promoted policy-head checksum and left the production logs and live bandit
weights unchanged.

The migration is reversible at the code level: older binaries ignore the new
columns and continue using the retained legacy fields. Keep the new columns
when rolling back; dropping them would discard feedback collected after this
change.
