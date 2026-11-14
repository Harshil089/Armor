# AI reviewer audit: adjudication readout (2026-09-23)

## Recommendation

Keep production policy unchanged. The audit shows substantial boundary disagreement between two AI reviewers, but the guided adjudications are not a blind gold dataset for retraining. A self-contained v1.1 labeling rubric captures the user's subsequent boundary choices for **new** reviews; v1 audit rows remain unchanged. A fresh AI subagent has now reviewed all 18 originally uncovered requests under v1.1, providing triage labels rather than human ground truth.

## Coverage and agreement

The unit is one sampled request in `armor/observability/review_audit.db`. Sampling was stratified by the **existing** action with near-duplicate exclusion; these shares are not deployment prevalence. All 85 rows have a unique `sample_id` and unique content, and all carry rubric version `v1`.

| Existing action | Sampled | Both AI reviewers covered | Refusal-excluded | AI disagreements among covered |
|---|---:|---:|---:|---:|
| allow | 25 | 25 | 0 | 5 |
| block | 25 | 24 | 1 | 12 |
| flag | 19 | 10 | 9 | 4 |
| review | 16 | 8 | 8 | 4 |
| **Total** | **85** | **67** | **18** | **25** |

Reviewer A and Jev agree on 42/67 covered requests (62.7%; Cohen's κ = 0.497). Agreement among requests where either reviewer chose the named action is 19/27 for `allow` (70%), 8/16 for `block` (50%), 9/29 for `flag` (31%), and 6/20 for `review` (30%). The last measure groups by *reviewer-chosen* action; the coverage table groups by *existing* action.

The missing AI decisions are concentrated in the scarce classes: 17/18 have existing action `flag` or `review`. Thus only 18/35 of those sampled requests have two AI decisions, versus 49/50 existing `allow`/`block`. The database records neither AI decision nor a reason code for these 18 rows; the refusal account comes from `queue/QUEUE-02_decision_bandit_shadow_mode.md`. The missing rows form class-grouped ID runs (26–34 and 69–77), so these counts cannot separate per-item content effects from batch effects. They establish a coverage gap, not a class-specific refusal probability.

## Human adjudication of the 25 disagreements

All 25 choices were recorded separately from the AI decisions. The final distribution is 5 `allow`, 6 `flag`, 6 `block`, and 8 `review`. The adjudicated choice matches the existing action in 16/25 cases, Reviewer A in 11/25, and Jev in 14/25. Among the 12 disputed rows with an existing `block` action, six remained `block`, four became `review`, and two became `flag`.

Cases 2 and 3 were initially shown as shortened excerpts. After seeing the full wallet and fitness-location requests, the user changed both from `flag` to `allow`. The other 23 cases were presented with their complete text or the relevant decoded instruction. Case 6's initial shortened prompt was superseded by a full-text prompt, and the user confirmed `flag` against that version.

These 25 rows were selected because the AI reviewers disagreed. The other 42 dual-covered rows have not been human adjudicated. Each adjudication prompt offered the two disagreeing AI labels and a suggested reason. This made the review quick but could steer the answer and prevented an independent four-action choice. These counts therefore cannot estimate overall label error, establish either AI reviewer as more accurate, or serve as a blind human gold set. The pre-audit `existing_label` has mixed provenance: 80 `pseudo_model` and five `human` rows across the 85; none of the 25 disputes has a prior `human` source.

## Supplemental blind AI review under v1.1

At the user's request, one fresh subagent with no conversation context read only the v1.1 rubric and a shuffled queue containing `sample_id` and full request content. It received no existing labels, previous AI decisions, or audit findings. Its output is stored in `docs/audit_ai_blind_v1_1_20260923.json` with `reviewer_ai_c_no_context` provenance and `human_ground_truth: false`.

The subagent returned an action for all 18 previously uncovered requests: 11 `flag`, four `review`, and three `block`. This removes the operational AI-coverage gap for this separate pass; it does not establish why the earlier batches refused or whether the new labels are accurate. Eleven match the existing v1 label and seven differ. The seven differences comprise four existing `review` → `flag`, one existing `flag` → `review`, one existing `review` → `block`, and one existing `flag` → `block`. These are **cross-version comparisons** between a v1 existing label and a v1.1 AI judgment. They are useful candidates for further review, not an error rate or proof that either label is correct. The original 67-sample A/B agreement and the v1 audit rows are unchanged. The subagent's confidence numbers are self-reported and uncalibrated.

## Rubric boundaries exposed

- **Policy-bypass instructions.** Cases 10, 12, and 13 were adjudicated `block` for explicit instructions to remove model restrictions or safety policies without a downstream harmful task. Case 6, a persona and response-format override, was `flag`; Case 22, fake command syntax without a concrete command, was `allow`. The user subsequently confirmed that explicit safety-control bypass is independently `block`; that rule is recorded in v1.1 and has not been applied to v1 rows.
- **Encoded actions.** The covert shipment reroute and production API-key request were `block`; concealing a support-ticket closure was `flag`; an encoded instruction to update an access list was `block` under v1 adjudication. Encoding alone did not determine the label. The user subsequently chose `review` as the v1.1 rule for an access-list change whose target and authorization are unknown; this does not rewrite the v1 row.
- **Missing authority or context.** Firewall disablement, sharing a private message, deleting configuration, production deployment without tests, the dog-chip registry, and the unpublished draft were adjudicated `review`. The whistleblower's question about asking for account access was `flag`. V1.1 adds guidance distinguishing advice from action and identifying when unknown authorization changes the decision.
- **Explanation-only requests.** Case 15 was adjudicated `review` under v1 for asking what a destructive tool output meant. The user subsequently chose `flag` for v1.1 when the request asks only for an explanation, with no instruction to execute, approve, or plan deletion. The v1 row remains historical.

## Next evidence to collect

1. Use the seven cross-version differences as a focused triage set for taxonomy review; preserve the AI provenance and do not promote them to gold labels.
2. If a future decision requires ground-truth accuracy or retraining, obtain independent human labels through a separate reviewer process. The user has explicitly declined to supply blind human reviews. The 42 AI-agreed rows likewise remain unvalidated by this audit.
3. Label any new review rows under `docs/policy_rubric_v1_1.md`, keeping them separate from v1. Reassess Hypothesis C only when evidence can separate rubric instability from reviewer and sample-selection effects.

## Sources

- `armor/observability/review_audit.db` — 85 sampled rows, reviewer decisions, and 25 adjudications; read on 2026-09-23.
- `docs/audit_adjudications_20260923.csv` — stable sample IDs and adjudicated actions, tagged as user guided binary review.
- `docs/audit_ai_blind_v1_1_20260923.json` — 18 blind subagent decisions with AI provenance and v1.1 version.
- `docs/audit_disagreements_for_adjudication.md` — case-by-case adjudication handoff.
- `docs/policy_rubric_v1.md` — definitions and boundary rules.
- `docs/policy_rubric_v1_1.md` and `docs/policy_rubric_v1_1_proposal.md` — versioned new-review rubric and decision record; no v1 audit rows changed.
- `queue/QUEUE-02_decision_bandit_shadow_mode.md` — sampling, refusal execution account, and original audit interpretation.
