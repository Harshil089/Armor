# AI-only policy review and shadow retraining — 2026-09-23

The user chose an AI-only evaluation path. These labels measure agreement with
the stated rubric and with other AI judgments. They are not human ground truth
or a measurement of real-world enforcement accuracy.

## Data and separation

- The original audit had 85 v1 rows. A fresh AI reviewer supplied v1.1 labels
  for the 18 rows missing one of the original AI reviews. The original audit
  decisions remain unchanged.
- For model experiments, the policy head's isolated baseline training DB held
  836 corrected log rows after excluding all 85 audit texts and all 100
  detector-regression texts. Source labels were 794 `pseudo_model` and 42
  `human` as stored in the logs; neither source was treated as a gold set.
- A first shadow retrain added the 18 blind v1.1 AI labels. It matched the
  baseline on 10/25 guided v1 adjudications and fell from 56/84 to 53/84 on
  regression-sourced texts with one unambiguous corrected **policy action**.
  The detector labels in `regression_set.py` were not used as action labels.
- A second, actively selected batch drew 40 unique uncorrected requests from
  413 eligible texts of length 20–4000 characters. Selection used the existing
  policy margin, reference-bank disagreement, flag similarity, OOD novelty,
  and embedding clusters. The logged decision only set sampling quotas
  (24 flag, 8 allow, 8 block); it was hidden from the blind reviewer. All 40
  lacked an exact-text match to corrected training rows.

## New blind AI batch

The first fresh v1.1 reviewer labeled the 40 cases as 26 allow, 6 flag,
2 block, and 6 review. Of the 24 sampled from the logged `flag` stratum,
only four were labeled `flag` by this reviewer; 17 were `allow` and three
were `review`. Logged decisions are weak sampling signals, not reference
labels. The batch is actively selected, so this is not a population estimate.

Twenty of the 40 AI labels were added to the isolated training DB, split by
sample ID within each new AI label; the other 20 were kept unseen. The new
shadow head used the same architecture, inverse-frequency weights, seed,
learning rate, and 10 epochs as the earlier heads. No serving checkpoint or
production decision path changed.

| AI/reference slice | Baseline | +18 AI rows | +18 and 20 new AI rows |
|---|---:|---:|---:|
| New blind v1.1 holdout, 20 cases | 11/20 | 8/20 | 11/20 |
| Guided v1 adjudications, 25 cases | 10/25 | 10/25 | 11/25 |
| Regression-sourced corrected actions, 84 cases | 56/84 | 53/84 | 56/84 |

On the new 20-case holdout, flag recall went from 0/3 to 2/3 after adding
the 20 AI labels, while review recall stayed 0/3. On the guided v1 slice,
flag recall went from 1/6 to 2/6 and review from 0/8 to 1/8. On the 84-case
regression action slice, flag recall remained 0/1 and block recall remained
12/15. The new holdout is too small and selected to claim a general gain;
the paired accuracy on that slice and on the regression slice did not rise.

## Decision

Keep all experimental heads in shadow. The added labels changed some
rare-class predictions but did not deliver a repeatable overall gain, and
`review` remains weak. A second independent blind v1.1 reviewer agreed with
the first on 37/40 cases (92.5%, Cohen's κ 0.859). All 26 `allow` cases
matched; the three disagreements were two `flag` ↔ `review` and one
`review` ↔ `block`. This selected batch is easier and more allow-heavy than
the original v1 audit, so its agreement is not directly comparable to the
original 63% A/B agreement.

A consensus-only retrain used the 19 matching labels from the original
training half and held out the 18 matching labels from the test half. It
improved the regression action slice from 56/84 to 63/84, but fell from
11/18 to 9/18 on the new blind holdout. It found none of the three
`review` holdout cases. The full 20-label retrain scored 10/18 on that
same consensus holdout and 56/84 on regression actions. The consensus
gain is therefore specific to one slice, not a reliable broad improvement.

The existing policy engine also turns low confidence into `review`, so raw
argmax alone understates its review capability. We evaluated that actual
abstention rule and the existing hierarchical System B on the 18 unseen
consensus cases. Review recall remained **0/3 for every tested path**:
the leak-free flat baseline, both AI retrains, and hierarchical System B.
The three review requests were mostly routed to `block`, which points to a
missing-context decision problem rather than only a softmax threshold
problem. One of the three has its decision-bearing final request clipped by
the encoder's 512-token limit; review recall on model-visible content is
0/2. Three examples are too few to establish a root cause.

No head is promoted. The next useful AI-only experiment should target
diverse, independently agreed `review` cases and compare an explicit
missing-context signal against the same frozen holdouts. Keep every
label's reviewer and rubric version visible; do not promote AI agreement
to ground truth.

Machine-readable evidence: `ai_only_policy_baseline_20260923.json`,
`ai_only_policy_comparison_20260923.json`,
`ai_only_candidate_mining_20260923.json`,
`ai_only_blind_batch_40_20260923.json`, and
`ai_only_policy_batch_40_comparison_20260923.json`,
`ai_only_blind_batch_40_agreement_20260923.json`,
`ai_only_policy_consensus_comparison_20260923.json`, and
`ai_only_policy_engine_comparison_20260923.json`.

The paired false-block regression and truncation check are documented in
`ai_only_policy_regression_investigation_20260923.md`.
