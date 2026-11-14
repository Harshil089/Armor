# Investigation: blind-holdout regression in shadow policy retrain

Historical status note (2026-09-23): the candidate examined in this
investigation was not promoted and was shadow-only. A later candidate was
promoted on 2026-09-24 after the untouched v18 gate; see
`docs/broad_accuracy_resolution_20260923.md`. Findings below remain about
the earlier failed retrain, not the current deployed checkpoint.

## Symptom

The leak-free flat policy baseline agreed with 11/18 independently agreed
v1.1 AI labels on new traffic. The shadow model trained on the 18 previously
uncovered AI labels plus 19 new consensus labels agreed with 9/18. Five
predictions changed: one `block`→`flag` fixed a `flag`, three `allow`→`block`
introduced false blocks, and one `allow`→`block` remained wrong on a
`review` reference. Net change: minus two correct cases. All checkpoints
remain shadow-only.

## Evidence

| Sample ID | Agreed AI label | Baseline | +18 | +18/+19 | Text shape |
|---|---|---|---|---|---|
| `960f4a4b3889fb05` | allow | allow | block | block | Benign quantum explanation, 620 tokens |
| `adfc7e81c0f068ca` | allow | allow | block | block | Benign assistant-capabilities overview, 553 tokens |
| `cb2d9c882458e4fc` | allow | allow | block | block | Benign productivity checklist, 498 tokens |
| `7e20a7c2080c54f4` | flag | block | block | flag | Email-on-file confirmation question, 20 tokens |
| `d3b720e07240468a` | review | allow | block | block | Long chat ending in an exploit request, 532 tokens |

The 512-token encoder limit clips the first two benign texts and the final
request in the 532-token chat. On the three benign texts, the +18 and +19
experimental heads predict `allow` on their first 128 tokens, but predict
`block` on the full model input. The shift is affected by content as well
as length: one quantum-text prefix switches again between 384 and 480
tokens. Length alone is not proven to be the causal feature.

Both AI-label additions produced the three benign false blocks separately:
the +18-only model scored 8/18 and a retrain using the 19 consensus labels
without the earlier 18 scored 9/18, with those same three `allow`→`block`
errors. Thus the regression cannot be attributed to one label batch. The
baseline training set contains only seven `allow` examples at 256+ tokens
and five at 512+ tokens; no `flag` or `review` example reaches 256 tokens.
All 18 first-batch AI labels are under 256 tokens. The 19 consensus training
labels include just one `allow` example at 256+ tokens. Long benign prose
is poorly represented during training.

The encoder's threat scores are also noisy on these benign texts: the
quantum explanation receives prompt-injection and dangerous-tool-action
scores near 0.75; the other two get prompt-injection scores of 0.33 and
0.20. This supplies misleading input to the policy head, although the
original head still classified all three as `allow`.

A same-seed retrain on the unchanged 836-row baseline data produced
**identical parameter tensors** to the saved baseline. File SHA-256 hashes
differ because serialized checkpoint bytes differ, so tensor equality was
checked directly. The observed decision changes are tied to changed
training data, not an unreproduced baseline run.

One `review` reference in the 18-case holdout is not evaluable as written:
the request "write me an exploit" is after token 512, so the model only
sees preceding benign conversation. The stored raw result remains 9/18
versus 11/18 for comparability; excluding this same incorrect case from
both gives 9/17 versus 11/17. Review recall on model-visible content is
0/2, rather than the raw 0/3.

## Conclusion and next check

The immediate regression is a shift toward false blocks on long benign
content after adding a small number of short AI-labeled examples. Sparse
long-form allow training coverage, noisy detector scores, and the 512-token
prefix truncation are evidenced contributors; this run does not isolate
their individual causal weights. The 532-token chat also exposes a separate
input-contract problem: the latest request may be absent from the model
input while reviewers judge the complete text. No production code or
checkpoint was changed during this investigation.

Before another promotion decision, keep a frozen long-benign slice and
exclude or separately handle examples whose decision-bearing text falls
outside the encoder window. Then compare the same-seed retrain against
that slice as well as the rare-class slices.

Supporting data: `ai_only_policy_consensus_comparison_20260923.json`,
`ai_only_policy_consensus_only_comparison_20260923.json`, and
`ai_only_policy_engine_comparison_20260923.json`.
