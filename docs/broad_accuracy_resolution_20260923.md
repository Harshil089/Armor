# Broad action accuracy, 2026-09-23

## Runtime architecture correction

**Armor runs on BERT only.** The production action path uses the existing
ArmorBase encoder and its small four-action `PolicyHead` over the embedding and
threat scores. It reuses the embedding already computed for inspection; there is
no second generative model call. Deterministic fast rules still run first, and
the proxy holds prompt `review` decisions before forwarding.

Qwen3 4B was an experiment added during this investigation. It improved some
rubric metrics but added roughly one second to median prompt latency and broke
the intended architecture. Its experimental serving code and Qwen model files
have been removed. Ollama is stopped. The corrected route passed the full test
suite: 348 passed, 6 skipped.

## AI-only data and limits

Jev labeled the rubric v1.1 gates blind through the TypeSafe development
connection. These are AI labels, not human ground truth. Existing logged
actions were used only to stratify gate selection and were hidden from Jev.
Gates v5-v7 have distinct request IDs, but all were sampled from corrected
logs. The existing `PolicyHead` checkpoint predates these gates and may have
trained on some of their source requests, so its gate scores are not a clean
unseen generalization estimate. Labels also include low-confidence
disagreements.

## Frozen gate comparison

The table compares the previous ArmorBase threshold route, the BERT policy
head, and the Qwen experiment. It is useful for tradeoffs, not a proof of
production accuracy.

| Gate | N | Base exact | BERT head exact | Qwen experiment exact | BERT block recall | BERT false hard blocks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v5 | 46 | 19 | 29 | 34 | 9/9 | 2 |
| v6 | 42 | 14 | 26 | 27 | 12/12 | 3 |
| v7 | 30 | 8 | 21 | 19 | 10/10 | 4 |

The BERT policy head was already present in shadow mode. These replays call the
same BERT embedding and policy head; enabling it as the action source adds no
second language-model inference. Its broad errors are still substantial:
especially `flag` versus `block`, with four AI-labeled non-block requests
classified as `block` in v7. The proxy holds `review` and blocks `block`, so
that distinction has real impact.

The BERT-only replay median latency was 79 ms on v5, 82 ms on v6, and 83 ms on
v7 (95th percentiles: 128, 179, and 233 ms). A live call after server restart
paid 2.2 seconds for cold BERT initialization; following warm calls measured
40-54 ms. Explicit fast-rule blocks returned without model inference.

The Qwen experiment's v7 result was 19/30 exact, with 2 AI-labeled false hard
blocks and 7/10 block recall. BERT is faster and preserves the required model
architecture; its higher block recall comes with more false hard blocks. These
AI-only results do not settle which route is closer to human judgment.

## Deployed state

- `com.armor.server` is configured with `ARMOR_USE_BASE_MODEL=1` and
  `ARMOR_BERT_POLICY_HEAD=1`.
- `ARMOR_LOCAL_POLICY_JUDGE` is absent. The running API no longer imports or
  calls Ollama.
- The Ollama service is stopped and the Qwen model files used in the experiment
  have been removed. Older unrelated Ollama models were left untouched.
- Prompt `review` is returned as a provider-shaped hold response and is not
  forwarded upstream. This hold is recorded for follow-up; it is not itself an
  adjudication.
- The flat BERT head abstains to `review` below its margin threshold. Explicit
  context review from task-scope checks remains decisive.

The corrected server returned BERT policy actions on live requests, preserved a
deterministic fast block, and returned `review` for a direct external access
grant with unknown authorization. The full corrected suite passed with 348
passed and 6 skipped. The result is operationally confirmed; policy accuracy
remains AI-evaluated and the false hard blocks above remain unresolved.

## Follow-up: request-scope REVIEW misses

A read-only replay of the current BERT head on the 18-case, independently
agreed AI holdout scored 13/18. The complete inspection path scored 12/18 and
missed all three \`review\` examples. Two involved disabling authentication
logging or monitoring without authorization context; one wrapped that action
in ROT13 with Cyrillic homoglyphs. A separate false \`review\` came from a
lookahead rule combining unrelated details across a multi-paragraph benign
request.

The request-scope path now evaluates boundary rules paragraph by paragraph,
recognizes a direct request to disable authentication monitoring as
authorization-dependent, and applies that REVIEW rule before a lower-severity
fast-scan flag. It also checks a ROT13 view only when the text explicitly
introduces a decode-and-follow/execute instruction. Explicit hard-block rules
keep precedence.

On replaying that same development slice, end-to-end agreement rose from
12/18 to 15/18, and REVIEW recall from 0/3 to 2/3. This slice was used to make
the change and is no longer an untouched holdout. Labels remain two-reviewer
AI consensus, not ground truth. The v5-v7 broad gates were unchanged at
29/46, 26/42, and 21/30 (zero decision changes across all 118 examples).
Validation: 349 passed, 6 skipped. A fresh blind AI holdout is required before
claiming general accuracy improvement.

## Broad-accuracy confirmation and bottleneck

The clean v10 holdout contains 26 prompts (10 allow, 5 flag, 10 block, and 1
review). It was selected from requests held out from the training rows and
removed exact or near-duplicate training texts at cosine similarity >= 0.98.
Jev labeled the prompts blind using rubric v1.1; this remains AI rubric
alignment, not human ground truth. The single REVIEW example makes REVIEW
recall too weak to generalize from.

| Candidate | Exact | Macro-F1 | Allow recall | Flag recall | Block recall | Review recall | False hard blocks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Same-data retrain, 1,029 rows | 20/26 (76.9%) | 0.526 | 9/10 | 1/5 | 10/10 | 0/1 | 1 |
| Add 52 active Jev labels | 17/26 (65.4%) | 0.448 | 9/10 | 1/5 | 7/10 | 0/1 | 2 |

The active-learning batch is rejected for promotion: it lowered exact and
macro accuracy, lowered block recall, and did not recover REVIEW. The
unmodified deployed checkpoint was not replaced. The experiment does not
support “more examples” as a fix by itself; selection and label consistency
matter. The training set and fresh holdout are heavily imbalanced/undersupplied
for `flag` and `review`, while the head still confuses neighboring action
classes, especially `flag`/`block`. The principal broad-accuracy bottleneck is
therefore boundary coverage and action-label consistency in the training
signal, followed by the flat head's inability to represent context-sensitive
decisions. It is not BERT encoder size or an extra inference call: the encoder
is reused, and the BERT-only warm path is tens of milliseconds. This conclusion
is diagnostic, not a claim that the current policy is accurate.

Artifacts: `broad_gate_v8_*`, `broad_gate_v9_*`,
`broad_gate_v10_*`, and `ai_policy_active_retrain_v{8_eval,10_confirm}_20260923.json`.
The v8 and v10 results are diagnostic sets after inspection; v9 is a training
batch. Do not report these as an untouched production benchmark.

## v11 blind evaluation and rubric stability

The fresh v11 gate contains 40 prompts from previously uncorrected traffic.
Selection excluded exact prior-gate/training matches, removed training
embedding neighbors at cosine >= 0.98, and balanced/ranked by the current
BERT-head predictions. The reviewers received only text and rubric v1.1.
Jev labeled 12 allow, 7 flag, 19 block, and 2 review. A fresh blind AI reviewer
labeled 10 allow, 7 flag, 13 block, and 10 review. Agreement was 70% (kappa
0.588), so these labels do not provide a single reliable gold set.

All policy candidates used the same API path and deterministic rules. Against
Jev's v1.1 labels, the deployed flat BERT head scored 18/40 exact (45.0%,
macro-F1 0.394); a same-data retrain scored 19/40 (47.5%, macro-F1 0.280); and
the retrain with 52 active Jev labels scored 18/40 (45.0%, macro-F1 0.370).
On the 28 reviewer-consensus examples, the deployed head scored 13/28 (46.4%,
macro-F1 0.404), the same-data retrain 13/28 (46.4%, macro-F1 0.284), the
active-label retrain 12/28 (42.9%, macro-F1 0.358), and the hierarchical
System B head 14/28 (50.0%, macro-F1 0.351). The base-threshold route matched
the deployed head's exact rate (13/28) but had macro-F1 0.298. No candidate
improved both exact accuracy and macro-F1 over the deployed head on consensus.
Review recall was 0/1 on that consensus subset for all tested routes.

The v1.1 disagreements exposed unclear applications of authorization,
production-secret disclosure, security-monitoring changes, and encoded
requests. A v1.2 proposal addendum was drafted, then tested correctly as the
full v1.1 rubric plus the proposed examples. The first proposal-only test is
marked incomplete because it omitted the v1.1 base definitions. In the
correct full-rubric development rerun, reviewer agreement rose to 75% (kappa
0.655), Jev's v1.1-to-v1.2 self-stability was 85%, and the second reviewer's
self-stability was 80%. This is promising evidence for clearer labeling, but
v11 was used to write the proposal and is development data, not a fresh
confirmation set. On the resulting 30-item consensus slice, the deployed head
scored 17/30 exact, macro-F1 0.523; same-data retrain also scored 17/30 but
macro-F1 0.396. The active-label retrain scored 15/30, macro-F1 0.444.
None supports promoting a retrain.

Training-pool audit (`broad_training_data_quality_20260923.json`) found 1,118
valid correction rows across 979 unique texts: 588 block, 349 allow, 109 flag,
and 72 review. 1,047 rows are marked `pseudo_model` and 71 `human` in the
database; these are provenance fields, not a ground-truth claim. Five unique
texts have conflicting corrections. The remaining uncorrected traffic could
not yield a v12 holdout after the same >=0.98 training/prior-gate neighbor
filters; see `broad_gate_v12_sampling_20260923.json`. That scarcity is a data
coverage limit, not a passing result.

The evidence continues to reject encoder size as the primary bottleneck.
Changing the head or adding more pseudo-labels has not produced a repeatable
gain; label stability, training-signal quality, and limited new boundary
examples remain the primary constraints. The BERT head sometimes overrides a
stronger base-model block with `allow` or `review`, while the base route also
creates false hard blocks. Their tradeoff needs a new, genuinely independent
confirmation gate before any serving change.

## System C conformal/OOD probe

A further non-serving diagnostic trained the hierarchical heads from scratch on
779 unique corrected texts, reserved 195 disjoint texts for calibration, and
excluded five conflicting text-label rows. The two-head hierarchical System B
route scored 20/30 (66.7%) exact on the v1.2 development consensus slice, but
macro-F1 was 0.443; it recalled 1/4 flag, 7/12 block, and 0/2 review cases.
This has higher aggregate accuracy than the deployed head on this development
slice (17/30, macro-F1 0.523), but lower macro-F1 and inadequate rare-class
recall, so it is not a broad improvement.

System C added split-conformal sets plus the embedding-neighbor OOD signal.
Across alpha 0.1–0.5 it recalled 0/2 review cases. At alpha 0.2 it scored
18/30 (60.0%) exact and macro-F1 0.411, with a 5% review rate; other thresholds
were worse overall. The conformal/OOD layer therefore did not turn REVIEW into
a reliable uncertainty outcome on this dataset. The queue was development
material used for the rubric proposal, and the calibration labels are
pseudo-model corrections, so none of these are confirmation estimates.

Artifact: `broad_gate_v11_system_c_probe_20260923.json`. This adds evidence
against switching to a hierarchical/conformal head as an accuracy fix. The
current confirmed bottleneck remains the quality and coverage of the action
labels at class boundaries. A next candidate needs better independently
sourced examples for flag/review and a new untouched gate; another threshold or
head sweep over v11 would only tune to the development set.

## v12 fresh gate and temporary candidate

A new hand-authored synthetic set was split before review into 48 training
candidates and 48 holdout prompts, with 12 examples per design stratum in each
split. The design strata are coverage targets, not labels. All items were
screened against correction texts and prior gates; none was an exact match or
had ModernBERT cosine similarity >= 0.98 to the 1,650 reference texts. The
reviewers saw only shuffled IDs and prompt text under the complete v1.1 rubric
plus the v1.2 proposal. These are AI labels, not human ground truth.

Stable identity mapping is now `reviewer_ai_a` = fresh blind AI subagent and
`reviewer_ai_b` = Jev via TypeSafe. Agreement over all 96 was 90.6% (kappa
0.858). On the 48-item holdout, the two reviewers agreed on 44: 22 allow, 12
block, 10 review, and **zero flag**. Jev marked two additional cases flag
where reviewer_ai_a chose allow. This gate therefore does not establish flag
recall.

The current live API baseline scored 25/44 exact (56.8%, macro-F1 0.310) on
those consensus items. A temporary flat BERT head, trained from scratch on
1,033 unique rows (974 existing unique texts after excluding five conflicts
and all 39 auto-corrected holdout rows, plus 59 reviewer-consensus training
examples), scored 34/44 (77.3%, macro-F1 0.554). By class, allow recall rose
from 19/22 to 20/22, block recall from 6/12 to 7/12, and review recall from
0/10 to 7/10. The candidate made no flag predictions; its flag recall is 0/2
against Jev and cannot be estimated on consensus. On the blind reviewer_ai_a
labels, the candidate scored 35/48 exact and macro-F1 0.537; on Jev's labels,
it scored 37/48 and macro-F1 0.549. Candidate API p50/p95 latency was about
41/43 ms versus about 40/42 ms for the live baseline. The candidate checkpoint
is temporary and has not replaced the deployed checkpoint.

A separate 24-prompt, training-only FLAG-boundary batch yielded only 66.7%
reviewer agreement (kappa 0.368): 7 agreed flag and 9 agreed allow, with 8
items disputed. This gives the temporary candidate seven new agreed flag
examples, but is direct evidence that the flag boundary remains unstable.

During the live baseline calls, a stale TypeSafe watcher auto-corrected 39
holdout log rows while omitting the source field; the admin API defaulted those
AI decisions to `human`. The watcher was stopped, only those 39 provenance
fields were changed to `pseudo_model`, and corrections/rewards were preserved.
A SQLite backup and row list are in `broad_gate_v12_db_provenance_correction_20260923.json`.
All 39 were explicitly excluded from candidate training. The additive review
store migration now records `reviewer_ai_a` / `reviewer_ai_b` IDs; it preserved
the prior 85 audit rows and imported the 120 v12/v13 AI decisions. Details are
in `review_audit_reviewer_identity_migration_20260923.json`.

This is a promising broad gain on allow/block/review but not a full four-class
confirmation. A separate untouched flag-focused holdout is being labeled
before any promotion decision. Do not deploy the temporary checkpoint unless
that gate also supports flag performance and the regression suite remains
clean.

## v14 flag-boundary holdout: candidate not promoted

The final 48-item v14 queue passed overlap screening against 1,650 prior texts: zero exact matches and maximum cosine 0.97895, below the 0.98 threshold. v14 was designed for low-severity FLAG boundary cases and adjacent benign controls; it is not a broad four-class holdout.

The two AI reviewers agreed on 41/48 (85.4%), but Cohen's kappa was only 0.152 because both reviewers overwhelmingly chose ALLOW. Their marginal labels were reviewer_ai_a: 42 ALLOW / 6 FLAG and reviewer_ai_b: 45 ALLOW / 3 FLAG. Seven disagreements were concentrated in mild hostility and privacy-risk discussions. The 41-item consensus contained 40 ALLOW, one FLAG, and no BLOCK or REVIEW examples. This sample cannot estimate broad macro-F1 or block/review recall.

On the 48 full-service calls, the live baseline predicted 35 ALLOW, 4 FLAG, 4 BLOCK, and 5 REVIEW. The temporary flat BERT candidate predicted 40 ALLOW, zero FLAG, one BLOCK, and 7 REVIEW. Against the single consensus FLAG, neither model predicted FLAG; on the consensus slice accuracy was 73.2% baseline and 80.5% candidate, while macro-F1 was 0.214 and 0.223. Those figures are too under-supported to claim a flag-boundary win; against reviewer_ai_a the candidate missed all six FLAG labels, and against reviewer_ai_b it missed all three. The candidate also shifted output heavily toward ALLOW. Its v12 gains do not establish a four-class improvement.

Decision: keep the candidate checkpoint temporary and do not promote it. The near-term bottleneck is training signal at the FLAG boundary and the skew/ambiguity of AI-generated action labels, not encoder size or inference latency. A useful next candidate needs more independently authored, diverse FLAG-vs-ALLOW training examples with reviewer consensus, followed by a larger fresh holdout that has meaningful consensus support in all four actions. Keep the new holdout sealed from training and report agreement, per-class support/recall, macro-F1, confusion matrices, and exact-service latency before considering promotion. All current evaluation labels remain AI-only and are not ground truth.

## v1.3 development data and fresh v16 confirmation

A v1.3 proposal makes the disputed FLAG/ALLOW boundaries explicit: generating bounded insulting or humiliating language is FLAG; respectful criticism is ALLOW; a concrete privacy exposure scenario discussed without carrying out the disclosure is FLAG; general privacy education is ALLOW. v15 was training-only. Its 48 hand-authored prompts had zero exact overlap and maximum embedding cosine 0.9761 against the prior 1,650-text reference set. The two AI reviewers agreed on 46/48 (95.8%, kappa 0.917); their labels were mostly FLAG/ALLOW, and the two disagreements were both mild hostile wording. This agreement is a development signal for this curated batch, not proof that the taxonomy matches human judgment.

A temporary flat BERT policy head was retrained with 46 v15 consensus labels (22 FLAG, 24 ALLOW) plus the prior v12 training split and v13 development consensuses. It used 1,079 unique rows: 974 prior rows after conflict/holdout exclusions and 105 new AI-consensus rows. The historical training source mix is disclosed in `broad_gate_v15_candidate_training_20260923.json`: 57 prior rows tagged `human`, 917 prior rows tagged `pseudo_model`, and 105 newly added AI-reviewer consensus rows. No human labels were collected for this work or used to score the confirmation gate; old source metadata remains intact. The 39 previously contaminated v12 holdout corrections remained excluded, and no v14 or v16 holdout text entered training.

The fresh v16 gate contains 80 shuffled prompts, with 20 prompts in each internal coverage stratum; those strata were not shown to reviewers and are not labels. Nine near-duplicate candidates were replaced during screening. The final queue has zero exact overlaps and maximum cosine 0.97899 against 1,746 prior training/evaluation texts. Both reviewers applied v1.1 plus the v1.3 proposal, independently and blind to predictions. Agreement was 79/80 (98.75%, kappa 0.983). Consensus support was 26 ALLOW, 13 FLAG, 20 BLOCK, and 20 REVIEW.

On the same full `/v1/inspect` path, the candidate scored 59/79 exact (74.7%, macro-F1 0.696) on consensus items, versus 37/79 (46.8%, macro-F1 0.335) for the current checkpoint. It also improved against each reviewer separately: accuracy 75.0% / 73.8% and macro-F1 0.698 / 0.686, versus baseline 47.5% / 46.3% and 0.337 / 0.333. Consensus recall moved from 96.2% to 96.2% ALLOW, 0% to 30.8% FLAG, 45% to 80% BLOCK, and 15% to 70% REVIEW. FLAG precision was 100% on 13 consensus cases; 9 of 13 FLAG cases are still missed, so rare-class work remains. Candidate p50/p95 API latency was 41.3/43.7 ms versus 40.6/43.8 ms baseline.

The separate 70-item long-text guard scored 83.6% accuracy / 0.438 macro-F1 with the candidate versus 80.7% / 0.425 baseline. It has no FLAG support and is only a regression guard, not broad confirmation. The complete test suite passed after promotion: 351 passed, 6 skipped.

## Local service promotion and rollback

Given the fresh four-class gain, the long-text guard, and unchanged warm-request latency, the candidate checkpoint was atomically copied to `armor/models/checkpoints/policy_head.pt` and the configured local `com.armor.server` launch agent was restarted. The prior checkpoint is backed up at `/tmp/armor_policy_head_pre_v15_promotion_20260923.pt`; both SHA-256 hashes and the rollback command are in `broad_gate_v15_promotion_20260923.json`. The restarted service loaded with `ARMOR_USE_BASE_MODEL=1` and `ARMOR_BERT_POLICY_HEAD=1`. Four post-restart `/v1/inspect` smoke requests, one per action, matched the frozen candidate predictions. First-request model warmup was 2.83 seconds; the other three requests took 0.3–50 ms. The temporary candidate service was stopped.

That was the v16 checkpoint state. The results below supersede its remaining-FLAG-gap status; v16 metrics and promotion history above remain unchanged.

## v17 FLAG-boundary development and v18 confirmation (2026-09-24)

The v17 training-only batch contained 80 fresh FLAG-boundary and ALLOW-control prompts. The blind reviewer and Jev agreed on 75/80 (93.75%); only the 75 consensus labels were eligible for training. The candidate was retrained from the frozen ArmorBase representation with the existing correction pool, v12 train/v13/v15 consensuses, v16 consensus now treated as development data, and v17 consensus. It used 1,234 unique rows: 975 prior rows (917 `pseudo_model`, 58 historically tagged `human`) plus 259 AI-consensus rows. New consensus labels added 79 FLAG, 116 ALLOW, 32 BLOCK, and 32 REVIEW examples. The v12 confirmation texts and conflicting prior labels were excluded. The training source metadata is in `broad_gate_v17_candidate_training_20260924.json`; the newly added consensus rows are in `broad_gate_v17_training_consensus_20260924.jsonl`.

The model was frozen before opening the separate 150-prompt v18 confirmation gate. Its five internal sampling strata were hidden from reviewers. After overlap screening, zero exact text matches remained against prior training and evaluation text; obvious near-duplicate phrasings were replaced before labeling. The blind reviewers agreed on 148/150 (98.67%, Cohen's kappa 0.982). Consensus support was 31 ALLOW, 59 FLAG, 30 BLOCK, and 28 REVIEW. All labels remain AI-only and are not human ground truth.

Both checkpoints were served through isolated `/v1/inspect` services with identical ArmorBase settings. On the 148 consensus cases, the promoted candidate scored 127/148 (85.8%) exact accuracy and 0.850 macro-F1, versus 68/148 (45.9%) and 0.446 for the previous checkpoint. Per-class recall changed as follows:

| Action | Prior recall | Candidate recall | Candidate precision | Consensus support |
|---|---:|---:|---:|---:|
| ALLOW | 93.5% | 93.5% | 76.3% | 31 |
| FLAG | 15.3% | 88.1% | 89.7% | 59 |
| BLOCK | 63.3% | 70.0% | 91.3% | 30 |
| REVIEW | 39.3% | 89.3% | 86.2% | 28 |

Against each reviewer separately, candidate accuracy / macro-F1 was 86.0% / 0.852 for reviewer A and 84.7% / 0.835 for Jev; the prior checkpoint scored 46.7% / 0.455 and 45.3% / 0.440. Warm API p50/p95 latency was 43.2/48.7 ms for the candidate and 43.6/50.3 ms for the prior checkpoint.

The separate 70-item long-text regression guard uses one AI reviewer and contains no FLAG examples. Through `/v1/inspect`, candidate accuracy improved from 80.0% to 87.1%, macro-F1 from 0.424 to 0.454, and BLOCK recall stayed at 100%. This is a regression guard, not a broad or FLAG estimate.

## Promotion and live verification

All promotion checks passed: reviewer agreement at least 95%; at least 25 consensus FLAG examples; FLAG recall at least 75% and precision at least 80%; at least a 15-point FLAG-recall gain; no material macro-F1 regression; no more than a 5-point recall loss in ALLOW/BLOCK/REVIEW; API p95 within 10%; and no long-text guard regression. The candidate was atomically installed as `armor/models/checkpoints/policy_head.pt` and the local `com.armor.server` service restarted. Its SHA-256 is `4fbf18ff1a6f48f4957beb743772440ad31d30f9d40e84cfa0c86652a5e1448d`. The previous checkpoint is backed up at `/tmp/armor_policy_head_pre_v17_promotion_20260924.pt`.

Four live `/v1/inspect` requests, one per action, matched the isolated candidate service after restart. The first request took 4.47 s for model warmup; the next three took 44–66 ms. The detailed gate, long-text guard, promotion checks, and live smoke are recorded in `broad_gate_v18_comparison_20260924.json`, `broad_gate_v18_long_text_guard_20260924.json`, `broad_gate_v18_promotion_20260924.json`, and `broad_gate_v18_live_smoke_20260924.json`.

This is a strong improvement on a fresh, stratified, hand-authored AI-only gate; it does not prove taxonomy correctness, production prevalence, or real-world accuracy. Keep provenance explicit and require a new untouched confirmation set before tuning on subsequent data.
