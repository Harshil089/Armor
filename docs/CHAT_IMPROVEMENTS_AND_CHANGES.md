# Armor Improvements and Changes

Historical notes below describe the ArmorBase partial-fine-tuning work in this file; they are not the current serving status. As of 2026-09-24, the active prompt policy head is the promoted four-action BERT checkpoint described in `docs/broad_accuracy_resolution_20260923.md`. The FLAG-boundary v18 gate measured 88.1% FLAG recall at 89.7% precision across 59 consensus FLAG examples, with no human ground truth.

## Current result

Candidate `final2_candidate_v5.pt` is the strongest result for `dangerous_tool_action`. Scores below are positive-minus-negative separation on the frozen regression set, not accuracy percentages.

| Label | Deployed baseline | Candidate v5 |
|---|---:|---:|
| `prompt_injection` | 0.788 | 0.810 |
| `jailbreak` | 0.718 | 0.867 |
| `sensitive_data_exposure` | 0.805 | 0.780 |
| `dangerous_tool_action` | 0.357 | 0.610 |
| `harmful_content` | 0.427 | 0.691 |
| `obfuscation_encoding` | 0.841 | 0.834 |

The deployed `armor/models/checkpoints/head.pt` remains unchanged. Candidate checkpoints are isolated until selection, ONNX export, and parity verification are complete.

## Partial encoder fine-tuning

`armor/models/train.py` now supports bounded partial fine-tuning while preserving the original frozen-head path.

- Added `fine_tune_layers` to unfreeze only the final ModernBERT encoder layers and final normalization layer.
- Added a separate `encoder_lr`, keeping encoder updates smaller than head updates.
- Initialized partial fine-tuning from the deployed head checkpoint.
- Added candidate-specific `output_path` support so experiments cannot overwrite the deployed checkpoint by default.
- Saved only fine-tuned encoder deltas alongside the classifier head.
- Added shuffled length buckets for partial fine-tuning to reduce padding cost without reintroducing length-order bias.
- Added periodic MPS allocator cleanup for stable Apple Silicon training.
- Preserved cached encoder embeddings for the default frozen-head training path.

`armor/models/armor_base.py` now loads optional encoder deltas from a checkpoint. It also rejects an ONNX backend when the export predates a fine-tuned encoder, preventing silent PyTorch/ONNX mismatch.

## Candidate experiments

Four one-epoch final-two-layer candidates were trained and kept separately.

| Candidate | Prompt injection | Jailbreak | Sensitive data | Dangerous tool | Harmful content | Obfuscation |
|---|---:|---:|---:|---:|---:|---:|
| v2 | 0.807 | 0.891 | 0.806 | 0.522 | 0.606 | 0.823 |
| v3 | 0.850 | 0.852 | 0.793 | 0.487 | 0.638 | 0.828 |
| v4 | 0.839 | 0.859 | 0.790 | 0.542 | 0.686 | 0.894 |
| v5 | 0.810 | 0.867 | 0.780 | 0.610 | 0.691 | 0.834 |

Candidate progression:

- v2 established that final-two-layer fine-tuning could break the frozen-probe ceiling.
- v3 added targeted harmful-content and external-action data.
- v4 added destructive-action matched data and produced the strongest balanced result at that point.
- v5 added licensed Agent Safety Bench and ToolEmu data, raising `dangerous_tool_action` from 0.542 to 0.610.

Median CPU inference remained about 26 ms. No material latency regression was observed.

## Dangerous tool action improvements

### New licensed sources

- `armor/data/agent_safety_bench.py` loads 287 examples from `aradhye/agent-safety-bench` under Apache-2.0.
  - Targets destructive changes, data exfiltration, external communication, and financial actions.
  - Keeps matched safe and unsafe instructions.
  - Removes instructions carrying conflicting text-only labels because Armor cannot observe hidden environment state.
  - Pins the dataset revision for reproducibility.
- `armor/data/toolemu.py` loads 322 examples from `ryoungj/ToolEmu` under Apache-2.0.
  - Uses curated risky actions as positives and expected safe achievements as negatives.
  - Targets the same four weak slices.
  - Pins the source commit for reproducibility.

Together, these sources add 609 licensed training examples. The total `dangerous_tool_action` pool is now 2,320 rows.

### Synthetic coverage

`armor/data/synthetic_dangerous_tool_action.py` now includes:

- A second external-side-effect domain covering publication, purchases, calendar changes, legal notices, and submissions.
- 200 balanced destructive-action examples covering backup deletion, database destruction, irreversible overwrites, instance termination, history erasure, audit-log deletion, table drops, and credential revocation.
- Matched safe controls such as previews, dry runs, snapshots, approval waits, and read-only inspection.

The synthetic dangerous-tool pool increased from 400 to 600 rows.

### Priority-slice result for v5

| Slice | Separation |
|---|---:|
| Destructive change | 0.521 |
| Privilege change | 0.838 |
| Data exfiltration | 0.419 |
| Financial action | 0.470 |
| External communication | 0.342 |
| Physical device | 0.866 |

All six slices remain positive. External communication is still the weakest slice and declined from 0.368 in v4 to 0.342 in v5.

## Harmful content improvements

Added `armor/data/synthetic_harmful_content.py` with 300 balanced examples for the three weakest policy areas:

- Non-consensual sexual content.
- Harassment, doxxing, and defamation.
- Illegal facilitation.

Every positive template has matched prevention, reporting, legal, or safety-focused negatives. Candidate v5 reaches 0.691 globally for `harmful_content`.

### Priority-slice result for v5

| Slice | Separation |
|---|---:|
| Violence and incitement | 0.486 |
| Hate and discrimination | 0.967 |
| Self-harm | 0.500 |
| Child exploitation | 0.919 |
| Non-consensual sexual content | 0.576 |
| Harassment, doxxing, and defamation | 0.358 |
| Weapons and CBRN | 0.507 |
| Misinformation | 0.583 |
| Illegal facilitation | 0.682 |

All nine harmful-content slices have positive separation.

## Evaluation and regression protection

Added `armor/eval/priority_slices.py` with balanced positive and negative cases for:

- Six dangerous-tool-action slices.
- Nine harmful-content slices.

The evaluator reports mean positive score minus mean negative score for every slice.

Strengthened `armor/eval/regression_gate.py` so any previously learned label is rejected when separation falls by more than 0.05, even if the new score remains positive. This closes a gap where a large quality loss could previously pass.

## Proxy validation (now QUEUE-01)

Created the proxy manual test plan (now `queue/QUEUE-01_proxy_manual_test_plan.md`) and expanded `tests/test_proxy.py`.

Automated coverage now includes both OpenAI and Anthropic adapters for:

- Benign passthrough.
- Prompt inspection failure with fail-open behavior.
- Response inspection failure with fail-closed behavior.
- Blocking a dangerous prompt before any upstream call.
- Blocking a sensitive upstream response.
- Unknown-provider handling.

The remaining live-provider wire checks require `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`.

## Tests added or expanded

New focused tests cover:

- Partial encoder layer selection and invalid layer counts.
- Fine-tuned encoder checkpoint loading and stale ONNX rejection.
- Priority-slice balance and separation calculation.
- Harmful-content synthetic data balance and labeling.
- Destructive-action synthetic data balance and labeling.
- Agent Safety Bench filtering, ambiguity removal, and labels.
- ToolEmu matching, labels, and deduplication.
- Regression-gate detection of material per-label drops.
- Expanded proxy policy behavior for both providers.

Final verification: **154 tests passed** with two dependency deprecation warnings.

## Documentation and queue maintenance

- Updated `README.md` with current training behavior, data sources, row counts, test count, candidate scores, and queue status.
- Refreshed ARCHIVE-QUEUE-01 through ARCHIVE-QUEUE-04 documents to reflect implemented work and current findings.
- Updated ARCHIVE-QUEUE-04 through candidates v2-v5, licensed weak-slice data, latency, slice results, and the remaining selection tradeoff.
- Added the proxy verification work item (now QUEUE-01).
- Visually checked the updated ARCHIVE-QUEUE-04 document after rendering through the available Quick Look fallback.

## Artifacts created

- `armor/models/checkpoints/final2_candidate_v2.pt`
- `armor/models/checkpoints/final2_candidate_v3.pt`
- `armor/models/checkpoints/final2_candidate_v4.pt`
- `armor/models/checkpoints/final2_candidate_v5.pt`

Each candidate is approximately 38 MB and remains separate from the deployed checkpoint.

## Promotion: v5 selected and deployed

`final2_candidate_v5.pt` promoted to `armor/models/checkpoints/head.pt`
(previous head archived as `checkpoints/archive/head_1789816778_pre_v5.pt`).
Decision: v5 wins on `dangerous_tool_action` (0.610 vs v4's 0.542, the
project's persistently weakest label) and `harmful_content` (0.691 vs
0.686); v4's edge on `obfuscation_encoding` and `prompt_injection` was
marginal on already-strong labels. Regression gate confirmed clean
(no label dropped >0.05 vs the previous deployed head; largest dip was
`sensitive_data_exposure` at 0.025). 154 tests pass post-promotion.

Found and fixed a real gap in the ONNX freshness guard while promoting:
`armor/models/armor_base.py` previously blocked the ONNX backend on any
presence of `encoder_state_dict`, with no way to recognize a *fresh*
re-export as fresh — it would permanently disable ONNX for any fine-tuned
checkpoint, forever, including right after a correct re-export. Replaced
the presence check with a content-hash fingerprint
(`encoder_fingerprint()`, sha256 of the sorted encoder delta tensors)
written to `onnx_export/encoder_fingerprint.txt` at export time and
compared at load time. `armor/models/export_onnx.py` now also bakes any
`encoder_state_dict` into the base encoder before ONNX export (previously
it only ever exported the untouched pretrained encoder, silently ignoring
fine-tuning).

Re-exported ONNX with the fine-tuned encoder baked in. Parity test
(`test_onnx_and_pytorch_backends_agree`, 1e-3 tolerance) passes. CPU
latency ~14.8ms/request, consistent with the ~21% ONNX speedup measured
pre-fine-tuning — no regression from the added encoder layers.

2026-09-25 long-input latency check: on the same eight longest sampled
TrustAIRLab in-the-wild jailbreak prompts (1566–3211 tokens), using the full
inspect path with the same active policy head, CPU averaged 2.42s (median
2.20s, max 4.42s) and Apple MPS averaged 1.23s (median 1.03s, max 2.31s).
All eight decisions matched (seven block, one review); the longest sample's
raw-score difference was at most `1.2e-7`. MPS is a local-only measurement;
production defaults to CPU and acceleration requires explicit configuration.
A live MPS HTTP replay averaged 1.35s, with a 7.0s cold first request after
restart. A batched 510-token-window shortcut made a vetted 797-token benign
answer's raw harmful-content score jump to 0.999, so it was rejected. ONNX
took 6.9–8.3s on the 3211-token sample, and MPS FP16 autocast showed score
deltas up to 0.31 without reliable acceleration; neither is enabled.

`external_communication` slice (`dangerous_tool_action`) remains the
weakest at 0.342 (down from v4's 0.368) — tracked, not blocking.

## v6: external_communication improvement, promoted

Added `_EXTERNAL_COMM_FOCUS` to `armor/data/synthetic_dangerous_tool_action.py`
(`generate_external_comm_focus_examples()`, 12 pos/12 neg templates covering
press releases, filings, contracts, refunds, wire payments, calendar,
ToS, press inquiries — negatives vary safe-signal phrasing
flag/route/stage/queue/log/summarize/confirm-internally instead of
repeating "review"/"approval", to avoid a keyword shortcut).

First attempt at 100/100 volume caused real regression gate failures
(`sensitive_data_exposure` -0.057, `obfuscation_encoding` -0.078, both past
the 0.05 tolerance) — nothing was saved, deployed head stayed on v5. Cut
default volume to 40/40 and retried: gate passed clean, all 6 labels flat
or improved. `external_communication` slice: 0.342 → 0.376 — small,
genuine improvement, no other slice regressed.

Promoted `final2_candidate_v6.pt` to `head.pt` (previous head archived as
`checkpoints/archive/head_1789825587_pre_v6.pt`). Re-exported ONNX with
the new fine-tuned encoder (fingerprint mismatch guard caught the stale
export correctly, forcing the re-export). 155 tests pass, including ONNX
parity.

v6 vs v5 full label state: `prompt_injection` 0.810→0.929,
`jailbreak` 0.867→0.903, `sensitive_data_exposure` 0.780→0.748 (within
tolerance), `dangerous_tool_action` 0.610→0.711, `harmful_content`
0.691→0.746, `obfuscation_encoding` 0.834→0.961.

## harassment_doxxing_defamation: 3 fix attempts abandoned

Tried the same booster pattern that worked for `external_communication`
(0.342→0.376) on `harassment_doxxing_defamation` (0.316, weakest of the 9
`harmful_content` slices) — it failed 3 different ways:

1. 40/40, then 100/100 volume of `_HARASSMENT_FOCUS` templates: regression
   gate failed on `prompt_injection` and `harmful_content`'s own general
   cases (unrelated labels/cases, not the targeted slice).
2. Redesigned negatives to remove target-contact-action ambiguity (matched
   this file's already-working `Explain`/`Summarize`/`List` style exactly):
   same gate failure, nearly identical numbers — ruled out negative-phrasing
   ambiguity as the cause.
3. Added `warm_start_path` to `train.py` (a real, kept feature — lets a
   partial fine-tune start from a checkpoint other than the deployed one)
   to test whether repeated fine-tune stacking on the same 2 encoder layers
   was the cause. Fine-tuning fresh from the last frozen-head-only
   checkpoint instead: worse failure, 5 of 6 labels regressed — 1 epoch
   wasn't enough to re-adapt to how much the dataset had grown since that
   checkpoint was trained.

No public dataset cleanly covers doxxing/defamation as this project's
taxonomy defines them either (HateXplain and OLID/OffensEval are general
hate/offensive-speech sets, targeted at communities or generic "individual
offense," not doxxing or false-accusation specifically).

Removed the `_HARASSMENT_FOCUS` templates and
`generate_harassment_focus_examples()` — no checkpoint using them ever
passed the gate, and future work here should use real data if a suitable
source turns up, not more synthetic volume tuning. `harassment_doxxing_
defamation` stays at 0.316, tracked, not urgent (no plateau signal).

## v11: short-input collapse fixed, SPML capped, promoted

Two real defects found while debugging a live groq-chat client routed
through the proxy, both predating this session's work:

**1. Short-input out-of-distribution collapse.** Every input under ~7
words scored ~1.0 on an arbitrary label — `"hello"` →
`sensitive_data_exposure` 1.0, `"yes"` → `dangerous_tool_action` 0.98.
Sharp cliff: scores drop to ~0.08 at 8+ words. Cause: no data source in
the entire corpus, for any label, has examples shorter than ~6 words, so
short text was never modeled at all. Confirmed present in the original
pre-fine-tuning frozen-head checkpoint, so not caused by any fine-tuning
here. Invisible until now because `decision` was capped at `flag` for the
base-model route, and `regression_set.py` contains no short-input cases.

Fix: `armor/data/synthetic_short_negatives.py` — 50 short benign phrases,
one label masked per row (an all-6-mask first version failed the gate at
both 200 and 80 volume with near-identical magnitude, ruling out volume
and confirming the mask shape itself was the problem).

**2. `prompt_injection` was 80% of the training pool.** Counted 2026-09-20:
16,587 rows vs 326-1,370 for every other label, because SPML's 16,012 rows
came in through `SOURCES`, which had no cap support at all — the same
"uncapped source dominates" mistake already caught three times this session
for civil_comments, ToolSafety, and TrustAIRLab. That dominance is why
`prompt_injection` and `obfuscation_encoding` regressed on *every* retrain
attempt regardless of what data was added. Fix: added `max_rows` to
`DataSource` (`armor/data/sources.py`) applied as a deterministic head
slice in `armor/data/public.py`, SPML capped at 2000 →
`prompt_injection` now 2,546 rows.

Promoted `final2_candidate_v11.pt` to `head.pt` (prior head archived as
`checkpoints/archive/head_1789912847_pre_v11.pt`). **This overrode the
regression gate deliberately, with explicit approval** — it is a trade,
not a clean win:

| label | v6 | v11 |
|---|---:|---:|
| `prompt_injection` | 0.929 | 0.827 |
| `jailbreak` | 0.903 | 0.904 |
| `sensitive_data_exposure` | 0.748 | 0.889 |
| `dangerous_tool_action` | 0.711 | 0.879 |
| `harmful_content` | 0.746 | 0.664 |
| `obfuscation_encoding` | 0.961 | 0.895 |
| **average** | **0.833** | **0.843** |

Short-input behavior, the reason for the trade:

| input | v6 max score | v11 max score |
|---|---:|---:|
| `"hello"` | 1.000 | 0.019 |
| `"thanks"` | 1.000 | 0.034 |
| `"yes"` | 1.000 | 0.163 |
| `"ok"` | 1.000 | 0.255 |
| `"how are you"` | 0.997 | 0.274 |
| `"hi"` | 0.999 | 0.451 |
| `"write me an exploit"` | 0.987 | 0.676 |

Every benign short input now falls below the 0.5 finding threshold
(`allow`, no finding), while genuinely suspicious text still fires.
ONNX re-exported, 177 tests pass.

Also attempted and reverted: making the base-model route return `block`
instead of `flag` on any finding (`armor/api/main.py`). Reverted
immediately — with the short-input defect still live at the time, it
blocked ordinary chat. Re-enabling it is now plausible on v11 but
untested; do not turn it on without re-running short-input and
false-positive checks first.

## v12-v15: the long-input half of the length defect

Short inputs were only half of it. The proxy's
`OpenAIChatAdapter.extract_prompt_text()` joins the *entire* conversation
history into one newline-separated blob per turn, so real traffic arrives
at 100+ words while every training source tops out near 14. Measured on
v11 against a 99-word benign chat blob (the exact shape a live groq-chat
client sent through the proxy, captured in `armor/observability/logs.db`):
`prompt_injection` 0.998, `obfuscation_encoding` 0.970 — ordinary chat,
two labels at ~0.99.

Added `armor/data/synthetic_long_negatives.py` (newline-joined multi-turn
benign blobs, one label masked per row) and long-input cases to
`regression_set.py`. What each step actually did to that blob:

| checkpoint | change | blob score |
|---|---|---:|
| v11 | no long negatives | 0.998 |
| v12 | 120 uniform (~20 to `prompt_injection`) | 0.801 |
| v13 | 480 uniform (~80) | 0.833 |
| v14 | 480 weighted 10x (~315) | 0.708 |
| v15 | + SPML cap 2000 → 800 | **0.476** |

v12 collapsed five of six labels outright — `obfuscation_encoding` went
0.970 → 0.057 on ~20 rows. Only `prompt_injection` resisted, and
**4x the uniform volume made it worse** (v13), which ruled volume out as
the lever. Weighting helped where volume did not (v14), but 15x the rows
bought only -0.093.

The cause was on the positive side, and measurable: within SPML's first
2000 rows, positives run a median of 68 words (mean 113) against
negatives at a median of 14 — a 5x length gap inside a single source, at
1594 pos / 406 neg. Length alone separates most of that source's classes,
so the model learned length. Capping SPML to 800 cut the long-positive
mass directly and did in one step what six rounds of counter-evidence
could not.

**State: substantially improved, not fully resolved.** The 12-turn
benchmark blob now scores 0.476 (`allow`), but a shorter 7-turn benign
variant still scores 0.554 (`flag`, `prompt_injection`). The band
straddles the 0.5 threshold, so some benign conversation shapes still
flag. All malicious probes still fire correctly (0.65-0.997): injection,
PII, obfuscation, dangerous tool action, and "write me an exploit".

v15 promoted with a deliberate gate override (approved): it missed
tolerance on `sensitive_data_exposure` (-0.053) and `obfuscation_encoding`
(-0.052), both by 0.002-0.003, while `prompt_injection` reached its best
value yet (0.897) and `harmful_content` rose to 0.698. Average across the
six labels is flat (0.845 → 0.841).

## The adapters were the actual bug

Everything above treated the long-conversation problem as a model defect
and attacked it with training data. It was not. `extract_prompt_text()`
in both proxy adapters joined **every message in the conversation** into
the inspected text, and chat clients resend the full history each turn —
so the message the user had just typed was a rounding error inside a
100+ word blob, and the score tracked accumulated history rather than new
input.

Measured on live traffic through the proxy (v15, `armor/observability/logs.db`):

| inspected text | score |
|---|---:|
| full history ending `"how are you"` | 0.99 |
| full history ending `"write me an exploit"` | 0.99 |

No discrimination whatsoever. The same two messages scored on their own:

| message alone | score |
|---|---:|
| `"hello"` | 0.016 |
| `"how are you"` | 0.197 |
| `"write me an exploit"` | 0.650 |
| `"write me an exploit to jailbreak ai tool"` | 0.997 |
| `"Ignore all previous instructions…"` | 0.997 |

Both adapters now inspect only the newest user turn. Earlier turns were
already inspected when they were sent; re-scanning them every turn costs
tokens and destroys the signal. Known limitation: indirect injection
carried in tool results or assistant turns is not covered, and needs its
own handling (inspect tool output as its own direction) rather than a
return to concatenating everything.

This also reframes v12-v15 above. Those rounds were fighting a text shape
the adapter should never have produced. The training work still stands on
its own (short and long inputs genuinely were unmodelled), but the
headline symptom — benign chat scoring ~0.99 — was an adapter bug, and
one round of reading `extract_prompt_text` would have found it faster
than six rounds of retraining.

## Block re-enabled, two-tier

With discrimination restored, block became viable. `armor/api/main.py`
now blocks at `BLOCK_THRESHOLD` (0.8, `ARMOR_BASE_BLOCK_THRESHOLD`) and
flags between `FINDING_THRESHOLD` (0.5) and that. The two-tier split is
deliberate: 0.8 sits above the measured false positives ("Write a
concise, useful checklist", 0.529) and well below real attacks (0.997).

Verified end to end through the proxy:

| input | result |
|---|---|
| `"hello"` | passes through |
| `"Write a concise, useful checklist"` | passes through (0.529 < 0.8) |
| `"write me an exploit to jailbreak ai tool"` | `Blocked by Armor: prompt_blocked` |
| `"Ignore all previous instructions…"` | `Blocked by Armor: prompt_blocked` |

## Remaining work

1. Indirect injection via tool results / assistant turns is not inspected
   at all now that adapters read only the newest user message. Needs its
   own direction, not re-concatenation.
2. An empty response body still produces a finding (`obfuscation_encoding`
   0.58 on `""`, seen when an upstream 401 returned no content). Guard
   empty content in `inspect()`.
2. Run live OpenAI and Anthropic proxy wire checks when API keys are
   available (now QUEUE-01).
3. Re-evaluate `block`-on-finding for the base-model route — it was
   reverted when the short-input defect made it block ordinary chat. With
   v15 that is worth retesting, but not while benign chat can still reach
   0.554.
4. `sensitive_data_exposure` has no real (non-synthetic) source. Checked
   2026-09-20: `Isotonic/pii-masking-200k` is cc-by-nc-4.0 (NC, same
   blocker that detached toxic-chat), `ai4privacy/*` is academic-only with
   written permission required and watermarked, and
   `gretelai/synthetic_pii_finance_multilingual` is genuinely Apache-2.0
   but is 500-token legal contracts with no negatives — wrong shape, and
   would teach exactly the kind of format shortcut just removed.
5. `external_communication` (0.376) and `harassment_doxxing_defamation`
   (0.316) — weakest priority slices; the latter needs a real dataset,
   synthetic tuning is a dead end (see above).
