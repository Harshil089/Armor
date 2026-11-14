# CPU latency investigation (2026-09-25)

## Decision

Keep the current PyTorch CPU backend. Fix the slow fast-scan rule at its source. Do not enable the existing ONNX export or the tested dynamic INT8 export: neither was a safe latency win on this host.

## Findings

- A 62,499-character logged prompt previously took 31.7 seconds end to end; a 62,498-character prompt took 30.2 seconds. Profiling reproduced the stall in `task_boundary_action()` at `_SECRET_VALUE_REQUEST.search()`. Two DOTALL positive-lookahead expressions independently scanned for keyword groups and could backtrack quadratically on long text that lacked either group.
- Replaced those lookaheads with two independent keyword searches. This preserves the same Boolean condition and policy behavior while making each scan linear in input length. After the change, the same local logged inputs took 23.0 ms (62,499 chars), 34.7 ms (62,498 chars), and 24.1 ms (43,298 chars) in `_run_fast_scan()`.
- Seven unique long prompts from the local log, 1,566–3,211 tokenizer tokens, took 2.09 seconds on average on the warm PyTorch CPU path (median 2.05 s, p95/max 4.06 s). All seven decisions matched their stored decisions. These are local development-machine numbers, not a production-host SLO.
- Live endpoint smoke test after restarting Armor: a 63,080-character explicit bypass was blocked in 56 ms; a 63,034-character benign input returned `review` in 2.18 s. That historical measurement did not isolate tokenization from model loading, logging, or the rest of the endpoint, so it does **not** establish that tokenization caused all 2.18 s. Both temporary test rows were removed afterward. The fast scan is no longer the bottleneck; very large undecidable inputs still need bounded handling.

## Backend comparison

Measured full `/v1/inspect` logic on the 70-item frozen long-text gate. Model load was excluded; the endpoint logger was disabled to avoid writing test prompts. The gate uses Jev AI labels, not human ground truth. Forty of 70 cases reached the encoder; the rest were decided by Fast Scan.

| Backend | Mean | p50 | p95 | Max | Accuracy vs AI labels | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch CPU | 88.64 ms | 121.74 ms | 195.55 ms | 339.13 ms | 87.14% | 0.6049 |
| Existing ONNX FP32 | 178.73 ms | 237.98 ms | 409.04 ms | 722.27 ms | 87.14% | 0.6049 |
| Temporary ONNX dynamic INT8 | 89.02 ms | 108.05 ms | 203.61 ms | 396.95 ms | 70.00% | 0.5021 |

ONNX FP32 made identical decisions but was about twice as slow on this 250–768-token gate. Dynamic INT8 did not improve mean or p95 latency and increased allow-to-block errors from 3/40 to 11/40. The quantized artifact was temporary and was not installed in Armor.

PyTorch intra-op thread counts of 1, 4, and 8 were also compared. One thread was slower (mean 124.95 ms, p95 298.21 ms); eight threads were effectively tied with the current four-thread run (mean 88.48 ms, p95 192.37 ms). That small p95 difference is within run noise, so no thread-count change was made.

## Architectural contributors

These explain why long requests cost more; the timing data above does not isolate the fraction caused by each one.

- **No adaptive compute:** after Fast Scan has no decisive result, `/v1/inspect` runs the encoder once over the full input. ModernBERT has 22 layers; there is no confidence-based early exit, so even an easy case pays for the full depth. ModernBERT's global-attention layers also compare tokens across the sequence, while its other layers use a bounded local window. More input tokens therefore increase encoder work throughout the stack.
- **Duplicate tokenization on one route (fixed in Generation A):** `predict_with_embedding()` tokenized content, then the long, uncertain model-only block guard tokenized it again to check the 256-token threshold. Both now consume the request's `TokenContext.token_count`.
- **Oversize detection followed full tokenization (bounded in Generation A):** Armor previously tokenized with `truncation=False`, then checked whether the count exceeded 8,192. It now rejects raw inputs above a configurable 65,536-character resource cap before tokenizer/model work. Below that cap, tokenizer output is limited to 8,193 tokens (one sentinel beyond the 8,192 model limit); that sentinel routes to `review` as `input_exceeds_model_context`. The raw-size cap routes separately as `input_exceeds_preflight_limit`, because Armor has not established an exact token count. This bounds tokenizer input and output, but the tokenizer still processes the complete under-cap string; it is not token-by-token interruptible. The 2.18-second endpoint measurement did not isolate this stage.
- **Fixed full-depth cost (staged interface now available, exits not promoted):** ArmorBase exposes embedding and layer-range continuation on the same PyTorch ModernBERT instance. An exact-output unit test checks parity with `ModernBertModel.forward()` and verifies that resuming does not rerun completed layers. The API still uses the legacy full-depth path; no early exit is enabled pending trained heads and frozen safety evaluation.
- **Training/serving context mismatch:** the base-model training collator truncates at 512 tokens, but inference passes up to 8,192 tokens to the encoder. The current base classifier therefore sees long inputs at serving lengths it was not exposed to in its standard training path. The long-prompt gates test some of this behavior, but they do not remove the mismatch in training examples.
- **Backend constraints:** production defaults to PyTorch CPU. The measured FP32 ONNX path was slower on the frozen long-text gate, while dynamic INT8 traded away accuracy without improving p95. There is no validated, portable CPU backend switch that currently removes the full-depth encoder cost.

## Verification and limits

- `pytest -q tests/test_request_scope.py tests/test_regex_complexity.py`: 27 passed.
- Full suite: 372 passed, 6 skipped.
- The large-input regression test checks that a 60 KB benign task-boundary scan completes in under one second; actual historical inputs completed in 23–35 ms.
- The 70-item gate is AI-labeled and small. The CPU benchmark is on this Apple Silicon development machine. Production latency still needs a matched benchmark on the intended server CPU and traffic mix. Long inputs that reach the encoder remain the main latency cost; do not truncate them to claim a faster decision.

## Adaptive inference refactor status (2026-09-25)

Generation A is implemented locally:

- `TokenContext` carries one request tokenization, token count, offsets, original input length, and stage timings.
- `/v1/inspect` passes that context into ArmorBase; the long-input policy guard reads the cached token count rather than calling the tokenizer again.
- Inputs over `ARMOR_MAX_PREFLIGHT_CHARS` (default 65,536 characters) return `review` before model loading with `review_reason=input_exceeds_preflight_limit`. Tokenizer output is capped at context limit + 1; the sentinel returns `review` with `review_reason=input_exceeds_model_context`, without running the encoder.
- SQLite stores additive `inference_trace` JSON for Fast Scan, preflight, tokenization, model load, encoder, classifier head, total latency, input size, token bucket, and compute-path metadata. Existing rows are preserved by an idempotent column migration.
- Focused verification after staged heads, windowing, API preflight, trace migration, and safety fusion: 29 tests passed. Staged encoder parity and no-recomputation checks are included.

The 65,536-character cap is a resource limit, not proof that each under-cap tokenizer call stops at token 8,193. A tokenizer-level early stop must preserve token-boundary semantics and be benchmarked before replacing this conservative cap.

### Live tokenizer call inventory

- **Required live call:** `ArmorBase.prepare_token_context()` tokenizes the full request once, after the raw character preflight. ArmorBase, the long-input guard, and telemetry consume its IDs/count/offsets.
- **Removed duplicate:** the long uncertain model-only block guard used to call `model.tokenizer(req.content)` again to test the 256-token rule. It now reads `TokenContext.token_count`.
- **Distinct secondary inspection:** when an explicit current task is extracted from a reference document, `main.py` calls `model.predict(task)` on that different task string. This is a second, scoped inspection, not a second tokenization of the same full request. It remains because current task and reference context are intentionally scored separately.
- **Optional, existing separate model path:** `armor/models/semantic_judge.py` owns its own tokenizer only when the pre-existing semantic-judge feature is explicitly enabled. This refactor does not add or call that model.
- **Training/test only:** the tokenizer in `armor/models/train.py` prepares training batches; tokenizer fakes in `tests/` assert call counts and limits. Neither is part of ordinary `/v1/inspect` tokenization.

### Generation B/C candidate results — not promoted

Generation B now has a staged ModernBERT interface and two tiny MLP exit heads at layers 6 and 12. Each head operates on the CLS state from the same encoder and produces six threat-logit estimates plus a safe-to-stop score. There is no second encoder or language model. `scripts/train_early_exit_heads.py` distilled the current full-depth ArmorBase classifier over 1,048 unique feedback prompts (960 `pseudo_model`, 55 `human`, 33 `reviewer_ai_b` provenance tags); labels are not used as teacher truth. The resulting candidate checkpoint is fingerprint-bound to the encoder and is only read by opt-in shadow experiments.

The internal 210-row validation split had 61 rows that the full-depth calibrated classifier labeled ALLOW. With asymmetric unsafe-exit loss weight 4, a threshold sweep found only 3/210 layer-6 candidates at a 0.4 safe score and 3/210 layer-12 candidates at a 0.5 safe score with zero teacher non-ALLOW candidates at each individual checkpoint. Those cutoffs are exploratory and were selected on the head validation split; they are not approved thresholds. With the default 0.999 confidence thresholds, neither checkpoint exited any example.

The frozen long-text evaluation corpus has 140 AI-labeled rows (not human ground truth), with a maximum of 785 tokens. It does not exercise the 1K–8K range. Its single-checkpoint policy used the exploratory 0.4/0.5 cutoffs and exited 5/140 requests. One of those changed a full-depth BLOCK to early ALLOW. Patience mode exited none. These results reject early-exit activation.

| Candidate path | Mean | p50 | p95 | p99 | Max | Agreement with full path | Block recall vs AI labels | Block-labeled ALLOWs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A: full-depth | 135.4 ms | 125.4 ms | 211.0 ms | 310.6 ms | 319.8 ms | 100.0% | 3.8% | 46 |
| B: adaptive depth, patience | 180.8 ms | 169.2 ms | 245.6 ms | 351.5 ms | 372.7 ms | 100.0% | 3.8% | 46 |
| B: adaptive depth, single checkpoint | 177.7 ms | 170.1 ms | 250.5 ms | 354.4 ms | 373.4 ms | 99.3% | 1.9% | 47 |
| C: adaptive windows, patience | 207.5 ms | 169.2 ms | 696.9 ms | 834.6 ms | 849.7 ms | 94.3% | 11.3% | 42 |
| C: adaptive windows, single checkpoint | 204.8 ms | 170.1 ms | 694.1 ms | 833.3 ms | 901.6 ms | 93.6% | 9.4% | 43 |

The comparison excludes model loading and measures tokenizer time separately. It is one warm local CPU run on an Apple Silicon development machine, not a production SLO. The full-path block recall is already low against this AI label set; these model-only numbers are not human-gold accuracy. Windowed inference reduced block-labeled ALLOWs in this corpus, but also created false blocks and was about 3.3× the full-path p95 in patience mode. It does not meet the latency gate. The 129–512 bucket contains 132/140 rows, so the corpus is especially weak evidence for genuinely long-input performance.

The window path uses existing token IDs/offsets and does not retokenize windows. It runs a shallow window batch, keeps hidden states for unresolved windows, continues selected windows, optionally builds bounded context-expanded regions, and fuses evidence conservatively. Uninspected windows and exhausted window/deep-window/context/time budgets route to REVIEW. Critical deterministic blocks remain authoritative. `ARMOR_ADAPTIVE_SHADOW=1` records candidate behavior while keeping the actual full-depth decision. Production defaults remain unchanged.

**Result:** Generation A's bounded preflight and token reuse are active in the code. Generation B/C mechanics exist, but their safety/latency gates fail on the current evidence. No early exits or selective windows are promoted, and no p95 improvement is claimed. Before promotion, expand the frozen gate to 1K–8K with beginning/middle/end, boundary-split, adjacent-context, one-attack-many-benign-windows, benign oversize, and malicious oversize cases; then improve safe-head coverage without any protected block-recall loss. Full per-sample traces, gate metrics, and head-training report are in `docs/adaptive_inference_gate_{single,patience}_20260925.json` and `docs/early_exit_candidate_20260925.json`.

## Frozen 1K–8K baseline — 2026-09-25

The earlier 140-row gate reached only 785 tokens and did not test the serving-length mismatch. A new pinned **105-row** challenge corpus now has 21 rows per bucket (1K, 2K, 4K, 6K, 8K), five same-carrier attack-position pairs per bucket, and a 512-token-boundary split case in every bucket. It also covers benign long text, separated action/target, quoted dangerous text, defensive security analysis, distributed indicators, obfuscation, ambiguity, and fabricated sensitive-data values. Its OASST1 carrier revision is pinned; every label is tagged `synthetic` with an explicit scenario-contract basis. It is **not human ground truth**. See [the frozen corpus](long_context_gate_v1_20260925.jsonl), [manifest](long_context_gate_v1_20260925_manifest.json), and [integrity-checked baseline report](long_context_full_depth_baseline_v1_20260925.md).

The current full-depth `/v1/inspect` path was measured with the local production ArmorBase checkpoint and policy action head, adaptive shadow/windowing disabled, PyTorch CPU, on an Apple M2 with 8 GB shared memory. Two warmups were excluded and prompt bodies were not persisted in benchmark logs. Latency is API compute time before the separate SQLite write.

| Tokens | Mean | p50 | p95 | Max | Encoder mean |
|---|---:|---:|---:|---:|---:|
| 1K | 678 ms | 677 ms | 720 ms | 759 ms | 643 ms |
| 2K | 1.40 s | 1.91 s | 2.08 s | 2.12 s | 1.36 s |
| 4K | 5.23 s | 6.07 s | 6.47 s | 7.74 s | 5.16 s |
| 6K | 10.48 s | 12.93 s | 13.42 s | 14.73 s | 10.40 s |
| 8K | 17.54 s | 20.39 s | 21.35 s | 28.15 s | 17.43 s |

Across all 105 rows, mean latency was 7.06 s, p50 4.87 s, p95 20.76 s, p99 21.34 s, and max 28.15 s. Encoder mean was 7.00 s; tokenization mean 15 ms, fast scan mean 22 ms, classifier head mean 0.33 ms, and the two small policy heads together averaged 1.25 ms. The long-input bottleneck is the full-depth encoder, rather than tokenization or the policy heads.

The production-action path agreed with 34.3% of the synthetic scenario contracts (macro F1 21.3%, block recall 40.0%, flag recall 0%). These are **agreement scores against authored challenge contracts**, not independent accuracy or ground truth. The frozen set is intended for paired architectural comparisons and position-stability checks. Do not tune the model to these rows, and do not use these agreement numbers as promotion evidence by themselves.

### Long-context training feasibility probe

The existing 20,153-row six-label training mix was previously tokenized without truncation: mean 117.8 tokens, median 42, p95 422, p99 873, max 4,674; 663 rows exceed 512 and 17 exceed 2,048. The default training collator still truncates at 512. A synchronized MPS feature-extraction probe on this M2 measured roughly 94 ms at 512, 232 ms at 1K, 576 ms at 2K, 1.72 s at 4K, 3.47 s at 6K, and 5.39 s at 8K. A two-row 8K batch took 11.76 s (5.88 s/row), slower than the measured singleton, so long examples should remain singleton feature batches until a larger batch is shown to improve throughput safely.

Under the proposed 40/25/15/10/7/3 curriculum and a deliberately conservative serial extrapolation, one frozen-feature pass over 20,153 examples is estimated at 4.25 hours on this device. This is a feasibility estimate from one pass per length, not a training measurement or production hardware result.

Two candidate-only head recalibration attempts completed feature extraction but failed the existing per-label regression gate: the fresh head regressed five protected labels; the warm-started head at `1e-4` regressed prompt injection, sensitive data, and obfuscation. The third warm-started head-only attempt at `1e-5` resumed on MPS from batch 10,500 and completed all 12,218 feature batches. Exact long-row shapes were bucket-padded to 128-token boundaries; MPS progress now prints every 25 rows and snapshots save every 100. The feature cache completed at about 63 MB. The new head failed the gate on prompt injection (0.784 → 0.679) and obfuscation encoding (0.851 → 0.755); other tracked labels did not regress. No candidate was promoted, and the deployed checkpoint and serving path remain unchanged. A later head-only attempt can reuse the completed feature cache when its dataset and base-checkpoint hashes match, skipping the expensive encoder pass. See [the MPS training run and Colab MCP findings](mps_training_and_colab_mcp_20260926.md) and [the curriculum manifest](long_context_training_curriculum_v3_20260925.json).
