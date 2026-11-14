# Early-exit feature extraction probe (2026-09-25)

## Result

Feature extraction is not a multi-day part of this experiment. On the local development machine, one batched pass through ModernBERT layers 1–18 over the current action-feedback corpus took **32.18 s on CPU** and **17.14 s on MPS**. The pass captured the 768-dimensional CLS vector at layers 6, 12, and 18; no encoder training or backpropagation was run.

| Device | Rows | Time to layer 18 | Per row | Throughput |
|---|---:|---:|---:|---:|
| CPU, 4 intra-op threads | 1,196 | 32.18 s | 26.91 ms | 37.2 rows/s |
| MPS | 1,196 | 17.14 s | 14.33 ms | 69.8 rows/s |

These are development-machine measurements, not portable production SLOs. The CPU result is the relevant hardware-independent baseline. MPS is an optional local training accelerator; the deployed request path remains CPU by default.

## Corpus used for action-head extraction

The current `load_training_rows(include_pseudo_model=True)` path yielded **1,196 rows**:

| Label | Rows |
|---|---:|
| Allow | 383 |
| Block | 636 |
| Flag | 110 |
| Review | 67 |

Label provenance was 1,086 `pseudo_model`, 65 `human`, and 45 `reviewer_ai_b` rows. These are training labels, not an independent evaluation set or a claim of human-ground-truth accuracy. `load_training_rows()` with its safe default filters down to 65 rows; reproducing the broader experiment requires explicit opt-in to pseudo-model rows.

Token lengths were measured without truncation using the ModernBERT tokenizer:

| Statistic | Tokens |
|---|---:|
| Mean | 66.4 |
| Median | 30 |
| p90 | 119 |
| p95 | 170 |
| p99 | 699 |
| Maximum | 3,394 |
| Over 512 | 18 (1.5%) |
| Over 2,048 | 4 (0.3%) |
| Over 8,192 | 0 |

Tokenizing all 1,196 rows took **45 ms**. For feature extraction, rows up to 512 tokens were sorted and batched in groups of 64; the 18 longer rows were singleton batches. That avoids padding ordinary examples to the 3,394-token maximum. The measured extraction pass therefore included all rows in the model's 8,192-token context.

Three float32 CLS vectors per row require **11,022,336 bytes (10.5 MiB)** of feature-cache storage, before small labels/metadata. Three proposed `768 → 128 → 4` action heads add about **1.13 MB** of parameters. These heads use only intermediate CLS features; feeding them the existing six final-layer threat scores would require completing the encoder and erase the early-exit benefit.

## Separate ArmorBase label-training corpus

The current `armor/models/train.py` source aggregation assembled **20,153 examples** from cached public sources and generated examples. This is a separate six-label taxonomy dataset; direct four-action exits do not need to re-extract it if trained from the action-feedback rows above.

The base trainer truncates inputs to 512 tokens. Before truncation, token lengths had mean **117.8**, median **42**, p90 **326**, p95 **422**, p99 **873**, and max **4,674**. **663 rows (3.29%)** exceed 512 tokens; **17** exceed 2,048, and none exceed the encoder's 8,192-token context. After the cap, mean length is **107.1** and max is **512**. Tokenizing the full corpus took **0.84 s** without truncation and **0.75 s** with the trainer's 512-token cap. Avoid treating this set as full-context training data for long-prompt exits: 663 long examples lose their suffix under the existing training collator.

## Method and limits

- Model: locally cached `answerdotai/ModernBERT-base`, 22 layers, hidden size 768; PyTorch 2.14.0, Transformers 4.57.6. The deployed `head.pt` has encoder deltas only for layers 20–21 plus final normalization; the probe exits at layer 18, so those deltas do not affect the measured exit features or compute path.
- CPU used four intra-op threads. Timed output was inference-only. A forward hook captured CLS features after layers 6, 12, and 18 and stopped the diagnostic forward after layer 18.
- MPS had one warm-up batch excluded from elapsed time. The CPU run had no warm-up exclusion. Timing noise and other machine activity can move these small totals.
- This measures encoder feature extraction only. It does not measure head fitting, calibration, API overhead, or serving latency improvements. The feature pass has been established as cheap for the current 1,196-row action corpus; whether early exit is safe and how much request latency it saves remain to be evaluated.
- The historical v17 candidate-training report records 1,234 rows, whereas the current database loader returns 1,196. They are different snapshots. Pin and record the exact row manifest before training an exit-head candidate.
