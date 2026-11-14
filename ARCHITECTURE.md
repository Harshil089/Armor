# Armor Architecture

Current serving state as of 2026-09-25; historical experiments retain their
original dates and decisions. This describes what's actually running,
not the original design spec (`docs/Armor_Architecture_and_Design_v0.1.docx`,
which predates most of what's below) — see `README.md`'s Open Work Queue
for what's still in flux and why.

## Inference cost architecture and current refactor (2026-09-25)

### Architectural bottleneck

The request path has a fixed-compute shape: deterministic Fast Scan either
returns a decisive result or `ArmorBase` tokenizes the request and runs the
same 22-layer ModernBERT stack over the whole token sequence. The six-label
head, policy head, and subsequent decision code receive the result only after
the encoder has completed. A clear benign request and an ambiguous request
therefore spend the same encoder depth. Encoder work also grows with sequence
length: every layer performs token-wise attention projections and MLP work;
ModernBERT's global-attention layers additionally relate tokens across the
sequence, while local-attention layers use a bounded neighborhood. Armor has
no current selective window path, so a long request drives this work through
all 22 layers.

There were two avoidable pre-encoder costs and one data-shape flaw:

1. One model-only long-input guard tokenized the same request again just to
   test its length.
2. Oversize detection happened after unrestricted tokenization, so a request
   that could only end in `review` still paid for a full tokenizer pass.
3. Training normally truncates at 512 tokens while serving permits up to
   8,192. A single CLS vector is then asked to summarize a much longer text
   than the usual training examples, without the window sampling and context
   expansion needed to test whether local malicious evidence survives.

Backend changes do not remove these architectural costs: the measured ONNX
FP32 CPU path was slower and dynamic INT8 did not improve p95 while reducing
accuracy. PyTorch CPU remains the validated serving backend.

### Generation A implemented locally

`armor.models.armor_base.TokenContext` now owns the model input IDs, attention
mask, one token count, offsets, input length, and timing data for a request.
`/v1/inspect` passes it through the full-depth ArmorBase call, and the long
input guard reads its cached count. The tokenizer is not called a second time
for that guard.

Oversize handling has two explicit boundaries. Raw input above the default
65,536-character cap (`ARMOR_MAX_PREFLIGHT_CHARS`) returns `review` with
`input_exceeds_preflight_limit` before model loading. Under the raw cap, the
tokenizer's returned sequence is capped at 8,193 tokens; token 8,193 is the
overflow sentinel and routes to `review` with
`input_exceeds_model_context`. No partial prefix is sent to ModernBERT. The
tokenizer still processes the complete string below the raw cap, so this is a
bounded request-size safeguard, not an interruptible tokenizer. Do not claim
that Armor stops tokenization exactly at token 8,193.

Inference traces are stored as an additive JSON column on the existing log
table. They carry stage timings and input/compute metadata without adding
prompt contents to metrics. The schema migration is additive and idempotent.

### Staged same-model interface and shadow candidate

`ArmorBase.begin_encoder()` runs ModernBERT embeddings once and prepares the
same global/sliding masks and positions used by its normal forward path.
`continue_encoder(state, end_layer)` mutates the retained hidden state and
executes only the next layer range. The final norm is applied only after the
final encoder layer, retaining full-depth pooling semantics. Unit tests compare
staged output with the full model output and count layer calls to prove no
recomputation. This interface is PyTorch-only; the ONNX path has no staged
hidden-state interface and remains on its existing full-depth route.

The API still serves the original full-depth path; staged execution is
available only through the opt-in shadow path. Two trained, tiny MLP exit
heads now exist at layers 6 and 12, each attached to this same encoder's CLS
state and producing six distilled threat logits plus a safe-to-stop score.
They are only a candidate artifact; they are not another encoder or language
model. `scripts/train_early_exit_heads.py` fits them against the current
full-depth ArmorBase classifier, with a validation split over exact-deduped
feedback prompts. That is self-distillation, not ground truth. The loaded
head checkpoint is tied to the ArmorBase encoder fingerprint so it cannot
silently outlive its teacher.

`ARMOR_ADAPTIVE_SHADOW=1` enables a comparison path. It runs layers 1–6,
records layer-6 and layer-12 exit candidates, continues the retained hidden
state through layer 22, and preserves the full-depth response. For requests
over the initial 512-token training context, it also runs a shadow candidate
with overlapping 512-token windows (96-token overlap), selective depth,
bounded context expansion, and request-level evidence fusion. These are
experimental defaults, not promoted serving settings. `ARMOR_EXIT_POLICY`
selects a single-checkpoint or two-checkpoint patience test; exit confidence,
window count, deep-window, context-expansion, and time budgets are
configurable. Unresolved windows and exhausted budgets yield `review` in the
candidate path. A deterministic critical block still wins request fusion.

**Measured candidate status: not promoted.** The training fit used 1,048
unique in-context rows (960 tagged `pseudo_model`, 55 `human`, 33
`reviewer_ai_b`); all targets came from the existing full-depth model, not
those row labels. On a 210-row internal validation split, the most permissive
tested thresholds with zero disagreements against the teacher reached only
3/210 candidates at layer 6 and 3/210 at layer 12. On the frozen 140-row
long-text corpus (AI labels, no human ground truth; maximum 785 tokens), the
patience policy produced no early exits. The single-checkpoint policy exited
5/140 times and changed one full-depth `BLOCK` into early `ALLOW`. It is
rejected on the safety gate.

On that same corpus, candidate-path p95 latency (tokenization separately
timed, model load excluded) was 211 ms for full depth, 246 ms for patience
adaptive depth, and 697 ms for adaptive windows. The windowed path continued
16 original windows to layer 22 across the corpus, and ran bounded context
expansion for some positive windows; shallow exits did not occur often enough
to pay for window setup. It changed labels relative to the full-depth
classifier and is not promoted. The opt-in API shadow path runs the full
reference and candidate work, so its latency is higher than these isolated
candidate timings. The corpus does not test any input above 785 tokens, so it
cannot support claims about the 1K–8K target range.

Production remains on the original full-depth PyTorch path. Generation A
does remove duplicate tokenization and returns `review` before model load for
oversize input. Generations B/C have an implementation and a shadow harness,
but they have **not** demonstrated lower p95 or protected block recall. The
next safe engineering work is to build a frozen 1K–8K long-input gate and
improve intermediate-head agreement/coverage against that gate; do not
activate exits or windows until both compute savings and the block-recall
guardrail pass. Full details and per-bucket numbers are in
[`docs/performance_latency_cpu_20260925.md`](docs/performance_latency_cpu_20260925.md).

## Decision architecture change: from threat scores to shared representation

**Previous limitation.** The bandit (and, briefly, a local-LLM semantic
judge bolted onto it) tried to make nuanced enforcement decisions from 6
scalar threat scores plus a keyword-regex framing feature. `ModernBERT`'s
own 768-dim pooled embedding — the actual semantic representation of the
request — was computed by `armor_base.py`'s `predict()` and discarded
every single call. Twelve red-team rounds proved this is a genuine
representation ceiling, not a data-volume one: "how do I access X?" can
mean completely different things depending on ownership, authorization,
and context that 6 numbers can't carry, so opposite-answer requests
produced near-identical vectors (measured: `dangerous_tool_action` 0.625
for both a `flag` case and a `block` case, `framing_signal` 0 for both).
More volume of the same representation made the bandit worse, not better
(76%→66% overall agreement) — see `queue/QUEUE-02` for the full history.

**Initial data flow (2026-09-22 shadow-mode design; superseded locally on 2026-09-24).**
```
ModernBERT encoder (one forward pass, reused, not duplicated)
        │
        ├──────────────┐
        ▼              ▼
  6 threat scores   pooled embedding (768-dim)
        │              │
        └──────┬───────┘
               ▼
         Policy head (small MLP, 768+6 → 128 → 4)
               │
               ▼
     softmax margin confidence
               │
      ┌────────┴────────┐
      ▼                 ▼
  confident          uncertain
      │                 │
      ▼                 ▼
 predicted action     REVIEW
      │                 │
      └────────┬────────┘
               ▼
   initial shadow recommendation (policy_action, policy_confidence,
   review_reason) — this was pre-deployment behavior
```

**Policy-head design**: `armor/models/policy_head.py` — one hidden layer
(128 units, ReLU), 4 output logits (`allow`/`flag`/`block`/`review`).
Deliberately small: a lightweight head, not a second large model.
Confidence is a softmax margin between the top-2 predicted actions
(`armor/models/uncertainty.py`) — genuinely low margin abstains to
`review` rather than forcing an arbitrary tie-break; this is what makes
`review` a real uncertainty outcome instead of an unused label.

**Review flow (2026-09-24 development path)**: proxy prompt/response
`review` decisions are sent to the localhost TypeSafe/Jev bridge only when
`ARMOR_JEV_REVIEW_ENABLED=1`. A valid explicit `allow` continues; `block`, an
invalid answer, timeout, or bridge failure blocks. Only an actual Jev verdict
is stored against the inspect row as `reviewer_ai_b`; fail-closed fallback
blocks are not mislabeled as Jev decisions. `GET /admin/reviews` lists only
unresolved production `decision='review'` inspect rows, excluding shadow-head
recommendations and duplicate proxy summaries. The existing decision-feedback
columns store the adjudication, and `--include-jev-reviews` makes those
provenance-tagged pseudo-labels available to an offline policy-head candidate
run. They remain distinct from human ground truth and do not trigger online
retraining or automatic checkpoint promotion. Streaming output holds each
uninspected batch until review completes; earlier approved batches may already
have reached the client before a later batch is blocked. The sandbox bridge
and launch setup are documented in `dev-tools/typesafe_labeler/README.md`.

**Review audit flow**: manual corrections still use the existing
`POST /admin/logs/{id}/correct` endpoint and persist explicit `label_source`
metadata (for example, `human` or `reviewer_consensus`). Training rows retain
their source tags.
The v15 candidate used 57 prior rows tagged `human`, 917 tagged
`pseudo_model`, and 105 new AI-reviewer consensus rows; these source tags
are provenance, not a claim that the AI-only v16 evaluation is human ground
truth.

**Bandit's role, unchanged in Phase 1.** Still runs in parallel shadow
mode with its own 12-dim context, logged separately (`bandit_action`) —
kept for direct comparison against the policy head, not replaced yet.
Repurposing it to adaptive inspection routing (§ Migration roadmap below)
is a later phase, not done.

**Migration status (historical decision as of 2026-09-22)**: the policy
head was then shadow-only and had not been promoted. The 2026-09-24 local
deployment status and fresh-gate evidence are recorded at the end of this
document and in `docs/broad_accuracy_resolution_20260923.md`.

**Evaluation results** — three separate measurements, in increasing order
of rigor (full detail and methodology in `queue/QUEUE-02`'s Phase 1/2
entries):

| Evaluation | Policy head | Naive-allow | Block recall |
|---|---|---|---|
| Training-adjacent 80/20 split (Phase 1's original claim) | 76% | 30% | 88% |
| Frozen holdout (independently-authored cases, never trained on) | 68% | **74%** | 84% |
| Ambiguity subset (the hard, previously-collapsed cases, held out) | **52%** | 31% | 83% |

The first number was optimistic — train and test shared distribution (both
drawn from the same red-team-generated pool). On a genuinely independent,
allow-heavy holdout, naive-allow actually wins on raw accuracy. On the
specific cases this whole change exists to fix (the ambiguity subset), the
policy head wins clearly. Block recall stays strong (83-88%) everywhere,
consistently far above naive's structural 0% — the value is real but
concentrated on threat-catching and hard-case separation, not a uniform
win, likely because `flag`/`review` training volume remains thin (8 and 57
rows respectively in the ambiguity holdout).

### Reformulation experiment: hierarchical policy head (tried, didn't help)

Since more data and reweighted loss both plateaued on `flag`/`review`, a
follow-up hypothesis was tested: maybe the flat 4-class softmax
*formulation* was the bottleneck, not the data. Built and A/B/C-tested a
replacement — two binary heads (`armor/models/policy_head.py`'s
`BinaryPolicyHead`: "does this need intervention," then "does it need
blocking") plus conformal prediction sets (`armor/models/conformal.py`,
from-scratch split-conformal, no new dependency) and OOD detection
(`armor/models/ood.py`, nearest-neighbor distance to training embeddings)
so `review` becomes genuine abstention rather than a predicted class.
Evaluated on the exact same frozen holdout and ambiguity subset as every
other measurement in this document (formalized into
`armor/eval/policy_eval.py`), no new data generation.

**Result: negative.** The hierarchical head made `flag` recall *worse*,
not better (18%→8% and 20%→3% across the two holdouts) — directly against
the hypothesis. Adding conformal+OOD on top was a clear failure: block
recall (the protected metric) dropped 79%→54% on the ambiguity subset,
with over half of all cases routed to `review` — excessive abstention, not
useful uncertainty. Neither system is promoted; `armor/api/main.py` logs
the hierarchical head's recommendation in shadow mode (harmless,
zero production effect) purely for continued observation. One caveat
worth naming: conformal calibration used the training set itself rather
than a genuinely held-out slice, which likely overstates its
over-abstention problem somewhat — but that doesn't explain the
hierarchical head's `flag` regression, which had no conformal layer
involved at all. Full numbers and methodology in `queue/QUEUE-02`'s
2026-09-22 reformulation-experiment entry.

**What this means**: the flag/review bottleneck really is a data problem
(volume and/or quality of ~40-100 rare-class examples), not a formulation
one — a genuine architecture change was tried and didn't move it, which is
stronger evidence than not having tried.

### Why architecture experimentation stopped, and the human-label audit

Per the finding above, architecture changes were paused (no new heads,
losses, or embeddings) and the remaining hypothesis — supervision
quality/volume — was tested directly. Built: a versioned rubric
(`docs/policy_rubric_v1.md`), a review-audit store separate from the
bandit's own logging (`armor/observability/review_store.py`), stratified
sampling with embedding-based dedup (`armor/eval/audit_sampling.py`),
from-scratch agreement metrics including Cohen's kappa
(`armor/eval/agreement.py`), and an active-learning candidate miner for
future large-scale labeling (`armor/eval/active_learning.py`, reuses
`ReferenceBank`/`OODReference` entirely — no new embedding
infrastructure).

**Scoping honesty**: the spec called for two independent human reviewers.
Asked the user; the actual reviewers this round were Claude (fresh
subagent, blind) and Jev (the new rubric, not the old bandit-correction
prompt) — **this tests whether the taxonomy holds up across independent
AI judges, not human ground truth**, and every label is tagged
accordingly (`reviewer_ai_a`, not `human`) rather than silently upgraded.

**An unplanned finding**: 18 of 85 sampled requests hard-refused by
Claude's own safety classifier during review — concentrated almost
entirely in `flag`(9)/`review`(8), barely touching `block`(1). The
content that makes these classes hard to label (dual-use framing, encoded
payloads, ambiguous authorization) is also what an AI safety classifier
is most likely to refuse outright — a structural argument that this
specific rare-class subset needs real human reviewers, not just a
convenience preference.

**Result**: 63% raw agreement between the two AI reviewers (Cohen's kappa
0.497 — "moderate," not "substantial"), with `flag`/`review` agreement
cratering to 30-31% (near chance) while `allow`/`block` held at 50-70%.
**Current evidence indicates** this points toward taxonomy/rubric
instability at the `flag`/`review` boundary specifically, on top of the
already-established volume problem — not mathematically proven on a
67-sample audit. 25 disagreements are queued for real human adjudication
(`docs/audit_disagreements_for_adjudication.md`), not fabricated. Full
numbers and methodology in `queue/QUEUE-02`'s 2026-09-22 human-label-
audit entry.

**Status at the 2026-09-22 checkpoint (historical):** large-scale
human-gold-dataset collection, baseline retraining on it, and the A/B/C
supervision comparison had not started; that plan depended on real human
adjudication. The user later explicitly declined human ground truth. The
2026-09-24 v17/v18 work instead used disclosed AI-reviewer consensus and
passed a fresh AI-only promotion gate; see the current deployment record
at the end of this document and `docs/broad_accuracy_resolution_20260923.md`.

### Historical migration roadmap (decision as of 2026-09-22)

At the 2026-09-22 checkpoint, the phases below were **blocked on more diverse/balanced training data**
(specifically `flag`/`review`, confirmed twice now — once by ruling out
loss weighting, once by ruling out the flat-softmax formulation itself) —
not on more engineering work. This is historical planning context; later
implementation and promotion status is recorded in the 2026-09-24 section
at the end of this document.

3. **Enable `review` for real** — at the 2026-09-22 checkpoint, `review`
   only came from a shadow recommendation; the production path had not yet
   returned it. The current prompt policy can return `review`, and the proxy
   holds those requests for review.
4. **Reduce global-threshold reliance** — `BLOCK_THRESHOLD`/
   `FINDING_THRESHOLD` stay as deterministic guardrails throughout, but
   stop being the *primary* mechanism once the policy head is trusted.
5. **Repurpose the bandit** — from choosing the final action to choosing
   *what additional inspection* (PII-NER, a specialist detector, deeper
   semantic inspection, direct escalation) gives the best security value
   per request. Different action space, different reward function —
   genuinely new work, not a parameter change.
6. **Streaming rolling-buffer inspection** — extend `armor/proxy/app.py`'s
   per-event SSE inspection with a bounded rolling buffer so a violation
   spanning multiple streaming events (not just one) gets caught. Not
   built — deferred alongside the phases above.

## Overview

Armor inspects text (a user prompt going to an LLM, or a model's response
coming back) and returns one of four decisions: `allow`, `flag`, `block`,
`review`. It sits in front of an LLM provider as a reverse proxy, or is
called directly as a standalone inspection API. No external network calls
are made at inspection time — every model Armor runs is local.

```
                              ┌─────────────────────┐
                              │   POST /v1/inspect   │
                              │  (content, direction)│
                              └──────────┬───────────┘
                                         │
                                         ▼
                          ┌───────────────────────────┐
                          │   Fast Scan Layer          │
                          │  (deterministic, regex +   │
                          │   checksum, no ML)         │
                          └──────────┬─────────────────┘
                    high-severity    │    everything else
                    structured PII   │
                    → block          │
                                     ▼
                    ARMOR_USE_BASE_MODEL=1 ?
                          │                    │
                         no                   yes
                          │                    ▼
                          │        ┌───────────────────────┐
                          │        │  ArmorBase classifier  │
                          │        │  (ModernBERT-base,     │
                          │        │   149M params, 6-label)│
                          │        └──────────┬─────────────┘
                          │                   │ raw scores
                          │                   ▼
                          │        ┌───────────────────────┐
                          │        │  Venn-Abers calibration│
                          │        │  (isotonic regression) │
                          │        └──────────┬─────────────┘
                          │                   │ calibrated scores
                          │        ┌──────────┴───────────┐
                          │        │  score >= ESCALATE_LOW│
                          │        │  and < FINDING_THRESH?│
                          │        └──────────┬───────────┘
                          │                  yes│no
                          │                    ▼ │
                          │        ┌────────────────────┐ │
                          │        │  PII-NER escalation │ │
                          │        │  (spaCy, unstructured│ │
                          │        │   PII only)          │ │
                          │        └──────────┬───────────┘ │
                          │                   │              │
                          ▼                   ▼              ▼
                    ┌──────────────────────────────────────────┐
                    │      decision = allow / flag / block       │
                    │   (BLOCK_THRESHOLD / FINDING_THRESHOLD)     │
                    └──────────────────┬───────────────────────┘
                                       │
                          max(scores) < BANDIT_FLOOR (0.95)?
                                       │yes
                                       ▼
                          ┌─────────────────────────────┐
                          │  Decision bandit (LinUCB)     │
                          │  SHADOW MODE — logs a         │
                          │  recommendation, never         │
                          │  controls the response          │
                          └─────────────────┬─────────────┘
                                            │
                                            ▼
                          ┌─────────────────────────────┐
                          │   logs.db (SQLite)            │
                          │   decision + bandit_action     │
                          │   logged separately             │
                          └─────────────────────────────┘
```

## Request path (`armor/api/main.py`)

1. **Fast Scan Layer** (`armor/detectors/registry.py`) — deterministic
   regex + checksum-validated detectors (credit cards, AWS keys, IBANs,
   etc.), locale-aware (`global`/`IN`/`US`/...). Zero ML, always runs
   first. A high-severity structured-PII match blocks immediately, no
   further inspection needed.
2. **ArmorBase classifier** (`armor/models/armor_base.py`) — ModernBERT-
   base (149M params) + a linear classification head, fine-tuned/trained
   via `armor/models/train.py`. Six labels, multi-label (not mutually
   exclusive): `prompt_injection`, `jailbreak`, `sensitive_data_exposure`,
   `dangerous_tool_action`, `harmful_content`, `obfuscation_encoding`
   (`armor/models/labels.py`). Opt-in via `ARMOR_USE_BASE_MODEL=1` — off
   by default so tests/CI don't pull the model. CPU inference,
   ~24-30ms/call; an optional ONNX Runtime backend
   (`ARMOR_BASE_BACKEND=onnx`) is ~21% faster, needs a one-time
   `export_onnx` run.
3. **Score calibration** (`armor/models/calibration.py`) — Inductive
   Venn-Abers Predictors, pure-numpy PAVA isotonic regression, no
   `scikit-learn` dependency. Corrects for classifier noise on
   out-of-distribution benign text (raw scores can be miscalibrated —
   e.g. 0.72 on a completely benign sentence). Keyed by
   `checkpoint_fingerprint()` (head + encoder hash), cached to
   `armor/models/calibration_cache/<fingerprint>.npz`. Requires real
   positive-class density per label (≥150, enforced by a hard guard) —
   too few positives lets isotonic regression get swamped by hard
   negatives and crush genuine attacks toward zero (a real regression
   found and fixed 2026-09-22, see `docs/archive/queue/ARCHIVE-QUEUE-07_venn_abers_calibration.md`).
4. **PII-NER escalation** (`armor/models/pii_ner.py`) — spaCy
   `en_core_web_sm`, catches *unstructured* PII (names, addresses) the
   regex layer can't pattern-match. Only invoked for scores in a narrow
   ambiguous band (`ARMOR_ESCALATE_LOW` to `FINDING_THRESHOLD`) — not run
   on every request.
5. **Decision thresholds**: `FINDING_THRESHOLD` (0.5, default) separates
   `allow` from `flag`; `BLOCK_THRESHOLD` (0.8) separates `flag` from
   `block`. One global threshold shared across all 6 labels — the known
   limitation this whole architecture is working around (see `docs/
   CHAT_IMPROVEMENTS_AND_CHANGES.md` and
   `docs/archive/queue/ARCHIVE-QUEUE-04_encoder_finetuning_evaluation.docx` for the v41-v53
   retrain history that tried and failed to fix this by retraining
   instead).
6. **Decision bandit** (`armor/models/bandit.py`) — LinUCB contextual
   bandit, **shadow mode only**: computes and logs a recommendation
   (`allow`/`flag`/`block`/`review`) alongside the real decision, never
   controls the response. Only invoked below `BANDIT_FLOOR` (0.95) — the
   genuinely ambiguous middle, not cases the threshold system already
   resolves confidently. Context vector is 12-dim: 6 threat scores +
   `framing_signal` (regex-based dual-use-framing detector) + 5 semantic
   features (currently always zero — see below). Learns online from
   analyst corrections via `POST /admin/logs/{id}/correct`; state
   persists to `armor/models/bandit_state/current.npz`.
7. **Semantic judge** (`armor/models/semantic_judge.py`) — a local
   `Qwen2.5-0.5B-Instruct` model that scores 5 scalar features
   (authorization confidence, target sensitivity, harm potential,
   actionability, ambiguity) for the bandit's context, gated to a narrow
   sub-band (`ARMOR_SEMANTIC_JUDGE_LOW` to `BANDIT_FLOOR`). **Disabled by
   default** (`ARMOR_SEMANTIC_JUDGE_ENABLED=0`) — a full replay of 1043
   logged decisions measured it making the bandit worse, not better
   (block recall 94%→74%). Superseded by the policy head below, which
   fixes the same problem at the source instead of bolting features onto
   the bandit's existing 6-score context.
8. **Policy head** (`armor/models/policy_head.py`,
   `armor/models/uncertainty.py`, `armor/policy/engine.py`) — the active
   prompt decision head when `ARMOR_BERT_POLICY_HEAD=1`. Consumes
   ModernBERT's pooled embedding directly (via
   `ArmorBase.predict_with_embedding()` — same forward pass, zero
   duplicate encoder calls) instead of the 6 scalar scores alone, and
   predicts an action with an explicit confidence/abstention gate
   (softmax margin; low margin → `review` rather than an arbitrary
   argmax). Logged (`policy_action`, `policy_confidence`, `review_reason`)
   and supplies the action to `InspectResponse`; explicit fast-scan and
   authorization guards still run first. The active checkpoint and
   validation are summarized below. The hierarchical experiment remains
   shadow-only and is evaluated separately.

Every inspection is logged to `logs.db` (`armor/observability/log_store.py`,
SQLite) regardless of path taken: content, decision, route, findings,
latency, and (when applicable) the bandit's shadow recommendation.

## Proxy layer (`armor/proxy/`)

A reverse proxy in front of real LLM providers (`armor/proxy/app.py`),
config-driven (`armor/proxy/config.py` reads `configs/proxy/providers.json`
— 12 providers seeded: OpenAI, Anthropic, Groq, Together, Fireworks,
Mistral, DeepSeek, OpenRouter, Gemini's OpenAI-compat endpoint, Perplexity,
Cerebras, xAI). Adding an OpenAI-compatible provider is a JSON-only change
(`POST /admin/providers`, no code, no restart).

- **Prompt leg**: inspects the newest user turn before forwarding
  upstream. A `block` decision stops the request before the provider is
  ever called — verified to actually happen, not just assumed.
- **Response leg, non-streaming**: inspects the full response before
  returning it to the caller.
- **Response leg, streaming**: parses SSE events (`_drain_sse_events`)
  and inspects per-event, not per-raw-network-chunk (a provider can bundle
  multiple events into one chunk — checking only the whole chunk would
  miss a violation split across the boundary). A kill-switch event plus
  `[DONE]` is emitted mid-stream if a block condition fires; bounds
  exposure to one inspection interval, not a full pre-buffer (a genuine
  zero-exposure guarantee would defeat streaming entirely).
- **Adapters** (`armor/proxy/adapters/`): `openai.py` and `anthropic.py`
  implement a shared `ProviderAdapter` protocol (`base.py`) —
  `extract_prompt_text`, `extract_response_text`, and their streaming
  counterparts. Both extract tool-call/function-call content (name +
  arguments), not just plain text — a response consisting only of a tool
  call previously inspected as an empty string and always allowed.
- **Fail-open configuration**: `fail_open_prompt`/`fail_open_response` —
  an inspection *exception* (not a `block` decision) can be configured to
  pass the request through rather than hang or 500. Distinct from a real
  `block`, which always stops the request regardless of this setting.

## Admin (`armor/proxy/admin.py`, `armor/proxy/admin_logs_ui.html`)

- `GET /admin/logs-page` — browsable log viewer, shows decision, route,
  findings, and (when present) the bandit's shadow recommendation.
- `POST /admin/logs/{id}/correct` — records an analyst's correction
  (`allow`/`flag`/`block`/`review`) against a logged bandit recommendation,
  computes the reward (`armor/models/bandit.py`'s asymmetric penalty
  table — a false negative costs far more than a false positive), and
  updates the live bandit in place.
- `POST /admin/providers` / `DELETE /admin/providers/{name}` — add/remove
  proxy providers at runtime, no restart.
- **`dev-tools/typesafe_labeler/`** (outside the `armor` package, own
  virtualenv, not a project dependency) — a dev-only automation that polls
  for unreviewed bandit recommendations and submits corrections through
  the same public endpoint a human would, using TypeSafe's Jev model as
  the judgment oracle. Runs continuously (`--watch --interval 30`) as a
  background process. Armor itself never imports TypeSafe — this tool
  calls Armor's own public API from the outside, same as a human clicking
  the same buttons would.

## Deployment

- `armor.api.main:app` is the real ASGI entry point (`armor.proxy.app`
  exports a router, not a standalone app — it's mounted under the same
  FastAPI instance).
- Persistent macOS service: `scripts/com.armor.server.plist`
  (launchd — `RunAtLoad`/`KeepAlive`, bakes in `ARMOR_USE_BASE_MODEL=1` so
  the base-model path survives restarts and reboots without a manual
  step).
- No committed `LICENSE` yet — Apache-2.0 recommended (see README's
  Licensing Notice section for the per-dataset compatibility check behind
  that recommendation).

## Training and evaluation

- `armor/models/train.py` — trains/fine-tunes the ArmorBase head (and
  optionally the final encoder layers). Every training run is checked
  against `armor/eval/regression_gate.py` (a frozen tripwire set,
  `armor/eval/regression_set.py`, 100 cases across all 6 labels) before a
  new checkpoint is allowed to overwrite the deployed one — a per-label
  score drop greater than 0.05 blocks promotion, even if the new model's
  overall score is higher. This gate only tests raw `model.predict()`
  output, not the full calibrated `/v1/inspect` path — a real testing gap
  found in the 2026-09-22 calibration regression (see
  `docs/archive/queue/ARCHIVE-QUEUE-07_venn_abers_calibration.md`),
  now separately covered by `tests/test_calibration_integration.py`
  (`ARMOR_RUN_SLOW_TESTS=1`, since it needs the real model + real
  calibration table).
- Training/calibration data sources are documented per-label in
  `README.md`'s Training Data table — a mix of licensed public datasets
  (deepset, SPML, jackhhao, TrustAIRLab jailbreak-wild, civil_comments,
  AgentHarm, InjecAgent, HarmBench, ToolSafety, CoEdIT, OASST1, Aegis) and
  purpose-built synthetic generators (`armor/data/synthetic_*.py`), each
  with its own ponytail comment explaining why it exists and what gap it
  closes.

## Current serving state and validation (2026-09-24)

- `com.armor.server` serves the promoted flat BERT policy head on prompt
  requests. The decision bandit and hierarchical head remain shadow-only.
- The v18 blind AI-only confirmation gate had 148 reviewer-consensus
  examples: 31 ALLOW, 59 FLAG, 30 BLOCK, and 28 REVIEW. Reviewer agreement
  was 148/150 (98.7%, Cohen's kappa 0.982). Candidate accuracy was 85.8%
  and macro-F1 0.850, compared with 45.9% and 0.446 before promotion.
  FLAG recall was 88.1% at 89.7% precision, compared with 15.3% recall
  before promotion. The separate 70-case, single-AI-reviewer long-text
  guard improved from 80.0% to 87.1% accuracy. These results are
  AI-reviewer evidence on hand-authored prompts, not human ground truth or
  proof of production accuracy.
- Promoted checkpoint SHA-256:
  `4fbf18ff1a6f48f4957beb743772440ad31d30f9d40e84cfa0c86652a5e1448d`.
  The prior checkpoint is backed up at
  `/tmp/armor_policy_head_pre_v17_promotion_20260924.pt`; the promotion
  record and rollback command are in
  `docs/broad_gate_v18_promotion_20260924.json`.
- Full per-class metrics, latency, data provenance, and serving smoke
  results are linked from `docs/broad_accuracy_resolution_20260923.md`.

## Known limitations (see `README.md`'s Open Work Queue for detail)

- The local `com.armor.server` launch agent now uses the four-action BERT
  policy head for prompt decisions when `ARMOR_BERT_POLICY_HEAD=1`; the
  bandit remains shadow-only. The current checkpoint and rollback record
  are documented in `docs/broad_accuracy_resolution_20260923.md`.
- FLAG improved substantially on v18 but remains measured only against AI
  reviewers. The 59 FLAG examples are stratified and hand-authored, so the
  result is not a natural-traffic prevalence estimate.
- Bandit feedback now records whether a correction applies to Armor's
  production decision, the shadow recommendation, or both. Individual AI
  judgments are provisional and do not change online bandit weights; only
  independently adjudicated feedback applies reward. Machine-only consensus
  is tagged `reviewer_ai_consensus` and remains provisional. The old
  `analyst_correction` values remain legacy and target-ambiguous. On the v18
  AI-agreement replay, the bandit matched only 48.1% of 108 shared cases and
  had 0% REVIEW recall; these are diagnostic, not ground-truth metrics. See
  `docs/feedback_targets_and_reward_loop.md`.
- The training corpus includes historical rows tagged `human` and
  `pseudo_model`; keep that provenance distinct. New v15-v18 labels were
  independently produced by AI reviewers. No human adjudication was used
  for the v18 confirmation metrics.
- Proxy live-provider wire checks (`QUEUE-01`) need real API keys, not
  available in the current environment — mock-transport coverage only.
- The proxy holds prompt requests classified as `review`; downstream
  review operations still depend on the configured proxy integration.
- Streaming inspection checks each SSE event independently; a violation
  deliberately spread across multiple events wouldn't be caught by either
  event alone — the rolling-buffer fix is Phase 5 of the roadmap, not
  built.

## Long-context phase: measured baseline and current status (2026-09-25)

The earlier 140-row adaptive gate reached at most 785 tokens, so it did not test the 1K–8K serving distribution. Armor now has a frozen 105-row challenge corpus with 21 rows in each of five length buckets, five paired attack positions per bucket, and a boundary-split case per bucket. Carrier prose is drawn from a pinned OASST1 revision. Labels are `synthetic` scenario contracts, not human-ground-truth labels. The corpus and its checksum are in `docs/long_context_gate_v1_20260925.jsonl` and `docs/long_context_gate_v1_20260925_manifest.json`.

The unchanged full-depth PyTorch CPU production path was measured first. On the local Apple M2 development machine, mean/p95 latency rose from 678/720 ms at 1K to 17.54/21.35 seconds at 8K; the largest observed request took 28.15 seconds. The encoder averaged 7.00 seconds across the full corpus, while tokenization averaged 15 ms and both small policy heads averaged 1.25 ms. The overall scenario-contract agreement was 34.3% (block recall 40.0%); these figures are model-to-synthetic-contract comparisons, not ground-truth quality claims. Full metrics, per-row hashes, source provenance, and limitations are in `docs/long_context_full_depth_baseline_v1_20260925.md` and its JSON companion.

The training/serving mismatch is confirmed: the current six-label training mix has 20,153 examples with mean 117.8 tokens, p95 422, and max 4,674, while the standard collator truncates at 512 and serving accepts up to 8,192. On the local MPS training device, a singleton frozen-feature pass took 5.39 seconds at 8K; a two-row 8K batch was slower per row. A conservative one-pass curriculum extrapolation is 4.25 hours for all 20,153 rows. `scripts/train_long_context_curriculum.py` prepares an isolated candidate mix with position-randomized reviewed carrier context and inherited source labels.

No early-exit or windowed candidate is promoted. Two earlier long-context head-only recalibration attempts failed the per-label regression gate. The third attempt resumed from batch 10,500 on MPS and completed all 12,218 feature batches. Long rows now pad to 128-token boundaries, MPS progress is logged every 25 rows, and resumable snapshots are saved every 100 rows. The completed frozen-feature cache is reusable for another head-only attempt when its data and base-checkpoint hashes match. The candidate still failed the gate on prompt injection and obfuscation encoding, so no candidate checkpoint was saved and the deployed head and full-depth serving remain unchanged. See [the MPS training run and Colab MCP findings](docs/mps_training_and_colab_mcp_20260926.md) and [the curriculum manifest](docs/long_context_training_curriculum_v3_20260925.json). Intermediate-depth probes, benign certification, and region selection remain out of serving.
