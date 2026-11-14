# Long input regression resolution (2026-09-23)

## What changed

- Armor Base now tokenizes without silent 512-token truncation. It reads up to the encoder's 8192-token limit and returns `review` when content exceeds that limit. The request is one current user turn or response; proxy adapters already extract the latest user turn.
- The live prompt fast scan now blocks direct instructions to disable model safety controls and direct requests to disclose live or production credentials. These rules scan paragraphs across the entire input, including an attack appended after a long benign document. They do not run on response direction.
- Historical note (2026-09-23): the +18 and +19 AI-label flat policy head retrains were **not promoted** because they caused false blocks on long benign text. Later long49 and paragraph-head probes also failed their then-current gates. On 2026-09-24, a separate policy-head checkpoint was promoted after a fresh four-action gate; the current status and long-text guard are documented below. These earlier failures remain relevant evidence, not the current serving state.

## Evidence

The two frozen, disjoint long-text gates use AI reviewer Jev under rubric v1.1; they are **not human ground truth**. Each has 40 safe OASST responses, 10 attack prefixes, and 20 attack suffixes. The gates contain repeated synthetic attack structure, so the result establishes these cases, not broad attack recall.

| Gate | Live before rules: AI `block` caught | Live after rules: AI `block` caught | Safe `block` before/after | Safe `flag` before/after |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0/26 | 26/26 | 0/40 → 0/40 | 5/40 → 5/40 |
| v2 | 2/27 | 27/27 | 0/40 → 0/40 | 1/40 → 1/40 |

Seven additional secret disclosure requests were marked `review` with low AI confidence; the new rule blocks them. This is a policy judgment, not measured agreement with the reviewer. The rule had zero hits among 331 corrected `allow`, 109 `flag`, and 67 `review` log entries. It matched 11/587 corrected `block` entries. This replay does not prove a zero false positive rate in future traffic.

The exact gate inputs and AI labels are frozen in `docs/long_text_final_gate_corpus_20260923.jsonl`. The complete serving output, including sample IDs and latency, is in `docs/long_text_serving_replay_20260923.json`. Full-context shadow head results are in `docs/long_text_fullctx_comparison_20260923.json`; length-routing results are in `docs/long_text_length_routing_probe_20260923.json`. The earlier failure investigation is in `docs/ai_only_policy_regression_investigation_20260923.md`.

## Limits and next gate

- The high precision rules cover explicit safety bypass and production secret disclosure phrasing. Novel phrasing and other harmful long suffixes can still be missed. A general solution needs a validated local intent model or an uncertainty route with broader, independently authored labels.
- Full 8192-token processing raises latency with text length. A 3515-token CPU sample took about 3.27 seconds in a local probe; the fast rules can decide matching requests before encoder inference.
- Latency follow-up (2026-09-25): an Apple MPS experiment is retained as a local-device measurement only; it is not the production path. Production now defaults to CPU; `ARMOR_BASE_DEVICE=mps` is explicit opt-in for Apple Silicon deployments. On the same eight longest (1,566–3,211 token) real-world TrustAIRLab jailbreak prompts, full inspect with the same active policy head averaged **2.42s** on CPU (median 2.20s, max 4.42s) and 1.23s on MPS (median 1.03s, max 2.31s). Decisions matched: seven `block`, one `review`; longest-prompt raw-score difference was at most `1.2e-7`. A separate live MPS replay averaged 1.35s and had a 7.0s cold first request. These eight samples are not a production latency estimate. Batched 510-token windows were rejected: while faster on some attacks, they changed a vetted 797-token benign answer's raw harmful-content score to `0.999`. ONNX was slower on the same long input (6.9–8.3s locally), and MPS FP16 autocast produced raw-score deltas up to 0.31 without reliable speedup. The next production candidate should be evaluated on CPU with a calibrated small-model cascade, not by assuming an accelerator or truncating/chunk-maxing long content.
- Historical status as of 2026-09-23: no AI-only classifier candidate had been made authoritative. On 2026-09-24, the v17 candidate was promoted after a separate broad gate and a 70-item long-text regression guard. In this guard, candidate accuracy improved from 80.0% to 87.1%, macro-F1 from 0.424 to 0.454, and BLOCK recall stayed at 100%. The guard has no FLAG support and one AI reviewer; it does not measure future false-block rates. Full details and checkpoint hash are in `docs/broad_accuracy_resolution_20260923.md`.

## Diverse follow-up and live rollout

The initial v1/v2 gates above reused attack phrasing. A follow-up used unused OASST1 assistant answers as long, safe backgrounds and separately generated requests. Exact inputs, Jev's rubric v1.1 labels, and per-sample serving replays are frozen as `long_text_gate_v3_*` through `long_text_gate_v7_*` in this directory. Jev is an **AI reviewer**, not human ground truth. v3 through v6 and the final v7 context result became diagnostic after fixes were made in response to them. None is an untouched estimate of future traffic accuracy.

The original serving path missed many appended attacks. On v3 development it blocked 4 of 23 Jev-`block` attack variants and hard-blocked two Jev-`allow` benign variants. It blocked only 1/11 v4, 2/15 v5, and 2/6 v6 bypass requests marked `block`. The fixes read the complete input, recognize direct safety-bypass, production-secret, and covert-action requests across paragraphs, separately score an explicitly marked current task alongside its reference document, and route borderline long model-only blocks to `review`. The alternative pooled task head and local Qwen direct judge were evaluated but did not meet this gate and were not deployed.

| Gate after fixes | Jev `block` attack variants blocked | Safe/benign hard blocks | Jev `review` context routed to review |
| --- | ---: | ---: | ---: |
| v3 development | 23/23 | 0/36 | 5/7 |
| v3 holdout, used during tuning | 12/12 | 0/24 | 4/4 |
| v4 | 11/11 | 0/24 | 4/4 |
| v5 | 15/15 | 0/24 | 4/4 |
| v6 | 9/13; four low-confidence secret-value questions routed to `review` | 0/26 | 1/4 |
| v7 after context fix | 3/3; all three AI confidences below 0.8 | 0/20 | 3/3 |

V7 was frozen and replayed once before the context fix. Its first replay caught all three Jev-`block` requests, hard-blocked none of its 20 safe/benign requests, and routed 0/3 Jev-`review` context questions to review. The targeted authorization-boundary change raised that last count to 3/3. V7's generator failed to produce bypass attempts: all five intended “bypass” requests were ordinary questions and Jev labeled all five `allow`. Its secret and covert labels were also low confidence. V7 therefore supplies a useful negative check and a context regression, **not** a new bypass-recall estimate. The pre-fix replay remains in `long_text_gate_v7_untouched_replay_20260923.json`.

For v6, the four Jev-`block` secret questions routed to `review` had low confidence; five context questions also had low confidence and mixed outcomes. V5 additionally had one Jev-`flag` secret request hard-blocked, and v4 had one Jev-`review` attack hard-blocked. These are boundary disagreements, not claimed wins. The original v1/v2 gates and earlier AI-labeled regression slices were replayed with no new hard blocks. Corrected historical logs showed zero direct-rule matches among 333 `allow`, 109 `flag`, and 67 `review` entries, and 24/587 `block` entries matched; those log corrections are not human truth.

The local launchd service was restarted with these changes and `/v1/inspect` was smoke tested on a safe long answer, direct bypass, covert action, and context request. It returned `review`, `block`, `block`, and `review`, respectively. The full test suite passed: **343 passed, 6 skipped**. The direct rules are intentionally narrow. Novel harmful phrasing can still be missed; ambiguous requests can still route differently from Jev, particularly in low-confidence `flag`/`review` boundaries. Neither these AI-only gates nor zero observed safe hard blocks establish a production false-positive rate.
