import json
import os
import time

from fastapi import FastAPI

from armor.api.schemas import Finding, InspectRequest, InspectResponse
from armor.detectors.masks import MASKS
from armor.detectors.registry import default_registry
from armor.observability import log_store
from armor.policy.bypass import explicit_covert_action, explicit_safety_bypass, explicit_secret_disclosure
from armor.policy.request_scope import explicit_current_task, task_boundary_action

# ponytail: module-qualified access (log_store.DB_PATH), not a copied
# `from ... import DB_PATH` name — tests monkeypatch that attribute to
# redirect writes to a tmp file, which only works read through the module
# each call, not a value bound once at import time (same lesson learned
# for armor/proxy/admin.py's PROVIDERS_PATH).

app = FastAPI(title="Armor", version="0.1.0")

# ponytail: opt-in so tests/CI don't pull a 149M-param model over the network
# by default. Set ARMOR_USE_BASE_MODEL=1 to route through armor/models/armor_base.py.
USE_BASE_MODEL = os.environ.get("ARMOR_USE_BASE_MODEL") == "1"
USE_BERT_POLICY_HEAD = os.environ.get("ARMOR_BERT_POLICY_HEAD") == "1"
ADAPTIVE_SHADOW = os.environ.get("ARMOR_ADAPTIVE_SHADOW") == "1"
ADAPTIVE_LIVE = os.environ.get("ARMOR_ADAPTIVE_LIVE") == "1"
FINDING_THRESHOLD = float(os.environ.get("ARMOR_BASE_THRESHOLD", "0.5"))
WINDOW_FUSION_MODE = os.environ.get("ARMOR_WINDOW_FUSION", "pi_max_others_second_max")


def _fuse_window_scores(window_raw) -> dict[str, float]:
    """Aggregate [n_windows, n_labels] raw scores into per-label request scores.

    Single-window `max` across 20+ windows causes isolated noise spikes on
    benign long documents. Using `max` for `prompt_injection` (single-window
    injection) and `second_max` for the remaining 5 labels preserves PI recall
    while suppressing single-window false positives.
    """
    from armor.models.labels import LABELS

    if WINDOW_FUSION_MODE == "max" or window_raw.shape[0] < 2:
        return dict(zip(LABELS, window_raw.max(dim=0).values.tolist()))
    pi_idx = LABELS.index("prompt_injection")
    second_max = window_raw.topk(2, dim=0).values[-1].tolist()
    second_max[pi_idx] = float(window_raw[:, pi_idx].max().item())
    return dict(zip(LABELS, second_max))

# ponytail: the base-model route could only ever return "flag", never "block",
# no matter the confidence — only the deterministic Fast Scan Layer could
# block. A first attempt at block-on-any-finding (2026-09-20) was reverted
# within the hour: the classifier was scoring "write me an exploit" and "how
# are you" identically at 0.99, so blocking on a finding blocked everything.
#
# That was not a model problem. The proxy adapters were concatenating the
# whole conversation history into the inspected text, so the newest message
# was a rounding error in the blob (see
# armor/proxy/adapters/openai.py::extract_prompt_text). With the adapters
# fixed to inspect only the newest user turn, the same two inputs separate
# cleanly: 0.016 vs 0.997.
#
# Block is therefore gated on a second, higher bar rather than on the mere
# existence of a finding. 0.8 sits above the measured false positives
# ("Write a concise, useful checklist" at 0.529) and well below the real
# attacks (0.997). Findings between FINDING_THRESHOLD and this stay "flag":
# logged and visible in /admin/logs-page, passed through untouched.
BLOCK_THRESHOLD = float(os.environ.get("ARMOR_BASE_BLOCK_THRESHOLD", "0.8"))
LONG_BLOCK_REVIEW_CEILING = 0.85

# ponytail: shadow-mode LinUCB bandit (armor/models/bandit.py) — logs a
# recommendation alongside the real decision but never controls the response
# (see docs/CHAT_IMPROVEMENTS_AND_CHANGES.md for why one global threshold
# keeps trading one label's separation for another). Skipped above this
# ceiling: the existing logic already blocks with certainty there, so the
# bandit only ever sees the genuinely ambiguous middle.
BANDIT_FLOOR = float(os.environ.get("ARMOR_BANDIT_FLOOR", "0.95"))
# v1.2: only pay the semantic judge's LLM-generation latency for the narrow
# band where the classifier is genuinely ambiguous, not the majority of
# clearly-benign traffic (near-zero scores) or clearly-obvious attacks
# (near BANDIT_FLOOR). See armor/models/semantic_judge.py.
SEMANTIC_JUDGE_LOW = float(os.environ.get("ARMOR_SEMANTIC_JUDGE_LOW", "0.3"))
# Default OFF: a full 1043-row replay (QUEUE-02's 2026-09-22 v1.2 entry)
# measured this making the bandit WORSE, not better -- block recall 94%
# to 74%, overall 66% to 56%. Kept as a documented, gated future direction
# (prompt/model quality is the likely next lever), not shipped live.
SEMANTIC_JUDGE_ENABLED = os.environ.get("ARMOR_SEMANTIC_JUDGE_ENABLED", "0") == "1"

# ponytail: DLP-6 specialist (armor/models/pii_ner.py), gated behind a real
# escalation rule, not another always-on layer (framework doc section 3).
# Only fires when Fast Scan found nothing decisive AND Armor Base reports a
# weak-but-nonzero sensitive_data_exposure score — conditional compute, not
# a second parallel detector.
PII_NER_ENABLED = os.environ.get("ARMOR_USE_PII_NER") == "1"
ESCALATE_LOW = float(os.environ.get("ARMOR_ESCALATE_LOW", "0.3"))

# ponytail: unset/empty = no filter, every locale pack active (current
# behavior, keeps existing deployments/tests unchanged). Set
# ARMOR_ACTIVE_LOCALES="US,GB" to scope national-ID detectors to the
# locales a deployment actually cares about (framework doc section 5).
_active_locales_env = os.environ.get("ARMOR_ACTIVE_LOCALES")
ACTIVE_LOCALES = set(_active_locales_env.split(",")) if _active_locales_env else None

_fast_scan = default_registry()

# Fast Scan Layer decision mapping (architecture spec section 5.2, DLP
# framework doc section 5 severity table). Only high/medium severities are
# decisive on their own; low-severity hits (e.g. a bare URL) are common
# enough that they shouldn't escalate alone — they're dropped here rather
# than merged with Armor Base findings, since there's no Policy Aggregator
# yet to combine multiple signal sources (architecture spec section 5 box).
_SEVERITY_DECISION = {"high": "block", "medium": "flag"}
_SEVERITY_SCORE = {"high": 0.95, "medium": 0.6, "low": 0.2}


def _run_fast_scan(content: str, direction: str = "prompt") -> InspectResponse | None:
    matches = _fast_scan.scan(content, locales=ACTIVE_LOCALES)
    decisive = [m for m in matches if m["severity"] in _SEVERITY_DECISION]
    bypass = direction == "prompt" and explicit_safety_bypass(content)
    secret_disclosure = direction == "prompt" and explicit_secret_disclosure(content)
    covert_action = direction == "prompt" and explicit_covert_action(content)
    boundary_review = (
        direction == "prompt"
        and task_boundary_action(explicit_current_task(content) or content) == "review"
    )
    if not decisive and not bypass and not secret_disclosure and not covert_action and not boundary_review:
        return None

    findings = [
        Finding(
            label="sensitive_data_exposure",
            score=_SEVERITY_SCORE.get(m["severity"], 0.5),
            evidence=f"{m['entity_id']}:{m['evidence']}",
        )
        for m in decisive
    ]
    if bypass:
        findings.append(Finding(label="jailbreak", score=0.95))
    if secret_disclosure:
        findings.append(Finding(label="sensitive_data_exposure", score=0.95))
    if covert_action:
        findings.append(Finding(label="dangerous_tool_action", score=0.95))
    hard_block = bypass or secret_disclosure or covert_action or any(m["severity"] == "high" for m in decisive)
    decision = "block" if hard_block else "review" if boundary_review else "flag"
    return InspectResponse(
        decision=decision,
        risk_score=max((f.score for f in findings), default=0.0),
        findings=findings,
        route="escalated" if decision == "review" else "fast",
        latency_ms=0.0,
        model_version=(
            "fast-scan-dlp-v0.1+policy-v2"
            if bypass or secret_disclosure or covert_action
            else "fast-scan-dlp-v0.1+task-scope-v1"
            if decision == "review"
            else "fast-scan-dlp-v0.1"
        ),
    )


@app.post("/v1/inspect", response_model=InspectResponse)
def inspect(req: InspectRequest) -> InspectResponse:
    start = time.perf_counter()

    # Empty upstream choices are common and contain no content to classify.
    # Do not ask the model to assign security labels to absent content.
    if not req.content.strip():
        result = InspectResponse(
            decision="allow",
            risk_score=0.0,
            findings=[],
            route="base" if USE_BASE_MODEL else "fast",
            latency_ms=(time.perf_counter() - start) * 1000,
            model_version="empty-input-bypass",
        )
        result._log_id = log_store.log_entry(
            log_store.DB_PATH,
            source="inspect",
            direction=req.direction,
            content=req.content,
            decision=result.decision,
            risk_score=result.risk_score,
            route=result.route,
            latency_ms=result.latency_ms,
            inference_trace={"fast_scan_ms": 0.0, "total_ms": result.latency_ms,
                             "final_decision": result.decision,
                             "approximate_compute_path": "empty_input_bypass"},
        )
        return result

    fast_scan_started = time.perf_counter()
    fast_result = _run_fast_scan(req.content, req.direction)
    fast_scan_ms = (time.perf_counter() - fast_scan_started) * 1000
    if fast_result is not None:
        fast_result.latency_ms = (time.perf_counter() - start) * 1000
        fast_result._log_id = log_store.log_entry(
            log_store.DB_PATH,
            source="inspect",
            direction=req.direction,
            content=req.content,
            decision=fast_result.decision,
            risk_score=fast_result.risk_score,
            route=fast_result.route,
            findings=[f.model_dump() for f in fast_result.findings],
            latency_ms=fast_result.latency_ms,
            inference_trace={"fast_scan_ms": fast_scan_ms, "total_ms": fast_result.latency_ms,
                             "final_decision": fast_result.decision,
                             "approximate_compute_path": "fast_scan_decisive"},
        )
        return fast_result

    if not USE_BASE_MODEL:
        latency_ms = (time.perf_counter() - start) * 1000
        result = InspectResponse(
            decision="allow",
            risk_score=0.0,
            findings=[],
            route="fast",
            latency_ms=latency_ms,
            model_version="stub-0.1",
        )
        result._log_id = log_store.log_entry(
            log_store.DB_PATH,
            source="inspect",
            direction=req.direction,
            content=req.content,
            decision=result.decision,
            risk_score=result.risk_score,
            route=result.route,
            latency_ms=result.latency_ms,
            inference_trace={"fast_scan_ms": fast_scan_ms, "total_ms": result.latency_ms,
                             "final_decision": result.decision,
                             "approximate_compute_path": "stub_allow"},
        )
        return result

    from armor.models.armor_base import MAX_PREFLIGHT_CHARS, InputTooLongError, get_armor_base
    from armor.models.calibration import calibrate_scores
    from armor.models.labels import LABELS

    # This separate raw-size guard bounds tokenizer work. It is deliberately
    # reported as a preflight limit, not as an exact model-token overflow.
    preflight_started = time.perf_counter()
    if len(req.content) > MAX_PREFLIGHT_CHARS:
        preflight_ms = (time.perf_counter() - preflight_started) * 1000
        latency_ms = (time.perf_counter() - start) * 1000
        result = InspectResponse(
            decision="review", risk_score=0.0, findings=[], route="escalated",
            latency_ms=latency_ms, model_version="armor-base-preflight-limit",
        )
        result._log_id = log_store.log_entry(
            log_store.DB_PATH, source="inspect", direction=req.direction,
            content=req.content, decision=result.decision, risk_score=result.risk_score,
            route=result.route, latency_ms=result.latency_ms,
            review_reason="input_exceeds_preflight_limit",
            inference_trace={
                "fast_scan_ms": fast_scan_ms, "preflight_ms": preflight_ms,
                "tokenization_ms": 0.0, "input_chars": len(req.content),
                "token_count": None, "reason": "input_exceeds_preflight_limit",
                "total_ms": latency_ms, "final_decision": "review",
                "window_count": 0, "approximate_compute_path": "preflight_review",
            },
        )
        return result
    preflight_ms = (time.perf_counter() - preflight_started) * 1000

    model_load_started = time.perf_counter()
    model = get_armor_base()
    model_load_ms = (time.perf_counter() - model_load_started) * 1000
    # Fake/mock models in tests only implement predict() -- fall back
    # rather than require every test double to grow an embedding method it
    # has no real use for. pooled stays None in that case, and the policy
    # engine (which needs it) is simply skipped below, the same fail-open
    # shape as the bandit/semantic judge calls.
    pooled = None
    token_context = None
    inference_trace = {
        "fast_scan_ms": fast_scan_ms, "preflight_ms": preflight_ms,
        "model_load_ms": model_load_ms, "input_chars": len(req.content),
        "window_count": 1, "windows_exited_at_6": 0,
        "windows_exited_at_12": 0, "windows_full_depth": 1,
        "context_expansion_count": 0,
        "approximate_compute_path": "full_depth_legacy",
    }
    try:
        if hasattr(model, "prepare_token_context"):
            token_context = model.prepare_token_context(req.content)
            inference_trace.update(token_context.timings_ms)
            inference_trace["token_count"] = token_context.token_count
            if token_context.oversize:
                raise InputTooLongError(
                    token_context.oversize_reason or "input exceeds inspection limits",
                    reason=token_context.oversize_reason or "input_exceeds_model_context",
                )
        if ADAPTIVE_SHADOW and token_context is not None and hasattr(model, "predict_with_adaptive_shadow"):
            adaptive_started = time.perf_counter()
            raw_scores, pooled, adaptive_trace = model.predict_with_adaptive_shadow(token_context)
            inference_trace.update(adaptive_trace)
            inference_trace["adaptive_shadow_ms"] = (time.perf_counter() - adaptive_started) * 1000
        elif token_context is not None and hasattr(model, "window_scores"):
            # window_scores degrades to one full-depth pass when the input
            # fits a single window, so this is safe at every length -- real
            # windowing (max-fused) only kicks in past ~512 tokens, which is
            # where the quadratic full-depth pass gets expensive.
            window_started = time.perf_counter()
            used_early_exit = False
            if ADAPTIVE_LIVE and hasattr(model, "begin_encoder") and hasattr(model, "continue_encoder"):
                from armor.inference.adaptive import run_adaptive_single, run_adaptive_windows

                win_size = int(os.environ.get("ARMOR_WINDOW_SIZE", "512"))
                exit_policy = os.environ.get("ARMOR_EXIT_POLICY", "patience").lower()
                if (token_context.token_count or 0) <= win_size:
                    adaptive_res = run_adaptive_single(
                        model, token_context, exit_policy=exit_policy,
                        max_inspection_time_ms=float(os.environ.get("ARMOR_MAX_INSPECTION_TIME_MS", "3000")),
                    )
                else:
                    adaptive_res = run_adaptive_windows(
                        model, token_context, model.tokenizer,
                        window_size=win_size,
                        overlap=int(os.environ.get("ARMOR_WINDOW_OVERLAP", "96")),
                        max_windows=int(os.environ.get("ARMOR_MAX_WINDOWS", "24")),
                        max_full_depth_windows=int(os.environ.get("ARMOR_MAX_FULL_DEPTH_WINDOWS", "4")),
                        max_context_expansions=int(os.environ.get("ARMOR_MAX_CONTEXT_EXPANSIONS", "2")),
                        max_expanded_tokens=int(os.environ.get("ARMOR_EXPANDED_CONTEXT_TOKENS", "768")),
                        max_inspection_time_ms=float(os.environ.get("ARMOR_MAX_INSPECTION_TIME_MS", "3000")),
                        exit_policy=exit_policy,
                    )
                if adaptive_res.raw_scores is not None:
                    raw_scores = adaptive_res.raw_scores
                    pooled = adaptive_res.pooled
                    inference_trace.update(adaptive_res.trace)
                    used_early_exit = True
            if not used_early_exit:
                if ADAPTIVE_LIVE:
                    stop_thresh = float(os.environ.get("ARMOR_WINDOW_EARLY_STOP_THRESHOLD", "0.85"))
                    window_raw, pooled_all = model.window_scores(
                        token_context, early_stop_threshold=stop_thresh
                    )
                else:
                    window_raw, pooled_all = model.window_scores(token_context)
                raw_scores = _fuse_window_scores(window_raw)
                pooled = pooled_all.mean(dim=0)
                inference_trace["window_count"] = window_raw.shape[0]
                inference_trace["windows_full_depth"] = window_raw.shape[0]
                inference_trace["approximate_compute_path"] = (
                    "full_depth_legacy" if window_raw.shape[0] == 1 else "windowed_full_depth"
                )
                inference_trace["windowing_ms"] = (time.perf_counter() - window_started) * 1000
        elif hasattr(model, "predict_with_embedding"):
            raw_scores, pooled = model.predict_with_embedding(
                token_context if token_context is not None else req.content
            )
        else:
            raw_scores = model.predict(req.content)
        if token_context is not None:
            inference_trace.update(token_context.timings_ms)
    except InputTooLongError as exc:
        if token_context is not None:
            inference_trace.update(token_context.timings_ms)
            inference_trace["token_count"] = token_context.token_count
        inference_trace.update({
            "input_chars": len(req.content), "reason": exc.reason,
            "final_decision": "review", "window_count": 0,
            "approximate_compute_path": "bounded_input_review",
        })
        result = InspectResponse(
            decision="review",
            risk_score=0.0,
            findings=[],
            route="escalated",
            latency_ms=(time.perf_counter() - start) * 1000,
            model_version=("armor-base-preflight-limit" if exc.reason == "input_exceeds_preflight_limit"
                           else "armor-base-context-limit"),
        )
        result._log_id = log_store.log_entry(
            log_store.DB_PATH,
            source="inspect",
            direction=req.direction,
            content=req.content,
            decision=result.decision,
            risk_score=result.risk_score,
            route=result.route,
            latency_ms=result.latency_ms,
            review_reason=exc.reason,
            inference_trace={**inference_trace, "total_ms": (time.perf_counter() - start) * 1000},
        )
        return result
    fingerprint = getattr(model, "fingerprint", None)
    # No fingerprint means no real checkpoint identity (a test/mock model) —
    # calibration is tied to a specific checkpoint's score distribution, so
    # skip it rather than fit one and write it to the shared on-disk cache.
    scores = (
        {label: c["score"] for label, c in calibrate_scores(raw_scores, fingerprint).items()}
        if fingerprint is not None
        else raw_scores
    )
    findings = [
        Finding(label=label, score=score)
        for label, score in scores.items()
        if score >= FINDING_THRESHOLD
    ]
    # Two-tier: findings at or above BLOCK_THRESHOLD stop the request,
    # weaker ones only flag it. See BLOCK_THRESHOLD's definition above for
    # why block is gated on a second bar instead of on any finding at all.
    if findings and max(f.score for f in findings) >= BLOCK_THRESHOLD:
        decision = "block"
    elif findings:
        decision = "flag"
    else:
        decision = "allow"
    risk_score = max(scores.values()) if scores else 0.0
    route = "base"
    model_version = "armor-base-seed-v0.1" if model.checkpoint_loaded else "armor-base-untrained"
    task_review_reason = None

    sde_score = scores.get("sensitive_data_exposure", 0.0)
    if PII_NER_ENABLED and decision == "allow" and ESCALATE_LOW <= sde_score < FINDING_THRESHOLD:
        from armor.models.pii_ner import get_pii_ner

        ner_matches = get_pii_ner().predict(req.content)
        if ner_matches:
            findings = findings + [
                Finding(
                    label="sensitive_data_exposure",
                    score=0.6,
                    evidence=f"{m['entity_id']}:{MASKS['last4'](m['evidence'])}",
                )
                for m in ner_matches
            ]
            decision = "flag"
            risk_score = max(risk_score, 0.6)
            route = "escalated"
            model_version = f"{model_version}+pii-ner"

    # An explicit current task and a reference document provide two useful
    # views. Keep the full-content result, then inspect the task separately;
    # a conflict involving block routes to review. Fast rules ran first.
    task = explicit_current_task(req.content) if req.direction == "prompt" else None
    if task:
        task_raw = model.predict(task)
        task_scores = (
            {label: c["score"] for label, c in calibrate_scores(task_raw, fingerprint).items()}
            if fingerprint is not None
            else task_raw
        )
        task_risk = max(task_scores.values()) if task_scores else 0.0
        task_action = "block" if task_risk >= BLOCK_THRESHOLD else (
            "flag" if task_risk >= FINDING_THRESHOLD else "allow"
        )
        if decision != task_action:
            if "block" in (decision, task_action):
                decision = "review"
                route = "escalated"
                task_review_reason = "full content and explicit current task disagree on block"
            elif "flag" in (decision, task_action):
                decision = "flag"
        boundary = task_boundary_action(task)
        if boundary == "review":
            decision = "review"
            route = "escalated"
            task_review_reason = "explicit task needs authorization or ownership context"
        elif boundary == "flag" and decision == "block":
            decision = "flag"
        combined = {
            label: max(scores.get(label, 0.0), task_scores.get(label, 0.0))
            for label in scores.keys() | task_scores.keys()
        }
        findings = [
            Finding(label=label, score=value)
            for label, value in combined.items()
            if value >= FINDING_THRESHOLD
        ]
        risk_score = max(risk_score, task_risk)
        model_version += "+task-scope-v1"

    elif req.direction == "prompt":
        # The same BERT policy boundary rules apply to an ordinary direct
        # request, not only to tasks extracted from a reference document.
        boundary = task_boundary_action(req.content)
        if boundary == "review":
            decision = "review"
            route = "escalated"
            task_review_reason = "request needs authorization or ownership context"
        elif boundary == "flag" and decision == "block":
            decision = "flag"

    # Long, unscoped model-only blocks just above the ordinary threshold
    # have produced benign false blocks. Keep the request intercepted while
    # the evidence is weak; explicit fast-rule blocks never reach this path.
    if (task is None and decision == "block" and route == "base"
            and risk_score < LONG_BLOCK_REVIEW_CEILING
            and token_context is not None
            and getattr(token_context, "token_count", 0) is not None
            and token_context.token_count >= 256):
        decision = "review"
        route = "escalated"
        task_review_reason = "long model-only block in uncertain score band"
        model_version += "+long-review-band-v1"

    bandit_action = None
    bandit_context_json = None
    if scores and max(scores.values()) < BANDIT_FLOOR:
        from armor.models.bandit import build_context, get_bandit

        semantic = None
        if SEMANTIC_JUDGE_ENABLED and SEMANTIC_JUDGE_LOW <= max(scores.values()) < BANDIT_FLOOR:
            try:
                from armor.models.semantic_judge import get_semantic_judge

                semantic = get_semantic_judge().analyze(req.content)  # None on failure -> neutral fill
            except Exception:
                # Model load failure (e.g. no cached weights, OOM) shouldn't
                # block a request any more than a generation failure does —
                # same fail-open contract as analyze()'s own try/except.
                semantic = None

        context = build_context(req.content, scores, semantic)
        bandit_action = get_bandit().choose(context)
        bandit_context_json = json.dumps(context.tolist())

    # Phase 1 policy head (queue/QUEUE-02) -- logged whenever a real
    # (non-mock) armor_base model produced an embedding, independent of
    # BANDIT_FLOOR/PII-NER's own gates -- the spec's own point that
    # unrelated decision-makers shouldn't share one classifier-score gate.
    policy_action = None
    policy_confidence_value = None
    review_reason = task_review_reason
    if pooled is not None:
        try:
            from armor.policy.engine import evaluate as evaluate_policy

            policy_result = evaluate_policy(scores, pooled)
            policy_action = policy_result["policy_action"]
            policy_confidence_value = policy_result["policy_confidence"]
            policy_reason = "; ".join(policy_result["decision_factors"])
            review_reason = f"{task_review_reason}; {policy_reason}" if task_review_reason else policy_reason
            if USE_BERT_POLICY_HEAD and req.direction == "prompt":
                # Reuse the already-computed ArmorBase embedding. The action
                # head adds no second language-model call; explicit context
                # review produced earlier in this function remains decisive.
                if task_review_reason:
                    decision = "review"
                elif decision == "block":
                    # A learned action head may refine non-block decisions,
                    # but it must not erase strong classifier block evidence.
                    # Only downgrade an isolated borderline single-label spike
                    # when the policy head is very confident the request is allow.
                    if (
                        policy_action == "allow"
                        and (policy_confidence_value or 0.0) >= 0.90
                        and len(findings) == 1
                        and max(scores.values(), default=0.0) < 0.95
                        and scores.get("prompt_injection", 0.0) < BLOCK_THRESHOLD
                    ):
                        decision = "allow"
                        findings = []
                        risk_score = min(risk_score, 0.45)
                elif "+pii-ner" in model_version:
                    # Keep the explicit entity detector's positive finding.
                    pass
                elif not findings and (
                    policy_action == "block"
                    or (policy_action in ("review", "flag") and max(scores.values(), default=0.0) < 0.45)
                ):
                    # Do not let a noisy/overfit policy head block when findings
                    # are empty, or hold/flag when calibrated classifier evidence
                    # is completely benign across all 6 threat labels.
                    pass
                else:
                    decision = policy_action
                    if decision == "allow" and findings:
                        findings = []
                        risk_score = min(risk_score, 0.45)
                route = "escalated"
                risk_score = max(risk_score, 0.8 if decision == "block" else 0.0)
                model_version += "+bert-policy-v1"
        except Exception:
            # Fail-open, same contract as every other optional inspection
            # stage here (bandit context build, semantic judge).
            policy_action = None

    # Hierarchical policy head, System B (queue/QUEUE-02's formulation-
    # bottleneck fix) -- logged alongside the flat head above for live
    # side-by-side comparison. System C (conformal + OOD + reference bank)
    # is not wired here yet: it needs saved calibration artifacts beyond
    # the two head checkpoints, not built until the A/B/C experiment
    # justifies that additional infrastructure.
    new_policy_action = None
    intervention_probability = None
    block_probability = None
    if pooled is not None:
        try:
            from armor.policy.engine import evaluate_hierarchical

            hier_result = evaluate_hierarchical(scores, pooled)
            new_policy_action = hier_result["policy_action"]
            intervention_probability = hier_result["intervention_probability"]
            block_probability = hier_result["block_probability"]
        except Exception:
            new_policy_action = None

    latency_ms = (time.perf_counter() - start) * 1000
    inference_trace["total_ms"] = latency_ms
    inference_trace["final_decision"] = decision
    if inference_trace.get("adaptive_shadow_status") == "evaluated":
        inference_trace["full_depth_decision"] = decision
        inference_trace["shadow_agreement"] = (
            inference_trace.get("shadow_decision") == decision
            if inference_trace.get("shadow_decision") is not None else None
        )
        if inference_trace.get("windowed_shadow_decision") is not None:
            inference_trace["windowed_shadow_agreement"] = (
                inference_trace["windowed_shadow_decision"] == decision
            )
    if token_context is not None:
        inference_trace["token_count"] = token_context.token_count
        inference_trace["input_chars"] = token_context.original_length
        inference_trace["token_bucket"] = (
            "0-128" if token_context.token_count is not None and token_context.token_count <= 128 else
            "129-512" if token_context.token_count is not None and token_context.token_count <= 512 else
            "513-1024" if token_context.token_count is not None and token_context.token_count <= 1024 else
            "1025-2048" if token_context.token_count is not None and token_context.token_count <= 2048 else
            "2049-4096" if token_context.token_count is not None and token_context.token_count <= 4096 else
            "4097-8192" if token_context.token_count is not None else "oversize"
        )
    result = InspectResponse(
        decision=decision,
        risk_score=risk_score,
        findings=findings,
        route=route,
        latency_ms=latency_ms,
        model_version=model_version,
    )
    result._log_id = log_store.log_entry(
        log_store.DB_PATH,
        source="inspect",
        direction=req.direction,
        content=req.content,
        decision=result.decision,
        risk_score=result.risk_score,
        route=result.route,
        findings=[f.model_dump() for f in result.findings],
        latency_ms=result.latency_ms,
        bandit_action=bandit_action,
        bandit_context=bandit_context_json,
        policy_action=policy_action,
        policy_confidence=policy_confidence_value,
        review_reason=review_reason,
        new_policy_action=new_policy_action,
        intervention_probability=intervention_probability,
        block_probability=block_probability,
        inference_trace=inference_trace,
    )
    return result


from armor.proxy.admin import router as admin_router  # noqa: E402
from armor.proxy.app import router as proxy_router  # noqa: E402

app.include_router(proxy_router)
app.include_router(admin_router)
