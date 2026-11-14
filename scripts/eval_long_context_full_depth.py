"""Measure the unchanged production /v1/inspect full-depth PyTorch path."""

import hashlib
import json
import os
import random
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "docs/long_context_gate_v1_20260925.jsonl"
MANIFEST = ROOT / "docs/long_context_gate_v1_20260925_manifest.json"
OUTPUT = ROOT / "docs/long_context_full_depth_baseline_v1_20260925.json"
REPORT = ROOT / "docs/long_context_full_depth_baseline_v1_20260925.md"


def _percentile(values, p):
    ordered = sorted(values)
    if not ordered:
        return None
    at = (len(ordered) - 1) * p
    lo, hi = int(at), min(int(at) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def _scores(rows):
    labels = ("allow", "flag", "block", "review")
    correct = sum(r["expected_action"] == r["decision"] for r in rows)
    f1s = []
    for label in labels:
        tp = sum(r["expected_action"] == label and r["decision"] == label for r in rows)
        fp = sum(r["expected_action"] != label and r["decision"] == label for r in rows)
        fn = sum(r["expected_action"] == label and r["decision"] != label for r in rows)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    block_tp = sum(r["expected_action"] == r["decision"] == "block" for r in rows)
    block_fn = sum(r["expected_action"] == "block" and r["decision"] != "block" for r in rows)
    block_fp = sum(r["expected_action"] != "block" and r["decision"] == "block" for r in rows)
    allow_tp = sum(r["expected_action"] == r["decision"] == "allow" for r in rows)
    allow_fp = sum(r["expected_action"] != "allow" and r["decision"] == "allow" for r in rows)
    flag_tp = sum(r["expected_action"] == r["decision"] == "flag" for r in rows)
    flag_fn = sum(r["expected_action"] == "flag" and r["decision"] != "flag" for r in rows)
    return {
        "n": len(rows), "accuracy": correct / len(rows) if rows else None,
        "macro_f1": statistics.mean(f1s) if rows else None,
        "block_recall": block_tp / (block_tp + block_fn) if block_tp + block_fn else None,
        "block_precision": block_tp / (block_tp + block_fp) if block_tp + block_fp else None,
        "allow_precision": allow_tp / (allow_tp + allow_fp) if allow_tp + allow_fp else None,
        "flag_recall": flag_tp / (flag_tp + flag_fn) if flag_tp + flag_fn else None,
        "review_rate": sum(r["decision"] == "review" for r in rows) / len(rows) if rows else None,
        "expected_counts": dict(Counter(r["expected_action"] for r in rows)),
        "predicted_counts": dict(Counter(r["decision"] for r in rows)),
    }


def _latency(rows):
    fields = ("total_ms", "fast_scan_ms", "preflight_ms", "tokenization_ms",
              "encoder_ms", "classifier_head_ms", "policy_head_ms")
    result = {}
    for field in fields:
        values = [float(r["trace"].get(field, 0.0) or 0.0) for r in rows]
        result[field] = {
            "mean": statistics.mean(values) if values else None,
            "p50": _percentile(values, .50), "p95": _percentile(values, .95),
            "p99": _percentile(values, .99), "max": max(values) if values else None,
        }
    return result


def main():
    # Mirror the local production launch profile and explicitly keep adaptive
    # paths off. No external review bridge is called by this benchmark.
    os.environ["ARMOR_USE_BASE_MODEL"] = "1"
    os.environ["ARMOR_BERT_POLICY_HEAD"] = "1"
    os.environ["ARMOR_ADAPTIVE_SHADOW"] = "0"
    os.environ["ARMOR_BASE_BACKEND"] = "pytorch"
    os.environ["ARMOR_BASE_DEVICE"] = "cpu"
    os.environ["ARMOR_SEMANTIC_JUDGE_ENABLED"] = "0"

    raw = CORPUS.read_bytes()
    manifest = json.loads(MANIFEST.read_text())
    if hashlib.sha256(raw).hexdigest() != manifest["corpus_sha256"]:
        raise SystemExit("Frozen corpus digest does not match manifest")
    samples = [json.loads(line) for line in raw.decode().splitlines()]
    if not samples or any(s.get("label_source") != "synthetic" for s in samples):
        raise SystemExit("Unexpected empty corpus or label provenance")

    from armor.api import main as api
    from armor.api.schemas import InspectRequest
    from armor.models.armor_base import CHECKPOINT_PATH, get_armor_base
    from armor.observability import log_store
    from armor.policy import engine as policy_engine

    captured = []
    stage_timings = {}

    def timed(function, name):
        def wrapped(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                stage_timings[name] = stage_timings.get(name, 0.0) + (time.perf_counter() - started) * 1000
        return wrapped

    # The existing API trace does not yet split its tiny policy heads from
    # encoder time. Instrument those existing functions here without changing
    # model decisions or live request code.
    policy_engine.evaluate = timed(policy_engine.evaluate, "flat_policy_ms")
    policy_engine.evaluate_hierarchical = timed(policy_engine.evaluate_hierarchical, "hierarchical_policy_ms")

    def capture_log(*args, **kwargs):
        trace = dict(kwargs.get("inference_trace") or {})
        trace.update(stage_timings)
        trace["policy_head_ms"] = sum(stage_timings.values())
        captured.append(trace)
        return len(captured)

    # Keep the actual inspect logic and response timing, but avoid writing full
    # source texts into a local SQLite log during a repeatable benchmark.
    log_store.log_entry = capture_log
    model = get_armor_base()
    if getattr(model, "backend", None) != "pytorch" or getattr(model, "device", None) != "cpu":
        raise RuntimeError("Baseline did not load the required PyTorch CPU encoder")

    warm = next(s for s in samples if s["bucket"] == "1k" and s["expected_action"] == "allow")
    api.inspect(InspectRequest(direction="prompt", content=warm["content"]))
    warm_long = next(s for s in samples if s["bucket"] == "8k" and s["expected_action"] == "allow")
    api.inspect(InspectRequest(direction="prompt", content=warm_long["content"]))

    ordered = list(samples)
    random.Random(20260925).shuffle(ordered)
    results = []
    for i, sample in enumerate(ordered, 1):
        before = len(captured)
        stage_timings.clear()
        started = time.perf_counter()
        response = api.inspect(InspectRequest(direction="prompt", content=sample["content"]))
        elapsed = (time.perf_counter() - started) * 1000
        trace = captured[-1] if len(captured) > before else {}
        results.append({
            "sample_id": sample["sample_id"], "bucket": sample["bucket"],
            "token_count": sample["token_count"], "scenario_id": sample["scenario_id"],
            "variant": sample["variant"], "label_source": sample["label_source"],
            "expected_action": sample["expected_action"], "decision": response.decision,
            "route": response.route, "risk_score": response.risk_score,
            "mismatch": response.decision != sample["expected_action"],
            "outer_elapsed_ms": elapsed, "trace": trace,
        })
        if i % 10 == 0 or i == len(ordered):
            OUTPUT.with_suffix(".progress.json").write_text(json.dumps(results) + "\n")
            print(f"completed {i}/{len(ordered)}", flush=True)

    groups = {bucket: [r for r in results if r["bucket"] == bucket]
              for bucket in ("1k", "2k", "4k", "6k", "8k")}
    summary = {
        "corpus_freeze_id": manifest["freeze_id"],
        "corpus_sha256": manifest["corpus_sha256"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "label_interpretation": "Scenario-contract agreement; all rows synthetic, not human-ground-truth accuracy.",
        "serving_profile": {
            "path": "armor.api.main.inspect via /v1/inspect implementation",
            "base_model_enabled": True, "adaptive_shadow": False,
            "adaptive_or_windowed_production": False, "backend": model.backend,
            "device": model.device, "policy_action_head": True,
            "semantic_judge": False, "checkpoint_path": str(CHECKPOINT_PATH),
            "checkpoint_sha256": hashlib.sha256(Path(CHECKPOINT_PATH).read_bytes()).hexdigest(),
            "checkpoint_loaded": bool(getattr(model, "checkpoint_loaded", False)),
            "warmup_requests_excluded": 2, "request_order_seed": 20260925,
            "database_writes_disabled_for_benchmark": True,
        },
        "overall": {**_scores(results), "latency": _latency(results)},
        "compute_path_counts": dict(Counter(r["trace"].get("approximate_compute_path", "missing") for r in results)),
        "route_counts": dict(Counter(r["route"] for r in results)),
        "by_bucket": {
            bucket: {**_scores(rows), "token_count_min": min(r["token_count"] for r in rows),
                     "token_count_max": max(r["token_count"] for r in rows),
                     "latency": _latency(rows)}
            for bucket, rows in groups.items()
        },
        "rows": results,
    }
    OUTPUT.write_text(json.dumps(summary, indent=2) + "\n")
    OUTPUT.with_suffix(".progress.json").unlink(missing_ok=True)

    def ms(v):
        return "—" if v is None else f"{v:.1f}"
    def pct(v):
        return "—" if v is None else f"{100*v:.1f}%"
    lines = [
        "# Frozen 1K–8K full-depth baseline (2026-09-25)", "",
        f"Corpus: `{manifest['freeze_id']}` · {manifest['rows']} rows · SHA-256 `{manifest['corpus_sha256']}`.",
        "All labels are **synthetic scenario contracts**, not human-ground-truth labels. Metrics below are model-to-contract agreement.",
        "Serving path: unchanged `/v1/inspect` logic, current ArmorBase checkpoint, PyTorch CPU, adaptive shadow/windowing off, local production policy action head on. Two warmups excluded. Database writes disabled so full prompt text is not duplicated in benchmark logs; response timing is the same pre-log timing returned by the API.", "",
        "## Overall", "",
        f"n={summary['overall']['n']} · accuracy {pct(summary['overall']['accuracy'])} · macro F1 {pct(summary['overall']['macro_f1'])} · block recall {pct(summary['overall']['block_recall'])} · block precision {pct(summary['overall']['block_precision'])} · allow precision {pct(summary['overall']['allow_precision'])} · flag recall {pct(summary['overall']['flag_recall'])} · review rate {pct(summary['overall']['review_rate'])}.",
        f"Compute paths: `{json.dumps(summary['compute_path_counts'], sort_keys=True)}`. Routes: `{json.dumps(summary['route_counts'], sort_keys=True)}`.", "",
        "| Bucket | n | tokens | mean ms | p50 ms | p95 ms | p99 ms | max ms | acc. | block recall | allow precision | review | tokenization mean | encoder mean | classifier head mean | policy heads mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bucket, data in summary["by_bucket"].items():
        lat = data["latency"]
        lines.append(
            f"| {bucket} | {data['n']} | {data['token_count_min']}–{data['token_count_max']} | "
            f"{ms(lat['total_ms']['mean'])} | {ms(lat['total_ms']['p50'])} | {ms(lat['total_ms']['p95'])} | "
            f"{ms(lat['total_ms']['p99'])} | {ms(lat['total_ms']['max'])} | {pct(data['accuracy'])} | "
            f"{pct(data['block_recall'])} | {pct(data['allow_precision'])} | {pct(data['review_rate'])} | "
            f"{ms(lat['tokenization_ms']['mean'])} | {ms(lat['encoder_ms']['mean'])} | "
            f"{ms(lat['classifier_head_ms']['mean'])} | {ms(lat['policy_head_ms']['mean'])} |"
        )
    lines += ["", "Stage timings are per-request trace values emitted by the current inspection path. `total_ms` is API compute/response latency before the separate SQLite audit-log write.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    print(f"wrote {REPORT.relative_to(ROOT)}")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
