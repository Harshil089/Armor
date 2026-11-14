"""Frozen A/B/C model-path comparison: full, adaptive depth, adaptive windows."""

import argparse
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from armor.inference.adaptive import run_adaptive_single, run_adaptive_windows
from armor.models.armor_base import get_armor_base
from armor.models.calibration import calibrate_scores


def _decision(scores):
    risk = max(scores.values(), default=0.0)
    finding = float(os.environ.get("ARMOR_BASE_THRESHOLD", "0.5"))
    block = float(os.environ.get("ARMOR_BASE_BLOCK_THRESHOLD", "0.8"))
    return "block" if risk >= block else "flag" if risk >= finding else "allow"


def _stats(values):
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered), "mean_ms": statistics.fmean(ordered),
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        "p99_ms": ordered[min(len(ordered) - 1, math.ceil(0.99 * len(ordered)) - 1)],
        "max_ms": max(ordered),
    }


def _bucket(tokens):
    for upper, name in ((128, "0-128"), (512, "129-512"), (1024, "513-1024"),
                        (2048, "1025-2048"), (4096, "2049-4096"), (8192, "4097-8192")):
        if tokens <= upper:
            return name
    return "oversize"


def _quality(rows, prediction_key):
    confusion = Counter()
    for row in rows:
        confusion[(row["label"], row[prediction_key])] += 1
    labels = sorted({row["label"] for row in rows} | {row[prediction_key] for row in rows})
    accuracy = sum(n for (expected, predicted), n in confusion.items() if expected == predicted) / max(1, len(rows))
    per_label_f1 = []
    for label in labels:
        tp = confusion[(label, label)]
        fp = sum(n for (expected, predicted), n in confusion.items() if predicted == label and expected != label)
        fn = sum(n for (expected, predicted), n in confusion.items() if expected == label and predicted != label)
        per_label_f1.append(2 * tp / max(1, 2 * tp + fp + fn))
    blocks = [r for r in rows if r["label"] == "block"]
    return {
        "accuracy_vs_gate_labels": accuracy,
        "macro_f1_vs_gate_labels": statistics.fmean(per_label_f1) if per_label_f1 else 0.0,
        "block_recall_vs_gate_labels": sum(r[prediction_key] == "block" for r in blocks) / max(1, len(blocks)),
        "dangerous_false_allow_count_vs_gate_labels": sum(
            r["label"] == "block" and r[prediction_key] == "allow" for r in rows
        ),
        "agreement_with_full_depth": sum(r[prediction_key] == r["full"] for r in rows) / max(1, len(rows)),
        "confusion": {f"{a}->{b}": n for (a, b), n in sorted(confusion.items())},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="docs/long_text_final_gate_corpus_20260923.jsonl")
    parser.add_argument("--report", default="docs/adaptive_inference_gate_20260925.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--max-deep-windows", type=int, default=4)
    parser.add_argument("--exit-policy", choices=("single", "patience"), default="patience")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    samples = [json.loads(line) for line in Path(args.corpus).read_text().splitlines() if line.strip()]
    model = get_armor_base()
    rows = []
    a_times, b_times, c_times, token_times = [], [], [], []
    by_bucket = defaultdict(lambda: {"a": [], "b": [], "c": []})
    exit_counts = Counter()
    window_counts = Counter()
    for sample in samples:
        text = sample["content"]
        context = model.prepare_token_context(text)
        token_times.append(context.timings_ms.get("tokenization_ms", 0.0))
        if context.oversize:
            full_decision = adaptive_single = adaptive_windowed = "review"
            a_ms = b_ms = c_ms = 0.0
            b_trace = c_trace = {"approximate_compute_path": "oversize_review"}
        else:
            started = time.perf_counter()
            raw, _embedding = model.predict_with_embedding(context)
            a_ms = (time.perf_counter() - started) * 1000
            calibrated = (
                {k: v["score"] for k, v in calibrate_scores(raw, model.fingerprint).items()}
                if model.fingerprint is not None else raw
            )
            full_decision = _decision(calibrated)
            started = time.perf_counter()
            b = run_adaptive_single(model, context, exit_policy=args.exit_policy)
            b_ms = (time.perf_counter() - started) * 1000
            adaptive_single, b_trace = b.decision, b.trace
            if context.token_count <= args.window_size:
                # The windowed variant uses the same single-request adaptive
                # path at or below its training-context window size.
                adaptive_windowed, c_trace, c_ms = adaptive_single, b_trace, b_ms
            else:
                started = time.perf_counter()
                c = run_adaptive_windows(
                    model, context, model.tokenizer,
                    window_size=args.window_size, overlap=args.overlap,
                    max_windows=24, max_full_depth_windows=args.max_deep_windows,
                    exit_policy=args.exit_policy,
                )
                c_ms = (time.perf_counter() - started) * 1000
                adaptive_windowed, c_trace = c.decision, c.trace
        bucket = _bucket(context.token_count if context.token_count is not None else 8193)
        row = {
            "sample_id": sample.get("sample_id"), "label": sample["label"],
            "token_count": context.token_count, "bucket": bucket,
            "full": full_decision, "adaptive_depth": adaptive_single,
            "adaptive_windows": adaptive_windowed,
            "latency_ms": {"full": a_ms, "adaptive_depth": b_ms, "adaptive_windows": c_ms,
                           "tokenization": token_times[-1]},
            "adaptive_depth_trace": b_trace, "window_trace": c_trace,
        }
        rows.append(row)
        a_times.append(a_ms); b_times.append(b_ms); c_times.append(c_ms)
        by_bucket[bucket]["a"].append(a_ms)
        by_bucket[bucket]["b"].append(b_ms)
        by_bucket[bucket]["c"].append(c_ms)
        exit_counts.update({"layer6": int(b_trace.get("windows_exited_at_6", 0)),
                            "layer12": int(b_trace.get("windows_exited_at_12", 0)),
                            "full_depth": int(b_trace.get("windows_full_depth", 0))})
        window_counts.update({"windows": int(c_trace.get("window_count", 0)),
                              "layer6": int(c_trace.get("windows_exited_at_6", 0)),
                              "layer12": int(c_trace.get("windows_exited_at_12", 0)),
                              "full_depth": int(c_trace.get("windows_full_depth", 0)),
                              "deep": int(c_trace.get("deep_window_count", 0))})
    report = {
        "gate": str(args.corpus), "human_ground_truth": False,
        "encoder_fingerprint": model.encoder_fingerprint,
        "rows": len(rows), "window_size": args.window_size, "overlap": args.overlap,
        "max_full_depth_windows": args.max_deep_windows,
        "exit_policy": args.exit_policy,
        "exit_safe_threshold_6": float(os.environ.get("ARMOR_EXIT_SAFE_THRESHOLD_6", "0.999")),
        "exit_safe_threshold_12": float(os.environ.get("ARMOR_EXIT_SAFE_THRESHOLD_12", "0.999")),
        "max_calibrated_threat_score": float(os.environ.get(
            "ARMOR_EXIT_MAX_THREAT", os.environ.get("ARMOR_BASE_THRESHOLD", "0.5")
        )),
        "timings": {"tokenization": _stats(token_times), "A_full_depth": _stats(a_times),
                    "B_adaptive_depth": _stats(b_times), "C_adaptive_windows": _stats(c_times)},
        "by_token_bucket": {
            key: {name: _stats(values[name]) for name in ("a", "b", "c")}
            for key, values in sorted(by_bucket.items())
        },
        "metrics": {
            "A_full_depth": _quality([{**r, "prediction": r["full"]} for r in rows], "prediction"),
            "B_adaptive_depth": _quality([{**r, "prediction": r["adaptive_depth"]} for r in rows], "prediction"),
            "C_adaptive_windows": _quality([{**r, "prediction": r["adaptive_windows"]} for r in rows], "prediction"),
        },
        "compute_distribution": {"B_adaptive_depth": dict(exit_counts),
                                 "C_adaptive_windows": dict(window_counts)},
        "per_sample": rows,
    }
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_sample"}, indent=2))
    print(f"report: {path}")


if __name__ == "__main__":
    main()
