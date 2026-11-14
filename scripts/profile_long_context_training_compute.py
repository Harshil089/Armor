"""Measure same-encoder feature-extraction cost at serving-length buckets."""

import json
import time
from pathlib import Path

import torch

from armor.models.armor_base import ArmorBase


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "docs/long_context_gate_v1_20260925.jsonl"
OUTPUT = ROOT / "docs/long_context_training_compute_v1_20260925.json"


def main():
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite {OUTPUT}")
    if not torch.backends.mps.is_available():
        print("MPS unavailable; no accelerator training profile recorded.")
        return
    rows = [json.loads(line) for line in CORPUS.read_text().splitlines()]
    chosen = {}
    for bucket in ("1k", "2k", "4k", "6k", "8k"):
        chosen[bucket] = next(r for r in rows if r["bucket"] == bucket and r["scenario_id"] == "benign_document")

    model = ArmorBase(device="mps", backend="pytorch")
    if not model.checkpoint_loaded:
        raise RuntimeError("Expected the current ArmorBase checkpoint")
    # Warm allocator/dispatch on a small input, then time one full-depth
    # frozen-feature pass per realistic length. This approximates the default
    # frozen-encoder training precompute; it does not measure backward/update.
    short_ids = model.tokenizer.encode(chosen["1k"]["content"], add_special_tokens=False)[:510]
    short = model.tokenizer.decode(short_ids)
    warm = model.prepare_token_context(short)
    model.predict_with_embedding(warm)
    torch.mps.synchronize()

    results = {"512": {"tokens": warm.token_count, "purpose": "warmup_only"}}
    context = model.prepare_token_context(short)
    torch.mps.synchronize()
    started = time.perf_counter()
    model.predict_with_embedding(context)
    torch.mps.synchronize()
    results["512"] = {"tokens": context.token_count,
                       "total_forward_ms": (time.perf_counter() - started) * 1000,
                       "encoder_ms": context.timings_ms.get("encoder_ms"),
                       "tokenization_ms": context.timings_ms.get("tokenization_ms")}
    for bucket, row in chosen.items():
        context = model.prepare_token_context(row["content"])
        if context.oversize:
            raise RuntimeError(f"Unexpected oversize profile input: {bucket}")
        torch.mps.synchronize()
        started = time.perf_counter()
        model.predict_with_embedding(context)
        torch.mps.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        results[bucket] = {
            "tokens": context.token_count,
            "total_forward_ms": elapsed,
            "encoder_ms": context.timings_ms.get("encoder_ms"),
            "tokenization_ms": context.timings_ms.get("tokenization_ms"),
        }
        print(f"{bucket}: {context.token_count} tokens, {elapsed:.1f} ms", flush=True)

    examples = 20153
    suggested = {"<=512": .40, "513-1024": .25, "1025-2048": .15,
                 "2049-4096": .10, "4097-6144": .07, "6145-8192": .03}
    counts = {k: round(examples * v) for k, v in suggested.items()}
    # MPS dispatch is asynchronous; use wall time around the synchronized
    # complete forward, not TokenContext's unsynchronized encoder timer.
    cost_map = {"<=512": results["512"]["total_forward_ms"],
                "513-1024": results["1k"]["total_forward_ms"],
                "1025-2048": results["2k"]["total_forward_ms"],
                "2049-4096": results["4k"]["total_forward_ms"],
                "4097-6144": results["6k"]["total_forward_ms"],
                "6145-8192": results["8k"]["total_forward_ms"]}
    estimated_hours = sum(counts[k] * cost_map[k] for k in counts) / 3_600_000
    report = {
        "measured_at": "2026-09-25", "device": "MPS", "backend": "PyTorch",
        "purpose": "frozen encoder feature extraction feasibility; not serving backend",
        "model": "existing ArmorBase / ModernBERT only", "checkpoint_loaded": True,
        "measured_forward_passes_per_bucket": 1, "warmup_passes_excluded": 1,
        "measurements": results, "source_training_rows": examples,
        "source_length_distribution": {
            "mean": 117.8, "median": 42, "p90": 326, "p95": 422, "p99": 873,
            "over_512": 663, "over_2048": 17, "max": 4674,
        },
        "illustrative_curriculum": suggested,
        "illustrative_rows_per_bucket": counts,
        "estimated_single_pass_wall_clock_hours": estimated_hours,
        "estimate_limit": "One sample per length; deliberately conservative sequential extrapolation of synchronized full-forward latency (not the unsynchronized MPS encoder timer). Excludes batching speedup, data assembly, backward work, validation, and retries; long-bucket variance may be substantial.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"estimated_single_pass_wall_clock_hours": estimated_hours,
                      "output": str(OUTPUT.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
