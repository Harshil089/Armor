"""Full-depth single pass vs batched 512-token windows: latency and decisions.

Corpus rows: {"content", "label"} JSONL, or `--longpibench N` for the first N
items of each LongPIBench suite (clean=allow, injected=block; the deployed
head never trained on it, so it is held out).
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

from armor.models.armor_base import get_armor_base
from armor.models.calibration import calibrate_scores
from armor.models.labels import LABELS


def decision(scores):
    risk = max(scores.values(), default=0.0)
    block = float(os.environ.get("ARMOR_BASE_BLOCK_THRESHOLD", "0.8"))
    flag = float(os.environ.get("ARMOR_BASE_THRESHOLD", "0.5"))
    return "block" if risk >= block else "flag" if risk >= flag else "allow"


def rows_from_longpibench(n):
    from armor.data.longpibench import load_grouped_examples

    rows = []
    for text, label, _mask, group in load_grouped_examples():
        if int(group.split(":")[1]) < n:
            rows.append({"content": text, "label": "block" if label[0] > 0.5 else "allow"})
    return rows


def stats(values):
    ordered = sorted(values)
    return {
        "mean_ms": round(statistics.fmean(ordered)),
        "p50_ms": round(statistics.median(ordered)),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]),
        "max_ms": round(ordered[-1]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus")
    p.add_argument("--longpibench", type=int)
    p.add_argument("--report", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--overlap", type=int, default=96)
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    if args.longpibench:
        rows = rows_from_longpibench(args.longpibench)
    else:
        rows = [json.loads(l) for l in Path(args.corpus).read_text().splitlines() if l.strip()]
    model = get_armor_base()

    def calibrated(raw):
        if model.fingerprint is None:
            return raw
        return {k: v["score"] for k, v in calibrate_scores(raw, model.fingerprint).items()}

    labels = list(LABELS)
    rules = {
        "max": lambda m: m.max(dim=0).values,
        "second_max": lambda m: m.topk(min(2, len(m)), dim=0).values[-1],
        "mean": lambda m: m.mean(dim=0),
        "pi_max_only": lambda m: torch.cat([m[:, :1].max(dim=0).values, torch.zeros(len(labels) - 1)]),
    }
    bars = {"0.5/0.8": (0.5, 0.8), "0.9/0.97": (0.9, 0.97)}

    def decide(vec, bar):
        risk = float(vec.max())
        return "block" if risk >= bar[1] else "flag" if risk >= bar[0] else "allow"

    full_ms, win_ms, out = [], [], []
    for row in rows:
        ctx = model.prepare_token_context(row["content"])
        if ctx.oversize:
            continue
        t = time.perf_counter()
        raw_full, _ = model.predict_with_embedding(ctx)
        full_ms.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        win, _ = model.window_scores(ctx, args.window, args.overlap, args.batch)
        win_ms.append((time.perf_counter() - t) * 1000)
        cal = torch.tensor([
            [calibrated(dict(zip(labels, r.tolist())))[k] for k in labels] for r in win
        ])
        rec = {"label": row["label"], "tokens": ctx.token_count, "windows": len(win),
               "full": decide(torch.tensor([calibrated(raw_full)[k] for k in labels]), bars["0.5/0.8"])}
        for rule, fn in rules.items():
            for bar_name, bar in bars.items():
                rec[f"{rule}@{bar_name}"] = decide(fn(cal), bar)
        out.append(rec)

    def quality(key):
        blocks = [r for r in out if r["label"] == "block"]
        allows = [r for r in out if r["label"] == "allow"]
        return {
            "block_recall": round(sum(r[key] != "allow" for r in blocks) / max(1, len(blocks)), 3),
            "false_allow": sum(r[key] == "allow" for r in blocks),
            "benign_flagged": sum(r[key] != "allow" for r in allows),
            "of_benign": len(allows), "of_block": len(blocks),
        }

    report = {
        "rows": len(out), "window": args.window, "overlap": args.overlap,
        "median_tokens": statistics.median(r["tokens"] for r in out),
        "full_depth_latency": stats(full_ms), "windowed_latency": stats(win_ms),
        "speedup_mean": round(statistics.fmean(full_ms) / statistics.fmean(win_ms), 2),
        "full_depth": quality("full"),
        "windowed": {k: quality(k) for k in out[0] if "@" in k},
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
