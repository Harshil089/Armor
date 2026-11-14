"""Build and train a candidate on realistic 512–8K context variants.

The generated carrier text stays in a local temporary gzip file because it is
derived from several upstream datasets with different reuse terms. Only the
recipe, provenance, digests, and aggregate counts are written into the repo.
"""

import argparse
import gzip
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from armor.data.long_context_curriculum import MIX, POSITIONS, assign_rows
from armor.models.armor_base import MODEL_NAME
from armor.models.labels import LABELS
from armor.models.train import load_training_examples_with_provenance, train
from scripts.build_long_context_gate import _body, _carriers, _compose


MANIFEST_PATH = ROOT / "docs/long_context_training_curriculum_v3_20260925.json"
DEFAULT_CACHE = Path("/tmp/armor_long_context_training_v3_20260925.jsonl.gz")
DEFAULT_FEATURES_CACHE = Path("/tmp/armor_long_context_features_v3_20260925.pt")
DEFAULT_CHECKPOINT = ROOT / "armor/models/checkpoints/long_context_candidate_v3.pt"
def _embed(tokenizer, candidates, row_text, row_index, target, low, high, seed):
    payload_tokens = len(tokenizer.encode(row_text, add_special_tokens=False))
    prefix = "Reference notes from the same conversation:\n\n"
    prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
    body_target = target - payload_tokens - prefix_tokens - 30
    if body_target < 64:
        raise RuntimeError("payload leaves too little room for a long carrier")
    for attempt in range(5):
        body, carrier_ids = _body(tokenizer, candidates, seed + row_index * 37 + attempt * 997,
                                  body_target)
        body_len = len(tokenizer.encode(body, add_special_tokens=False))
        available = max(0, target - payload_tokens - prefix_tokens - 8)
        positions = [p for p in POSITIONS if p <= available]
        position_seed = int(hashlib.sha256(
            f"{seed}|{row_index}|{row_text}".encode("utf-8")
        ).hexdigest()[:16], 16)
        position = positions[random.Random(position_seed).randrange(len(positions))] if positions else 0
        fraction = min(0.999, position / max(1, body_len))
        content, offsets = _compose(tokenizer, body, prefix, [(fraction, row_text)])
        token_count = len(tokenizer.encode(content, add_special_tokens=True))
        if low <= token_count <= high:
            return content, token_count, position, carrier_ids, offsets
        body_target += low - token_count if token_count < low else high - token_count
    raise RuntimeError(f"could not fit payload into {low}..{high} token bucket")


def build_curriculum(seed=20260925, limit=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    source_rows = load_training_examples_with_provenance()
    if limit:
        source_rows = source_rows[:limit]
    source_lengths = [len(tokenizer.encode(row[0], add_special_tokens=True)) for _source, row in source_rows]
    if max(source_lengths) > 8192:
        raise RuntimeError(f"existing training row exceeds ModernBERT context: {max(source_lengths)}")
    assignments = assign_rows(source_rows, source_lengths, seed)
    carriers = _carriers(tokenizer)
    rng = random.Random(seed)
    rng.shuffle(carriers)

    rows = []
    position_counts = Counter()
    label_counts = Counter()
    mask_counts = Counter()
    carrier_ids_used = Counter()
    positive_by_position = {label: Counter() for label in LABELS}
    masked_by_position = {label: Counter() for label in LABELS}
    processed = 0
    for bucket_index, bucket_rows in enumerate(assignments):
        bucket_name, _share, low, high = MIX[bucket_index]
        for row_index in bucket_rows:
            source, (text, labels, masks) = source_rows[row_index]
            original_count = source_lengths[row_index]
            position = None
            carrier_ids = []
            offsets = []
            if bucket_index == 0 or original_count > 512:
                content = text
                token_count = original_count
            else:
                target = random.Random(seed + row_index * 101).randint(low + 128, high - 128)
                content, token_count, position, carrier_ids, offsets = _embed(
                    tokenizer, carriers, text, row_index, target, low, high, seed
                )
            if not low <= token_count <= high:
                raise RuntimeError(
                    f"row {row_index} assigned {bucket_name} but measured {token_count} tokens"
                )
            sample_id = hashlib.sha256(
                f"{source}|{row_index}|{hashlib.sha256(text.encode()).hexdigest()}".encode()
            ).hexdigest()[:24]
            for i, label in enumerate(LABELS):
                if masks[i]:
                    mask_counts[label] += 1
                    if position is not None:
                        masked_by_position[label][str(position)] += 1
                    if labels[i]:
                        label_counts[label] += 1
                        if position is not None:
                            positive_by_position[label][str(position)] += 1
            if position is not None:
                position_counts[str(position)] += 1
            carrier_ids_used.update(carrier_ids)
            rows.append({
                "sample_id": sample_id, "source_training_dataset": source,
                "label_source": "inherited_from_existing_training_row",
                "label_basis": "source module and label/mask vectors from the existing ArmorBase training mix",
                "augmentation": "synthetic_position_randomized_reviewed_carrier" if carrier_ids else "none",
                "bucket": bucket_name, "token_count": token_count,
                "original_token_count": original_count,
                "inserted_payload_token_offset": offsets[0].get("token_offset") if offsets else None,
                "requested_position_tokens": position,
                "carrier_message_ids": carrier_ids,
                "labels": labels, "masks": masks, "content": content,
            })
            processed += 1
            if processed % 1000 == 0:
                print(f"assembled {processed}/{len(source_rows)} curriculum rows", flush=True)

    rows.sort(key=lambda row: row["sample_id"])
    counts = Counter(row["bucket"] for row in rows)
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps({k: v for k, v in row.items() if k != "content"},
                                 sort_keys=True, separators=(",", ":")).encode())
        digest.update(hashlib.sha256(row["content"].encode()).digest())
    report = {
        "curriculum_id": "armor_long_context_training_v3_20260925",
        "created_at": "2026-09-25", "seed": seed, "rows": len(rows),
        "training_example_sha256": digest.hexdigest(),
        "label_source": "inherited from existing training sources; augmented text itself is synthetic",
        "original_training_rows_by_source": dict(Counter(source for source, _ in source_rows)),
        "original_length_min": min(source_lengths), "original_length_max": max(source_lengths),
        "original_rows_over_512": sum(n > 512 for n in source_lengths),
        "bucket_target_shares": {name: share for name, share, _low, _high in MIX},
        "target_length_policy": "seeded uniform target within each non-short bucket, with 128-token edge margins; retain every existing >512-token source row intact",
        "bucket_counts": dict(counts),
        "actual_token_ranges": {name: [min(r["token_count"] for r in rows if r["bucket"] == name),
                                        max(r["token_count"] for r in rows if r["bucket"] == name)]
                                for name, *_ in MIX},
        "positioned_augmented_rows": sum(position_counts.values()),
        "requested_payload_position_counts": dict(position_counts),
        "supervised_positive_counts": dict(label_counts),
        "supervised_masked_row_counts": dict(mask_counts),
        "positive_position_counts_by_label": {
            label: dict(counts) for label, counts in positive_by_position.items()
        },
        "masked_position_counts_by_label": {
            label: dict(counts) for label, counts in masked_by_position.items()
        },
        "carrier_dataset": "OpenAssistant/oasst1",
        "carrier_revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
        "carrier_filter": "assistant; accepted; not deleted; >=2 reviewers; toxicity <0.05; Armor DLP-clean",
        "carrier_message_id_count": len(carrier_ids_used),
        "carrier_text_not_adjudicated_for_full_Armor_taxonomy": True,
        "full_depth_candidate_only": True, "same_existing_model_only": True,
    }
    return rows, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--limit", type=int, help="small deterministic smoke-build; never used for training")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--features-cache", type=Path, default=DEFAULT_FEATURES_CACHE)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warm-start-head", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train", action="store_true", dest="do_train")
    args = parser.parse_args()
    if args.limit and args.do_train:
        raise SystemExit("refusing to train from a limited smoke-build")
    if not args.prepare_only and not args.do_train:
        parser.error("choose --prepare-only or --train")
    if args.do_train:
        if not args.manifest.exists() or not args.cache.exists():
            raise SystemExit("prepare the full curriculum first with --prepare-only")
        report = json.loads(args.manifest.read_text())
        if report.get("curriculum_id") != "armor_long_context_training_v3_20260925":
            raise SystemExit("prepared manifest is not the expected frozen curriculum v3")
        cache_digest = hashlib.sha256()
        with gzip.open(args.cache, "rt", encoding="utf-8") as stream:
            for line in stream:
                cache_digest.update(line.encode())
        cache_sha = cache_digest.hexdigest()
        if cache_sha != report.get("cache_sha256"):
            raise SystemExit("prepared curriculum cache digest mismatch")
        report.setdefault("training_attempts", []).append({
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "warm_start_head": args.warm_start_head,
            "fine_tune_layers": 0,
            "candidate_only": True,
        })
        report["training_status"] = "training"
        args.manifest.write_text(json.dumps(report, indent=2) + "\n")
    else:
        if args.manifest.exists() or args.cache.exists():
            raise SystemExit("refusing to overwrite an existing curriculum manifest/cache")
        rows, report = build_curriculum(args.seed, args.limit)
        report.update({"cache_path": str(args.cache), "cache_is_local_ephemeral": True,
                       "prepare_only": True, "training_status": "prepared"})
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with gzip.open(args.cache, "wt", encoding="utf-8") as stream:
            for row in rows:
                line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                stream.write(line)
                digest.update(line.encode())
        report["cache_sha256"] = digest.hexdigest()
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if args.prepare_only:
            return

    if args.checkpoint.exists():
        raise SystemExit(f"refusing to overwrite candidate checkpoint {args.checkpoint}")
    examples = []
    with gzip.open(args.cache, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            examples.append((row["content"], row["labels"], row["masks"]))
    try:
        train(
            seed=args.seed, epochs=args.epochs, lr=args.lr,
            output_path=str(args.checkpoint), fine_tune_layers=0,
            warm_start_head=args.warm_start_head,
            precompute_batch_size=64, max_length=8192, examples_override=examples,
            embeddings_cache_path=str(args.features_cache),
        )
    except KeyboardInterrupt:
        report["training_status"] = "interrupted"
        report["last_training_attempt_status"] = "interrupted"
        report["training_attempts"][-1].update({
            "outcome": "interrupted",
            "feature_cache_saved": args.features_cache.exists(),
            "checkpoint_saved": args.checkpoint.exists(),
        })
        report["candidate_training_note"] = (
            "Training was paused by the user during frozen feature extraction; "
            "no candidate checkpoint was saved."
        )
        args.manifest.write_text(json.dumps(report, indent=2) + "\n")
        raise
    except SystemExit as exc:
        report["training_status"] = "regression_gate_failed" if exc.code else "failed"
        report["training_exit_code"] = exc.code
        report["last_training_attempt_status"] = report["training_status"]
        args.manifest.write_text(json.dumps(report, indent=2) + "\n")
        raise
    except Exception as exc:
        report["training_status"] = "failed"
        report["last_training_attempt_status"] = "failed"
        report["last_training_error"] = f"{type(exc).__name__}: {exc}"
        args.manifest.write_text(json.dumps(report, indent=2) + "\n")
        raise
    report["training_status"] = "candidate_trained"
    report["candidate_checkpoint"] = str(args.checkpoint)
    report["candidate_checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    args.manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"training_status": report["training_status"],
                      "candidate_checkpoint_sha256": report["candidate_checkpoint_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
