"""Fit small layer-6/layer-12 heads to the existing full-depth ArmorBase.

This produces an offline shadow candidate only. It does not modify the
deployed ArmorBase checkpoint and does not enable early exit in serving.
The target is the full-depth base classifier: an example is marked safe only
when every final threat probability is below a deliberately low ceiling.
"""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from armor.models.armor_base import get_armor_base
from armor.models.early_exit import (
    CHECKPOINT_PATH,
    EarlyExitHeads,
    candidate_from_logits,
    distillation_loss,
    save_exit_heads,
)
from armor.data.longpibench import load_grouped_examples as load_longpibench_examples
from armor.models.labels import LABELS
from armor.models.train_policy import load_training_rows


def _decision(scores: dict[str, float]) -> str:
    risk = max(scores.values(), default=0.0)
    return "block" if risk >= 0.8 else "flag" if risk >= 0.5 else "allow"


def collect_features(armor, rows: list[tuple[str, str, str]], safe_ceiling: float):
    from armor.models.armor_base import MAX_PREFLIGHT_CHARS, TokenContext

    eligible = [(i, row) for i, row in enumerate(rows) if len(row[0]) <= MAX_PREFLIGHT_CHARS]
    tokenizer_output = armor.tokenizer(
        [row[0] for _, row in eligible],
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )
    model_limit = min(
        armor.tokenizer.model_max_length,
        getattr(armor.encoder.config, "max_position_embeddings", armor.tokenizer.model_max_length),
    )
    encoded = []
    for (row_index, row), token_ids in zip(eligible, tokenizer_output["input_ids"]):
        if len(token_ids) <= model_limit:
            encoded.append((row_index, row, token_ids))
    encoded.sort(key=lambda item: len(item[2]))
    groups = []
    short = [item for item in encoded if len(item[2]) <= 512]
    groups.extend(short[start:start + 64] for start in range(0, len(short), 64))
    groups.extend([[item] for item in encoded if len(item[2]) > 512])

    hidden6 = [None] * len(rows)
    hidden12 = [None] * len(rows)
    teacher_logits = [None] * len(rows)
    teacher_safe = [False] * len(rows)
    identities = [None] * len(rows)
    sources = [None] * len(rows)
    for group in groups:
        max_len = max(len(item[2]) for item in group)
        pad_id = int(getattr(armor.tokenizer, "pad_token_id", 0) or 0)
        ids = torch.full((len(group), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros((len(group), max_len), dtype=torch.long)
        for local_i, (_row_i, _row, token_ids) in enumerate(group):
            ids[local_i, :len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
            mask[local_i, :len(token_ids)] = 1
        context = TokenContext(
            inputs={"input_ids": ids, "attention_mask": mask},
            token_count=max_len,
            original_length=max(len(item[1][0]) for item in group),
        )
        state = armor.begin_encoder(context)
        armor.continue_encoder(state, 6)
        features6 = state.hidden_states[:, 0, :].detach().cpu()
        armor.continue_encoder(state, 12)
        features12 = state.hidden_states[:, 0, :].detach().cpu()
        armor.continue_encoder(state, len(armor.encoder.layers))
        logits = armor.head(state.hidden_states[:, 0, :]).detach().cpu()
        for local_i, (row_index, row, _token_ids) in enumerate(group):
            hidden6[row_index] = features6[local_i]
            hidden12[row_index] = features12[local_i]
            teacher_logits[row_index] = logits[local_i]
            raw = dict(zip(LABELS, torch.sigmoid(logits[local_i]).tolist()))
            if armor.fingerprint is not None:
                from armor.models.calibration import calibrate_scores

                full_scores = {
                    label: value["score"]
                    for label, value in calibrate_scores(raw, armor.fingerprint).items()
                }
            else:
                full_scores = raw
            is_safe = bool(max(full_scores.values(), default=0.0) < safe_ceiling
                           and _decision(full_scores) == "allow")
            teacher_safe[row_index] = is_safe
            identities[row_index] = hashlib.sha256(row[0].encode("utf-8")).hexdigest()
            sources[row_index] = row[2]
    valid = [i for i, value in enumerate(hidden6) if value is not None]
    if not valid:
        raise RuntimeError("no in-context training examples available")
    return (
        torch.stack([hidden6[i] for i in valid]),
        torch.stack([hidden12[i] for i in valid]),
        torch.stack([teacher_logits[i] for i in valid]),
        torch.tensor([teacher_safe[i] for i in valid], dtype=torch.float32),
        [identities[i] for i in valid], [sources[i] for i in valid],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default="docs/early_exit_candidate_20260925.json")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=413)
    parser.add_argument("--safe-ceiling", type=float, default=0.5)
    parser.add_argument("--features-cache", default="/tmp/armor_early_exit_features_20260925.npz")
    parser.add_argument("--unsafe-weight", type=float, default=8.0)
    args = parser.parse_args()
    torch.set_num_threads(max(1, int(os.environ.get("ARMOR_TRAIN_THREADS", "4"))))

    if args.epochs < 1 or not 0 < args.safe_ceiling <= 0.5:
        parser.error("epochs must be positive and safe-ceiling must be in (0, 0.5]")
    if args.unsafe_weight < 1:
        parser.error("unsafe-weight must be at least 1")
    rows = load_training_rows(include_pseudo_model=True, include_jev_reviews=True)
    # Real long-context signal: logged production rows rarely exceed a few
    # hundred tokens, so exit heads never saw content anywhere near the
    # 8,192-token serving limit. LongPIBench's papers genuinely run
    # 6,000-7,400 tokens with a real (not padded) injected instruction.
    rows += [
        (text, "allow", f"longpibench:{group}")
        for text, _label, _mask, group in load_longpibench_examples()
    ]
    # Keep one representative per exact prompt, preventing duplicate rows
    # from leaking between the fit and validation partitions.
    unique = {}
    for row in rows:
        unique.setdefault(hashlib.sha256(row[0].encode("utf-8")).hexdigest(), row)
    rows = list(unique.values())
    row_manifest = hashlib.sha256(
        "\n".join(sorted(unique)).encode("ascii")
    ).hexdigest()
    random.Random(args.seed).shuffle(rows)
    if len(rows) < 20:
        raise RuntimeError(f"need at least 20 unique feedback rows, got {len(rows)}")

    armor = get_armor_base()
    if not armor.checkpoint_loaded:
        raise RuntimeError("refusing to distill an untrained ArmorBase checkpoint")
    cache_path = Path(args.features_cache)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
    else:
        cached = None
    if (
        cached is not None
        and str(cached["encoder_fingerprint"].item()) == armor.encoder_fingerprint
        and str(cached["row_manifest"].item()) == row_manifest
        and float(cached["safe_ceiling"].item()) == args.safe_ceiling
    ):
        f6 = torch.from_numpy(cached["layer6"].copy())
        f12 = torch.from_numpy(cached["layer12"].copy())
        teacher = torch.from_numpy(cached["teacher_logits"].copy())
        safe = torch.from_numpy(cached["teacher_safe"].copy())
        identities = cached["identities"].tolist()
        sources = cached["sources"].tolist()
        elapsed = float(cached["feature_extraction_seconds"].item())
    else:
        started = time.perf_counter()
        f6, f12, teacher, safe, identities, sources = collect_features(
            armor, rows, args.safe_ceiling
        )
        elapsed = time.perf_counter() - started
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            layer6=f6.numpy(), layer12=f12.numpy(), teacher_logits=teacher.numpy(),
            teacher_safe=safe.numpy(), identities=np.asarray(identities),
            sources=np.asarray(sources), encoder_fingerprint=np.asarray(armor.encoder_fingerprint),
            row_manifest=np.asarray(row_manifest), safe_ceiling=np.asarray(args.safe_ceiling),
            feature_extraction_seconds=np.asarray(elapsed),
        )
    # Split by document group: all variants of one LongPIBench item stay on
    # the same side (they differ by one injected sentence). Other rows are
    # their own group. Rebuild sources from rows so a stale cache still works.
    source_by_id = {hashlib.sha256(r[0].encode("utf-8")).hexdigest(): r[2] for r in rows}
    sources = [source_by_id[i] for i in identities]
    groups = [s if s.startswith("longpibench:") else i for s, i in zip(sources, identities)]
    unique_groups = sorted(set(groups))
    random.Random(args.seed).shuffle(unique_groups)
    val_groups = set(unique_groups[int(len(unique_groups) * 0.8):])
    indices = list(range(len(identities)))
    train_idx = torch.tensor([i for i in indices if groups[i] not in val_groups])
    val_idx = torch.tensor([i for i in indices if groups[i] in val_groups])
    if val_idx.numel() == 0:
        raise RuntimeError("validation split is empty")

    torch.manual_seed(args.seed)
    heads = EarlyExitHeads(f6.shape[-1])
    optimizer = torch.optim.AdamW(heads.parameters(), lr=2e-3, weight_decay=1e-4)
    best_state = None
    best_loss = float("inf")
    for _epoch in range(args.epochs):
        heads.train()
        optimizer.zero_grad()
        out6 = heads.layer6(f6[train_idx])
        out12 = heads.layer12(f12[train_idx])
        loss = distillation_loss(
            out6, teacher[train_idx], safe[train_idx], unsafe_weight=args.unsafe_weight
        )
        loss = loss + distillation_loss(
            out12, teacher[train_idx], safe[train_idx], unsafe_weight=args.unsafe_weight
        )
        loss.backward()
        optimizer.step()
        heads.eval()
        with torch.no_grad():
            val_loss = distillation_loss(
                heads.layer6(f6[val_idx]), teacher[val_idx], safe[val_idx],
                unsafe_weight=args.unsafe_weight,
            ) + distillation_loss(
                heads.layer12(f12[val_idx]), teacher[val_idx], safe[val_idx],
                unsafe_weight=args.unsafe_weight,
            )
        score = float(val_loss)
        if score < best_loss:
            best_loss = score
            best_state = {k: v.detach().clone() for k, v in heads.state_dict().items()}

    heads.load_state_dict(best_state)
    heads.eval()
    checkpoint_path = args.output or CHECKPOINT_PATH
    save_exit_heads(heads, checkpoint_path, encoder_fingerprint=armor.encoder_fingerprint)
    metrics = {"candidate_only": True, "teacher": "full_depth_ArmorBase_classifier",
               "source_rows": len(rows), "in_context_rows": len(identities),
               "train_rows": int(train_idx.numel()), "validation_rows": int(val_idx.numel()),
               "validation_teacher_safe_rows": int(safe[val_idx].sum()),
               "validation_non_safe_rows": int(val_idx.numel() - safe[val_idx].sum()),
               "teacher_safe_ceiling": args.safe_ceiling, "best_validation_loss": best_loss,
               "unsafe_weight": args.unsafe_weight,
               "encoder_fingerprint": armor.encoder_fingerprint,
               "feature_extraction_seconds": elapsed,
               "source_provenance_counts": {
                   s: [x.split(":")[0] for x in sources].count(s)
                   for s in sorted({x.split(":")[0] for x in sources})},
               "split": "grouped_by_longpibench_item",
               "checkpoints": {}}
    for depth, features, head in ((6, f6, heads.layer6), (12, f12, heads.layer12)):
        with torch.no_grad():
            outputs = head(features[val_idx])
            candidates = []
            teacher_agrees = []
            for row_i, output in zip(val_idx.tolist(), outputs):
                raw = dict(zip(LABELS, torch.sigmoid(output[:-1]).tolist()))
                if armor.fingerprint is not None:
                    from armor.models.calibration import calibrate_scores

                    calibrated = {
                        label: value["score"]
                        for label, value in calibrate_scores(raw, armor.fingerprint).items()
                    }
                else:
                    calibrated = raw
                candidates.append(candidate_from_logits(
                    output, depth=depth, calibrated_threat_scores=calibrated
                ))
                teacher_raw = dict(zip(LABELS, torch.sigmoid(teacher[row_i]).tolist()))
                teacher_scores = {
                    label: value["score"]
                    for label, value in calibrate_scores(teacher_raw, armor.fingerprint).items()
                } if armor.fingerprint is not None else teacher_raw
                teacher_agrees.append(_decision(teacher_scores) == "allow")
        emitted = [i for i, candidate in enumerate(candidates) if candidate is not None]
        unsafe = [i for i in emitted if not bool(safe[val_idx[i]])]
        safe_probs = torch.sigmoid(outputs[:, -1]).tolist()
        sweeps = []
        calibrated_per_row = []
        for output in outputs:
            raw = dict(zip(LABELS, torch.sigmoid(output[:-1]).tolist()))
            calibrated_per_row.append({
                label: value["score"]
                for label, value in calibrate_scores(raw, armor.fingerprint).items()
            } if armor.fingerprint is not None else raw)
        for safe_cutoff in (0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.9, 0.99):
            chosen = [
                i for i, probability in enumerate(safe_probs)
                if probability >= safe_cutoff
                and max(calibrated_per_row[i].values(), default=0.0) < 0.5
            ]
            sweeps.append({
                "safe_confidence_threshold": safe_cutoff,
                "candidate_allow_count": len(chosen),
                "candidate_allow_rate": len(chosen) / max(1, len(candidates)),
                "teacher_non_safe_early_allow_count": sum(not bool(safe[val_idx[i]]) for i in chosen),
                "teacher_allow_agreement_count": sum(bool(safe[val_idx[i]]) for i in chosen),
            })
        metrics["checkpoints"][str(depth)] = {
            "candidate_allow_count": len(emitted),
            "candidate_allow_rate": len(emitted) / max(1, len(candidates)),
            "teacher_non_safe_early_allow_count": len(unsafe),
            "teacher_non_safe_early_allow_ids": [identities[int(val_idx[i])] for i in unsafe],
            "mean_teacher_agreement_on_candidates": sum(
                teacher_agrees[i] for i in emitted
            ) / max(1, len(emitted)),
            "safe_confidence_quantiles": {
                "p50": float(torch.tensor(safe_probs).quantile(0.5)),
                "p90": float(torch.tensor(safe_probs).quantile(0.9)),
                "p99": float(torch.tensor(safe_probs).quantile(0.99)),
                "max": max(safe_probs),
            },
            "confidence_threshold_sweep_without_threat_ceiling": sweeps,
        }
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"candidate checkpoint: {checkpoint_path}")
    print(f"report: {report}")


if __name__ == "__main__":
    main()
