"""Trainer for the Armor Base classifier.

The default path keeps the ModernBERT encoder frozen and trains only the
linear head. ``fine_tune_layers`` opts into bounded partial fine-tuning of
the final encoder layers after the data-first ARCHIVE-QUEUE-03 work plateaued.

Run: python -m armor.models.train
"""

import glob
import gc
import json
import os
import random
import shutil
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from armor.data.agentdojo import load_examples as load_agentdojo_examples
from armor.data.agentharm import load_examples as load_agentharm_examples
from armor.data.agent_safety_bench import load_examples as load_agent_safety_bench_examples
from armor.data.civil_comments import load_examples as load_civil_comments_examples
from armor.data.coedit_noisy_negatives import load_examples as load_coedit_noisy_examples
from armor.data.harmbench_harmful_content import (
    load_examples as load_harmbench_harmful_content_examples,
)
from armor.data.harm_request_sources import LOADERS as HARM_REQUEST_LOADERS
from armor.data.injecagent import load_examples as load_injecagent_examples
from armor.data.jailbreak_wild import load_examples as load_jailbreak_wild_examples
from armor.data.longpibench import load_examples as load_longpibench_examples
from armor.data.synthetic_jailbreak_canonical import (
    generate_examples as generate_jailbreak_canonical_examples,
)
from armor.data.synthetic_harmful_content import (
    generate_examples as generate_harmful_content_examples,
)
from armor.data.aegis_content_safety import load_examples as load_aegis_examples
from armor.data.oasst_chat_negatives import load_examples as load_oasst_chat_examples
from armor.data.synthetic_short_negatives import (
    generate_examples as generate_short_negative_examples,
)
from armor.data.synthetic_long_negatives import (
    generate_examples as generate_long_negative_examples,
)
from armor.data.synthetic_topic_adjacent_negatives import (
    generate_examples as generate_topic_adjacent_negative_examples,
)
from armor.data.public import load_training_examples
from armor.data.synthetic_dangerous_tool_action import (
    generate_destructive_focus_examples,
    generate_examples as generate_dangerous_tool_action_examples,
    generate_external_comm_focus_examples,
)
from armor.data.synthetic_obfuscation import generate_examples as generate_obfuscation_examples
from armor.data.synthetic_pii import generate_examples
from armor.data.toolsafety_dangerous_tool_action import (
    load_examples as load_toolsafety_examples,
)
from armor.data.toolemu import load_examples as load_toolemu_examples
from armor.eval.regression_gate import check_regression, evaluate as regression_evaluate
from armor.models.armor_base import MODEL_NAME
from armor.models.labels import LABELS

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "head.pt")
ARCHIVE_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "archive")
MAX_ARCHIVED = 5


class _ExampleDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_training_examples_with_provenance() -> list[tuple[str, tuple]]:
    """Return the standard training mix with its existing loader provenance.

    This preserves the historical row order and labels. Long-context
    augmentation uses the source name to make lineage auditable instead of
    collapsing the assembled rows into an anonymous list.
    """
    groups = [
        ("registered_public_sources", load_training_examples("train")),
        ("synthetic_pii", generate_examples()),
        ("agentharm", load_agentharm_examples()),
        ("agentdojo", load_agentdojo_examples()),
        ("agent_safety_bench", load_agent_safety_bench_examples()),
        ("civil_comments", load_civil_comments_examples()),
        ("coedit_noisy_negatives", load_coedit_noisy_examples()),
        ("harmbench", load_harmbench_harmful_content_examples()),
        ("injecagent", load_injecagent_examples()),
        ("synthetic_dangerous_tool_action", generate_dangerous_tool_action_examples()),
        ("synthetic_destructive_tool_action", generate_destructive_focus_examples()),
        ("synthetic_external_communication_action", generate_external_comm_focus_examples()),
        ("synthetic_obfuscation", generate_obfuscation_examples()),
        ("toolsafety", load_toolsafety_examples()),
        ("toolemu", load_toolemu_examples()),
        ("jailbreak_wild", load_jailbreak_wild_examples()),
        ("longpibench", load_longpibench_examples()),
        ("synthetic_jailbreak_canonical", generate_jailbreak_canonical_examples()),
        ("synthetic_harmful_content", generate_harmful_content_examples()),
        ("aegis", load_aegis_examples()),
        ("oasst_chat_negatives", load_oasst_chat_examples()),
        ("synthetic_short_negatives", generate_short_negative_examples()),
        ("synthetic_long_negatives", generate_long_negative_examples()),
        ("synthetic_topic_adjacent_negatives", generate_topic_adjacent_negative_examples()),
        *((name, load()) for name, load in HARM_REQUEST_LOADERS.items()),
    ]
    return [(source, example) for source, examples in groups for example in examples]


def _collate(batch, tokenizer):
    return _collate_at_length(batch, tokenizer, max_length=512)


def _collate_at_length(batch, tokenizer, *, max_length: int):
    texts, labels, masks = zip(*batch)
    inputs = tokenizer(
        list(texts), return_tensors="pt", padding=True,
        truncation=True, max_length=max_length,
        # MPS SDPA is sensitive to continually changing sequence shapes.
        # Long curriculum rows are singletons, so exact padding creates a new
        # shape for nearly every forward; bounded bucketing limits allocator
        # churn while adding at most 127 masked tokens per row.
        pad_to_multiple_of=128 if max_length > 512 else None,
    )
    return inputs, torch.tensor(labels), torch.tensor(masks)


def _resolve_device() -> str:
    # MPS is the preferred development accelerator when available. Long
    # curriculum rows use singleton forwards, so _collate_at_length pads
    # them to fixed 128-token buckets to avoid thousands of distinct SDPA
    # shapes. Progress is reported every 25 MPS rows so a long command-buffer
    # wait is observable instead of looking like a deadlock.
    #
    # ARMOR_TRAIN_DEVICE=cpu forces CPU if this ever needs re-diagnosing.
    override = os.environ.get("ARMOR_TRAIN_DEVICE")
    if override:
        return override
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def _precompute_embeddings(
    examples, encoder, tokenizer, device: str, batch_size: int, max_length: int = 512,
    partial_cache_path: str | None = None,
    partial_cache_metadata: dict[str, str | int] | None = None,
    partial_cache_interval: int = 500,
):
    """Runs the frozen encoder over every example exactly once and caches
    the pooled embeddings, instead of recomputing them every epoch.

    ponytail: the encoder never updates (frozen, wrapped in no_grad already)
    so its output for a given input is identical in every epoch — repeating
    that forward pass 5x (once per epoch) was pure wasted compute. This
    turns an O(epochs * dataset) encoder cost into O(dataset), leaving only
    the tiny linear head to actually retrain per epoch.

    Three things keep this tractable on MPS, all load-bearing (see
    _resolve_device for the measurements that motivated them):

    1. Length-sorted batching. `_collate` pads to the longest member of a
       batch, and this corpus spans ~326 to ~11977 characters, so randomly
       ordered batches pad nearly everything up to a near-max sequence —
       a 64x512 forward costs 10.5s versus 1.8s at 64x128. Sorting by
       length first makes each batch roughly uniform, so almost none of
       that padding compute gets spent.
    2. Per-batch offload to CPU. The embedding cache is ~25k x 768 floats
       and only the tiny head needs it on-device per step; holding every
       batch's activations on the GPU for the whole pass is what pushed
       into memory pressure.
    3. Draining the allocator cache periodically, for the same reason.

    The sort is undone before returning, and that matters: it must stay a
    pure speed optimization with no effect on the result. A first version
    returned rows still in sorted order, which silently made training
    order depend on every row's text length — so editing one label's
    wording reshuffled which rows batched together for *every other*
    label, changing their converged weights. That surfaced as
    dangerous_tool_action falling 0.213 -> 0.045 on a run that only
    touched obfuscation_encoding's carrier phrasing (caught by the
    regression gate, which refused the checkpoint). Restoring the original
    order keeps batch composition stable against unrelated edits.
    """
    if max_length > 512:
        lengths = [
            len(tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length))
            for text, _labels, _masks in examples
        ]
        order = sorted(range(len(examples)), key=lambda i: lengths[i])
    else:
        lengths = None
        order = sorted(range(len(examples)), key=lambda i: len(examples[i][0]))
    ordered = [examples[i] for i in order]
    if max_length > 512:
        # The MPS profile confirmed an 8K singleton forward, while a batch of
        # two 8K rows was slower per row. Keep proven <=512 batching, then run
        # longer sequences singly until higher long-length batch sizes show a
        # measured throughput win and fit the 8 GB development device.
        groups, short_group = [], []
        for position, original_index in enumerate(order):
            if lengths[original_index] <= 512:
                short_group.append(position)
                if len(short_group) == batch_size:
                    groups.append(short_group)
                    short_group = []
            else:
                if short_group:
                    groups.append(short_group)
                    short_group = []
                groups.append([position])
        if short_group:
            groups.append(short_group)
        total_batches = len(groups)
    else:
        loader = DataLoader(
            _ExampleDataset(ordered), batch_size=batch_size, shuffle=False,
            collate_fn=lambda batch: _collate_at_length(batch, tokenizer, max_length=max_length),
        )
        total_batches = len(loader)
    all_pooled, all_labels, all_masks = [], [], []
    completed_batches = 0
    if partial_cache_path and os.path.exists(partial_cache_path):
        partial = torch.load(partial_cache_path, map_location="cpu", weights_only=True)
        if partial.get("metadata") != partial_cache_metadata:
            raise RuntimeError("partial long-context feature cache does not match model/data/checkpoint")
        completed_batches = partial["next_batch"]
        if not 0 <= completed_batches <= total_batches:
            raise RuntimeError("partial long-context feature cache has invalid progress")
        all_pooled = partial["pooled_chunks"]
        all_labels = partial["label_chunks"]
        all_masks = partial["mask_chunks"]
        print(f"resuming frozen feature extraction at batch {completed_batches}/{total_batches}", flush=True)

    if max_length > 512:
        loader = DataLoader(
            _ExampleDataset(ordered), batch_sampler=groups[completed_batches:],
            collate_fn=lambda batch: _collate_at_length(batch, tokenizer, max_length=max_length),
        )
    for batch_index, (inputs, labels, masks) in enumerate(loader, start=completed_batches):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # On CPU, the CLS slice is a view into the full sequence activation.
        # For singleton batches, `.contiguous()` may be a no-op because the
        # leading dimension has size one; clone to keep only the pooled vector.
        pooled = encoder(**inputs).last_hidden_state[:, 0, :].clone()
        all_pooled.append(pooled.cpu())
        all_labels.append(labels)
        all_masks.append(masks)
        if (partial_cache_path and max_length > 512
                and (batch_index + 1) % partial_cache_interval == 0):
            parent = os.path.dirname(partial_cache_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            temporary_path = partial_cache_path + ".tmp"
            torch.save({
                "metadata": partial_cache_metadata,
                "next_batch": batch_index + 1,
                "pooled_chunks": all_pooled,
                "label_chunks": all_labels,
                "mask_chunks": all_masks,
            }, temporary_path)
            os.replace(temporary_path, partial_cache_path)
        if device == "mps" and batch_index % 25 == 0:
            torch.mps.empty_cache()
        if max_length > 512:
            interval = 25 if device == "mps" else 100
            if (batch_index + 1) % interval == 0 or batch_index + 1 == total_batches:
                print(
                    f"long-context feature batches: {batch_index + 1}/{total_batches} "
                    f"(padded_tokens={inputs['input_ids'].shape[1]})",
                    flush=True,
                )

    # Undo the sort: inverse[original_index] = position in sorted output.
    inverse = torch.empty(len(order), dtype=torch.long)
    inverse[torch.as_tensor(order, dtype=torch.long)] = torch.arange(len(order))
    return (
        torch.cat(all_pooled)[inverse],
        torch.cat(all_labels)[inverse],
        torch.cat(all_masks)[inverse],
    )


def _embedding_cache_metadata(examples, max_length: int) -> dict[str, str | int]:
    import hashlib

    digest = hashlib.sha256()
    for text, labels, masks in examples:
        digest.update(hashlib.sha256(text.encode("utf-8")).digest())
        digest.update(json.dumps([labels, masks], separators=(",", ":"), allow_nan=False).encode())
    checkpoint_digest = hashlib.sha256()
    with open(CHECKPOINT_PATH, "rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            checkpoint_digest.update(chunk)
    return {
        "model_name": MODEL_NAME,
        "max_length": max_length,
        "examples_sha256": digest.hexdigest(),
        "checkpoint_sha256": checkpoint_digest.hexdigest(),
    }


def _multilabel_loss(
    logits: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor, loss_fn, pos_weight=None
) -> torch.Tensor:
    """Mean loss per label (only over rows masked for that label), then
    averaged equally across labels — not a single sum/divide over every
    (row, label) cell in the batch.

    ponytail: the naive version (sum all masked-cell losses / total masked
    cells) lets whichever label has the most rows in the dataset dominate
    the gradient every batch, regardless of how many labels are actually
    being trained. With harmful_content at ~64% of all examples, that
    starved sparser labels (dangerous_tool_action, ~7%) of real gradient
    share — this is what was silently causing its weak/inverted signal.

    Equal per-label weighting alone doesn't protect a label whose positives
    are a small minority *within that label's own rows* — a flood of true
    negatives (e.g. chat/moderation datasets with no positives for a sparse
    label like prompt_injection) can still pull its decision boundary toward
    under-predicting. `pos_weight[i]` (BCEWithLogitsLoss-style: applied only
    to positive rows) counters that per label.
    """
    per_label_losses = []
    for i in range(logits.shape[1]):
        label_mask = masks[:, i]
        if label_mask.sum() > 0:
            cell_loss = loss_fn(logits[:, i], labels[:, i])
            if pos_weight is not None:
                weight = 1.0 + (pos_weight[i] - 1.0) * labels[:, i]
                cell_loss = cell_loss * weight
            per_label_losses.append((cell_loss * label_mask).sum() / label_mask.sum())
    if not per_label_losses:
        return torch.tensor(0.0, requires_grad=True)
    return torch.stack(per_label_losses).mean()


def _archive_current_checkpoint() -> None:
    if not os.path.exists(CHECKPOINT_PATH):
        return
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    dest = os.path.join(ARCHIVE_DIR, f"head_{int(time.time())}.pt")
    shutil.copy2(CHECKPOINT_PATH, dest)
    archived = sorted(glob.glob(os.path.join(ARCHIVE_DIR, "head_*.pt")))
    for old in archived[:-MAX_ARCHIVED]:
        os.remove(old)


def _configure_encoder(encoder, fine_tune_layers: int) -> list[torch.nn.Parameter]:
    """Freeze the encoder, then optionally unfreeze its final N layers."""
    if fine_tune_layers < 0 or fine_tune_layers > len(encoder.layers):
        raise ValueError(f"fine_tune_layers must be between 0 and {len(encoder.layers)}")

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)

    if fine_tune_layers:
        for layer in encoder.layers[-fine_tune_layers:]:
            layer.train()
            for param in layer.parameters():
                param.requires_grad_(True)
        encoder.final_norm.train()
        for param in encoder.final_norm.parameters():
            param.requires_grad_(True)

    return [param for param in encoder.parameters() if param.requires_grad]


def train(
    epochs: int = 5,
    batch_size: int = 16,
    precompute_batch_size: int = 64,
    lr: float = 1e-3,
    encoder_lr: float = 2e-5,
    fine_tune_layers: int = 0,
    seed: int = 42,
    force: bool = False,
    output_path: str = CHECKPOINT_PATH,
    warm_start_path: str | None = None,
    warm_start_head: bool = False,
    examples_override: list | None = None,
    max_length: int = 512,
    embeddings_cache_path: str | None = None,
) -> None:
    """warm_start_path lets a partial fine-tune start from a different
    checkpoint than the deployed one (default: same as CHECKPOINT_PATH,
    unchanged behavior). Needed because fine_tune_layers continuing on top
    of an already-fine-tuned checkpoint compounds encoder drift across
    rounds — stacking fine-tunes on the same 2 layers 3 rounds deep caused
    a real regression-gate failure that persisted even after the new
    round's training data was redesigned (2026-09-20). Point this at the
    last frozen-head-only checkpoint (no encoder_state_dict) to fine-tune
    fresh from the pretrained encoder instead, using the full current
    example pool in one shot — the regression gate still always compares
    against the currently *deployed* checkpoint, not the warm-start source.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    device = _resolve_device()
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)
    encoder.eval()
    head = torch.nn.Linear(encoder.config.hidden_size, len(LABELS)).to(device)
    checkpoint = None
    deployed_checkpoint = None
    baseline_metrics = None
    if os.path.exists(CHECKPOINT_PATH):
        deployed_checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        if deployed_checkpoint.get("encoder_state_dict"):
            encoder.load_state_dict(deployed_checkpoint["encoder_state_dict"], strict=False)
        encoder.eval()
        baseline_head = torch.nn.Linear(encoder.config.hidden_size, len(LABELS)).to(device)
        baseline_head.load_state_dict(deployed_checkpoint["state_dict"])
        baseline_head.eval()
        # Reuse the candidate encoder for baseline scoring. Loading two full
        # ModernBERT copies at once needlessly doubles training memory, which
        # is particularly costly when feature extraction includes 8K inputs.
        baseline_metrics = regression_evaluate(baseline_head, encoder, tokenizer, device=device)
        print(f"regression set (current checkpoint): {baseline_metrics}")
        del baseline_head

    warm_start_source = warm_start_path or CHECKPOINT_PATH
    if os.path.exists(warm_start_source):
        checkpoint = (
            deployed_checkpoint if os.path.abspath(warm_start_source) == os.path.abspath(CHECKPOINT_PATH)
            and deployed_checkpoint is not None
            else torch.load(warm_start_source, map_location="cpu")
        )
        if os.path.abspath(warm_start_source) != os.path.abspath(CHECKPOINT_PATH):
            # A distinct warm start requires fresh pretrained weights. Drop
            # the baseline-loaded instance before constructing that same
            # single encoder again; never hold both encoder copies together.
            del encoder
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)
        if checkpoint.get("encoder_state_dict"):
            encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=False)
        if fine_tune_layers or warm_start_head:
            head.load_state_dict(checkpoint["state_dict"])
    elif fine_tune_layers:
        raise RuntimeError("partial fine-tuning requires an existing checkpoint")

    encoder_params = _configure_encoder(encoder, fine_tune_layers)
    optimizer_groups = [{"params": head.parameters(), "lr": lr}]
    if encoder_params:
        optimizer_groups.append({"params": encoder_params, "lr": encoder_lr})
    optimizer = torch.optim.Adam(optimizer_groups)

    examples = examples_override if examples_override is not None else [
        example for _source, example in load_training_examples_with_provenance()
    ]
    print(f"training examples: {len(examples)}")

    # Per-label loss is already averaged equally across labels (see
    # _multilabel_loss), but within a label a large corpus of hard negatives
    # (e.g. chat/moderation datasets that resemble injection-adjacent text)
    # can still pull the decision boundary toward under-predicting a sparse
    # positive class. pos_weight counteracts that by penalizing false
    # negatives on each label in proportion to how rare its positives are.
    pos_counts = [0] * len(LABELS)
    neg_counts = [0] * len(LABELS)
    for example in examples:
        labels, mask = example[1], example[2]
        for i in range(len(LABELS)):
            if mask[i] <= 0:
                continue
            if labels[i] > 0.5:
                pos_counts[i] += 1
            else:
                neg_counts[i] += 1
    pos_weight = torch.tensor(
        [
            min(neg_counts[i] / pos_counts[i], 20.0) if pos_counts[i] else 1.0
            for i in range(len(LABELS))
        ],
        device=device,
    )
    print(f"pos_weight per label: {dict(zip(LABELS, pos_weight.tolist()))}")
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    if fine_tune_layers:
        # Length buckets keep padding cost bounded. Shuffle whole buckets each
        # epoch so long examples do not always dominate the final updates.
        ordered = sorted(examples, key=lambda example: len(example[0]))
        fine_tune_batches = [
            ordered[start : start + batch_size]
            for start in range(0, len(ordered), batch_size)
        ]
        num_batches = len(fine_tune_batches)
        print(f"fine-tuning final {fine_tune_layers} encoder layers")
    else:
        # The frozen encoder is deterministic, so compute its output once.
        cache_metadata = (
            _embedding_cache_metadata(examples, max_length)
            if embeddings_cache_path and max_length > 512 else None
        )
        cached = None
        if cache_metadata and os.path.exists(embeddings_cache_path):
            cached = torch.load(embeddings_cache_path, map_location="cpu", weights_only=True)
            if cached.get("metadata") != cache_metadata:
                raise RuntimeError("long-context feature cache does not match model/data/checkpoint")
        if cached is not None:
            pooled_all = cached["pooled"]
            labels_all = cached["labels"]
            masks_all = cached["masks"]
            print(f"loaded frozen feature cache: {embeddings_cache_path}")
        else:
            pooled_all, labels_all, masks_all = _precompute_embeddings(
                examples, encoder, tokenizer, device, precompute_batch_size, max_length=max_length,
                partial_cache_path=(embeddings_cache_path + ".partial" if cache_metadata else None),
                partial_cache_metadata=cache_metadata,
                partial_cache_interval=100 if device == "mps" else 500,
            )
            if cache_metadata:
                cache_parent = os.path.dirname(embeddings_cache_path)
                if cache_parent:
                    os.makedirs(cache_parent, exist_ok=True)
                temporary_path = embeddings_cache_path + ".tmp"
                torch.save({"metadata": cache_metadata, "pooled": pooled_all,
                            "labels": labels_all, "masks": masks_all}, temporary_path)
                os.replace(temporary_path, embeddings_cache_path)
                partial_path = embeddings_cache_path + ".partial"
                if os.path.exists(partial_path):
                    os.unlink(partial_path)
                print(f"saved frozen feature cache: {embeddings_cache_path}")
        pooled_all = pooled_all.to(device)
        labels_all = labels_all.to(device)
        masks_all = masks_all.to(device)
        n = pooled_all.shape[0]
        num_batches = (n + batch_size - 1) // batch_size

    for epoch in range(epochs):
        total_loss = 0.0
        per_label_loss_sum = [0.0] * len(LABELS)
        per_label_batch_count = [0] * len(LABELS)
        if fine_tune_layers:
            bucket_order = list(range(num_batches))
            random.Random(seed + epoch).shuffle(bucket_order)
            batches = (
                _collate_at_length(fine_tune_batches[index], tokenizer, max_length=max_length)
                for index in bucket_order
            )
        else:
            permutation = torch.randperm(n, device=device)
            batches = (
                (pooled_all[idx], labels_all[idx], masks_all[idx])
                for start in range(0, n, batch_size)
                for idx in [permutation[start : start + batch_size]]
            )

        for batch_index, batch in enumerate(batches):
            if fine_tune_layers:
                inputs, labels, masks = batch
                inputs = {key: value.to(device) for key, value in inputs.items()}
                labels, masks = labels.to(device), masks.to(device)
                pooled = encoder(**inputs).last_hidden_state[:, 0, :]
            else:
                pooled, labels, masks = batch

            logits = head(pooled)
            loss = _multilabel_loss(logits, labels, masks, loss_fn, pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if device == "mps" and batch_index % 25 == 0:
                torch.mps.empty_cache()

            with torch.no_grad():
                for i in range(len(LABELS)):
                    label_mask = masks[:, i]
                    if label_mask.sum() > 0:
                        per_label_loss_sum[i] += (
                            (loss_fn(logits[:, i], labels[:, i]) * label_mask).sum() / label_mask.sum()
                        ).item()
                        per_label_batch_count[i] += 1

        per_label_avg = {
            LABELS[i]: (per_label_loss_sum[i] / per_label_batch_count[i] if per_label_batch_count[i] else float("nan"))
            for i in range(len(LABELS))
        }
        print(f"epoch {epoch + 1}/{epochs} loss={total_loss / num_batches:.4f} per_label={per_label_avg}")

    encoder.eval()
    head.eval()
    new_metrics = regression_evaluate(head, encoder, tokenizer, device=device)
    print(f"regression set (new head): {new_metrics}")

    if baseline_metrics is not None:
        regressed = check_regression(new_metrics, baseline_metrics)
        if regressed and not force:
            print(f"REGRESSION GATE FAILED on: {regressed} — refusing to overwrite {CHECKPOINT_PATH}")
            print("Pass force=True to train() to override.")
            sys.exit(1)
        if regressed:
            print(f"REGRESSION GATE FAILED on: {regressed} — overwriting anyway (force=True)")

    if output_path == CHECKPOINT_PATH:
        _archive_current_checkpoint()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "state_dict": {key: value.cpu() for key, value in head.state_dict().items()},
        "labels": LABELS,
        "model_name": MODEL_NAME,
    }
    encoder_state = dict(checkpoint.get("encoder_state_dict", {})) if checkpoint else {}
    if fine_tune_layers:
        encoder_state.update(
            {
                name: param.detach().cpu()
                for name, param in encoder.named_parameters()
                if param.requires_grad
            }
        )
    if encoder_state:
        payload["encoder_state_dict"] = encoder_state
        payload["fine_tune_layers"] = fine_tune_layers
    torch.save(payload, output_path)
    print(f"saved checkpoint to {output_path}")


if __name__ == "__main__":
    train()
