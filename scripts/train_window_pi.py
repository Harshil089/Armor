"""Window-level prompt_injection head: only the 512-token window holding the
injected sentence is positive; every other window (clean docs, and the rest of
injected docs) is negative. LongPIBench items < HOLDOUT are never trained on.

Replaces only the prompt_injection row of the deployed head; the other five
labels keep their weights. Candidate-only: writes --out, never the deployed head.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("ARMOR_LONGPIBENCH_HOLDOUT", "15")

import torch

from armor.data import longpibench as lpb
from armor.eval.regression_gate import check_regression, evaluate as regression_evaluate
from armor.models.armor_base import get_armor_base
from armor.models.labels import LABELS
from armor.models.train import CHECKPOINT_PATH, load_training_examples_with_provenance

PI = LABELS.index("prompt_injection")
ATTACKS = (None, "naive", "combine", "authority_spoof")
WINDOW, OVERLAP = 512, 96


def window_labels(ctx, text, payload):
    """(starts, labels) per window; label 1 = holds >=50% of the payload,
    0 = none of it, -1 = ambiguous partial overlap. None if payload not found."""
    body_n = ctx.inputs["input_ids"].shape[1] - 2
    span = WINDOW - 2
    starts = [0] if body_n <= span else list(range(0, max(1, body_n - OVERLAP), span - OVERLAP))
    labels = [0] * len(starts)
    if payload:
        p0 = text.find(payload)
        if p0 < 0:
            return None
        p1 = p0 + len(payload)
        off = ctx.offsets[0, 1:-1]
        hit = [i for i in range(body_n) if p0 <= int(off[i, 0]) < p1]
        if not hit:
            return None
        for w, s in enumerate(starts):
            inside = sum(s <= i < s + span for i in hit) / len(hit)
            labels[w] = 1 if inside >= 0.5 else (0 if inside == 0 else -1)
    return starts, labels


def doc_windows(model, text, payload):
    """(pooled [n, hidden] on cpu, labels [n]; -1 = ambiguous partial overlap)."""
    ctx = model.prepare_token_context(text)
    if ctx.oversize:
        return None
    found = window_labels(ctx, text, payload)
    if found is None:
        return None
    _scores, pooled = model.window_scores(ctx, WINDOW, OVERLAP, 8)
    return pooled.float().cpu(), found[1]


def iter_docs():
    """(text, group, attack-or-None, payload) for every LongPIBench row."""
    for n, (text, _l, _m, group) in enumerate(lpb.load_grouped_examples()):
        attack = ATTACKS[n % 4]
        suite = group.split(":")[0]
        payload = lpb._injection(suite, attack, lpb._DEFAULT_GOAL[suite]) if attack else ""
        if suite == "code":
            payload = payload.replace("\n", " ")
        yield text, group, attack, payload


def build_windows(model):
    pooled, labels, groups, skipped = [], [], [], 0
    for n, (text, group, attack, payload) in enumerate(iter_docs()):
        out = doc_windows(model, text, payload)
        if out is None:
            skipped += 1
            continue
        pooled.append(out[0])
        labels += out[1]
        groups += [(group, attack or "clean")] * len(out[1])
        if n % 100 == 0:
            print(f"windows: doc {n}/1600", flush=True)
    print(f"docs skipped (payload not found / oversize): {skipped}")
    return torch.cat(pooled), torch.tensor(labels), groups


def auc(pos, neg):
    pos, neg = torch.tensor(pos), torch.tensor(neg)
    return float((pos[:, None] > neg[None]).float().mean() + 0.5 * (pos[:, None] == neg[None]).float().mean())


def report_doc_level(name, scores, groups, holdout):
    """Doc score = max window PI score; clean vs injected, held-out items only."""
    docs = {}
    for s, (g, kind) in zip(scores.tolist(), groups):
        if int(g.split(":")[1]) < holdout:
            key = (g, kind)
            docs[key] = max(docs.get(key, 0.0), s)
    clean = [v for (g, k), v in docs.items() if k == "clean"]
    inj = [v for (g, k), v in docs.items() if k != "clean"]
    fpr5 = sorted(clean)[int(0.95 * len(clean)) - 1]
    print(f"{name}: doc AUC={auc(inj, clean):.3f} recall@clean-FPR<=5%={sum(v > fpr5 for v in inj) / len(inj):.3f} "
          f"| @0.5 recall={sum(v >= .5 for v in inj) / len(inj):.3f} clean_flagged={sum(v >= .5 for v in clean)}/{len(clean)} "
          f"({len(inj)} injected, {len(clean)} clean)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/window_pi_head.pt")
    ap.add_argument("--features", default="/content/feats_8k.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    holdout = int(os.environ["ARMOR_LONGPIBENCH_HOLDOUT"])
    torch.manual_seed(42)

    model = get_armor_base()
    deployed = torch.load(CHECKPOINT_PATH, map_location="cpu")
    W0, b0 = deployed["state_dict"]["weight"][PI].clone(), deployed["state_dict"]["bias"][PI].clone()

    pooled, wl, groups = build_windows(model)
    keep = wl >= 0
    is_train = torch.tensor([int(g.split(":")[1]) >= holdout for g, _ in groups])
    print(f"windows: {len(wl)} | train pos={(wl[is_train] == 1).sum()} neg={(wl[is_train] == 0).sum()} "
          f"| ambiguous dropped={(~keep).sum()}")

    # short-text PI rows keep the head calibrated on the original distribution
    cache = torch.load(args.features, map_location="cpu", weights_only=True)
    provenance = [s for s, _ex in load_training_examples_with_provenance()]
    assert len(provenance) == len(cache["pooled"]), "feature cache does not match training pool"
    other = torch.tensor([s != "longpibench" for s in provenance]) & (cache["masks"][:, PI] > 0)
    X = torch.cat([pooled[keep & is_train], cache["pooled"][other]])
    y = torch.cat([wl[keep & is_train].float(), cache["labels"][other, PI]])
    pos_weight = min((y == 0).sum().item() / max(1, (y == 1).sum().item()), 20.0)
    print(f"train rows={len(y)} pos={(y == 1).sum().item()} pos_weight={pos_weight:.2f}")

    lin = torch.nn.Linear(X.shape[1], 1)
    lin.weight.data, lin.bias.data = W0[None].clone(), b0[None].clone()
    opt = torch.optim.Adam(lin.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    for epoch in range(args.epochs):
        perm, total = torch.randperm(len(y)), 0.0
        for i in range(0, len(y), 256):
            idx = perm[i:i + 256]
            loss = loss_fn(lin(X[idx]).squeeze(1), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        print(f"epoch {epoch + 1}/{args.epochs} loss={total / len(y):.4f}")

    with torch.no_grad():
        old = torch.sigmoid(pooled @ W0 + b0)
        new = torch.sigmoid(lin(pooled).squeeze(1))
    report_doc_level("old head  (held-out)", old, groups, holdout)
    report_doc_level("window head (held-out)", new, groups, holdout)
    held = ~is_train & keep
    print("held-out window-level: pos mean old/new = "
          f"{old[held & (wl == 1)].mean():.3f}/{new[held & (wl == 1)].mean():.3f}, "
          f"neg mean old/new = {old[held & (wl == 0)].mean():.3f}/{new[held & (wl == 0)].mean():.3f}")

    state = {k: v.clone() for k, v in deployed["state_dict"].items()}
    state["weight"][PI], state["bias"][PI] = lin.weight.data[0], lin.bias.data[0]
    payload = {**deployed, "state_dict": state}
    torch.save(payload, args.out)
    print(f"saved candidate: {args.out}")

    head = torch.nn.Linear(state["weight"].shape[1], len(LABELS)).to(model.device)
    head.load_state_dict(state)
    head.eval()
    new_metrics = regression_evaluate(head, model.encoder, model.tokenizer, device=str(model.device))
    old_metrics = regression_evaluate(model.head, model.encoder, model.tokenizer, device=str(model.device))
    print("regression set old:", old_metrics)
    print("regression set new:", new_metrics)
    print("REGRESSION GATE:", check_regression(new_metrics, old_metrics) or "pass")


if __name__ == "__main__":
    main()
