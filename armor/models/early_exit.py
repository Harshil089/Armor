"""Tiny, same-encoder exit heads for conservative Armor inference.

These linear heads read ModernBERT intermediate CLS representations. They do
not tokenize, encode, or route requests themselves and do not constitute a
second language model. Promotion stays opt-in until the frozen safety gates
pass; default serving remains the full-depth reference path.
"""

import os
from dataclasses import dataclass

import torch
from torch import nn

from armor.models.labels import LABELS

DEFAULT_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "early_exit_heads.pt")
CHECKPOINT_PATH = os.environ.get("ARMOR_EARLY_EXIT_HEADS_PATH", DEFAULT_CHECKPOINT_PATH)
EXIT_LAYERS = (6, 12)


class EarlyExitHead(nn.Module):
    """Predict six teacher threat logits plus a binary safe-to-stop logit."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, len(LABELS) + 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim == 3:
            hidden_states = hidden_states[:, 0, :]
        elif hidden_states.ndim != 2:
            raise ValueError("early-exit head expects [batch, sequence, hidden] or [batch, hidden]")
        return self.projection(hidden_states)


class EarlyExitHeads(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layer6 = EarlyExitHead(hidden_size)
        self.layer12 = EarlyExitHead(hidden_size)


@dataclass(frozen=True)
class ExitCandidate:
    depth: int
    decision: str
    confidence: float
    threat_scores: dict[str, float]


def candidate_from_logits(
    logits: torch.Tensor,
    *,
    depth: int,
    safe_threshold: float | None = None,
    max_threat_threshold: float | None = None,
    calibrated_threat_scores: dict[str, float] | None = None,
    structural_warning: bool = False,
) -> ExitCandidate | None:
    """Return only a very-low-risk ALLOW exit candidate; otherwise continue.

    Thresholds are runtime-configurable and deliberately conservative. This
    helper never emits BLOCK, FLAG, or REVIEW. Deterministic and policy checks
    remain authoritative around it.
    """
    if depth not in EXIT_LAYERS or structural_warning:
        return None
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim != 2 or logits.shape[-1] != len(LABELS) + 1 or logits.shape[0] != 1:
        raise ValueError("early-exit head expects one request and seven logits")
    safe_threshold = safe_threshold if safe_threshold is not None else float(
        os.environ.get(f"ARMOR_EXIT_SAFE_THRESHOLD_{depth}", "0.999")
    )
    max_threat_threshold = max_threat_threshold if max_threat_threshold is not None else float(
        os.environ.get("ARMOR_EXIT_MAX_THREAT", os.environ.get("ARMOR_BASE_THRESHOLD", "0.5"))
    )
    if not 0.0 <= safe_threshold <= 1.0 or not 0.0 <= max_threat_threshold <= 1.0:
        raise ValueError("early-exit thresholds must be between zero and one")
    probabilities = torch.sigmoid(logits[0]).detach().cpu().tolist()
    threat_scores = calibrated_threat_scores or dict(zip(LABELS, probabilities[:-1]))
    if set(threat_scores) != set(LABELS):
        raise ValueError("calibrated early-exit scores must contain every Armor threat label")
    confidence = float(probabilities[-1])
    if confidence < safe_threshold or max(threat_scores.values(), default=0.0) >= max_threat_threshold:
        return None
    return ExitCandidate(depth, "allow", confidence, threat_scores)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_safe: torch.Tensor,
    *,
    unsafe_weight: float = 8.0,
) -> torch.Tensor:
    """Distill full-depth risk logits and heavily penalize unsafe exits.

    A false safe-to-stop target on a full-depth non-ALLOW example receives
    `unsafe_weight`; the helper is for offline candidate fitting only.
    """
    if student_logits.ndim != 2 or student_logits.shape[-1] != len(LABELS) + 1:
        raise ValueError("student logits must have six threat outputs and one safety output")
    if teacher_logits.shape != student_logits[:, :-1].shape:
        raise ValueError("teacher logits must contain the six final threat-head logits")
    teacher_targets = torch.sigmoid(teacher_logits.detach())
    threat_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        student_logits[:, :-1], teacher_targets
    )
    safe_targets = teacher_safe.to(dtype=student_logits.dtype).reshape(-1)
    per_row = torch.nn.functional.binary_cross_entropy_with_logits(
        student_logits[:, -1], safe_targets, reduction="none"
    )
    weights = torch.where(safe_targets > 0.5, 1.0, float(unsafe_weight))
    return threat_loss + (per_row * weights).mean()


def load_exit_heads(
    hidden_size: int,
    path: str = CHECKPOINT_PATH,
    *,
    expected_encoder_fingerprint: str | None = None,
    device: str | torch.device = "cpu",
) -> EarlyExitHeads | None:
    if not os.path.exists(path):
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 2 or payload.get("hidden_size") != hidden_size:
        raise ValueError("early-exit checkpoint format or hidden size is incompatible")
    if (
        expected_encoder_fingerprint is not None
        and payload.get("encoder_fingerprint") != expected_encoder_fingerprint
    ):
        raise ValueError("early-exit checkpoint was trained against a different ArmorBase encoder")
    heads = EarlyExitHeads(hidden_size)
    heads.load_state_dict(payload["state_dict"])
    heads.to(device)
    heads.eval()
    return heads


def save_exit_heads(
    heads: EarlyExitHeads,
    path: str = CHECKPOINT_PATH,
    *,
    encoder_fingerprint: str | None = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {"format_version": 2, "hidden_size": heads.layer6.projection[0].in_features,
         "encoder_fingerprint": encoder_fingerprint,
         "state_dict": heads.state_dict()},
        path,
    )
