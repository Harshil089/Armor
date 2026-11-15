"""Policy decision head -- consumes ModernBERT's pooled embedding (not just
the 6 scalar threat scores) to predict an enforcement action. Phase 1 of
the representation-bottleneck fix documented in queue/QUEUE-02: the bandit
and the (disabled) semantic judge both tried to recover semantic nuance
from 6 numbers after the actual representation had already been computed
and discarded in armor/models/armor_base.py's predict().

When `ARMOR_BERT_POLICY_HEAD=1`, this head supplies prompt decisions in
`armor/api/main.py`; otherwise its output is logged for comparison. It uses
the pooled embedding already computed by ArmorBase, so it does not add a
second language-model call.

Small head, not a second big model: pooled embedding (768-dim for
ModernBERT-base) + 6 calibrated scores -> one hidden layer -> 4 action
logits. Same checkpoint-loading shape as armor_base.py's own ArmorBase
class, not a new pattern.
"""

import logging
import os

import torch
from torch import nn

from armor.models.labels import LABELS

POLICY_ACTIONS = ["allow", "flag", "block", "review"]

DEFAULT_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "policy_head.pt")
CHECKPOINT_PATH = os.environ.get("ARMOR_POLICY_HEAD_PATH", DEFAULT_CHECKPOINT_PATH)

HIDDEN_SIZE = 128


class PolicyHead(nn.Module):
    def __init__(self, embedding_dim: int, n_threat_scores: int = len(LABELS), hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + n_threat_scores, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(POLICY_ACTIONS)),
        )

    def forward(self, pooled_embedding: torch.Tensor, threat_scores: torch.Tensor) -> torch.Tensor:
        x = torch.cat([pooled_embedding, threat_scores], dim=-1)
        return self.net(x)


def build_input(pooled_embedding: torch.Tensor, scores: dict[str, float]) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalizes a scores dict (in LABELS order, missing labels -> 0.0)
    into the tensor shape PolicyHead.forward() expects."""
    threat_scores = torch.tensor([scores.get(label, 0.0) for label in LABELS], dtype=torch.float32)
    return pooled_embedding, threat_scores


_policy_head: PolicyHead | None = None


def _load_or_warn(head: nn.Module, checkpoint_path: str, head_name: str) -> None:
    """Load checkpoint weights if present, or warn before using random weights."""
    if os.path.exists(checkpoint_path):
        head.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    else:
        logging.warning(
            "%s checkpoint not found at %s; using randomly initialized weights.",
            head_name,
            checkpoint_path,
        )


def get_policy_head(embedding_dim: int = 768) -> PolicyHead:
    global _policy_head
    if _policy_head is None:
        head = PolicyHead(embedding_dim)
        _load_or_warn(head, CHECKPOINT_PATH, "PolicyHead")
        head.eval()
        _policy_head = head
    return _policy_head


def save_policy_head(head: PolicyHead, path: str = CHECKPOINT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(head.state_dict(), path)


# --- Hierarchical heads (queue/QUEUE-02's formulation-bottleneck fix) ---
#
# Replaces the single flat 4-way softmax above (kept as "system A", the
# baseline) with two binary questions asked in sequence: does this need
# intervention at all, and if so, does it need blocking. flag/review
# stayed stuck at 5-17% recall through two independent fixes (more
# targeted data, inverse-frequency loss weighting) -- the same null
# result twice pointed at the *formulation*, not the data: allow/block are
# broad regions, flag is a narrow one between them, and review isn't a
# semantic class at all. See armor/models/uncertainty.py for how review is
# now produced (abstention), not predicted directly by either head below.

INTERVENTION_CHECKPOINT_PATH = os.environ.get(
    "ARMOR_INTERVENTION_HEAD_PATH",
    os.path.join(os.path.dirname(__file__), "checkpoints", "intervention_head.pt"),
)
BLOCK_CHECKPOINT_PATH = os.environ.get(
    "ARMOR_BLOCK_HEAD_PATH",
    os.path.join(os.path.dirname(__file__), "checkpoints", "block_head.pt"),
)


class BinaryPolicyHead(nn.Module):
    """Shared shape for both hierarchical heads -- same size class as the
    flat PolicyHead (single hidden layer, ~128 units): this is a
    formulation change, not a capacity change."""

    def __init__(self, embedding_dim: int, n_threat_scores: int = len(LABELS), hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + n_threat_scores, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, pooled_embedding: torch.Tensor, threat_scores: torch.Tensor) -> torch.Tensor:
        """Returns a single logit (not a probability) -- apply sigmoid at
        the call site, matching how the flat head returns raw logits."""
        x = torch.cat([pooled_embedding, threat_scores], dim=-1)
        return self.net(x).squeeze(-1)


InterventionHead = BinaryPolicyHead
BlockHead = BinaryPolicyHead

_intervention_head: BinaryPolicyHead | None = None
_block_head: BinaryPolicyHead | None = None


def get_intervention_head(embedding_dim: int = 768) -> BinaryPolicyHead:
    global _intervention_head
    if _intervention_head is None:
        head = BinaryPolicyHead(embedding_dim)
        _load_or_warn(head, INTERVENTION_CHECKPOINT_PATH, "InterventionHead")
        head.eval()
        _intervention_head = head
    return _intervention_head


def get_block_head(embedding_dim: int = 768) -> BinaryPolicyHead:
    global _block_head
    if _block_head is None:
        head = BinaryPolicyHead(embedding_dim)
        _load_or_warn(head, BLOCK_CHECKPOINT_PATH, "BlockHead")
        head.eval()
        _block_head = head
    return _block_head


def save_intervention_head(head: BinaryPolicyHead, path: str = INTERVENTION_CHECKPOINT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(head.state_dict(), path)


def save_block_head(head: BinaryPolicyHead, path: str = BLOCK_CHECKPOINT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(head.state_dict(), path)
