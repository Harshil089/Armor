"""LinUCB contextual bandit — shadow-mode decision layer over ArmorBase scores.

ponytail: the classifier's decision logic (armor/api/main.py) uses one global
BLOCK_THRESHOLD/FINDING_THRESHOLD across all 6 labels. Retrains keep trading
one label's separation for another at that shared boundary (see
docs/CHAT_IMPROVEMENTS_AND_CHANGES.md v41-v52). This is a contextual bandit,
not full RL: each request is a single-shot decision with no state carried
between requests, so LinUCB (disjoint linear bandit, closed-form per-action
update) fits without the instability of policy-gradient RL.

v1 is shadow mode only: choose() is called and logged, but never controls the
response (see armor/api/main.py's bandit wiring). It has no reward signal
until an analyst corrects a logged recommendation via
POST /admin/logs/{id}/correct, which calls update().
"""

import os
import re
from pathlib import Path

import numpy as np

from armor.models.labels import LABELS
from armor.models.semantic_judge import SEMANTIC_FEATURES

# Reuse InspectResponse's existing decision vocabulary (allow/flag/block/review)
# instead of inventing a second one (ALLOW/SANITIZE/BLOCK/ESCALATE) that every
# log reader would have to reconcile against `decision`. "review" is declared
# in armor/api/schemas.py but has zero producers today — this bandit is the
# first one.
ACTIONS = ["allow", "flag", "block", "review"]

_DEFAULT_STATE_PATH = Path(__file__).with_name("bandit_state") / "current.npz"
BANDIT_STATE_PATH = Path(os.environ.get("ARMOR_BANDIT_STATE_PATH", _DEFAULT_STATE_PATH))

# v1.1: the 6 raw ArmorBase label scores alone can't separate "block" from
# "review" (dual-use) requests — "write a keylogger" and "how does a keylogger
# work, for a security course" land on near-identical score patterns since the
# classifier judges topic, not framing. Bootstrapping this bandit with ~90
# hand-labeled corrections showed validation agreement getting WORSE with more
# data (38% -> 6% -> 25%), because contradictory labels were being fit onto
# the same 6-dim region — a feature-representation ceiling, not a data-volume
# one. This 7th feature is a cheap keyword heuristic for authorized/defensive
# framing, giving the linear model something to actually separate on.
_FRAMING_PATTERN = re.compile(
    r"\b("
    r"for (a |my )?(research|class|course|ctf|pentest(ing)?|red[- ]?team(ing)?)"
    r"|to defend against|to prevent|to protect against|so (i|we) can (defend|prevent|protect)"
    r"|penetration test(ing)?|authorized|with permission|i own|on my own (account|machine|network)"
    r"|academic|educational purposes|security (course|training|research)"
    r")\b",
    re.IGNORECASE,
)


def framing_signal(text: str) -> float:
    return 1.0 if _FRAMING_PATTERN.search(text) else 0.0


# v1.2: neither the 6 threat scores nor framing_signal carry any signal
# about real-world authorization ambiguity -- "my roommate left their
# phone" (flag) and "my co-signer stopped paying, show me their bank
# statements" (block) produced near-identical 7-dim contexts, confirmed by
# inspecting fitted vectors directly after 12 red-team rounds made
# agreement WORSE (76% -> 66%) instead of better. See
# armor/models/semantic_judge.py and QUEUE-02's 2026-09-22 v1.2 entry.
def build_context(content: str, scores: dict, semantic: list[float] | None = None) -> np.ndarray:
    semantic_scores = semantic if semantic is not None else [0.0] * len(SEMANTIC_FEATURES)
    return np.array(
        [scores.get(label, 0.0) for label in LABELS] + [framing_signal(content)] + list(semantic_scores)
    )


N_FEATURES = len(LABELS) + 1 + len(SEMANTIC_FEATURES)

# Asymmetric penalty table: bandit_action (row) x analyst_correction (column).
# False negatives (bandit under-reacted relative to a "block" correction) are
# the worst cell in their row; false positives (bandit said block, correction
# says allow) are mild. Values are tunable constants, not load-bearing math.
_REWARD_TABLE = {
    "allow": {"allow": 1.0, "flag": -1.0, "block": -5.0, "review": -2.0},
    "flag": {"allow": -0.3, "flag": 1.0, "block": -3.0, "review": -1.0},
    "block": {"allow": -1.0, "flag": -0.5, "block": 1.0, "review": -0.3},
    "review": {"allow": -0.3, "flag": -0.2, "block": -0.5, "review": 1.0},
}


def compute_reward(bandit_action: str, analyst_correction: str) -> float:
    return _REWARD_TABLE[bandit_action][analyst_correction]


class LinUCBBandit:
    """Disjoint LinUCB: one (A, b) pair per action, pure numpy.linalg (no
    scipy — not installed in this project)."""

    def __init__(self, n_actions: int = len(ACTIONS), n_features: int = N_FEATURES, alpha: float = 1.0):
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = alpha
        self.A = [np.eye(n_features) for _ in range(n_actions)]
        self.b = [np.zeros(n_features) for _ in range(n_actions)]
        self.applied_feedback_ids: set[str] = set()

    def choose(self, context: np.ndarray) -> str:
        # ponytail: "review" is only ever justified by the framing_signal feature
        # (index len(LABELS), NOT the last dim -- v1.2 appended 5 semantic
        # features after it, see armor.models.semantic_judge) -- that's its
        # entire reason for existing (see the comment on framing_signal above).
        # With few review corrections, its theta stays near-zero everywhere
        # else, and near-zero beats the OTHER arms' correctly-large-negative
        # theta on contexts those arms HAVE seen (e.g. a clear
        # dangerous_tool_action attack) but "review" hasn't -- measured this
        # directly: held-out block recall was 18% with review eligible
        # everywhere, 46% once gated. Ridge and alpha were both grid-searched
        # first (7 ridge values x 5 alpha values, alpha=0 included) and
        # neither moved the number -- this isn't an exploration-bonus problem,
        # it's an out-of-coverage extrapolation problem that a hard structural
        # gate fixes and a regularization dial can't.
        review_idx = ACTIONS.index("review")
        eligible = (
            range(len(ACTIONS))
            if context[len(LABELS)] == 1.0
            else (i for i in range(len(ACTIONS)) if i != review_idx)
        )
        best_score = -np.inf
        best_action = ACTIONS[0]
        for i in eligible:
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]
            score = theta @ context + self.alpha * np.sqrt(context @ A_inv @ context)
            if score > best_score:
                best_score = score
                best_action = ACTIONS[i]
        return best_action

    def update(self, context: np.ndarray, action: str, reward: float, feedback_id: str | None = None) -> bool:
        if context.shape != (self.n_features,):
            raise ValueError(f"expected context shape {(self.n_features,)}, got {context.shape}")
        if action not in ACTIONS:
            raise ValueError(f"invalid action: {action}")
        if feedback_id is not None and feedback_id in self.applied_feedback_ids:
            return False
        i = ACTIONS.index(action)
        self.A[i] += np.outer(context, context)
        self.b[i] += reward * context
        if feedback_id is not None:
            self.applied_feedback_ids.add(feedback_id)
        return True

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        for i, action in enumerate(ACTIONS):
            payload[f"A_{action}"] = self.A[i]
            payload[f"b_{action}"] = self.b[i]
        payload["applied_feedback_ids"] = np.array(sorted(self.applied_feedback_ids), dtype="U")
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: Path) -> "LinUCBBandit":
        data = np.load(path)
        bandit = cls()
        for i, action in enumerate(ACTIONS):
            bandit.A[i] = data[f"A_{action}"]
            bandit.b[i] = data[f"b_{action}"]
        if "applied_feedback_ids" in data.files:
            bandit.applied_feedback_ids = set(data["applied_feedback_ids"].astype(str).tolist())
        return bandit


_bandit: LinUCBBandit | None = None


def get_bandit() -> LinUCBBandit:
    global _bandit
    if _bandit is None:
        _bandit = LinUCBBandit.load(BANDIT_STATE_PATH) if BANDIT_STATE_PATH.exists() else LinUCBBandit()
    return _bandit


def save_bandit() -> None:
    get_bandit().save(BANDIT_STATE_PATH)
