"""jinjinyien/ToolSafety training examples for dangerous_tool_action.

ponytail: licensing resolved directly with the authors (ARCHIVE-QUEUE-03) — Prof.
Pinjia He confirmed MIT is intentional, and the HF dataset owner accepted
the usage request. Best conceptual fit found for this label: multi-turn
tool-invocation requests spanning direct harm, indirect harm, and
multi-step scenarios (paper: EMNLP 2025, 10.18653/v1/2025.emnlp-main.714).

Every row is a risky user request paired with an assistant refusal — no
benign counterpart, same shape as HarmBench/AgentHarm-standalone. Capped
at a fixed slice (deterministic, no shuffle) so this doesn't swamp the
label's existing ~1000-row pool the way an uncapped civil_comments once
would have for harmful_content.
"""

from datasets import load_dataset

from armor.models.labels import LABELS

_LABEL_INDEX = LABELS.index("dangerous_tool_action")
_MAX_ROWS = 500


def load_examples() -> list[tuple[str, list[float], list[float]]]:
    dataset = load_dataset("jinjinyien/ToolSafety", split=f"train[:{_MAX_ROWS}]")

    examples = []
    for row in dataset:
        user_turn = next((t["value"] for t in row["conversations"] if t["from"] == "user"), None)
        if not user_turn:
            continue
        label_vec = [0.0] * len(LABELS)
        mask_vec = [0.0] * len(LABELS)
        label_vec[_LABEL_INDEX] = 1.0
        mask_vec[_LABEL_INDEX] = 1.0
        examples.append((user_turn, label_vec, mask_vec))
    return examples
