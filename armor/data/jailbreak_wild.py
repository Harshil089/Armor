"""TrustAIRLab/in-the-wild-jailbreak-prompts training examples for jailbreak.

ponytail: real academic dataset (CCS 2024, "Do Anything Now": Shen, Chen,
Backes, Shen, Zhang — github.com/TrustAIRLab/JailbreakLLMs, MIT confirmed at
the project level, matching the HF card). 15140 ChatGPT prompts collected
from Reddit/Discord/websites Dec 2022-Dec 2023, split into two configs:
`jailbreak_2023_12_25` (1405 rows, real jailbreak attempts) and
`regular_2023_12_25` (13735 rows, ordinary prompts) — genuine negatives,
not synthetic filler, unlike most sources added this session.

Both pools capped, not just the negative one. A first version capped only
the negative side and let all 1405 positives in — that made TrustAIRLab
73% of the positive class, and its style (long, elaborate persona prompts)
is different enough from jackhhao's short canonical DAN-style phrasing that
the retrained head lost sensitivity to exactly that canonical shape: 4 of 5
short-phrase cases in the frozen regression set ("Enable developer mode",
"roleplay: unfiltered model") flipped from caught to missed, while every
negative stayed correct — a style-dilution effect, not a labeling error.
Same "uncapped source dominates" mistake already made and fixed twice this
session for other labels (civil_comments, ToolSafety); missed here because
only one side of this source was capped.
"""

from datasets import load_dataset

from armor.models.labels import LABELS

_LABEL_INDEX = LABELS.index("jailbreak")
_MAX_POSITIVE = 600
_MAX_NEGATIVE = 600


def load_examples() -> list[tuple[str, list[float], list[float]]]:
    examples = []

    positive = load_dataset(
        "TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25", split=f"train[:{_MAX_POSITIVE}]"
    )
    for row in positive:
        label_vec = [0.0] * len(LABELS)
        mask_vec = [0.0] * len(LABELS)
        label_vec[_LABEL_INDEX] = 1.0
        mask_vec[_LABEL_INDEX] = 1.0
        examples.append((row["prompt"], label_vec, mask_vec))

    negative = load_dataset(
        "TrustAIRLab/in-the-wild-jailbreak-prompts", "regular_2023_12_25", split=f"train[:{_MAX_NEGATIVE}]"
    )
    for row in negative:
        label_vec = [0.0] * len(LABELS)
        mask_vec = [0.0] * len(LABELS)
        mask_vec[_LABEL_INDEX] = 1.0
        examples.append((row["prompt"], label_vec, mask_vec))

    return examples
