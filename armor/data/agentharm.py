"""AgentHarm training examples for dangerous_tool_action.

ponytail: not a standard HF load_dataset table (no loading script, just raw
JSON assets per file) — hence its own loader instead of going through
armor/data/public.py's generic path, same reasoning as synthetic_pii.py.
License: MIT + "restricted to improving AI safety/security" clause, matches
Armor's stated purpose (M1 source survey, RECOMMENDED verdict).
"""

import json

from huggingface_hub import hf_hub_download

from armor.models.labels import LABELS

_LABEL_INDEX = LABELS.index("dangerous_tool_action")

_FILES = [
    ("benchmark/harmful_behaviors_validation.json", 1.0),
    ("benchmark/benign_behaviors_validation.json", 0.0),
    ("benchmark/harmful_behaviors_test_public.json", 1.0),
    ("benchmark/benign_behaviors_test_public.json", 0.0),
]


def load_examples() -> list[tuple[str, list[float], list[float]]]:
    examples = []
    for filename, positive_value in _FILES:
        path = hf_hub_download("ai-safety-institute/AgentHarm", filename, repo_type="dataset")
        data = json.loads(open(path).read())
        for behavior in data["behaviors"]:
            text = behavior["prompt"]
            label_vec = [0.0] * len(LABELS)
            mask_vec = [0.0] * len(LABELS)
            label_vec[_LABEL_INDEX] = positive_value
            mask_vec[_LABEL_INDEX] = 1.0
            examples.append((text, label_vec, mask_vec))
    return examples
