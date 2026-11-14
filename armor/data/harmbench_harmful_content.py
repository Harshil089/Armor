"""walledai/HarmBench training examples for harmful_content.

ponytail: only the categories mapped to Armor's 9-category harmful_content
policy (ARCHIVE-QUEUE-01, docs/... Threat Taxonomy v0.1 section 3.6) are pulled here
— chemical_biological, harassment_bullying, misinformation_disinformation,
illegal, harmful (generic), cybercrime_intrusion. cybercrime_intrusion was
excluded under ARCHIVE-QUEUE-03 (routed to dangerous_tool_action) but never
actually loaded anywhere, leaving bare malware requests unsupervised; as of
2026-10-19 harmful_content owns cyber requests (see harm_request_sources.py)
and dangerous_tool_action stays for tool calls. copyright (100 rows) is excluded
entirely — no current taxonomy label covers IP-infringement content.

License: MIT (repo-level, per walledai/HarmBench's HF card), auto-gated
(click-through accepted).

Known gap, not resolved by this loader (see ARCHIVE-QUEUE-01's own caveat): HarmBench
supplies zero hard negatives — every row here is a positive example.
Negatives for this label still come entirely from civil_comments
(armor/data/civil_comments.py).
"""

from datasets import load_dataset

from armor.models.labels import LABELS

_LABEL_INDEX = LABELS.index("harmful_content")
_CONFIGS = ["standard", "contextual"]
_TARGET_CATEGORIES = {
    "chemical_biological",
    "harassment_bullying",
    "misinformation_disinformation",
    "illegal",
    "harmful",
    "cybercrime_intrusion",
}


def load_examples() -> list[tuple[str, list[float], list[float]]]:
    examples = []
    for config in _CONFIGS:
        dataset = load_dataset("walledai/HarmBench", config, split="train")
        for row in dataset:
            if row["category"] not in _TARGET_CATEGORIES:
                continue
            label_vec = [0.0] * len(LABELS)
            mask_vec = [0.0] * len(LABELS)
            label_vec[_LABEL_INDEX] = 1.0
            mask_vec[_LABEL_INDEX] = 1.0
            examples.append((row["prompt"], label_vec, mask_vec))
    return examples
