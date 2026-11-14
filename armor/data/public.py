from datasets import load_dataset

from armor.data.sources import SOURCES
from armor.models.labels import LABELS


def load_training_examples(split: str = "train") -> list[tuple[str, list[float], list[float]]]:
    """(text, label_vector, mask_vector) triples from the registered public sources.

    ponytail: only the label a source actually supervises gets mask=1; every
    other label is left unknown (mask=0), not assumed negative, since each
    source here covers exactly one taxonomy label. Real multi-label coverage
    waits for the M1 dataset build (architecture spec section 9).
    """
    examples = []
    for source in SOURCES:
        label_index = LABELS.index(source.label)
        # ponytail: same deterministic head-slice cap the non-SOURCES loaders
        # already use (civil_comments, ToolSafety, jailbreak_wild). Added after
        # counting the pool 2026-09-20: SPML's 16,012 rows made prompt_injection
        # 80% of all training rows, ~12x every other label. That dominance is
        # why prompt_injection (and obfuscation_encoding, the highest-scoring
        # and most drift-sensitive label) regressed on every single retrain
        # attempt regardless of what new data was added.
        source_split = f"{split}[:{source.max_rows}]" if source.max_rows else split
        if source.config:
            rows = load_dataset(source.hf_id, source.config, split=source_split)
        else:
            rows = load_dataset(source.hf_id, split=source_split)
        for row in rows:
            text = row[source.text_field]
            if not isinstance(text, str) or not text.strip():
                continue
            positive = row[source.label_field] in source.positive_values
            label_vec = [0.0] * len(LABELS)
            mask_vec = [0.0] * len(LABELS)
            label_vec[label_index] = 1.0 if positive else 0.0
            mask_vec[label_index] = 1.0
            examples.append((text, label_vec, mask_vec))
    return examples
