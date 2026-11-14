from armor.data.synthetic_topic_adjacent_negatives import generate_examples
from armor.models.labels import LABELS


def test_generate_examples_shape_and_labels():
    examples = generate_examples(n=300)
    assert len(examples) == 300

    for text, label_vec, mask_vec in examples:
        assert isinstance(text, str) and text
        assert label_vec == [0.0] * len(LABELS)
        assert sum(mask_vec) == 1.0


def test_generate_examples_weighted_toward_harmful_content():
    examples = generate_examples(n=1000)
    counts = {label: 0 for label in LABELS}
    for _, _, mask_vec in examples:
        counts[LABELS[mask_vec.index(1.0)]] += 1
    # weight 6 vs weight 1 (e.g. prompt_injection) should show up clearly
    assert counts["harmful_content"] > counts["prompt_injection"] * 3


def test_generate_examples_has_phrasing_variety():
    examples = generate_examples(n=300)
    unique_texts = {e[0] for e in examples}
    assert len(unique_texts) > 15
