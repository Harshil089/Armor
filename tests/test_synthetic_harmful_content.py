from armor.data.synthetic_harmful_content import generate_examples
from armor.models.labels import LABELS


def test_generates_balanced_harmful_content_pairs():
    examples = generate_examples(n_positive=30, n_negative=30)
    index = LABELS.index("harmful_content")

    assert len(examples) == 60
    assert sum(labels[index] for _, labels, _ in examples) == 30
    assert all(masks[index] == 1.0 for _, _, masks in examples)
    assert all(sum(masks) == 1.0 for _, _, masks in examples)
