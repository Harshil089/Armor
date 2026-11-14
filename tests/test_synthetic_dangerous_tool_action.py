from armor.data.synthetic_dangerous_tool_action import (
    generate_destructive_focus_examples,
    generate_examples,
    generate_external_comm_focus_examples,
)
from armor.models.labels import LABELS

_DTA_INDEX = LABELS.index("dangerous_tool_action")


def test_generate_examples_shape_and_labels():
    examples = generate_examples(n_positive=50, n_negative=50)
    assert len(examples) == 100

    positives = [e for e in examples if e[1][_DTA_INDEX] == 1.0]
    negatives = [e for e in examples if e[1][_DTA_INDEX] == 0.0]
    assert len(positives) == 50
    assert len(negatives) == 50

    for text, label_vec, mask_vec in examples:
        assert isinstance(text, str) and text
        assert mask_vec[_DTA_INDEX] == 1.0
        assert sum(mask_vec) == 1.0


def test_generate_examples_has_phrasing_variety():
    examples = generate_examples(n_positive=100, n_negative=0)
    unique_texts = {e[0] for e in examples}
    assert len(unique_texts) > 20


def test_destructive_focus_is_balanced_and_single_label():
    examples = generate_destructive_focus_examples(n_positive=20, n_negative=20)

    assert len(examples) == 40
    assert sum(labels[_DTA_INDEX] for _, labels, _ in examples) == 20
    assert all(masks[_DTA_INDEX] == 1.0 for _, _, masks in examples)
    assert all(sum(masks) == 1.0 for _, _, masks in examples)


def test_external_comm_focus_is_balanced_and_single_label():
    examples = generate_external_comm_focus_examples(n_positive=20, n_negative=20)

    assert len(examples) == 40
    assert sum(labels[_DTA_INDEX] for _, labels, _ in examples) == 20
    assert all(masks[_DTA_INDEX] == 1.0 for _, _, masks in examples)
    assert all(sum(masks) == 1.0 for _, _, masks in examples)
