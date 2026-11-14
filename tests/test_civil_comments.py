from armor.data.civil_comments import load_examples
from armor.models.labels import LABELS

_HC_INDEX = LABELS.index("harmful_content")


def test_load_examples_shape_and_labels():
    examples = load_examples()
    assert len(examples) > 0

    positives = [e for e in examples if e[1][_HC_INDEX] == 1.0]
    negatives = [e for e in examples if e[1][_HC_INDEX] == 0.0]
    assert len(positives) > 0
    assert len(negatives) > 0
    assert len(positives) <= 2500
    assert len(negatives) <= 2500

    for text, label_vec, mask_vec in examples:
        assert isinstance(text, str) and text
        assert mask_vec[_HC_INDEX] == 1.0
        assert sum(mask_vec) == 1.0
