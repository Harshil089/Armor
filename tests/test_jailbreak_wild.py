from armor.data.jailbreak_wild import load_examples
from armor.models.labels import LABELS

_JB_INDEX = LABELS.index("jailbreak")


def test_load_examples_shape_and_labels():
    examples = load_examples()
    assert len(examples) == 600 + 600

    positives = [e for e in examples if e[1][_JB_INDEX] == 1.0]
    negatives = [e for e in examples if e[1][_JB_INDEX] == 0.0]
    assert len(positives) == 600
    assert len(negatives) == 600

    for text, label_vec, mask_vec in examples:
        assert isinstance(text, str) and text
        assert mask_vec[_JB_INDEX] == 1.0
        assert sum(mask_vec) == 1.0
