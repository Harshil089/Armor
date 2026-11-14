from armor.data.synthetic_jailbreak_canonical import generate_examples
from armor.models.labels import LABELS

_JB_INDEX = LABELS.index("jailbreak")


def test_generate_examples_shape_and_labels():
    examples = generate_examples(n_positive=50, n_negative=50)
    assert len(examples) == 100

    positives = [e for e in examples if e[1][_JB_INDEX] == 1.0]
    negatives = [e for e in examples if e[1][_JB_INDEX] == 0.0]
    assert len(positives) == 50
    assert len(negatives) == 50

    for text, label_vec, mask_vec in examples:
        assert isinstance(text, str) and text
        assert mask_vec[_JB_INDEX] == 1.0
        assert sum(mask_vec) == 1.0
