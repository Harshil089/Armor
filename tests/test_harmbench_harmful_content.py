from armor.data.harmbench_harmful_content import load_examples
from armor.models.labels import LABELS

_HC_INDEX = LABELS.index("harmful_content")


def test_load_examples_shape_and_labels():
    examples = load_examples()
    assert len(examples) == 300  # 233 + 67 cybercrime_intrusion

    for text, label_vec, mask_vec in examples:
        assert isinstance(text, str) and text
        assert label_vec[_HC_INDEX] == 1.0
        assert mask_vec[_HC_INDEX] == 1.0
        assert sum(mask_vec) == 1.0
