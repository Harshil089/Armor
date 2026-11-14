from armor.data.toolsafety_dangerous_tool_action import load_examples
from armor.models.labels import LABELS

_DTA_INDEX = LABELS.index("dangerous_tool_action")


def test_load_examples_shape_and_labels():
    examples = load_examples()
    assert 0 < len(examples) <= 500

    for text, label_vec, mask_vec in examples:
        assert isinstance(text, str) and text
        assert label_vec[_DTA_INDEX] == 1.0
        assert mask_vec[_DTA_INDEX] == 1.0
        assert sum(mask_vec) == 1.0
