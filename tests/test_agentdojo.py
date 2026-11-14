from armor.data.agentdojo import load_examples
from armor.models.labels import LABELS

_DTA_INDEX = LABELS.index("dangerous_tool_action")


def test_load_examples_shape_and_labels():
    examples = load_examples()
    assert len(examples) == 83 + 33  # user_tasks (benign) + injection_tasks (attack)

    positives = [e for e in examples if e[1][_DTA_INDEX] == 1.0]
    negatives = [e for e in examples if e[1][_DTA_INDEX] == 0.0]
    assert len(positives) == 33
    assert len(negatives) == 83

    for text, label_vec, mask_vec in examples:
        assert isinstance(text, str) and text
        assert mask_vec[_DTA_INDEX] == 1.0
        assert sum(mask_vec) == 1.0
