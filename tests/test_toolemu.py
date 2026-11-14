import io
import json

import armor.data.toolemu as source
from armor.models.labels import LABELS


def test_loads_matched_examples_for_all_target_slices(monkeypatch):
    cases = [
        {
            "Toolkits": ["BankManager", "Gmail"],
            "Potential Risky Outcomes": ["Sensitive data leak and financial loss."],
            "Potential Risky Actions": ["Send funds and private records without confirmation."],
            "Expected Achievements": ["Verify the payee and recipient before acting."],
        },
        {
            "Toolkits": ["Terminal"],
            "Potential Risky Outcomes": ["Data loss from deletion."],
            "Potential Risky Actions": ["Delete important files without checking."],
            "Expected Achievements": ["Inspect file importance and request confirmation."],
        },
    ]
    monkeypatch.setattr(
        source.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(json.dumps(cases).encode()),
    )

    examples = source.load_examples()
    index = LABELS.index("dangerous_tool_action")
    assert {example[1][index] for example in examples} == {0.0, 1.0}
    assert all(example[2][index] == 1.0 and sum(example[2]) == 1.0 for example in examples)
    assert len({(text, labels[index]) for text, labels, _ in examples}) == len(examples)
