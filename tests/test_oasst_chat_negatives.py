from armor.data import oasst_chat_negatives
from armor.models.labels import LABELS


def _row(message_id, role, text, **overrides):
    row = {
        "message_id": message_id,
        "role": role,
        "text": text,
        "lang": "en",
        "review_result": True,
        "deleted": False,
        "review_count": 3,
        "rank": 0 if role == "assistant" else None,
        "detoxify": {"toxicity": 0.001, "threat": 0.001},
    }
    row.update(overrides)
    return row


def test_load_examples_uses_reviewed_clean_turns_with_role_specific_masks(monkeypatch):
    structured = "\n".join(f"line {index}" for index in range(8))
    code = "Here is a safe example:\n```html\n<form></form>\n```"
    rows = [
        _row("b", "assistant", "Reviewed assistant response"),
        _row("a", "prompter", "Reviewed user prompt"),
        _row("aa", "assistant", structured),
        _row("ab", "assistant", structured + "\ncontact me@example.com"),
        _row("ac", "assistant", code),
        _row("c", "assistant", "Lower ranked response", rank=1),
        _row("d", "prompter", "Toxic prompt", detoxify={"toxicity": 0.2}),
        _row("e", "prompter", "Unreviewed prompt", review_result=False),
        _row("f", "assistant", "word " * 351),
    ]
    monkeypatch.setattr(oasst_chat_negatives, "load_dataset", lambda *args, **kwargs: rows)

    examples = oasst_chat_negatives.load_examples()

    observed = {(text, LABELS[mask.index(1.0)]) for text, _, mask in examples}
    assert ("Reviewed assistant response", "prompt_injection") in observed
    assert ("Reviewed assistant response", "obfuscation_encoding") in observed
    assert ("Reviewed user prompt", "obfuscation_encoding") in observed
    assert (structured, "sensitive_data_exposure") in observed
    assert (code, "prompt_injection") in observed
    assert (code, "sensitive_data_exposure") in observed
    assert (structured + "\ncontact me@example.com", "sensitive_data_exposure") not in observed
    assert all(sum(labels) == 0.0 and sum(mask) == 1.0 for _, labels, mask in examples)


def test_load_examples_passes_pinned_revision(monkeypatch):
    call = {}

    def fake_load_dataset(dataset_id, **kwargs):
        call.update(dataset_id=dataset_id, **kwargs)
        return []

    monkeypatch.setattr(oasst_chat_negatives, "load_dataset", fake_load_dataset)
    oasst_chat_negatives.load_examples()

    assert call == {
        "dataset_id": "OpenAssistant/oasst1",
        "revision": oasst_chat_negatives._REVISION,
        "split": "train",
    }
