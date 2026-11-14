from armor.data import aegis_content_safety
from armor.models.labels import LABELS


def _row(row_id, text_type, votes, text="A normal reviewed message"):
    row = {
        "id": row_id,
        "text": text,
        "text_type": text_type,
        "num_annotations": len(votes),
    }
    row.update({f"labels_{index}": vote for index, vote in enumerate(votes)})
    return row


def test_load_examples_uses_only_unanimous_single_leg_safe_rows(monkeypatch):
    rows = [
        _row("b", "user_message", ["Safe", "Safe", "Safe"], "safe prompt"),
        _row("a", "llm_response", ["Safe", "Safe", "Safe"], "safe response"),
        _row("c", "combined", ["Safe", "Safe", "Safe"]),
        _row("d", "user_message", ["Safe", "Harassment", "Safe"]),
        _row("e", "llm_response", ["Safe", "Safe"], "   "),
        _row("f", "llm_response", ["Safe", "Safe"], "safe   response"),
    ]
    monkeypatch.setattr(aegis_content_safety, "load_dataset", lambda *args, **kwargs: rows)

    examples = aegis_content_safety.load_examples()

    assert [text for text, _, _ in examples] == ["safe prompt"]
    label_indexes = [
        LABELS.index("harmful_content"),
        LABELS.index("sensitive_data_exposure"),
        LABELS.index("obfuscation_encoding"),
    ]
    for _, labels, mask in examples:
        assert all(labels[index] == 0.0 for index in label_indexes)
        assert all(mask[index] == 1.0 for index in label_indexes)
        assert sum(mask) == 3.0


def test_load_examples_samples_stably_by_source_id(monkeypatch):
    rows = [
        _row("z", "user_message", ["Safe"]),
        _row("a", "user_message", ["Safe"], "selected"),
    ]
    monkeypatch.setattr(aegis_content_safety, "load_dataset", lambda *args, **kwargs: rows)

    assert aegis_content_safety.load_examples(max_rows=1)[0][0] == "selected"
