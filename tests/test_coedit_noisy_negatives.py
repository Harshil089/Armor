from armor.data import coedit_noisy_negatives
from armor.models.labels import LABELS


def test_load_examples_uses_complete_gec_pairs_only(monkeypatch):
    rows = [
        {"_id": "2", "task": "gec", "src": "Fix grammar: This sentnce is noisy.", "tgt": "This sentence is noisy."},
        {"_id": "1", "task": "paraphrase", "src": "Rewrite: Skip me.", "tgt": "Ignore me."},
        {"_id": "3", "task": "gec", "src": "Fix grammar: unchanged", "tgt": "unchanged"},
    ]
    monkeypatch.setattr(coedit_noisy_negatives, "load_dataset", lambda *args, **kwargs: rows)

    examples = coedit_noisy_negatives.load_examples()

    assert [text for text, _, _ in examples] == ["This sentnce is noisy.", "This sentence is noisy."]
    index = LABELS.index("obfuscation_encoding")
    assert all(sum(labels) == 0.0 and mask[index] == 1.0 and sum(mask) == 1.0 for _, labels, mask in examples)


def test_load_examples_passes_pinned_revision(monkeypatch):
    call = {}

    def fake_load_dataset(dataset_id, **kwargs):
        call.update(dataset_id=dataset_id, **kwargs)
        return []

    monkeypatch.setattr(coedit_noisy_negatives, "load_dataset", fake_load_dataset)
    coedit_noisy_negatives.load_examples()

    assert call == {
        "dataset_id": "grammarly/coedit",
        "revision": coedit_noisy_negatives._REVISION,
        "split": "train",
    }
