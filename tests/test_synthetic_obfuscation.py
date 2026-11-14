from armor.data.synthetic_obfuscation import generate_examples
from armor.models.labels import LABELS

_OE_INDEX = LABELS.index("obfuscation_encoding")


def test_generate_examples_shape_and_labels():
    examples = generate_examples(n_positive=50, n_negative=50)
    assert len(examples) == 100

    positives = [e for e in examples if e[1][_OE_INDEX] == 1.0]
    negatives = [e for e in examples if e[1][_OE_INDEX] == 0.0]
    assert len(positives) == 50
    assert len(negatives) == 50

    for text, label_vec, mask_vec in examples:
        assert isinstance(text, str) and text
        assert mask_vec[_OE_INDEX] == 1.0
        assert sum(mask_vec) == 1.0


def test_generate_examples_has_phrasing_variety():
    examples = generate_examples(n_positive=100, n_negative=0)
    unique_texts = {e[0] for e in examples}
    assert len(unique_texts) > 30


def test_negatives_include_hard_negative_transformed_payloads():
    # Hard negatives reuse the exact same payload/transform vocabulary as
    # positives (base64-looking, rot13-looking, etc.) but with a neutral
    # carrier — this is the actual fix for the base64-presence shortcut, so
    # assert the pairing exists, not just phrasing variety in general.
    from armor.data.synthetic_obfuscation import _PAYLOADS, _b64

    examples = generate_examples(n_positive=0, n_negative=200)
    hard_negatives = [t for t, _, _ in examples if any(_b64(p) in t for p in _PAYLOADS)]
    assert len(hard_negatives) > 0


def test_negatives_do_not_use_concealment_transforms():
    # Negative carriers should never contain the concealment framing words
    # used in positive carrier templates.
    concealment_markers = ["decode", "bypass", "no questions", "quietly", "skip the usual review", "filters"]
    examples = generate_examples(n_positive=0, n_negative=100)
    for text, _, _ in examples:
        lowered = text.lower()
        assert not any(marker in lowered for marker in concealment_markers)
