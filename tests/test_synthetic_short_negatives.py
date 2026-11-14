from armor.data.synthetic_short_negatives import generate_examples
from armor.models.labels import LABELS


def test_generate_examples_shape_and_labels():
    examples = generate_examples(n=100)
    assert len(examples) == 100

    for text, label_vec, mask_vec in examples:
        assert isinstance(text, str) and text
        assert len(text.split()) <= 6
        assert label_vec == [0.0] * len(LABELS)
        assert sum(mask_vec) == 1.0


def test_generate_examples_spreads_across_all_labels():
    examples = generate_examples(n=600)
    masked_labels = {mask_vec.index(1.0) for _, _, mask_vec in examples}
    assert masked_labels == set(range(len(LABELS)))


def test_generate_examples_has_phrasing_variety():
    examples = generate_examples(n=200)
    unique_texts = {e[0] for e in examples}
    assert len(unique_texts) > 20
