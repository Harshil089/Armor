from armor.data.synthetic_long_negatives import generate_examples
from armor.models.labels import LABELS


def test_generate_examples_shape_and_labels():
    examples = generate_examples(n=100)
    assert len(examples) == 100

    for text, label_vec, mask_vec in examples:
        assert isinstance(text, str) and text
        # The point of this source: rows are long, unlike every other generator.
        assert len(text.split()) >= 30
        assert label_vec == [0.0] * len(LABELS)
        assert sum(mask_vec) == 1.0


def test_rows_are_newline_joined_multi_turn():
    # Must match how OpenAIChatAdapter.extract_prompt_text joins a conversation.
    for text, _, _ in generate_examples(n=20):
        assert text.count("\n") >= 5


def test_generate_examples_spreads_across_all_labels():
    examples = generate_examples(n=600)
    masked = {mask_vec.index(1.0) for _, _, mask_vec in examples}
    assert masked == set(range(len(LABELS)))
