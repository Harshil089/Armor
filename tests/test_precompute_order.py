"""The length-sort inside _precompute_embeddings is a speed optimization
only. If it leaks into the returned order, training order starts depending
on every row's text length, so editing one label's wording silently
reshuffles batch composition for every other label — which is exactly how
dangerous_tool_action fell 0.213 -> 0.045 on a run that only touched
obfuscation_encoding's phrasing. These tests pin the contract.
"""

import torch
import pytest

from armor.models.train import _precompute_embeddings

_HIDDEN = 4


class _FakeTokenizer:
    """Encodes each text's length into input_ids so the fake encoder can
    echo an identity back, letting the test verify embedding order too and
    not just the labels riding alongside it."""

    def __call__(self, texts, return_tensors=None, padding=None, truncation=None,
                 max_length=None, pad_to_multiple_of=None):
        longest = max(len(t) for t in texts)
        if pad_to_multiple_of:
            longest = ((longest + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        input_ids = torch.zeros(len(texts), longest, dtype=torch.long)
        for row, text in enumerate(texts):
            input_ids[row, 0] = len(text)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def encode(self, text, add_special_tokens=True, truncation=False, max_length=None):
        return list(range(len(text)))


class _FakeEncoderOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _FakeEncoder:
    def __call__(self, input_ids=None, attention_mask=None):
        batch, seq = input_ids.shape
        hidden = torch.zeros(batch, seq, _HIDDEN)
        # Broadcast the identity marker across the CLS position.
        hidden[:, 0, :] = input_ids[:, 0].unsqueeze(1).float()
        return _FakeEncoderOutput(hidden)


class _FailOnSecondCallEncoder(_FakeEncoder):
    def __init__(self):
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("simulated interruption")
        return super().__call__(**kwargs)


def _examples():
    # Deliberately unsorted lengths so a leaked sort would reorder them.
    texts = ["dddd", "a", "ccccccc", "bb", "eeeeeeeeeeee", "ffff"]
    examples = []
    for index, text in enumerate(texts):
        labels = [float(index), 0.0]
        masks = [1.0, 0.0]
        examples.append((text, labels, masks))
    return examples


def test_returns_rows_in_original_order_not_sorted_order():
    examples = _examples()
    pooled, labels, masks = _precompute_embeddings(
        examples, _FakeEncoder(), _FakeTokenizer(), "cpu", batch_size=2
    )

    for index, (text, expected_labels, expected_masks) in enumerate(examples):
        # The fake encoder echoes text length, so this pins embedding order.
        assert pooled[index, 0].item() == len(text)
        assert labels[index].tolist() == expected_labels
        assert masks[index].tolist() == expected_masks


def test_row_contents_stay_aligned_with_each_other():
    examples = _examples()
    pooled, labels, _ = _precompute_embeddings(
        examples, _FakeEncoder(), _FakeTokenizer(), "cpu", batch_size=4
    )

    # labels[i][0] was set to the row's original index, so an embedding
    # paired with the wrong labels row would break this.
    for index in range(len(examples)):
        assert labels[index][0].item() == index
        assert pooled[index, 0].item() == len(examples[index][0])


def test_partial_feature_cache_resumes_without_reordering(tmp_path):
    examples = _examples()
    cache = str(tmp_path / "features.partial")
    metadata = {"fixture": "v1"}
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _precompute_embeddings(
            examples, _FailOnSecondCallEncoder(), _FakeTokenizer(), "cpu", batch_size=2,
            max_length=8192, partial_cache_path=cache,
            partial_cache_metadata=metadata, partial_cache_interval=1,
        )
    partial = torch.load(cache, map_location="cpu", weights_only=True)
    for features in partial["pooled_chunks"]:
        assert features.untyped_storage().nbytes() == features.numel() * features.element_size()

    pooled, labels, masks = _precompute_embeddings(
        examples, _FakeEncoder(), _FakeTokenizer(), "cpu", batch_size=2,
        max_length=8192, partial_cache_path=cache,
        partial_cache_metadata=metadata, partial_cache_interval=1,
    )
    for index, (text, expected_labels, expected_masks) in enumerate(examples):
        assert pooled[index, 0].item() == len(text)
        assert labels[index].tolist() == expected_labels
        assert masks[index].tolist() == expected_masks


def test_singleton_feature_cache_drops_full_sequence_storage(tmp_path):
    text = "z" * 600
    examples = [(text, [1.0, 0.0], [1.0, 1.0])]
    cache = str(tmp_path / "singleton.partial")
    _precompute_embeddings(
        examples, _FakeEncoder(), _FakeTokenizer(), "cpu", batch_size=2,
        max_length=8192, partial_cache_path=cache,
        partial_cache_metadata={"fixture": "singleton"}, partial_cache_interval=1,
    )
    partial = torch.load(cache, map_location="cpu", weights_only=True)
    features = partial["pooled_chunks"][0]
    assert features.shape == (1, _HIDDEN)
    assert features.untyped_storage().nbytes() == features.numel() * features.element_size()


def test_long_singleton_shapes_are_padded_to_bounded_mps_buckets():
    text = "z" * 600

    class RecordingEncoder(_FakeEncoder):
        sequence_lengths = []

        def __call__(self, input_ids=None, attention_mask=None):
            self.sequence_lengths.append(input_ids.shape[1])
            return super().__call__(input_ids=input_ids, attention_mask=attention_mask)

    encoder = RecordingEncoder()
    _precompute_embeddings(
        [(text, [1.0, 0.0], [1.0, 1.0])], encoder, _FakeTokenizer(), "cpu",
        batch_size=2, max_length=8192,
    )
    assert encoder.sequence_lengths == [640]
