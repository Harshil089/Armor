import torch
from types import SimpleNamespace

from armor.models.train import _collate, _collate_at_length, _precompute_embeddings
from armor.data.long_context_curriculum import MIX, assign_rows, target_counts


class _CharTokenizer:
    def __init__(self):
        self.batch_sizes = []

    def encode(self, text, *, add_special_tokens=True, truncation=False, max_length=None):
        size = len(text) + (2 if add_special_tokens else 0)
        return list(range(min(size, max_length) if max_length else size))

    def __call__(self, texts, *, return_tensors, padding, truncation, max_length, **kwargs):
        self.batch_sizes.append(len(texts))
        encoded = [list(range(min(len(text), max_length))) for text in texts]
        width = max(map(len, encoded))
        ids = torch.zeros((len(texts), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(encoded):
            ids[i, :len(row)] = torch.tensor(row)
            mask[i, :len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _TinyEncoder:
    def __call__(self, input_ids, attention_mask):
        hidden = torch.zeros((input_ids.shape[0], input_ids.shape[1], 2))
        return SimpleNamespace(last_hidden_state=hidden)


def test_existing_512_training_cap_stays_the_default_but_curriculum_can_keep_long_rows():
    tokenizer = _CharTokenizer()
    batch = [("x" * 700, [1.0], [1.0])]

    old_inputs, _, _ = _collate(batch, tokenizer)
    long_inputs, _, _ = _collate_at_length(batch, tokenizer, max_length=8192)

    assert old_inputs["input_ids"].shape == (1, 512)
    assert long_inputs["input_ids"].shape == (1, 700)


def test_curriculum_assignment_preserves_every_row_and_requested_mix():
    count = 1000
    lengths = [100] * 900 + [700] * 80 + [1500] * 15 + [3000] * 5
    rows = [("source", (str(i), [], [])) for i in range(count)]

    groups = assign_rows(rows, lengths, seed=7)

    assert [len(group) for group in groups] == target_counts(count)
    flattened = [index for group in groups for index in group]
    assert len(flattened) == len(set(flattened)) == count
    assert set(flattened) == set(range(count))
    for group_index, group in enumerate(groups):
        for index in group:
            if lengths[index] > 512:
                low, high = MIX[group_index][2:]
                assert low <= lengths[index] <= high


def test_long_feature_precompute_batches_only_verified_short_rows():
    tokenizer = _CharTokenizer()
    examples = [("x" * size, [0.0], [1.0]) for size in (100, 200, 600, 700)]

    pooled, labels, masks = _precompute_embeddings(
        examples, _TinyEncoder(), tokenizer, "cpu", batch_size=2, max_length=8192
    )

    assert tokenizer.batch_sizes == [2, 1, 1]
    assert pooled.shape == (4, 2)
    assert labels.shape == masks.shape == (4, 1)
