import torch

from armor.eval.audit_sampling import stratified_sample


class _FakeModel:
    """Distinct, non-similar embeddings per distinct text (via hash), so
    dedup logic doesn't accidentally collapse everything -- and two exact
    duplicate strings get identical embeddings, so dedup DOES catch them."""

    def predict_with_embedding(self, text: str):
        seed = hash(text) % 1000
        gen = torch.Generator().manual_seed(seed)
        return {}, torch.randn(16, generator=gen)


def test_stratified_sample_respects_per_class_cap():
    rows = [(f"allow text {i}", "allow", "pseudo_model") for i in range(30)]
    rows += [(f"block text {i}", "block", "pseudo_model") for i in range(30)]
    result = stratified_sample(rows, _FakeModel(), per_class=10)
    allow_count = sum(1 for r in result if r["existing_label"] == "allow")
    block_count = sum(1 for r in result if r["existing_label"] == "block")
    assert allow_count == 10
    assert block_count == 10


def test_stratified_sample_excludes_near_duplicates():
    rows = [("identical text", "flag", "pseudo_model") for _ in range(10)]
    result = stratified_sample(rows, _FakeModel(), per_class=10)
    # All 10 rows are byte-identical -> identical embedding -> only the
    # first should survive the similarity filter.
    assert len(result) == 1


def test_stratified_sample_handles_missing_classes_gracefully():
    rows = [("only allow text", "allow", "pseudo_model")]
    result = stratified_sample(rows, _FakeModel(), per_class=25)
    assert len(result) == 1
    assert result[0]["existing_label"] == "allow"


def test_sample_id_is_deterministic_for_same_content():
    from armor.eval.audit_sampling import _sample_id
    assert _sample_id("hello") == _sample_id("hello")
    assert _sample_id("hello") != _sample_id("world")
