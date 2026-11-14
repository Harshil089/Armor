import torch

from armor.models.reference_bank import REFERENCE_CLASSES, ReferenceBank


def _bank():
    bank = ReferenceBank()
    bank.fit([
        (torch.tensor([1.0, 0.0]), "allow", "pseudo_model"),
        (torch.tensor([0.0, 1.0]), "block", "human"),
        (torch.tensor([0.7, 0.7]), "flag", "pseudo_model"),
        (torch.tensor([-1.0, 0.0]), "review", "pseudo_model"),  # excluded, not a reference class
    ])
    return bank


def test_nearest_similarity_high_for_matching_class():
    bank = _bank()
    sims = bank.nearest_similarity(torch.tensor([1.0, 0.0]))
    assert sims["allow"] > 0.99
    assert sims["block"] < 0.5


def test_review_examples_excluded_from_reference_classes():
    bank = _bank()
    sims = bank.nearest_similarity(torch.tensor([-1.0, 0.0]))
    assert set(sims.keys()) == set(REFERENCE_CLASSES)


def test_label_source_distribution_tracks_provenance():
    bank = _bank()
    dist = bank.label_source_distribution()
    assert dist["allow"] == {"pseudo_model": 1}
    assert dist["block"] == {"human": 1}


def test_empty_bank_returns_zero_not_error():
    bank = ReferenceBank()
    sims = bank.nearest_similarity(torch.tensor([1.0, 0.0]))
    assert sims == {c: 0.0 for c in REFERENCE_CLASSES}
