import torch

from armor.eval.active_learning import (
    boundary_margin_score,
    cluster_for_diversity,
    disagreement_score,
    neighbor_to_rare_class,
    novelty_score,
    select_diverse_candidates,
)
from armor.models.ood import OODReference
from armor.models.reference_bank import ReferenceBank


def test_boundary_margin_low_for_near_tie():
    assert boundary_margin_score([0.4, 0.35, 0.25]) < 0.1


def test_boundary_margin_high_for_confident_prediction():
    assert boundary_margin_score([0.95, 0.03, 0.02]) > 0.8


def _bank():
    bank = ReferenceBank()
    bank.fit([
        (torch.tensor([1.0, 0.0]), "allow", "pseudo_model"),
        (torch.tensor([0.0, 1.0]), "block", "pseudo_model"),
    ])
    return bank


def test_disagreement_score_high_when_nearest_differs_from_prediction():
    bank = _bank()
    score = disagreement_score("allow", bank, torch.tensor([0.0, 1.0]))  # nearest is "block"
    assert score == 1.0


def test_disagreement_score_zero_when_prediction_matches_nearest():
    bank = _bank()
    score = disagreement_score("allow", bank, torch.tensor([1.0, 0.0]))
    assert score == 0.0


def test_neighbor_to_rare_class_uses_similarity():
    bank = ReferenceBank()
    bank.fit([(torch.tensor([1.0, 0.0]), "flag", "pseudo_model")])
    score = neighbor_to_rare_class(torch.tensor([1.0, 0.0]), bank, rare_classes=("flag",))
    assert score > 0.99


def test_novelty_score_wraps_ood_reference():
    ood = OODReference()
    ood.fit([torch.tensor([1.0, 0.0])])
    assert novelty_score(torch.tensor([1.0, 0.0]), ood) < 1e-6
    assert novelty_score(torch.tensor([0.0, 1.0]), ood) > 0.9


def test_cluster_for_diversity_returns_one_label_per_embedding():
    embeddings = [torch.tensor([1.0, 0.0]), torch.tensor([1.1, 0.0]), torch.tensor([0.0, 1.0])]
    labels = cluster_for_diversity(embeddings, n_clusters=2)
    assert len(labels) == 3
    assert labels[0] == labels[1]  # the two near-identical points cluster together
    assert labels[2] != labels[0]


def test_select_diverse_candidates_caps_per_cluster():
    embeddings = [torch.tensor([1.0, 0.0]), torch.tensor([1.05, 0.0]), torch.tensor([0.0, 1.0])]
    candidates = [{"id": 1}, {"id": 2}, {"id": 3}]
    selected = select_diverse_candidates(candidates, embeddings, n_clusters=2, per_cluster=1)
    assert len(selected) == 2  # one per cluster, not all 3
