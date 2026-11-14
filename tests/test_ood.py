import numpy as np
import torch

from armor.models.ood import OODReference


def test_score_zero_for_exact_match():
    ref = OODReference()
    ref.fit([torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0])])
    score = ref.score(torch.tensor([1.0, 0.0, 0.0]))
    assert score < 1e-6


def test_score_high_for_orthogonal_vector():
    ref = OODReference()
    ref.fit([torch.tensor([1.0, 0.0, 0.0])])
    score = ref.score(torch.tensor([0.0, 1.0, 0.0]))
    assert score > 0.9


def test_score_is_one_when_unfitted():
    ref = OODReference()
    assert ref.score(torch.tensor([1.0, 0.0])) == 1.0


def test_accepts_numpy_input():
    ref = OODReference()
    ref.fit([np.array([1.0, 0.0])])
    score = ref.score(np.array([1.0, 0.0]))
    assert score < 1e-6
