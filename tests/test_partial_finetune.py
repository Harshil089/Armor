import pytest
import torch

from armor.models.train import _configure_encoder


class _FakeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Linear(2, 2)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])
        self.final_norm = torch.nn.LayerNorm(2)


def test_only_requested_final_layers_and_norm_are_trainable():
    encoder = _FakeEncoder()

    trainable = _configure_encoder(encoder, fine_tune_layers=2)

    assert not any(param.requires_grad for param in encoder.embeddings.parameters())
    assert not any(param.requires_grad for layer in encoder.layers[:2] for param in layer.parameters())
    assert all(param.requires_grad for layer in encoder.layers[2:] for param in layer.parameters())
    assert all(param.requires_grad for param in encoder.final_norm.parameters())
    assert set(trainable) == {param for param in encoder.parameters() if param.requires_grad}


def test_rejects_more_layers_than_encoder_has():
    with pytest.raises(ValueError):
        _configure_encoder(_FakeEncoder(), fine_tune_layers=5)
