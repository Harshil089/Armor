import torch

from armor.models.train import _multilabel_loss

loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")


def test_sparse_label_not_drowned_out_by_dense_label():
    # label 0: dense (3 of 3 rows masked), predictions perfect -> ~0 loss
    # label 1: sparse (1 of 3 rows masked), prediction wrong -> high loss
    logits = torch.tensor([[10.0, -10.0], [10.0, 0.0], [-10.0, 0.0]])
    labels = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    masks = torch.tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]])

    loss = _multilabel_loss(logits, labels, masks, loss_fn)

    # naive sum/divide over all masked cells would be dominated by label 0's
    # near-zero loss (2 of 3 masked cells belong to label 0); per-label
    # averaging must still reflect label 1's large error.
    assert loss.item() > 1.0


def test_label_with_zero_masked_rows_is_excluded_not_nan():
    logits = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    masks = torch.tensor([[1.0, 0.0], [1.0, 0.0]])  # label 1 never masked

    loss = _multilabel_loss(logits, labels, masks, loss_fn)

    assert torch.isfinite(loss)


def test_equal_coverage_matches_plain_mean_of_per_label_means():
    logits = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    masks = torch.ones(2, 2)

    loss = _multilabel_loss(logits, labels, masks, loss_fn)

    label0_loss = loss_fn(logits[:, 0], labels[:, 0]).mean()
    label1_loss = loss_fn(logits[:, 1], labels[:, 1]).mean()
    expected = (label0_loss + label1_loss) / 2

    assert abs(loss.item() - expected.item()) < 1e-6


def test_no_masked_rows_at_all_returns_zero_not_crash():
    logits = torch.zeros(2, 2)
    labels = torch.zeros(2, 2)
    masks = torch.zeros(2, 2)

    loss = _multilabel_loss(logits, labels, masks, loss_fn)

    assert loss.item() == 0.0
