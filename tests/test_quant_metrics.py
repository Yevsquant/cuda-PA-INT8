"""Unit tests for the cosine / rel-L2 output-error metrics. Pure-torch, CPU."""

import torch

from quant_metrics import cosine_sim, rel_l2


def test_identical_vectors():
    x = torch.randn(4, 8, 128)
    torch.testing.assert_close(cosine_sim(x, x), torch.ones(4, 8))
    torch.testing.assert_close(rel_l2(x, x), torch.zeros(4, 8))


def test_scaled_vector_is_cosine_invariant():
    """Cosine ignores magnitude; rel_l2 does not."""
    x = torch.randn(3, 128)
    torch.testing.assert_close(cosine_sim(x, 2.0 * x), torch.ones(3))
    torch.testing.assert_close(rel_l2(x, 2.0 * x), torch.full((3,), 0.5))


def test_near_zero_reference_is_not_noisy():
    """The whole-vector metrics stay finite even when the reference has elements
    at exactly zero — the failure mode of the old elementwise metric."""
    b = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    a = torch.tensor([[1.0, 1e-3, 0.0, 0.0]])
    assert torch.isfinite(cosine_sim(a, b)).all()
    assert torch.isfinite(rel_l2(a, b)).all()
    assert rel_l2(a, b).item() < 1e-2  # a small perturbation stays small
