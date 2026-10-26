"""
Tests for the cdist-based distance in RBFKernelLayer / MemRBFKernelLayer.

The layers compute distances with `torch.cdist` (direct differences). These
tests check that the result equals the original Euclidean distance
formulation, ||a||^2 + ||b||^2 - 2ab, on normal-scale data, and show where
the two differ: the identity loses precision for large-magnitude inputs.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from kdm.layers import MemRBFKernelLayer, RBFKernelLayer

RNG = np.random.default_rng(2024)

DTYPES = [torch.float64, torch.float32]
DTYPE_IDS = ["float64", "float32"]
ATOL = {torch.float64: 1e-12, torch.float32: 1e-5}


def _rand(shape, dtype):
    return torch.as_tensor(RNG.standard_normal(shape), dtype=dtype)


def _cast_layer(layer: torch.nn.Module, dtype: torch.dtype) -> torch.nn.Module:
    # Re-assign sigma after cast so softplus_inv runs at the new dtype.
    sigma = float(layer.sigma.detach())
    layer = layer.double() if dtype == torch.float64 else layer.float()
    layer.sigma = sigma
    return layer


def _identity_dist2(A, B):
    """Original squared Euclidean distance: ||a||^2 + ||b||^2 - 2ab.

    A: (bs, n, d); B: (m, d) or (bs, m, d). Returns (bs, n, m).
    """
    A_norm = (A ** 2).sum(dim=-1, keepdim=True)  # (bs, n, 1)
    if B.ndim == 2:
        AB = torch.matmul(A, B.transpose(-1, -2))
        B_norm = (B ** 2).sum(dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, m)
    else:
        AB = torch.matmul(A, B.transpose(1, 2))
        B_norm = (B ** 2).sum(dim=-1).unsqueeze(1)  # (bs, 1, m)
    return (A_norm + B_norm - 2.0 * AB).clamp(min=0.0)


def _identity_rbf(A, B, sigma):
    """Original RBF kernel, built on the identity distance."""
    return torch.exp(-_identity_dist2(A, B) / (2.0 * sigma ** 2))


# ---------------------------------------------------------------------------
# cdist distance == original Euclidean distance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_cdist_equals_identity_distance(dtype):
    A = _rand((4, 5, 3), dtype)
    B = _rand((7, 3), dtype)
    B_batched = B.unsqueeze(0).expand(A.shape[0], -1, -1)
    dist = torch.cdist(A, B_batched, compute_mode="donot_use_mm_for_euclid_dist")
    assert torch.allclose(dist ** 2, _identity_dist2(A, B), atol=ATOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_cdist_equals_identity_distance_batched_b(dtype):
    A = _rand((4, 5, 3), dtype)
    B = _rand((4, 7, 3), dtype)
    dist = torch.cdist(A, B, compute_mode="donot_use_mm_for_euclid_dist")
    assert torch.allclose(dist ** 2, _identity_dist2(A, B), atol=ATOL[dtype])


# ---------------------------------------------------------------------------
# RBF layers == RBF built on the original distance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_rbf_kernel_matches_identity(dtype):
    layer = _cast_layer(RBFKernelLayer(sigma=0.8, dim=3), dtype)
    A = _rand((4, 5, 3), dtype)
    B = _rand((7, 3), dtype)
    expected = _identity_rbf(A, B, layer.sigma.detach())
    assert torch.allclose(layer(A, B), expected, atol=ATOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_mem_rbf_kernel_matches_identity(dtype):
    layer = _cast_layer(MemRBFKernelLayer(sigma=0.8, dim=3), dtype)
    A = _rand((4, 5, 3), dtype)
    B = _rand((4, 7, 3), dtype)
    expected = _identity_rbf(A, B, layer.sigma.detach())
    assert torch.allclose(layer(A, B), expected, atol=ATOL[dtype])


# ---------------------------------------------------------------------------
# Where cdist and the identity differ
# ---------------------------------------------------------------------------

def test_rbf_kernel_precise_for_large_inputs():
    # Points near 1e4 with unit spread: in float32 the identity subtracts
    # numbers of order 1e8 to recover distances of order 1, so it loses
    # nearly all precision. cdist stays close to the float64 reference.
    layer = RBFKernelLayer(sigma=1.0, dim=4)
    A32 = _rand((1, 5, 4), torch.float32) + 1e4
    B32 = _rand((6, 4), torch.float32) + 1e4
    sigma = layer.sigma.detach()

    # Reference: the same float32 inputs, evaluated in float64 with explicit
    # differences (independent of both methods under test).
    A64, B64 = A32.double(), B32.double()
    dist2 = ((A64[:, :, None, :] - B64[None, None, :, :]) ** 2).sum(dim=-1)
    reference = torch.exp(-dist2 / (2.0 * sigma.double() ** 2))

    cdist_err = (layer(A32, B32).double() - reference).abs().max()
    identity_err = (_identity_rbf(A32, B32, sigma).double() - reference).abs().max()
    assert cdist_err < 1e-5
    assert identity_err > 1e-2


def test_rbf_kernel_gradient_finite_at_zero_distance():
    # cdist must not produce NaN gradients when two points coincide.
    layer = RBFKernelLayer(sigma=0.5, dim=3)
    A = _rand((1, 5, 3), torch.float32).requires_grad_(True)
    K = layer(A, A[0].detach())  # diagonal entries have zero distance
    K.sum().backward()
    assert torch.isfinite(A.grad).all()
    assert torch.isfinite(layer.raw_sigma.grad)
