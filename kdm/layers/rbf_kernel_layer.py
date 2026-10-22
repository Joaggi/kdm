import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import Kernel


def _softplus_inv(y: torch.Tensor) -> torch.Tensor:
    # inverse of softplus: log(exp(y) - 1), numerically stable for y > 0
    return torch.log(torch.expm1(y))


class RBFKernelLayer(Kernel):
    """RBF (Gaussian) kernel with a trainable, strictly positive bandwidth.

    sigma is parameterized as `softplus(raw_sigma) + min_sigma`, which keeps
    it positive without any per-step clamping. Direct assignment (via the
    property setter) inverts softplus so that `kernel.sigma = new_value`
    behaves intuitively at initialization time.

    Arguments:
        sigma: initial value of the RBF scale parameter (must be > min_sigma).
        dim: dimensionality of the input vectors (used by `log_weight`).
        trainable: whether sigma is learnable.
        min_sigma: lower bound enforced structurally via the softplus offset.
    """

    def __init__(self, sigma, dim, trainable=True, min_sigma=1e-3):
        super().__init__()
        self.dim = dim
        self.min_sigma = min_sigma
        sigma_t = torch.as_tensor(float(sigma))
        if sigma_t <= min_sigma:
            raise ValueError(
                f"Initial sigma ({float(sigma_t):g}) must be > min_sigma ({min_sigma:g})"
            )
        raw = _softplus_inv(sigma_t - min_sigma)
        self.raw_sigma = nn.Parameter(raw, requires_grad=trainable)

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma) + self.min_sigma

    @sigma.setter
    def sigma(self, value) -> None:
        with torch.no_grad():
            value_t = torch.as_tensor(value, dtype=self.raw_sigma.dtype,
                                      device=self.raw_sigma.device)
            if (value_t <= self.min_sigma).any():
                raise ValueError(
                    f"Assigned sigma must be > min_sigma ({self.min_sigma:g})"
                )
            self.raw_sigma.copy_(_softplus_inv(value_t - self.min_sigma))

    def forward(self, A, B):
        """
        Arguments:
            A: tensor of shape (bs, n, d)
            B: tensor of shape (m, d)
        Returns:
            K: tensor of shape (bs, n, m)
        """
        B = B.unsqueeze(0).expand(A.shape[0], -1, -1)  # (bs, m, d)
        dist = torch.cdist(A, B, compute_mode="donot_use_mm_for_euclid_dist")  # (bs, n, m)
        return torch.exp(-dist ** 2 / (2.0 * self.sigma ** 2))

    def log_weight(self):
        return -self.dim * torch.log(self.sigma + 1e-12) - self.dim * math.log(math.pi) / 2


class MemRBFKernelLayer(RBFKernelLayer):
    """Memory-variant RBF kernel: B carries a batch dimension.

    Used by `MemKDMLayer`, where each query sample has its own set of
    neighbors (rather than a shared support set).
    """

    def forward(self, A, B):
        """
        Arguments:
            A: tensor of shape (bs, n, d)
            B: tensor of shape (bs, m, d)
        Returns:
            K: tensor of shape (bs, n, m)
        """
        dist = torch.cdist(A, B, compute_mode="donot_use_mm_for_euclid_dist")  # (bs, n, m)
        return torch.exp(-dist ** 2 / (2.0 * self.sigma ** 2))
