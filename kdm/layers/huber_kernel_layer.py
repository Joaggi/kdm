import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import Kernel


def _softplus_inv(y: torch.Tensor) -> torch.Tensor:
    # inverse of softplus: log(exp(y) - 1), numerically stable for y > 0
    return torch.log(torch.expm1(y))


class HuberKernelLayer(Kernel):
    """Huber robust kernel with a trainable, strictly positive scale.

        k(x, y) = -1/4 * r**2                 if r / c <= 1
                  -c/2 * r + c**2 / 4         otherwise,    r = ||x - y||

    The values are <= 0 with a maximum of 0 at x = y, so it is meant to be
    used through `SchoenbergKernelLayer` to obtain a valid kernel.

    c is parameterized as `softplus(raw_c) + min_c`, like sigma in
    `RBFKernelLayer`, so it stays positive without per-step clamping.

    Distances are computed with `torch.cdist` (direct differences) rather
    than the `||a||^2 + ||b||^2 - 2ab` shortcut used by `RBFKernelLayer`,
    which loses precision for large-magnitude inputs.

    Arguments:
        c: initial value of the scale (must be > min_c).
        dim: dimensionality of the input vectors.
        trainable: whether c is learnable.
        min_c: lower bound enforced structurally via the softplus offset.
    """

    def __init__(self, c, dim, trainable=True, min_c=1e-3):
        super().__init__()
        self.dim = dim
        self.min_c = min_c
        c_t = torch.as_tensor(float(c))
        if c_t <= min_c:
            raise ValueError(
                f"Initial c ({float(c_t):g}) must be > min_c ({min_c:g})"
            )
        raw = _softplus_inv(c_t - min_c)
        self.raw_c = nn.Parameter(raw, requires_grad=trainable)

    @property
    def c(self) -> torch.Tensor:
        return F.softplus(self.raw_c) + self.min_c

    @c.setter
    def c(self, value) -> None:
        with torch.no_grad():
            value_t = torch.as_tensor(value, dtype=self.raw_c.dtype,
                                      device=self.raw_c.device)
            if (value_t <= self.min_c).any():
                raise ValueError(f"Assigned c must be > min_c ({self.min_c:g})")
            self.raw_c.copy_(_softplus_inv(value_t - self.min_c))

    def _kernel(self, dist):
        c = self.c
        return torch.where(dist / c <= 1, -0.25 * dist ** 2, -c / 2.0 * dist + c ** 2 / 4.0)

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
        return self._kernel(dist)


class MemHuberKernelLayer(HuberKernelLayer):
    """Memory-variant Huber kernel: B carries a batch dimension.

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
        return self._kernel(dist)
