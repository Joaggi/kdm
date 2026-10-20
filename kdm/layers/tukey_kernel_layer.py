import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import Kernel


def _softplus_inv(y: torch.Tensor) -> torch.Tensor:
    # inverse of softplus: log(exp(y) - 1), numerically stable for y > 0
    return torch.log(torch.expm1(y))


class TukeyKernelLayer(Kernel):
    """Tukey (biweight) robust kernel with a trainable, strictly positive cutoff.

        k(x, y) = 1/2 * (1 - (||x-y|| / c)**2)**3   if ||x-y|| / c <= 1
                  0                                  otherwise

    c is parameterized as `softplus(raw_c) + min_c`, like sigma in
    `RBFKernelLayer`, so it stays positive without per-step clamping.

    Distances are computed with `torch.cdist` (direct differences) rather
    than the `||a||^2 + ||b||^2 - 2ab` shortcut used by `RBFKernelLayer`,
    which loses precision for large-magnitude inputs.

    Arguments:
        c: initial value of the cutoff (must be > min_c).
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
        # Clamping the ratio to 1 gives exactly 0 beyond the cutoff, matching
        # the piecewise definition.
        ratio2 = ((dist / self.c) ** 2).clamp(max=1.0)
        return 0.5 * (1.0 - ratio2) ** 3

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


class MemTukeyKernelLayer(TukeyKernelLayer):
    """Memory-variant Tukey kernel: B carries a batch dimension.

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
