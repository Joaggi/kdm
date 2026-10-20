import torch

from ..base import Kernel


class SchoenbergKernelLayer(Kernel):
    """Wraps a robust kernel layer with the Schoenberg construction::

        K_PD(x, y) = exp(-gamma * Phi(x, y)),   gamma > 0

    where Phi(x, y) = -base_kernel(x, y). The base layer is registered as a
    submodule, so its parameters (e.g. the cutoff `c`) stay trainable and
    are saved with the model. Works with any layer taking `(A, B)`, including
    the `Mem...` variants.

    Arguments:
        base_kernel: a kernel layer, e.g. `TukeyKernelLayer(c=3.0, dim=d)`.
        gamma: fixed (not learned) positive scale. Default 1.0.
    """

    def __init__(self, base_kernel, gamma=1.0):
        super().__init__()
        if not isinstance(base_kernel, torch.nn.Module):
            raise TypeError("base_kernel must be a kernel layer (nn.Module).")
        if gamma <= 0:
            raise ValueError("gamma must be > 0.")
        self.base_kernel = base_kernel
        self.gamma = gamma

    def forward(self, A, B):
        """
        Arguments:
            A: tensor of shape (bs, n, d)
            B: tensor of shape (m, d), or (bs, m, d) for `Mem` base layers
        Returns:
            K: tensor of shape (bs, n, m)
        """
        phi = -self.base_kernel(A, B)
        return torch.exp(-self.gamma * phi)
