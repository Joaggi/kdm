from typing import Optional

import torch.nn as nn

from ..base import Kernel
from ..layers import KDMLayer, RBFKernelLayer
from ..utils import dm2discrete, pure2dm


class KDMSequentialClassModel(nn.Module):
    """Chained KDM layers for classification.

    The first layer uses `kernel` on the encoded input (an RBF built from
    `sigma` if None); subsequent
    layers are configured via a list of dicts with keys
    `kernel`, `dim_x`, `dim_y`, `n_comp`.
    """

    def __init__(
        self,
        encoded_size: int,
        dim_y: int,
        encoder: nn.Module,
        n_comp: int,
        sigma: float = 0.1,
        sequence=None,
        kernel: Optional[Kernel] = None,
    ):
        super().__init__()
        self.encoded_size = encoded_size
        self.dim_y = dim_y
        self.n_comp = n_comp
        self.encoder = encoder
        if kernel is None:
            kernel = RBFKernelLayer(sigma=sigma, dim=encoded_size, trainable=True)
        first = KDMLayer(
            kernel=kernel,
            dim_x=encoded_size,
            dim_y=dim_y,
            n_comp=n_comp,
        )
        stack = [first]
        for cfg in (sequence or []):
            stack.append(
                KDMLayer(
                    kernel=cfg["kernel"],
                    dim_x=cfg["dim_x"],
                    dim_y=cfg["dim_y"],
                    n_comp=cfg["n_comp"],
                )
            )
        self.kdm_stack = nn.Sequential(*stack)

    def forward(self, x):
        encoded = self.encoder(x)
        rho_x = pure2dm(encoded)
        rho_y = self.kdm_stack(rho_x)
        return dm2discrete(rho_y)
