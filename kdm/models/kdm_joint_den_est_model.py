from typing import Optional

import torch
import torch.nn as nn

from ..base import Kernel
from ..layers import (
    CosineKernelLayer,
    CrossProductKernelLayer,
    KDMProjLayer,
    RBFKernelLayer,
)


class KDMJointDenEstModel(nn.Module):
    """Joint density estimation over (x, y) with a cross-product kernel.

    Uses an RBF kernel on the x slice and a cosine kernel on the y slice.
    `kernel_x` replaces the x-slice kernel; if None, an `RBFKernelLayer` is
    built from `sigma`, `trainable_sigma` and `min_sigma`.
    `forward(xy)` returns per-sample log-density; training minimizes
    `-forward(xy).mean()`.
    """

    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        sigma: float,
        n_comp: int,
        trainable_sigma: bool = True,
        min_sigma: float = 1e-3,
        kernel_x: Optional[Kernel] = None,
    ):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_comp = n_comp
        if kernel_x is None:
            kernel_x = RBFKernelLayer(
                sigma=sigma, dim=dim_x,
                trainable=trainable_sigma, min_sigma=min_sigma,
            )
        self.kernel_x = kernel_x
        self.kernel_y = CosineKernelLayer()
        self.kernel = CrossProductKernelLayer(
            dim1=dim_x, kernel1=self.kernel_x, kernel2=self.kernel_y
        )
        self.kdmproj = KDMProjLayer(
            kernel=self.kernel, dim_x=dim_x + dim_y, n_comp=n_comp
        )
        self.eps = 1e-7

    def forward(self, xy):
        return torch.log(self.kdmproj(xy) + self.eps) + self.kernel.log_weight()
