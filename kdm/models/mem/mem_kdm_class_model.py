from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...base import Kernel
from ...layers import MemKDMLayer, MemRBFKernelLayer
from ...utils import dm2discrete, pure2dm


class MemKDMClassModel(nn.Module):
    """Memory-based KDM classifier.

    Input is a tuple `(x_enc, x_neigh, y_neigh)` produced by the wrapper:
        x_enc:   (bs, encoded_size)
        x_neigh: (bs, n_comp, encoded_size)  — per-sample neighbors from faiss
        y_neigh: (bs, n_comp)                — integer class labels

    `kernel` must be a memory-variant kernel (B of shape (bs, m, d)), e.g.
    `MemTukeyKernelLayer`. If None, a `MemRBFKernelLayer` is built from `sigma`.
    """

    def __init__(
        self,
        encoded_size: int,
        dim_y: int,
        n_comp: int,
        sigma: float = 0.1,
        kernel: Optional[Kernel] = None,
    ):
        super().__init__()
        self.encoded_size = encoded_size
        self.dim_y = dim_y
        self.n_comp = n_comp
        if kernel is None:
            kernel = MemRBFKernelLayer(
                sigma=sigma, dim=encoded_size, trainable=True
            )
        self.kernel = kernel
        self.mkdm = MemKDMLayer(
            kernel=self.kernel, dim_x=encoded_size, dim_y=dim_y, n_comp=n_comp
        )

    def forward(self, inputs):
        x_enc, x_neigh, y_neigh = inputs
        rho_x = pure2dm(x_enc)
        y_neigh_ohe = F.one_hot(y_neigh.long(), num_classes=self.dim_y).to(x_enc.dtype)
        rho_y = self.mkdm((rho_x, x_neigh, y_neigh_ohe))
        return dm2discrete(rho_y)
