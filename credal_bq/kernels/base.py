"""
credal_bq.kernels.base
----------------------
Minimal kernel base class used by cosine kernels.

Subclasses should implement forward(W1, W2) returning a kernel matrix.
This base provides __call__ and gram(W) conveniences.
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Protocol


class Kernel:
    """
    Base kernel with convenience helpers.

    Subclasses must implement forward(W1, W2) → Tensor of shape (..., n, m).
    """

    def forward(self, W1: Tensor, W2: Tensor) -> Tensor:  # pragma: no cover
        raise NotImplementedError

    def __call__(self, W1: Tensor, W2: Tensor) -> Tensor:
        return self.forward(W1, W2)

    def gram(self, W: Tensor) -> Tensor:
        """Return Gram matrix K(W, W)."""
        return self.forward(W, W)

