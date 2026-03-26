"""
kernels/cosine_kernels.py
--------------------------
Kernels on S^{d-1} parameterised by the cosine similarity
(equivalently geodesic distance) between directions.

For two L2-normalised vectors w_i, w_j:
    cos_sim = w_i · w_j ∈ [-1, 1]
    geodesic = arccos(cos_sim) ∈ [0, π]

We define:
    d²(w_i, w_j) = 1 - cos_sim     (half the squared chord distance)

CosineRBF:
    k_ℓ(w_i, w_j) = exp(-(1 - cos_sim) / ℓ²)

    At ℓ→∞: k_∞ = 1 (constant kernel, CBM degenerate limit)
    At ℓ→0: k_0 = δ(i=j) (identity kernel)

CosineMaterN32:
    k_ℓ(w_i, w_j) = (1 + √3 · r/ℓ) exp(-√3 · r/ℓ),  r = √(1 - cos_sim)

CosineMaterN52:
    k_ℓ(w_i, w_j) = (1 + √5 · r/ℓ + 5r²/(3ℓ²)) exp(-√5 · r/ℓ)

For the discrete uniform measure  P̃ = (1/k) Σ_j δ(w - w_j),
the BQ integrals reduce to row-sum operations:

    z_i = ∫ k(w_i, w) dP̃(w) = (1/k) Σ_j k(w_i, w_j) = (1/k) [G 1]_i
    K_PP = ∫∫ k(w, w') dP̃(w) dP̃(w') = (1/k²) 1ᵀ G 1

Both are O(k) after G is computed (no additional integration needed).
"""

import math
import torch
from torch import Tensor

from .base import Kernel


# ─────────────────────────────────────────────────────────────────────────────
class CosineRBF(Kernel):
    """
    k_ℓ(w_i, w_j) = exp(-(1 - w_i·w_j) / ℓ²)

    Args:
        lengthscale: ℓ > 0.  Controls how quickly similarity decays
                     with angular separation.
    """

    def __init__(self, lengthscale: float = 0.7):
        if lengthscale <= 0:
            raise ValueError(f"lengthscale must be > 0, got {lengthscale}")
        self.lengthscale = lengthscale

    def forward(self, W1: Tensor, W2: Tensor) -> Tensor:
        """
        Args:
            W1: (..., n, d)  L2-normalised
            W2: (..., m, d)  L2-normalised
        Returns:
            K:  (..., n, m)
        """
        # cos_sim[..., i, j] = W1[..., i, :] · W2[..., j, :]
        cos_sim = torch.matmul(W1, W2.transpose(-2, -1))  # (..., n, m)
        # clamp for numerical safety (should be in [-1,1] for unit vectors)
        cos_sim = cos_sim.clamp(-1.0, 1.0)
        d_sq = 1.0 - cos_sim                               # ∈ [0, 2]
        return torch.exp(-d_sq / (self.lengthscale ** 2))

    def __repr__(self) -> str:
        return f"CosineRBF(ℓ={self.lengthscale})"


# ─────────────────────────────────────────────────────────────────────────────
class CosineMatern32(Kernel):
    """
    Matérn-3/2 on S^{d-1}:
        r = sqrt(1 - w_i·w_j)
        k_ℓ(w_i, w_j) = (1 + √3·r/ℓ) · exp(-√3·r/ℓ)

    Differentiable everywhere; rougher than RBF.
    """
    _sqrt3 = math.sqrt(3.0)

    def __init__(self, lengthscale: float = 0.7):
        if lengthscale <= 0:
            raise ValueError(f"lengthscale must be > 0, got {lengthscale}")
        self.lengthscale = lengthscale

    def forward(self, W1: Tensor, W2: Tensor) -> Tensor:
        cos_sim = torch.matmul(W1, W2.transpose(-2, -1)).clamp(-1.0, 1.0)
        # r = sqrt(max(1 - cos_sim, 0)) — clamp avoids sqrt(neg) from float noise
        r = (1.0 - cos_sim).clamp(min=0.0).sqrt()
        scaled = self._sqrt3 * r / self.lengthscale
        return (1.0 + scaled) * torch.exp(-scaled)

    def __repr__(self) -> str:
        return f"CosineMatern32(ℓ={self.lengthscale})"


# ─────────────────────────────────────────────────────────────────────────────
class CosineMatern52(Kernel):
    """
    Matérn-5/2 on S^{d-1}:
        r = sqrt(1 - w_i·w_j)
        k_ℓ(w_i, w_j) = (1 + √5·r/ℓ + 5r²/(3ℓ²)) · exp(-√5·r/ℓ)

    Twice differentiable; smoother than Matérn-3/2, rougher than RBF.
    """
    _sqrt5 = math.sqrt(5.0)

    def __init__(self, lengthscale: float = 0.7):
        if lengthscale <= 0:
            raise ValueError(f"lengthscale must be > 0, got {lengthscale}")
        self.lengthscale = lengthscale

    def forward(self, W1: Tensor, W2: Tensor) -> Tensor:
        cos_sim = torch.matmul(W1, W2.transpose(-2, -1)).clamp(-1.0, 1.0)
        r = (1.0 - cos_sim).clamp(min=0.0).sqrt()
        scaled = self._sqrt5 * r / self.lengthscale
        r_sq_term = 5.0 * (1.0 - cos_sim).clamp(min=0.0) / (3.0 * self.lengthscale ** 2)
        return (1.0 + scaled + r_sq_term) * torch.exp(-scaled)

    def __repr__(self) -> str:
        return f"CosineMatern52(ℓ={self.lengthscale})"


# ─────────────────────────────────────────────────────────────────────────────
class ConstantKernel(Kernel):
    """
    k_∞(w_i, w_j) = σ²   (constant; CBM degenerate limit)

    Under this kernel every pair of concept directions is treated as
    pairwise orthogonal, and BQ posterior variance = 0 identically.
    Included so that CQ can verify containment (Prop. 1).
    """

    def __init__(self, value: float = 1.0):
        self.value = value

    def forward(self, W1: Tensor, W2: Tensor) -> Tensor:
        n = W1.shape[-2]
        m = W2.shape[-2]
        batch = W1.shape[:-2]
        return W1.new_full(batch + (n, m), self.value)

    def __repr__(self) -> str:
        return f"ConstantKernel(σ²={self.value})"


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def default_imprecise_prior() -> list[Kernel]:
    """
    Π_5 from the paper:
        { RBF(0.3), RBF(0.7), RBF(1.5), Matern52(0.7), Matern32(0.7) }
    """
    return [
        CosineRBF(0.3),
        CosineRBF(0.7),
        CosineRBF(1.5),
        CosineMatern52(0.7),
        CosineMatern32(0.7),
    ]
