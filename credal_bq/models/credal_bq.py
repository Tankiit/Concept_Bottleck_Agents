"""
models/credal_bq.py
--------------------
CredalBQ: Credal Quadrature as described in §3 of the paper.

Given:
    W  ∈ R^{k×d}   L2-normalised concept directions
    c  ∈ R^{k}     concept activations
    Π = {κ₁,...,κ_M}  imprecise prior (list of kernels)
    γ              coverage factor (default 2.0)

For each kernel κ_m, run BQIntegrator to get (μ_m, σ_m).
Then:

    C⁻ = min_m (μ_m − γ σ_m)     lower bound
    C⁺ = max_m (μ_m + γ σ_m)     upper bound

    σ_num   = mean_m(σ_m)         numerical uncertainty (node geometry)
    σ_prior = std_m(μ_m)          prior uncertainty (kernel ambiguity)

The Gram matrices and Cholesky factors for all M kernels are computed
once in setup() and cached.  Per-image cost: O(M k²).
"""

import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
from typing import Sequence

from ..kernels.base import Kernel
from ..kernels.cosine_kernels import default_imprecise_prior
from .bq_integrator import BQIntegrator, _ensure_unit


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CQOutput:
    """
    Container for all outputs of a single CredalBQ forward pass.

    Shapes are (B,) for a batch of B images, or scalars if B=1.
    """
    c_lower:     Tensor   # C⁻: lower bound on class score
    c_upper:     Tensor   # C⁺: upper bound on class score
    sigma_num:   Tensor   # σ_num:   numerical uncertainty
    sigma_prior: Tensor   # σ_prior: prior (kernel ambiguity) uncertainty
    mu_per_kernel: Tensor # (M, B) posterior means per kernel
    sigma_per_kernel: Tensor  # (M, B) posterior stds per kernel


# ─────────────────────────────────────────────────────────────────────────────
class CredalBQ(nn.Module):
    """
    Credal Quadrature over an imprecise prior Π = {κ₁,...,κ_M}.

    Args:
        kernels:  List of Kernel instances.  Defaults to Π_5 from the paper.
        gamma:    Coverage factor γ.  Default 2.0 (≈ 95% for Gaussian).
        jitter:   Diagonal regularisation for each Gram matrix.
    """

    def __init__(
        self,
        kernels: Sequence[Kernel] | None = None,
        gamma: float = 2.0,
        jitter: float = 1e-6,
    ):
        super().__init__()
        if kernels is None:
            kernels = default_imprecise_prior()
        self.kernels = list(kernels)
        self.gamma = gamma
        self.jitter = jitter
        self.M = len(self.kernels)

        # One BQIntegrator per kernel
        self._integrators: list[BQIntegrator] = [
            BQIntegrator(k, jitter=jitter) for k in self.kernels
        ]
        self._ready = False

    # ── Setup ─────────────────────────────────────────────────────────────
    def setup(self, W: Tensor, S: Tensor | None = None) -> None:
        """
        Pre-compute Gram matrices and Cholesky factors for all M kernels.

        Args:
            W: (k, d) tensor of L2-normalised concept directions.
               Typically the L2-normalised rows of the trained W_c matrix.
        """
        W = _ensure_unit(W)
        for integrator in self._integrators:
            integrator.setup(W, S)
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, c: Tensor) -> CQOutput:
        """
        Compute the credal interval and uncertainty decomposition.

        Args:
            c: (B, k) or (k,) concept activation vector.
               Values should be in [0, 1].

        Returns:
            CQOutput with all interval and decomposition quantities.
        """
        if not self._ready:
            raise RuntimeError("Call setup(W) before forward().")

        squeeze = c.dim() == 1
        if squeeze:
            c = c.unsqueeze(0)   # (1, k)
        B = c.shape[0]

        # Collect (μ_m, σ_m) for each kernel
        mus    = []
        sigmas = []
        for integrator in self._integrators:
            mu_m, sigma2_m = integrator(c)              # (B,), (B,)
            mus.append(mu_m)
            sigmas.append(sigma2_m.clamp(min=0.0).sqrt())

        mu_stack    = torch.stack(mus,    dim=0)        # (M, B)
        sigma_stack = torch.stack(sigmas, dim=0)        # (M, B)

        # Credal interval
        lower = (mu_stack - self.gamma * sigma_stack).min(dim=0).values  # (B,)
        upper = (mu_stack + self.gamma * sigma_stack).max(dim=0).values  # (B,)

        # Uncertainty decomposition
        # σ_num:   mean within-kernel standard deviation (node geometry only)
        # σ_prior: std of posterior means across kernels (kernel ambiguity)
        sigma_num   = sigma_stack.mean(dim=0)                             # (B,)
        sigma_prior = mu_stack.std(dim=0, unbiased=False)                 # (B,)
        # Note: std(dim=0) with M=1 returns 0, which is correct.

        if squeeze:
            lower       = lower.squeeze(0)
            upper       = upper.squeeze(0)
            sigma_num   = sigma_num.squeeze(0)
            sigma_prior = sigma_prior.squeeze(0)
            # mu_stack / sigma_stack are (M, 1) → squeeze to (M,)
            mu_per_k    = mu_stack.squeeze(-1)      # (M,)
            sig_per_k   = sigma_stack.squeeze(-1)   # (M,)
        else:
            mu_per_k    = mu_stack.transpose(0, 1)   # (B, M)
            sig_per_k   = sigma_stack.transpose(0, 1) # (B, M)

        return CQOutput(
            c_lower=lower,
            c_upper=upper,
            sigma_num=sigma_num,
            sigma_prior=sigma_prior,
            mu_per_kernel=mu_per_k,
            sigma_per_kernel=sig_per_k,
        )

    # ── Convenience ───────────────────────────────────────────────────────
    def interval_width(self, c: Tensor) -> Tensor:
        """Return C⁺ − C⁻ for each image in batch."""
        out = self.forward(c)
        return out.c_upper - out.c_lower

    def cbm_prediction(self, c: Tensor) -> Tensor:
        """
        Return the CBM point prediction (midpoint approximation).
        For the exact CBM output, use the label-weight dot product directly.
        Here we return the mean of the BQ means as an approximation.
        """
        out = self.forward(c)
        return (out.c_lower + out.c_upper) / 2.0

    def __repr__(self) -> str:
        kernel_strs = ", ".join(str(k) for k in self.kernels)
        return (f"CredalBQ(M={self.M}, γ={self.gamma}, "
                f"kernels=[{kernel_strs}])")
