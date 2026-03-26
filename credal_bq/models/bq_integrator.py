"""
models/bq_integrator.py
------------------------
BQIntegrator: Bayesian Quadrature over a discrete uniform measure
on concept direction space S^{d-1}.

Given:
    W  ∈ R^{k×d}   L2-normalised concept directions (quadrature nodes)
    c  ∈ R^{k}     concept activations (function values, per image)
    κ               a kernel on S^{d-1}

Under the uniform prior measure P̃ = (1/k) Σ_j δ(w − w_j):

    z_i  = ∫ κ(w_i, w) dP̃(w) = (1/k) [G 1]_i
    K_PP = ∫∫ κ(w, w') dP̃² = (1/k²) 1ᵀ G 1

BQ posterior:
    μ_BQ  = zᵀ G⁻¹ c          (posterior mean)
    σ²_BQ = K_PP − zᵀ G⁻¹ z  (posterior variance = WCE²)

The Gram matrix G and its Cholesky factor are computed once at
setup and cached; per-image inference is O(k²).

BQIntegratorWithNoise (Option C) additionally accepts per-node
aleatoric noise estimates and returns an extended decomposition:
    σ²_total = σ²_epistemic + σ²_numerical
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from ..kernels.base import Kernel


# ─────────────────────────────────────────────────────────────────────────────
class BQIntegrator(nn.Module):
    """
    Single-kernel BQ over concept direction space.

    Args:
        kernel:   A Kernel instance (e.g. CosineRBF(0.7))
        jitter:   Small diagonal added to G for numerical stability.
    """

    def __init__(self, kernel: Kernel, jitter: float = 1e-6):
        super().__init__()
        self.kernel = kernel
        self.jitter = jitter

        # Cached after setup(); both None until setup() is called
        self._L: Optional[Tensor] = None   # Cholesky factor of G,  (k, k)
        self._G_full: Optional[Tensor] = None  # fallback for rank-deficient G
        self._use_cholesky: bool = True
        self._z: Optional[Tensor] = None   # kernel mean vector,     (k,)
        self._K_PP: Optional[Tensor] = None  # double integral,       scalar
        self._k: int = 0

    # ── Setup (called once per model, not per image) ──────────────────────
    def setup(self, W: Tensor, S: Tensor | None = None) -> None:
        """
        Pre-compute and cache Gram matrix, Cholesky factor, kernel mean,
        and double integral for the given concept directions W.

        Args:
            W: Tensor of shape (k, d), L2-normalised concept directions.
        """
        W = _ensure_unit(W)   # (k, d)
        k = W.shape[0]
        self._k = k

        # Gram matrix G ∈ R^{k×k}
        G = self.kernel.gram(W)                           # (k, k)
        G = G + self.jitter * torch.eye(k, device=G.device, dtype=G.dtype)

        # Cholesky: G = L Lᵀ
        # For rank-deficient kernels (e.g. ConstantKernel), Cholesky fails.
        # We detect this and fall back to the pseudo-inverse via lstsq.
        try:
            self._L = torch.linalg.cholesky(G)           # (k, k)
            self._use_cholesky = True
        except torch.linalg.LinAlgError:
            # Store the full G for lstsq-based solve
            self._G_full = G
            self._L = None
            self._use_cholesky = False

        # Kernel mean z and K_PP under the prior measure.
        # If a separate prior support S is provided, use Monte Carlo on S.
        # Otherwise, fall back to the empirical discrete prior over W.
        if S is not None:
            S = _ensure_unit(S)
            m = S.shape[0]
            # z_i = (1/m) sum_j k(w_i, s_j) = (1/m) [K_WS 1]_i
            K_WS = self.kernel.forward(W, S)  # (k, m)
            ones_m = torch.ones(m, 1, device=K_WS.device, dtype=K_WS.dtype)
            self._z = (K_WS @ ones_m).squeeze(-1) / m        # (k,)
            # K_PP = (1/m^2) 1^T K_SS 1
            K_SS = self.kernel.forward(S, S)                 # (m, m)
            self._K_PP = (ones_m.t() @ K_SS @ ones_m).squeeze() / (m * m)
        else:
            # Empirical prior over W (previous behaviour)
            ones = torch.ones(k, 1, device=G.device, dtype=G.dtype)
            self._z = (G @ ones).squeeze(-1) / k             # (k,)
            self._K_PP = (ones.t() @ G @ ones).squeeze() / (k * k)  # scalar

    def is_ready(self) -> bool:
        return self._z is not None

    def _solve(self, G_or_L: Tensor, rhs: Tensor, cholesky: bool) -> Tensor:
        """
        Solve G x = rhs.
        - If cholesky=True: rhs is (B, k, 1) or (k, 1), uses cholesky_solve.
        - If cholesky=False: uses lstsq (pseudo-inverse path for rank-deficient G).
        """
        if cholesky:
            return torch.cholesky_solve(rhs, G_or_L)
        else:
            # lstsq: solve G x = rhs for potentially rank-deficient G
            # rhs can be (k, 1) or (B, k, 1)
            if rhs.dim() == 2:
                sol = torch.linalg.lstsq(G_or_L, rhs).solution
            else:
                B = rhs.shape[0]
                G_exp = G_or_L.unsqueeze(0).expand(B, -1, -1)
                sol = torch.linalg.lstsq(G_exp, rhs).solution
            return sol

    # ── Per-image inference ────────────────────────────────────────────────
    def forward(self, c: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute BQ posterior mean and variance for a batch of activation
        vectors.

        Args:
            c: Tensor of shape (B, k) or (k,) — concept activations.

        Returns:
            mu:     Tensor of shape (B,) or scalar — BQ posterior mean.
            sigma2: Tensor of shape (B,) or scalar — BQ posterior variance.
        """
        if not self.is_ready():
            raise RuntimeError("Call setup(W) before forward().")

        squeeze = c.dim() == 1
        if squeeze:
            c = c.unsqueeze(0)          # (1, k)

        B, k = c.shape
        use_chol = self._use_cholesky
        solver   = self._L if use_chol else self._G_full
        z        = self._z            # (k,)
        K_PP     = self._K_PP         # scalar

        # Solve G α = c  →  α = G⁻¹ c
        c_col = c.unsqueeze(-1)                                    # (B, k, 1)
        if use_chol:
            L_exp = solver.unsqueeze(0).expand(B, -1, -1)
            alpha = torch.cholesky_solve(c_col, L_exp)             # (B, k, 1)
        else:
            G_exp = solver.unsqueeze(0).expand(B, -1, -1)
            alpha = torch.linalg.lstsq(G_exp, c_col).solution      # (B, k, 1)

        # μ = zᵀ α
        z_row = z.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)     # (B, k, 1)
        mu = torch.bmm(z_row.transpose(1, 2), alpha).squeeze(-1).squeeze(-1)  # (B,)

        # σ² = K_PP − zᵀ G⁻¹ z  (same for every image in batch)
        z_col = z.unsqueeze(-1)                                     # (k, 1)
        if use_chol:
            beta = torch.cholesky_solve(z_col, solver)              # (k, 1)
        else:
            beta = torch.linalg.lstsq(solver, z_col).solution       # (k, 1)

        sigma2_val = (K_PP - (z @ beta.squeeze(-1))).clamp(min=0.0)
        sigma2 = sigma2_val.expand(B)                               # (B,)

        if squeeze:
            return mu.squeeze(0), sigma2.squeeze(0)
        return mu, sigma2


# ─────────────────────────────────────────────────────────────────────────────
class BQIntegratorWithNoise(BQIntegrator):
    """
    Option C: BQ integrator that additionally accepts per-node aleatoric
    noise estimates and returns an additive uncertainty decomposition.

    The noise model assumes the concept activation at node j has
    additional independent Gaussian noise with variance η_j:

        f_j_observed = f_j_true + ε_j,   ε_j ~ N(0, η_j)

    The noisy Gram matrix is:
        G_noisy = G + diag(η)

    This separates:
        σ²_numerical  = BQ posterior variance under G_noisy
                       (integration error from node geometry + noise)
        σ²_epistemic  = (noise propagation into the integral)
                       = zᵀ G_noisy⁻¹ diag(η) G_noisy⁻¹ z

    Additive decomposition (proved in paper appendix):
        σ²_total = σ²_numerical + σ²_epistemic

    Args:
        kernel:      A Kernel instance.
        jitter:      Diagonal stabilisation jitter.
    """

    def setup_with_noise(self, W: Tensor, eta: Tensor) -> None:
        """
        Pre-compute cached quantities for the noisy model.

        Args:
            W:   (k, d) L2-normalised concept directions.
            eta: (k,)   per-node aleatoric noise variances  ≥ 0.
        """
        W = _ensure_unit(W)
        k = W.shape[0]
        self._k = k

        G = self.kernel.gram(W)
        G_noisy = (G
                   + torch.diag(eta)
                   + self.jitter * torch.eye(k, device=G.device, dtype=G.dtype))

        self._L = torch.linalg.cholesky(G_noisy)

        ones = torch.ones(k, 1, device=G.device, dtype=G.dtype)
        # kernel mean uses the *clean* G (prior measure still uniform)
        self._z = (G @ ones).squeeze(-1) / k
        self._K_PP = (ones.t() @ G @ ones).squeeze() / (k * k)

        # store eta for epistemic variance computation
        self._eta = eta                                   # (k,)
        self._G_noisy = G_noisy                          # (k, k)

    def forward_with_decomposition(
        self, c: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            mu:           (B,) BQ posterior mean
            sigma2_num:   (B,) numerical uncertainty
            sigma2_epi:   (B,) epistemic (noise) uncertainty
        """
        if not self.is_ready():
            raise RuntimeError("Call setup_with_noise(W, eta) before forward.")

        squeeze = c.dim() == 1
        if squeeze:
            c = c.unsqueeze(0)

        B, k = c.shape
        L    = self._L
        z    = self._z
        K_PP = self._K_PP
        eta  = self._eta    # (k,)

        # α = G_noisy⁻¹ c
        c_col  = c.unsqueeze(-1)
        L_exp  = L.unsqueeze(0).expand(B, -1, -1)
        alpha  = torch.cholesky_solve(c_col, L_exp)          # (B, k, 1)

        # μ = zᵀ α
        z_row  = z.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)
        mu     = torch.bmm(z_row.transpose(1, 2), alpha).squeeze(-1).squeeze(-1)

        # σ²_numerical = K_PP − zᵀ G_noisy⁻¹ z
        z_col  = z.unsqueeze(-1)
        beta   = torch.cholesky_solve(z_col, L)               # (k, 1)
        sigma2_num = (K_PP - (z @ beta.squeeze())).clamp(min=0.0).expand(B)

        # σ²_epistemic = zᵀ G_noisy⁻¹ diag(η) G_noisy⁻¹ z
        # = βᵀ diag(η) β     where β = G_noisy⁻¹ z
        sigma2_epi = (beta.squeeze() * eta * beta.squeeze()).sum().clamp(min=0.0)
        sigma2_epi = sigma2_epi.expand(B)

        if squeeze:
            return mu.squeeze(0), sigma2_num.squeeze(0), sigma2_epi.squeeze(0)
        return mu, sigma2_num, sigma2_epi


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_unit(W: Tensor) -> Tensor:
    """L2-normalise along the last dimension; warn if already normalised."""
    norms = W.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return W / norms
