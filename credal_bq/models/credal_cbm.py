"""
models/credal_cbm.py
---------------------
CredalCBM: full CBM pipeline with Credal Quadrature replacing the
standard linear aggregation step.

Pipeline:
    x  →  backbone f_θ  →  h ∈ R^d
       →  concept encoder g  →  ĉ ∈ [0,1]^k
       →  CredalBQ  →  [C⁻_y, C⁺_y] per class y
       →  prediction ŷ = argmax_y μ_y

The concept encoder g produces activations via:
    ĉ_j = σ(h · w_j)

where w_j are the concept direction vectors (rows of W_c).
These same normalised directions w̃_j = w_j / ‖w_j‖ serve as the
quadrature nodes for CredalBQ.

At training time, the CBM label-predictor weights W_Λ are still used
to compute the standard CBM loss.  At inference time, CredalBQ uses
the uniform prior measure (not W_Λ) to compute integration uncertainty.

Args:
    backbone:         nn.Module  x → h ∈ R^d  (frozen in standard use)
    concept_encoder:  nn.Module  h → ĉ ∈ [0,1]^k
    label_predictor:  nn.Linear  ĉ → ŷ ∈ R^C
    credal_bq:        CredalBQ   per-class CQ wrapper
    freeze_backbone:  If True, backbone gradients are disabled.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from typing import Optional

from .credal_bq import CredalBQ, CQOutput
from .bq_integrator import _ensure_unit


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CBMOutput:
    """
    All outputs from a single CredalCBM forward pass.

    Shapes: B = batch size, k = concepts, C = classes, M = kernels.
    """
    logits:          Tensor          # (B, C)  standard CBM logits
    concept_acts:    Tensor          # (B, k)  ĉ ∈ [0,1]
    # Per-class credal intervals
    c_lower:         Tensor          # (B, C)
    c_upper:         Tensor          # (B, C)
    # Aggregated uncertainty signals (mean over classes)
    sigma_num:       Tensor          # (B,)
    sigma_prior:     Tensor          # (B,)
    # Full CQ outputs per class, if needed
    cq_outputs:      Optional[list[CQOutput]] = None


# ─────────────────────────────────────────────────────────────────────────────
class ConceptEncoder(nn.Module):
    """
    Lightweight concept encoder: linear projection + sigmoid.

    Maps frozen backbone features h ∈ R^d to k concept activations.
    Weights W_c ∈ R^{k×d}; ĉ_j = σ(h · w_j).
    """

    def __init__(self, d: int, k: int):
        super().__init__()
        self.W_c = nn.Linear(d, k, bias=True)

    def forward(self, h: Tensor) -> Tensor:
        """h: (B, d) → ĉ: (B, k) ∈ [0, 1]"""
        return torch.sigmoid(self.W_c(h))

    @property
    def concept_directions(self) -> Tensor:
        """L2-normalised rows of W_c.  Shape: (k, d)."""
        return _ensure_unit(self.W_c.weight)   # W_c.weight is (k, d)


# ─────────────────────────────────────────────────────────────────────────────
class CredalCBM(nn.Module):
    """
    Full CBM pipeline with Credal Quadrature aggregation.

    Args:
        backbone:         Frozen (or trainable) feature extractor.
        concept_encoder:  ConceptEncoder instance.
        label_predictor:  nn.Linear(k, C).
        credal_bq:        CredalBQ instance (one per class is handled internally).
        freeze_backbone:  Whether to disable backbone gradients at init.
        num_classes:      C — number of output classes.
    """

    def __init__(
        self,
        backbone: nn.Module,
        concept_encoder: ConceptEncoder,
        label_predictor: nn.Linear,
        credal_bq: CredalBQ,
        freeze_backbone: bool = True,
        num_classes: int = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.concept_encoder = concept_encoder
        self.label_predictor = label_predictor
        self.credal_bq = credal_bq
        self.C = num_classes or label_predictor.out_features

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    # ── Setup  (call once after concept encoder is trained) ───────────────
    def setup_quadrature(self) -> None:
        """
        Pre-compute Gram matrices for all kernels in the imprecise prior,
        using the current concept directions from the trained W_c.
        Must be called before inference.
        """
        W = self.concept_encoder.concept_directions.detach()  # (k, d)
        self.credal_bq.setup(W)

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(
        self,
        x: Tensor,
        return_cq_outputs: bool = False,
    ) -> CBMOutput:
        """
        Args:
            x:                 (B, *) input images.
            return_cq_outputs: If True, attach per-class CQOutput list.

        Returns:
            CBMOutput containing logits, concept activations, and
            per-class credal intervals with uncertainty decomposition.
        """
        # ── Stage 1: Backbone ─────────────────────────────────────────────
        with torch.set_grad_enabled(self.backbone.training):
            h = self.backbone(x)    # (B, d)

        # ── Stage 2: Concept encoder ──────────────────────────────────────
        c_hat = self.concept_encoder(h)    # (B, k)

        # ── Stage 3a: Standard CBM logits (for classification + training) ─
        logits = self.label_predictor(c_hat)    # (B, C)

        # ── Stage 3b: Credal Quadrature per class ─────────────────────────
        if not self.credal_bq.is_ready():
            raise RuntimeError(
                "Call setup_quadrature() before inference to pre-compute "
                "Gram matrices."
            )

        # For each class y, the CBM prediction is:
        #   ŷ_y = Σ_j |W_Λ[y,j]| / ‖W_Λ[y,:]‖₁  ·  ĉ_j
        # CredalBQ instead uses the uniform measure over concept directions;
        # it returns a set-valued estimate [C⁻_y, C⁺_y] per class.
        #
        # We run CQ on the activation vector c_hat weighted by the label
        # predictor weights: effective function value for class y at node j
        # is W_Λ[y,j] * ĉ_j — but for the *integration uncertainty*, we use
        # the normalised W_Λ row as the function values passed to CQ.

        W_lp = self.label_predictor.weight   # (C, k)

        c_lower_list  = []
        c_upper_list  = []
        snum_list     = []
        sprior_list   = []
        cq_list       = []

        for y in range(self.C):
            # Effective function value: concept activations weighted by class y
            w_y    = W_lp[y].abs()                         # (k,)
            w_y    = w_y / w_y.sum().clamp(min=1e-8)       # normalise → Δ^{k-1}
            c_eff  = c_hat * w_y.unsqueeze(0)              # (B, k)

            cq_out = self.credal_bq(c_eff)                 # CQOutput

            c_lower_list.append(cq_out.c_lower)            # (B,)
            c_upper_list.append(cq_out.c_upper)            # (B,)
            snum_list.append(cq_out.sigma_num)
            sprior_list.append(cq_out.sigma_prior)
            if return_cq_outputs:
                cq_list.append(cq_out)

        c_lower  = torch.stack(c_lower_list,  dim=-1)      # (B, C)
        c_upper  = torch.stack(c_upper_list,  dim=-1)      # (B, C)
        sigma_num   = torch.stack(snum_list,   dim=-1).mean(dim=-1)  # (B,)
        sigma_prior = torch.stack(sprior_list, dim=-1).mean(dim=-1)  # (B,)

        return CBMOutput(
            logits=logits,
            concept_acts=c_hat,
            c_lower=c_lower,
            c_upper=c_upper,
            sigma_num=sigma_num,
            sigma_prior=sigma_prior,
            cq_outputs=cq_list if return_cq_outputs else None,
        )

    # ── Prediction helpers ────────────────────────────────────────────────
    def predict(self, x: Tensor) -> Tensor:
        """Return class predictions ŷ = argmax logits.  Shape: (B,)."""
        out = self.forward(x)
        return out.logits.argmax(dim=-1)

    def predict_with_uncertainty(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            ŷ:           (B,) class predictions
            sigma_num:   (B,) numerical uncertainty (averaged over classes)
            sigma_prior: (B,) prior uncertainty (averaged over classes)
        """
        out = self.forward(x)
        return out.logits.argmax(dim=-1), out.sigma_num, out.sigma_prior

    # ── Loss ─────────────────────────────────────────────────────────────
    @staticmethod
    def loss(
        output: CBMOutput,
        y_true: Tensor,
        c_true: Optional[Tensor] = None,
        lambda_concept: float = 1.0,
    ) -> Tensor:
        """
        Combined CBM loss:
            L = L_task + λ_c · L_concept

        Args:
            output:        CBMOutput from forward().
            y_true:        (B,) integer class labels.
            c_true:        (B, k) binary concept labels, or None.
            lambda_concept: Weight on concept supervision loss.

        Returns:
            Scalar loss tensor.
        """
        loss_task = F.cross_entropy(output.logits, y_true)

        if c_true is not None:
            # Binary cross-entropy on concept predictions
            loss_concept = F.binary_cross_entropy(
                output.concept_acts, c_true.float()
            )
            return loss_task + lambda_concept * loss_concept

        return loss_task


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_credal_cbm(
    backbone: nn.Module,
    d: int,
    k: int,
    C: int,
    freeze_backbone: bool = True,
    gamma: float = 2.0,
    jitter: float = 1e-6,
    kernels=None,
) -> CredalCBM:
    """
    Convenience factory for building a CredalCBM with default components.

    Args:
        backbone:        Feature extractor  (output dim d).
        d:               Feature dimension.
        k:               Number of concepts.
        C:               Number of classes.
        freeze_backbone: Whether to freeze backbone parameters.
        gamma:           Coverage factor for credal interval.
        jitter:          Gram matrix diagonal jitter.
        kernels:         List of Kernel instances, or None for Π_5 default.

    Returns:
        CredalCBM ready for training (setup_quadrature() must be called
        before inference once W_c is trained).
    """
    concept_encoder = ConceptEncoder(d=d, k=k)
    label_predictor = nn.Linear(k, C, bias=True)
    credal_bq = CredalBQ(kernels=kernels, gamma=gamma, jitter=jitter)

    return CredalCBM(
        backbone=backbone,
        concept_encoder=concept_encoder,
        label_predictor=label_predictor,
        credal_bq=credal_bq,
        freeze_backbone=freeze_backbone,
        num_classes=C,
    )
