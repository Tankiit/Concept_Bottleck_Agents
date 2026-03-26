"""
train_sequential.py
====================
Three-phase sequential Credal CBM training on CUB-200-2011.

Rationale over end-to-end:
  Phase 1 gives concept directions W_c their meaning by supervising x→c
  independently.  Only once W_c is stable do we commit to it as the set of
  quadrature nodes.  Phase 2 trains W_labels from those fixed concept scores.
  Phase 3 then fine-tunes W_labels against a credal loss — teaching the
  classifier to produce predictions that are robust to the BQ interval, not
  just to the CBM point estimate.  The backbone and concept encoder stay frozen
  throughout Phases 2–3, so W_c never drifts away from its Phase-1 meaning.

Three phases:
  Phase 1 — x → c
    Standard Koh et al. independent training.
    ResNet backbone + linear concept head, BCE loss with class weights.
    Two-stage within Phase 1: warmup with frozen backbone, then unfrozen.
    Saves: checkpoints/encoder_best.pth
           artifacts/W_concepts.npy

  Phase 2 — c → y (linear predictor, standard CE)
    Frozen backbone + concept encoder.
    Trains W_labels from extracted concept activations.
    Establishes a strong W_labels baseline and the concept activation arrays
    c_train / c_val / c_test that feed Phase 3 and the ablation.
    Saves: checkpoints/predictor_best.pth
           artifacts/W_labels.npy
           artifacts/c_{train,val,test}.npy
           artifacts/y_{train,val,test}.npy

  Phase 3 — W_labels fine-tuning with credal loss
    Frozen backbone + concept encoder (W_c does NOT change).
    Only W_labels is updated.
    Loss:  L = CE(midpoint_logit, y)  +  λ · width_penalty
    width_penalty = mean over classes of max(0, logit_width - target_width)
    — penalises intervals that are wider than a target, encouraging
      W_labels to concentrate weight on concepts with low σ_num.
    Saves: checkpoints/predictor_credal_best.pth
           artifacts/W_labels_credal.npy

Ablation:
  After all three phases, computes and prints a comparison table:
    Phase 1 only   — concept accuracy, no task predictor
    Phase 1+2      — standard CBM (W_labels, CE loss)
    Phase 1+2+3    — credal fine-tuned W_labels
  Metrics: task accuracy, σ_num, σ_prior, interval width, quadrant fractions

Usage:
  python train_sequential.py                          # all three phases
  python train_sequential.py --no-credal              # Phases 1+2 only (standard CBM baseline)
  python train_sequential.py --phase 1                # Phase 1 only
  python train_sequential.py --phase 2                # skip Phase 1 (needs encoder)
  python train_sequential.py --phase 3                # skip Phases 1–2 (needs both)
  python train_sequential.py --ablation-only          # load existing artifacts → table
  python train_sequential.py --backbone resnet50
  python train_sequential.py --lambda-width 0.5       # credal loss weight (default 0.1)
  python train_sequential.py --target-width 0.3       # max allowed interval width

Recommended run order:
  1. python train_sequential.py --no-credal           # train CBM, save W_labels
  2. python train_sequential.py --phase 3             # credal fine-tune W_labels only
  3. python train_sequential.py --ablation-only       # compare all three phases
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

from cub_loader import (
    CUBDataset,
    CONCEPT_NAMES,
    CONCEPT_GROUP_MAP,
    N_CLASSES,
    N_CONCEPTS,
)

from credal_bq.kernels.cosine_kernels import (
    CosineRBF, CosineMatern52, default_imprecise_prior
)
from credal_bq.models.credal_bq import CredalBQ
from credal_bq.models.bq_integrator import _ensure_unit


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CFG = dict(
    cub_root=os.environ.get(
        "CUB_DIR", "/Users/tanmoy/research/data/CUB_200_2011/CUB_200_2011"
    ),
    backbone=os.environ.get("BACKBONE", "resnet18"),
    checkpoint_dir="./checkpoints",
    artifact_dir="./artifacts",
    # Phase 1 — concept encoder
    phase1_warmup=10,
    phase1_epochs=100,
    phase1_lr={"resnet18": 0.01, "resnet50": 0.005},
    phase1_wd=4e-5,
    phase1_momentum=0.9,
    phase1_patience=5,
    phase1_min_lr=1e-5,
    phase1_early_stop=10,
    phase1_save_freq=10,
    # Phase 2 — standard linear predictor
    phase2_epochs=500,          # needs more epochs for 200-class linear problem
    phase2_lr=0.01,             # kept for reference / SGD fallback
    phase2_lr_adam=1e-2,        # Adam LR — converges without scheduler tuning
    phase2_wd=4e-5,
    phase2_momentum=0.9,        # unused with Adam, kept for clarity
    # Phase 3 — credal fine-tuning of W_labels only
    phase3_epochs=100,
    phase3_lr=5e-3,
    phase3_wd=1e-4,
    phase3_momentum=0.9,
    phase3_lambda_width=0.1,    # weight on the width penalty
    phase3_target_width=0.3,    # target interval width (logit scale)
    phase3_patience=10,
    phase3_min_lr=1e-5,
    # Credal Quadrature
    credal_gamma=2.0,
    credal_m=5,
    credal_jitter=1e-6,
    credal_threshold=0.3,           # fixed fallback; overridden by adaptive τ
    credal_target_active_frac=0.20, # adaptive τ target: 20% of concepts active
    batch_size=64,
    n_concepts=N_CONCEPTS,
    n_classes=N_CLASSES,
    num_workers=4,
    seed=42,
    device=(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

class ConceptEncoder(nn.Module):
    """ResNet backbone + linear concept head. Identical to train.py."""

    def __init__(self, backbone: str, n_concepts: int):
        super().__init__()
        self._bname = backbone
        if backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif backbone == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(backbone)
        d = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.head = nn.Linear(d, n_concepts)
        self._d = d

    def forward(self, x):
        return self.head(self.backbone(x)), None

    def freeze_backbone(self, freeze=True):
        for p in self.backbone.parameters():
            p.requires_grad_(not freeze)

    def get_W_concepts(self):
        return self.head.weight.detach().cpu().numpy()

    def concept_directions(self):
        """L2-normalised concept directions. Shape: (k, d)."""
        return _ensure_unit(self.head.weight.detach())


class LabelPredictor(nn.Module):
    """Linear c → y."""

    def __init__(self, n_concepts, n_classes):
        super().__init__()
        self.fc = nn.Linear(n_concepts, n_classes)

    def forward(self, c):
        return self.fc(c)

    def get_W_labels(self):
        return self.fc.weight.detach().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_ckpt(model, opt, sched, epoch, tr, vl, path):
    key = "encoder_state_dict" if isinstance(model, ConceptEncoder) else "predictor_state_dict"
    torch.save({
        "epoch": epoch + 1,
        key: model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sched.state_dict() if sched else None,
        "train_loss": tr, "val_loss": vl,
    }, path)


def _load_ckpt(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    key = "encoder_state_dict" if isinstance(model, ConceptEncoder) else "predictor_state_dict"
    model.load_state_dict(ckpt[key] if key in ckpt else ckpt)


def _make_sgd(model, lr, wd, momentum):
    return torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, momentum=momentum, weight_decay=wd,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: concept encoder (unchanged from Koh et al.)
# ─────────────────────────────────────────────────────────────────────────────

def run_phase1(encoder, train_loader, val_loader, cfg):
    """Train x→c. Identical to the original train.py Stage 1."""
    device  = cfg["device"]
    encoder = encoder.to(device)
    warmup  = cfg["phase1_warmup"]
    base_lr = cfg["phase1_lr"].get(encoder._bname, 0.01)

    pos_w = torch.tensor(
        train_loader.dataset.concept_weights(), dtype=torch.float32
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    encoder.freeze_backbone(True)
    opt   = _make_sgd(encoder, base_lr, cfg["phase1_wd"], cfg["phase1_momentum"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.1,
        patience=cfg["phase1_patience"], min_lr=cfg["phase1_min_lr"],
    )

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    best_path = os.path.join(cfg["checkpoint_dir"], "encoder_best.pth")
    best_loss, no_improve, phase_b = float("inf"), 0, False

    print(f"  Backbone: {encoder._bname}  LR: {base_lr}  Device: {device}")
    print(f"  Warmup: {warmup} epochs frozen backbone")

    for epoch in range(cfg["phase1_epochs"]):
        if epoch == warmup and not phase_b:
            print("  Unfreezing backbone (LR×0.1)")
            encoder.freeze_backbone(False)
            opt   = _make_sgd(encoder, base_lr * 0.1, cfg["phase1_wd"], cfg["phase1_momentum"])
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.1,
                patience=cfg["phase1_patience"], min_lr=cfg["phase1_min_lr"],
            )
            phase_b = True

        encoder.train()
        tr = 0.0
        for imgs, cpts, _ in tqdm(train_loader, desc=f"P1 E{epoch+1}", leave=False):
            imgs, cpts = imgs.to(device), cpts.to(device)
            opt.zero_grad(set_to_none=True)
            logits, _ = encoder(imgs)
            loss = criterion(logits, cpts)
            loss.backward(); opt.step()
            tr += loss.item()
        tr /= len(train_loader)

        encoder.eval()
        vl = 0.0
        with torch.no_grad():
            for imgs, cpts, _ in val_loader:
                logits, _ = encoder(imgs.to(device), )
                vl += criterion(logits, cpts.to(device)).item()
        vl /= len(val_loader)
        sched.step(vl)

        ph = "A" if epoch < warmup else "B"
        lr = opt.param_groups[0]["lr"]
        print(f"  P1[{epoch+1:3d}]{ph}  tr={tr:.4f}  vl={vl:.4f}  lr={lr:.1e}")

        if vl < best_loss:
            best_loss, no_improve = vl, 0
            _save_ckpt(encoder, opt, sched, epoch, tr, vl, best_path)
        else:
            no_improve += 1
            if no_improve >= cfg["phase1_early_stop"] and phase_b:
                print(f"  Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % cfg["phase1_save_freq"] == 0:
            p = os.path.join(cfg["checkpoint_dir"], f"encoder_epoch_{epoch+1}.pth")
            _save_ckpt(encoder, opt, sched, epoch, tr, vl, p)

    _load_ckpt(encoder, best_path, device)
    print(f"Phase 1 done. Best val BCE: {best_loss:.4f}")
    return encoder


# ─────────────────────────────────────────────────────────────────────────────
# Concept score extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_scores(encoder, loader, device):
    encoder.eval()
    all_c, all_y = [], []
    with torch.no_grad():
        for imgs, _, labels in tqdm(loader, desc="Extract", leave=False):
            logits, _ = encoder(imgs.to(device))
            all_c.append(torch.sigmoid(logits).cpu())
            all_y.append(labels if isinstance(labels, torch.Tensor)
                         else torch.tensor(labels))
    return (
        torch.cat(all_c).numpy().astype(np.float32),
        torch.cat(all_y).numpy().astype(np.int64),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: standard linear predictor (W_labels baseline)
# ─────────────────────────────────────────────────────────────────────────────

def run_phase2(predictor, c_train, y_train, c_val, y_val, cfg):
    """
    Train c→y with standard cross-entropy.

    Uses Adam rather than SGD: for a full-batch linear problem over 200 classes,
    SGD with ReduceLROnPlateau kills the LR before the loss has moved meaningfully
    (val_acc stays at random-chance for the first ~20 epochs because a linear
    classifier needs many steps to separate 200 classes).  Adam is invariant to
    gradient scale and converges reliably on this problem without tuning.

    W_labels from this phase serves as both the baseline and the warm-start
    for Phase 3.
    """
    device = cfg["device"]
    predictor = predictor.to(device)

    xtr = torch.tensor(c_train).to(device)
    ytr = torch.tensor(y_train, dtype=torch.long).to(device)
    xvl = torch.tensor(c_val).to(device)
    yvl = torch.tensor(y_val, dtype=torch.long).to(device)

    opt = torch.optim.Adam(
        predictor.parameters(),
        lr=cfg["phase2_lr_adam"],
        weight_decay=cfg["phase2_wd"],
    )
    # Cosine annealing: slow warm decay that doesn't kill LR early
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["phase2_epochs"], eta_min=cfg["phase1_min_lr"]
    )
    crit      = nn.CrossEntropyLoss()
    best_path = os.path.join(cfg["checkpoint_dir"], "predictor_best.pth")
    best_acc  = 0.0

    for epoch in range(cfg["phase2_epochs"]):
        predictor.train()
        opt.zero_grad()
        loss = crit(predictor(xtr), ytr)
        loss.backward(); opt.step()
        sched.step()

        predictor.eval()
        with torch.no_grad():
            val_acc = (predictor(xvl).argmax(1) == yvl).float().mean().item()

        if (epoch + 1) % 50 == 0:
            lr = opt.param_groups[0]["lr"]
            print(f"  P2[{epoch+1:3d}]  loss={loss.item():.4f}  "
                  f"val_acc={val_acc:.4f}  best={best_acc:.4f}  lr={lr:.1e}")

        if val_acc > best_acc:
            best_acc = val_acc
            _save_ckpt(predictor, opt, sched, epoch, loss.item(), val_acc, best_path)

    _load_ckpt(predictor, best_path, device)
    print(f"Phase 2 done. Best val acc: {best_acc:.4f}")
    return predictor


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: credal fine-tuning of W_labels
# ─────────────────────────────────────────────────────────────────────────────

def _credal_loss(
    logits:  torch.Tensor,      # (B, C) — standard CBM logits W_lp · c
    y_true:  torch.Tensor,      # (B,) int
    sigma_total: torch.Tensor,  # (B,) per-image total CQ uncertainty
    W_lp:    torch.Tensor,      # (C, k) label predictor weights
    cfg:     dict,
) -> tuple[torch.Tensor, dict]:
    """
    Credal fine-tuning loss:

        L = CE(logits, y) + λ · width_penalty

    width_penalty = mean over batch of:
        mean over classes of max(0, interval_width_y - target_width)

    where interval_width_y = |W_y| · sigma_total  (interval arithmetic,
    same formula as in train_credal.py).

    This penalises W_labels rows that have large L1 norm in directions
    with high σ_total (uncertain concepts), pushing the predictor toward
    weight concentration on concepts that BQ considers well-covered.

    Returns scalar loss and a dict of component values for logging.
    """
    lam    = cfg["phase3_lambda_width"]
    target = cfg["phase3_target_width"]

    # Standard CE on the midpoint (= standard CBM logit, no change)
    loss_ce = F.cross_entropy(logits, y_true)

    # Per-class interval width = |W_y| · σ_total
    # sigma_total: (B,) → (B, 1)
    # W_lp.abs().sum(dim=1): (C,) — L1 norm of each class row
    W_l1    = W_lp.abs().sum(dim=1)          # (C,)
    # width[b, y] = W_l1[y] * sigma_total[b]
    width   = sigma_total.unsqueeze(1) * W_l1.unsqueeze(0)  # (B, C)

    # Penalty: hinge on width exceeding target
    penalty = F.relu(width - target).mean()

    loss = loss_ce + lam * penalty

    return loss, {
        "loss_ce":  loss_ce.item(),
        "penalty":  penalty.item(),
        "loss":     loss.item(),
        "width_mean": width.mean().item(),
    }


def _compute_sigma_batch(
    c_batch: torch.Tensor,    # (B, k)
    cq:      CredalBQ,
) -> torch.Tensor:
    """Return per-image sigma_total = sigma_num + sigma_prior for a batch."""
    out = cq(c_batch)
    return out.sigma_num + out.sigma_prior    # (B,)


def run_phase3(
    predictor: LabelPredictor,
    encoder:   ConceptEncoder,
    c_train:   np.ndarray,
    y_train:   np.ndarray,
    c_val:     np.ndarray,
    y_val:     np.ndarray,
    cq:        CredalBQ,
    cfg:       dict,
) -> LabelPredictor:
    """
    Fine-tune W_labels against the credal loss.

    ONLY W_labels is updated. The backbone, concept encoder, and quadrature
    kernels are all frozen.  The concept directions W_c do not change,
    so the BQ nodes remain stable throughout.

    The credal loss teaches W_labels to:
      (a) Maintain task accuracy (CE term)
      (b) Prefer weight on concepts where σ_num is low — i.e. concepts
          the BQ considers well-represented given the current activation
          pattern (width penalty term)
    """
    device    = cfg["device"]
    predictor = predictor.to(device)

    # Pre-compute sigma_total for the whole training set once per epoch
    # (sigma_total is a function of c only, not of W_labels, so we can
    # cache it rather than recomputing inside the loop)
    print("  Pre-computing CQ uncertainties for training set ...")
    c_tr_t  = torch.tensor(c_train)
    s_tr    = []
    with torch.no_grad():
        for i in range(0, len(c_train), 256):
            s_tr.append(
                _compute_sigma_batch(c_tr_t[i:i+256], cq)
            )
    sigma_train = torch.cat(s_tr)    # (N_train,)
    print(f"  σ_total train: mean={sigma_train.mean():.4f}  "
          f"std={sigma_train.std():.4f}  max={sigma_train.max():.4f}")

    xtr = torch.tensor(c_train).to(device)
    ytr = torch.tensor(y_train, dtype=torch.long).to(device)
    xvl = torch.tensor(c_val).to(device)
    yvl = torch.tensor(y_val, dtype=torch.long).to(device)

    opt   = torch.optim.SGD(
        predictor.parameters(),
        lr=cfg["phase3_lr"], momentum=cfg["phase3_momentum"],
        weight_decay=cfg["phase3_wd"],
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5,
        patience=cfg["phase3_patience"], min_lr=cfg["phase3_min_lr"],
    )

    best_path = os.path.join(cfg["checkpoint_dir"], "predictor_credal_best.pth")
    best_acc  = 0.0
    no_improve = 0
    W_lp_dev  = torch.tensor(predictor.get_W_labels(), device=device)

    for epoch in range(cfg["phase3_epochs"]):
        predictor.train()
        W_lp_dev = predictor.fc.weight   # live reference, updated by opt

        # Full-batch update (same as Phase 2 — dataset fits in GPU RAM)
        opt.zero_grad()
        logits = predictor(xtr)           # (N_train, C)
        sigma_dev = sigma_train.to(device)

        loss, components = _credal_loss(
            logits, ytr, sigma_dev, W_lp_dev, cfg
        )
        loss.backward()
        opt.step()

        # Validation accuracy (standard CBM logit, no CQ)
        predictor.eval()
        with torch.no_grad():
            val_acc = (predictor(xvl).argmax(1) == yvl).float().mean().item()
        sched.step(val_acc)

        if (epoch + 1) % 10 == 0:
            lr = opt.param_groups[0]["lr"]
            print(
                f"  P3[{epoch+1:3d}]  "
                f"CE={components['loss_ce']:.4f}  "
                f"pen={components['penalty']:.4f}  "
                f"w_mean={components['width_mean']:.4f}  "
                f"val_acc={val_acc:.4f}  best={best_acc:.4f}  lr={lr:.1e}"
            )

        if val_acc > best_acc:
            best_acc, no_improve = val_acc, 0
            _save_ckpt(predictor, opt, sched, epoch, loss.item(), val_acc, best_path)
        else:
            no_improve += 1
            if no_improve >= cfg["phase3_patience"] * 2:
                print(f"  Early stop at epoch {epoch+1}")
                break

    _load_ckpt(predictor, best_path, device)
    print(f"Phase 3 done. Best val acc: {best_acc:.4f}")
    return predictor


# ─────────────────────────────────────────────────────────────────────────────
# CQ evaluation pass
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_with_cq(
    c_test:    np.ndarray,
    y_test:    np.ndarray,
    predictor: LabelPredictor,
    cq:        CredalBQ,
    cfg:       dict,
    label:     str = "",
) -> dict:
    """
    Run CQ on test set with the given predictor's W_labels.
    Returns accuracy, sigma stats, quadrant fractions.
    """
    device = cfg["device"]
    C      = cfg["n_classes"]
    gamma  = cfg["credal_gamma"]

    c_t  = torch.tensor(c_test)
    W_lp = torch.tensor(predictor.get_W_labels(), dtype=torch.float32)
    y_t  = torch.tensor(y_test, dtype=torch.long)

    all_preds, all_snum, all_sprior = [], [], []

    batch = 256
    for i in range(0, len(c_test), batch):
        c_b = c_t[i:i+batch]
        out = cq(c_b)

        # Standard CBM logits for prediction
        logits_b = c_b @ W_lp.T
        all_preds.append(logits_b.argmax(dim=-1))
        all_snum.append(out.sigma_num)
        all_sprior.append(out.sigma_prior)

    preds  = torch.cat(all_preds).numpy()
    snum   = torch.cat(all_snum).numpy().astype(np.float32)
    sprior = torch.cat(all_sprior).numpy().astype(np.float32)

    acc = (preds == y_test).mean()

    # Quadrant routing (median thresholds)
    sn_thr = float(np.median(snum))
    sp_thr = float(np.median(sprior))
    trust   = float(((snum <= sn_thr) & (sprior <= sp_thr)).mean())
    collect = float(((snum >  sn_thr) & (sprior <= sp_thr)).mean())
    review  = float(((snum <= sn_thr) & (sprior >  sp_thr)).mean())
    abstain = float(((snum >  sn_thr) & (sprior >  sp_thr)).mean())

    result = dict(
        label=label,
        accuracy=float(acc),
        sigma_num_mean=float(snum.mean()),
        sigma_prior_mean=float(sprior.mean()),
        sigma_num_p75=float(np.percentile(snum, 75)),
        sigma_prior_p75=float(np.percentile(sprior, 75)),
        total_width_mean=float((snum + sprior).mean()),
        trust=trust, collect=collect, review=review, abstain=abstain,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Ablation table
# ─────────────────────────────────────────────────────────────────────────────

def print_ablation_table(rows: list[dict]) -> None:
    """Print a formatted comparison table for the three phases."""
    cols = [
        ("Phase",          "label",             12),
        ("Acc",            "accuracy",           6),
        ("σ_num",          "sigma_num_mean",      7),
        ("σ_prior",        "sigma_prior_mean",    8),
        ("Width",          "total_width_mean",    6),
        ("TRUST%",         "trust",               7),
        ("COLLECT%",       "collect",             9),
        ("REVIEW%",        "review",              8),
        ("ABSTAIN%",       "abstain",             8),
    ]

    # Header
    header = "  ".join(f"{h:{w}}" for h, _, w in cols)
    sep    = "  ".join("-" * w for _, _, w in cols)
    print("\n" + "=" * len(header))
    print("  ABLATION: Sequential Phase Comparison")
    print("=" * len(header))
    print(header)
    print(sep)

    for row in rows:
        cells = []
        for _, key, w in cols:
            val = row.get(key, "")
            if isinstance(val, float):
                if key in ("trust", "collect", "review", "abstain"):
                    cells.append(f"{val*100:>{w}.1f}")
                elif key == "accuracy":
                    cells.append(f"{val:>{w}.4f}")
                else:
                    cells.append(f"{val:>{w}.5f}")
            else:
                cells.append(f"{str(val):<{w}}")
        print("  ".join(cells))

    print("=" * len(header))
    print()
    print("  Interpretation:")
    print("  Phase 1 only   — no task predictor; accuracy meaningless")
    print("  Phase 1+2      — standard CBM (CE loss)")
    print("  Phase 1+2+3    — credal fine-tuned (CE + width penalty)")
    print("  σ_num ↑ = more sparse concept activations (sparsity signal)")
    print("  σ_prior ↑ = more kernel ambiguity (similarity structure signal)")
    print("  Width  = σ_num + σ_prior (total CQ uncertainty)")
    print("  TRUST  = low σ_num & low σ_prior → automate")
    print("  COLLECT= high σ_num, low σ_prior  → add concept vocabulary")
    print("  REVIEW = low σ_num, high σ_prior  → human review")
    print("  ABSTAIN= high both               → escalate")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Artifact saving
# ─────────────────────────────────────────────────────────────────────────────

def save_artifacts(encoder, pred_standard, pred_credal,
                   c_test, y_test, c_train, y_train, c_val, y_val, cfg):
    d = cfg["artifact_dir"]
    os.makedirs(d, exist_ok=True)

    np.save(os.path.join(d, "W_concepts.npy"),     encoder.get_W_concepts())
    np.save(os.path.join(d, "c_train.npy"),        c_train)
    np.save(os.path.join(d, "y_train.npy"),        y_train)
    np.save(os.path.join(d, "c_val.npy"),          c_val)
    np.save(os.path.join(d, "y_val.npy"),          y_val)
    np.save(os.path.join(d, "c_test.npy"),         c_test)
    np.save(os.path.join(d, "y_test.npy"),         y_test)

    if pred_standard is not None:
        np.save(os.path.join(d, "W_labels.npy"),   pred_standard.get_W_labels())
    if pred_credal is not None:
        np.save(os.path.join(d, "W_labels_credal.npy"), pred_credal.get_W_labels())

    with open(os.path.join(d, "concept_names.json"), "w") as f:
        json.dump(CONCEPT_NAMES, f, indent=2)
    with open(os.path.join(d, "concept_group_map.json"), "w") as f:
        json.dump({k: list(v) for k, v in CONCEPT_GROUP_MAP.items()}, f, indent=2)

    print(f"\nArtifacts saved to {d}/")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sequential (Phase 1→2→3) Credal CBM on CUB-200-2011"
    )
    parser.add_argument("--phase",         type=int,   default=None,
                        choices=[1, 2, 3],
                        help="Start from this phase (needs earlier checkpoints)")
    parser.add_argument("--no-credal",     action="store_true",
                        help="Run Phases 1+2 only (standard CBM, no credal fine-tuning). "
                             "Equivalent to --phase 2 with Phase 3 disabled. "
                             "Useful as the baseline before running Phase 3 separately.")
    parser.add_argument("--ablation-only", action="store_true",
                        help="Skip training; load existing artifacts and print ablation table")
    parser.add_argument("--backbone",      default=None, choices=["resnet18", "resnet50"])
    parser.add_argument("--resume",        type=str,   help="Encoder checkpoint to resume from")
    parser.add_argument("--lambda-width",  type=float, default=None,
                        help="Phase 3 width penalty weight λ (default 0.1)")
    parser.add_argument("--target-width",  type=float, default=None,
                        help="Phase 3 target interval width (default 0.3)")
    parser.add_argument("--credal-gamma",  type=float, default=None)
    parser.add_argument("--credal-m",      type=int,   default=None)
    parser.add_argument("--tau-frac",      type=float, default=None,
                        help="Target fraction of concepts active per image "
                             "for adaptive τ (default 0.20 = 20%%). "
                             "Set to 0 to use the fixed --credal-threshold.")
    args = parser.parse_args()

    if args.backbone:        CFG["backbone"]                  = args.backbone
    if args.lambda_width:    CFG["phase3_lambda_width"]        = args.lambda_width
    if args.target_width:    CFG["phase3_target_width"]        = args.target_width
    if args.credal_gamma:    CFG["credal_gamma"]               = args.credal_gamma
    if args.credal_m:        CFG["credal_m"]                   = args.credal_m
    if args.tau_frac is not None:
        CFG["credal_target_active_frac"] = args.tau_frac

    # --no-credal caps the pipeline at Phase 2 regardless of --phase
    run_credal = not args.no_credal

    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    print(f"Device:   {CFG['device']}")
    print(f"Backbone: {CFG['backbone']}")
    print(f"Mode:     {'Phases 1+2 only (--no-credal)' if not run_credal else 'Phases 1+2+3 (credal fine-tuning)'}")
    if run_credal:
        print(f"CQ:       M={CFG['credal_m']}  γ={CFG['credal_gamma']}  τ={CFG['credal_threshold']}")
        print(f"Phase 3:  λ={CFG['phase3_lambda_width']}  target_width={CFG['phase3_target_width']}")

    # ── Datasets ──────────────────────────────────────────────────────────
    start_phase = args.phase or 1
    need_loaders = (not args.ablation_only) and start_phase <= 2

    if need_loaders:
        print("\nLoading datasets ...")
        pin = CFG["device"] == "cuda"
        nw  = CFG["num_workers"]
        kw  = dict(num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))

        train_ds = CUBDataset("train", CFG["cub_root"])
        val_ds   = CUBDataset("val",   CFG["cub_root"])
        test_ds  = CUBDataset("test",  CFG["cub_root"])

        train_loader = DataLoader(train_ds, CFG["batch_size"], shuffle=True,  **kw)
        val_loader   = DataLoader(val_ds,   CFG["batch_size"], shuffle=False, **kw)
        test_loader  = DataLoader(test_ds,  CFG["batch_size"], shuffle=False, **kw)

        imgs, cpts, labels = next(iter(train_loader))
        assert imgs.abs().mean() > 0.01, "Images look black — check CUB_DIR"
        print(f"Sanity OK: imgs={imgs.shape}  concepts={cpts.shape}  labels={labels.shape}")

    # ── Encoder ───────────────────────────────────────────────────────────
    encoder = ConceptEncoder(CFG["backbone"], CFG["n_concepts"])

    # ── Phase 1 ───────────────────────────────────────────────────────────
    if not args.ablation_only and start_phase <= 1:
        print("\n" + "="*50)
        print("PHASE 1 — concept encoder  (x → c)")
        print("="*50)
        if args.resume:
            print(f"Resuming from {args.resume}")
            _load_ckpt(encoder, args.resume, CFG["device"])
        encoder = run_phase1(encoder, train_loader, val_loader, CFG)
    else:
        enc_path = os.path.join(CFG["checkpoint_dir"], "encoder_best.pth")
        print(f"\nLoading encoder from {enc_path}")
        _load_ckpt(encoder, enc_path, CFG["device"])
    encoder = encoder.to(CFG["device"])

    # ── Extract concept scores ─────────────────────────────────────────────
    art = CFG["artifact_dir"]
    os.makedirs(art, exist_ok=True)

    if not args.ablation_only and start_phase <= 2:
        print("\nExtracting concept scores ...")
        c_train, y_train = extract_scores(encoder, train_loader, CFG["device"])
        c_val,   y_val   = extract_scores(encoder, val_loader,   CFG["device"])
        c_test,  y_test  = extract_scores(encoder, test_loader,  CFG["device"])
        print(f"  train={c_train.shape}  val={c_val.shape}  test={c_test.shape}")

        concept_acc = (
            (c_val > 0.5).astype(float)
            == np.stack([it["concepts"] for it in val_ds.items])
        ).mean()
        print(f"  Val concept accuracy: {concept_acc:.4f}")

        # Save for reuse
        for name, arr in [("c_train", c_train), ("y_train", y_train),
                          ("c_val",   c_val),   ("y_val",   y_val),
                          ("c_test",  c_test),  ("y_test",  y_test)]:
            np.save(os.path.join(art, f"{name}.npy"), arr)
    else:
        print("\nLoading concept scores from artifacts ...")
        c_train = np.load(os.path.join(art, "c_train.npy"))
        y_train = np.load(os.path.join(art, "y_train.npy"))
        c_val   = np.load(os.path.join(art, "c_val.npy"))
        y_val   = np.load(os.path.join(art, "y_val.npy"))
        c_test  = np.load(os.path.join(art, "c_test.npy"))
        y_test  = np.load(os.path.join(art, "y_test.npy"))
        print(f"  train={c_train.shape}  val={c_val.shape}  test={c_test.shape}")

    # ── Set up CredalBQ (uses W_c from Phase 1, never changes again) ──────
    print("\nSetting up Credal Quadrature ...")
    kernels = default_imprecise_prior()[:CFG["credal_m"]]

    # ── Adaptive threshold ─────────────────────────────────────────────────
    # Fixed τ=0.3 is calibrated for bimodal activations (Koh et al. ResNet50).
    # When encoder weights are small, activations cluster near 0.5 and nearly
    # all 112 concepts exceed τ=0.3, collapsing σ_num to zero (interpolation
    # regime: all k nodes active → BQ posterior variance = 0).
    #
    # Fix: compute τ adaptively so that a target fraction of concepts are
    # active per image on average.  We aim for 20% active (≈ 22/112), matching
    # the sparsity assumption in the paper.  τ is set to the 80th percentile
    # of the concept activation distribution over the training set.
    target_active_frac = CFG.get("credal_target_active_frac", 0.20)
    tau_pct = (1.0 - target_active_frac) * 100          # e.g. 80th percentile
    tau_adaptive = float(np.percentile(c_train, tau_pct))
    tau_adaptive = max(tau_adaptive, 0.05)               # floor: never below 0.05

    # Report the activation distribution so the user can verify
    frac_above_fixed  = float((c_train > CFG["credal_threshold"]).mean())
    frac_above_adapt  = float((c_train > tau_adaptive).mean())
    n_active_fixed    = float((c_train > CFG["credal_threshold"]).sum(axis=1).mean())
    n_active_adapt    = float((c_train > tau_adaptive).sum(axis=1).mean())
    print(f"  Activation distribution (train): mean={c_train.mean():.3f}  "
          f"std={c_train.std():.3f}  p50={np.median(c_train):.3f}")
    print(f"  Fixed τ={CFG['credal_threshold']:.2f}: "
          f"frac active={frac_above_fixed:.3f}  mean n_active={n_active_fixed:.1f}/{CFG['n_concepts']}")
    print(f"  Adaptive τ={tau_adaptive:.3f} (target {target_active_frac*100:.0f}% active): "
          f"frac active={frac_above_adapt:.3f}  mean n_active={n_active_adapt:.1f}/{CFG['n_concepts']}")

    # Use adaptive τ if fixed τ leaves too few or too many nodes active
    if frac_above_fixed > 0.70:
        print(f"  ⚠ Fixed τ leaves {frac_above_fixed*100:.0f}% of concepts active "
              f"(interpolation regime). Using adaptive τ={tau_adaptive:.3f}.")
        effective_tau = tau_adaptive
    elif frac_above_fixed < 0.05:
        print(f"  ⚠ Fixed τ leaves {frac_above_fixed*100:.0f}% of concepts active "
              f"(too sparse). Using adaptive τ={tau_adaptive:.3f}.")
        effective_tau = tau_adaptive
    else:
        print(f"  Fixed τ={CFG['credal_threshold']:.2f} is in valid range. Using it.")
        effective_tau = CFG["credal_threshold"]

    CFG["effective_tau"] = effective_tau   # store for reporting

    cq = CredalBQ(
        kernels=kernels,
        gamma=CFG["credal_gamma"],
        jitter=CFG["credal_jitter"],
        threshold=effective_tau,
    )
    W = encoder.concept_directions().cpu()
    cq.setup(W)
    print(f"  Nodes: {W.shape}  Kernels: {[str(k) for k in kernels]}"
          f"  Effective τ: {effective_tau:.3f}")

    # ── Phase 2 ───────────────────────────────────────────────────────────
    pred_standard = LabelPredictor(CFG["n_concepts"], CFG["n_classes"])

    if not args.ablation_only and start_phase <= 2:
        print("\n" + "="*50)
        print("PHASE 2 — standard linear predictor  (c → y)")
        print("="*50)
        pred_standard = run_phase2(pred_standard, c_train, y_train, c_val, y_val, CFG)
        np.save(os.path.join(art, "W_labels.npy"), pred_standard.get_W_labels())
    else:
        p2_path = os.path.join(CFG["checkpoint_dir"], "predictor_best.pth")
        print(f"\nLoading standard predictor from {p2_path}")
        _load_ckpt(pred_standard, p2_path, CFG["device"])

    # ── Phase 3 ───────────────────────────────────────────────────────────
    # Initialise Phase 3 predictor from Phase 2 weights (warm start)
    pred_credal = LabelPredictor(CFG["n_concepts"], CFG["n_classes"])
    pred_credal.load_state_dict(pred_standard.state_dict())   # warm start

    if run_credal and not args.ablation_only and start_phase <= 3:
        print("\n" + "="*50)
        print("PHASE 3 — credal fine-tuning  (W_labels only, credal loss)")
        print("="*50)
        print(f"  λ_width={CFG['phase3_lambda_width']}  "
              f"target_width={CFG['phase3_target_width']}")
        print("  (backbone + concept encoder FROZEN — W_c does not change)")

        pred_credal = run_phase3(
            pred_credal, encoder,
            c_train, y_train, c_val, y_val,
            cq, CFG,
        )
        np.save(os.path.join(art, "W_labels_credal.npy"), pred_credal.get_W_labels())
    elif run_credal:
        # ablation-only or start_phase > 3: try to load from checkpoint
        p3_path = os.path.join(CFG["checkpoint_dir"], "predictor_credal_best.pth")
        if os.path.exists(p3_path):
            print(f"\nLoading credal predictor from {p3_path}")
            _load_ckpt(pred_credal, p3_path, CFG["device"])
        else:
            print("\n(No Phase 3 checkpoint found — ablation will show Phase 1+2 only)")
            run_credal = False   # suppress Phase 3 row from ablation table
    else:
        print("\nPhase 3 skipped (--no-credal).  Standard CBM only.")

    # ── Ablation table ─────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("ABLATION EVALUATION")
    print("="*50)

    rows = []

    # Row 0: Phase 1 only (CQ uncertainty stats with random W_labels)
    out_p1 = evaluate_with_cq(c_test, y_test, pred_standard, cq, CFG,
                               label="P1 only (W_labels=uninit)")
    out_p1["accuracy"] = None   # meaningless without trained predictor
    rows.append(out_p1)

    # Row 1: Phase 1+2 (standard CBM)
    pred_standard.eval()
    with torch.no_grad():
        test_logits = pred_standard(torch.tensor(c_test).to(CFG["device"]))
        p2_acc = (test_logits.argmax(1).cpu().numpy() == y_test).mean()
    out_p2 = evaluate_with_cq(c_test, y_test, pred_standard, cq, CFG,
                               label="Phase 1+2  (standard CE)")
    out_p2["accuracy"] = float(p2_acc)
    rows.append(out_p2)

    # Row 2: Phase 1+2+3 (credal fine-tuned) — only if credal was run
    if run_credal:
        pred_credal.eval()
        with torch.no_grad():
            test_logits_c = pred_credal(torch.tensor(c_test).to(CFG["device"]))
            p3_acc = (test_logits_c.argmax(1).cpu().numpy() == y_test).mean()
        out_p3 = evaluate_with_cq(c_test, y_test, pred_credal, cq, CFG,
                                   label="Phase 1+2+3 (CE+width)")
        out_p3["accuracy"] = float(p3_acc)
        rows.append(out_p3)
    else:
        print("  (Phase 3 row omitted — run without --no-credal to add it)")

    print_ablation_table(rows)

    # Save JSON report
    report_path = os.path.join(art, "ablation_sequential.json")
    with open(report_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Ablation report saved to {report_path}")

    # ── Save all artifacts ─────────────────────────────────────────────────
    save_artifacts(encoder, pred_standard,
                   pred_credal if run_credal else None,
                   c_test, y_test, c_train, y_train, c_val, y_val, CFG)


if __name__ == "__main__":
    main()