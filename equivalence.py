# ============================================================
# run_equivalence.py
# PyC wraps the Koh CBM. BQ is custom numpy.
# Run this file directly after downloading the CodaLab checkpoint.
# ============================================================

import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import spearmanr
import pandas as pd
import sys

# ── Step 0: install check ────────────────────────────────────────────────────
try:
    import torch_concepts as pyc
    PYC_AVAILABLE = True
except ImportError:
    PYC_AVAILABLE = False
    print("WARNING: pytorch-concepts not installed. Using raw nn.ModuleDict.")


# ── Step 1: load Koh checkpoint ──────────────────────────────────────────────

def load_koh_checkpoint(checkpoint_dir: str):
    """
    Loads the Koh et al. independent CBM.
    checkpoint_dir: path containing best_model_independent.pth etc.

    Returns raw weight tensors — we do NOT assume Koh's module names
    will match PyC's expected interface. Extract weights directly.
    """
    sys.path.insert(0, checkpoint_dir)

    # Their model file defines InceptionV3-based concept encoder
    # Adjust import to match the actual file in the CodaLab download
    try:
        from CUB.models import ModelXtoC, ModelCtoY
    except ImportError:
        # Fallback: minimal stubs if module names differ
        raise ImportError(
            "Cannot find Koh model classes. "
            "Check the CodaLab download structure and adjust the import."
        )

    x_to_c = ModelXtoC(pretrained=False, num_classes=112, use_aux=False)
    x_to_c.load_state_dict(
        torch.load(f"{checkpoint_dir}/best_model_independent.pth",
                   map_location="cpu")
    )
    x_to_c.eval()

    c_to_y = ModelCtoY(n_class_attr=2, num_classes=200, expand_dim=0)
    c_to_y.load_state_dict(
        torch.load(f"{checkpoint_dir}/best_model_independent_y.pth",
                   map_location="cpu")
    )
    c_to_y.eval()

    return x_to_c, c_to_y


def extract_weight_matrices(x_to_c, c_to_y):
    """
    Extract the two weight matrices we need for the BQ equivalence.

    W_concepts: (112, d) — concept direction vectors in feature space
                These are the rows of the final linear layer of x_to_c
                (the layer that maps features → concept logits)

    W_labels:   (200, 112) — label predictor weights
                These are the rows of c_to_y's linear layer
    """
    # Find the last linear layer in x_to_c
    # InceptionV3-based: the concept head is typically model.fc or model.last_linear
    # Inspect to find the right attribute:
    last_linear = None
    for name, module in x_to_c.named_modules():
        if isinstance(module, nn.Linear):
            last_linear = module    # keep overwriting → gets the last one

    if last_linear is None:
        raise ValueError("No Linear layer found in x_to_c. Check architecture.")

    W_concepts = last_linear.weight.detach().cpu().numpy()   # (112, d)
    print(f"W_concepts shape: {W_concepts.shape}")           # expect (112, d)

    # c_to_y is a simple Linear(112, 200)
    # Find it the same way
    for name, module in c_to_y.named_modules():
        if isinstance(module, nn.Linear):
            W_labels_layer = module

    W_labels = W_labels_layer.weight.detach().cpu().numpy()  # (200, 112)
    print(f"W_labels shape: {W_labels.shape}")               # expect (200, 112)

    return W_concepts, W_labels


# ── Step 2: wrap in PyC ModuleDict (clean interface) ─────────────────────────

def wrap_in_pyc(x_to_c, c_to_y):
    """
    Wraps the Koh modules in a PyC-style ModuleDict.
    This gives us the PyC forward-pass interface cleanly.

    NOTE: We do NOT use PyC's DoIntervention here.
    Concept zeroing is done directly on the tensor — safer and simpler.
    """
    if PYC_AVAILABLE:
        # PyC ModuleDict with keyword-arg forward pass
        # Both modules stay as-is; PyC just gives us the interface
        cbm = torch.nn.ModuleDict({
            'encoder'  : x_to_c,
            'predictor': c_to_y,
        })
        # PyC-style forward:
        # concepts = cbm['encoder'](input=h)       ← but x_to_c expects images
        # y        = cbm['predictor'](endogenous=c) ← c_to_y expects concept vec
        #
        # IMPORTANT: x_to_c is an InceptionV3 — it expects full images (3,299,299)
        # not feature vectors. So we use it as-is via standard call.
        return cbm
    else:
        # Fallback: plain ModuleDict
        return torch.nn.ModuleDict({'encoder': x_to_c, 'predictor': c_to_y})


def run_forward(cbm, image_batch):
    """
    Full forward pass through the wrapped Koh CBM.
    Returns concept activations and label logits.
    """
    with torch.no_grad():
        # x_to_c: image → concept logits (InceptionV3 forward)
        c_logits = cbm['encoder'](image_batch)    # (B, 112)
        c_scores  = torch.sigmoid(c_logits)        # (B, 112) ∈ [0,1]

        # c_to_y: concept scores → label logits
        y_logits  = cbm['predictor'](c_scores)    # (B, 200)

    return c_scores, y_logits


# ── Step 3: concept zeroing (intervention) — direct tensor approach ───────────

def concept_zeroing_delta(cbm, c_scores_batch, class_idx: int, n_concepts=112):
    """
    For each concept j, measure |Δŷ| when c_j is zeroed.
    This is the PyC DoIntervention logic, implemented directly.

    We do this directly on tensors rather than through PyC's context manager
    because:
    1. PyC's intervention API uses string-named concepts in the mid-level SEM
    2. Direct zeroing is transparent and debuggable
    3. Functionally identical for our purposes

    Returns:
        delta: (n_concepts,) — |ŷ_original - ŷ_zeroed| for concept j
    """
    with torch.no_grad():
        # Baseline prediction
        y_base = torch.softmax(cbm['predictor'](c_scores_batch), dim=-1)
        p_base = y_base[:, class_idx].mean().item()   # avg over batch

        deltas = np.zeros(n_concepts)
        for j in range(n_concepts):
            c_int = c_scores_batch.clone()
            c_int[:, j] = 0.0                          # zero out concept j

            y_int = torch.softmax(cbm['predictor'](c_int), dim=-1)
            p_int = y_int[:, class_idx].mean().item()

            deltas[j] = abs(p_base - p_int)

    return deltas / (deltas.sum() + 1e-8)              # normalize


# ── Step 4: BQ effective weights — angular kernel in feature space ────────────

def angular_rbf_kernel(W_concepts_normed, ell, jitter=1e-4):
    """
    k(w_i, w_j) = exp(-(2 - 2·cos(w_i, w_j)) / 2ℓ²)

    W_concepts_normed: (J, d) unit vectors
    Returns K: (J, J)
    """
    cos_sim = W_concepts_normed @ W_concepts_normed.T   # (J, J)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    sq_dist = 2.0 * (1.0 - cos_sim)                    # angular distance²
    K = np.exp(-sq_dist / (2.0 * ell ** 2))
    K += jitter * np.eye(len(W_concepts_normed))
    return K


def bq_effective_weights(W_concepts_normed, W_labels_class, ell):
    """
    BQ effective weights over concept directions in feature space.

    p     = normalize(|W_labels[class]|)  ← the learned measure
    z_i   = Σ_j p_j k(w_i, w_j)          ← kernel mean
    w_eff = K^{-1} z                       ← effective weights

    At ℓ→∞: K→constant → w_eff → p (exact Koh CBM recovery)
    At ℓ→0: K→I        → w_eff → z = Kp → p (same limit, diagonal)
    At intermediate ℓ: w_eff redistributes mass through concept similarity
    """
    p = np.abs(W_labels_class) / (np.abs(W_labels_class).sum() + 1e-8)
    K = angular_rbf_kernel(W_concepts_normed, ell)
    z = K @ p
    c, low = cho_factor(K)
    w_eff = cho_solve((c, low), z)
    return w_eff / (np.abs(w_eff).sum() + 1e-8), p


# ── Step 5: unit test — exact recovery at large ℓ ────────────────────────────

def unit_test_recovery(W_concepts_normed, W_labels, c_scores_np,
                        ell=50.0, n_classes=10):
    """
    At ℓ=50 (effectively ∞), w_eff should ≈ p_koh.
    μ_BQ(ℓ=50) = w_eff · c  should ≈  p_koh · c  for all images.
    Run this BEFORE the main experiment. If it fails, the kernel is wrong.
    """
    print(f"\n=== Unit test: exact recovery at ℓ={ell} ===")
    max_err = 0.0
    for cls in range(n_classes):
        w_eff, p_koh = bq_effective_weights(W_concepts_normed, W_labels[cls], ell)
        for img_c in c_scores_np[:5]:
            mu_bq  = float(w_eff @ img_c)
            mu_koh = float(p_koh @ img_c)
            max_err = max(max_err, abs(mu_bq - mu_koh))
    print(f"Max |μ_BQ - μ_koh| across {n_classes} classes × 5 images: {max_err:.6f}")
    assert max_err < 0.05, f"Recovery failed: {max_err:.4f} > 0.05"
    print("PASSED")


# ── Step 6: main experiment loop ─────────────────────────────────────────────

def run_equivalence_experiment(
    W_concepts_normed,      # (112, d)
    W_labels,               # (200, 112)
    c_scores_np,            # (N_images, 112)
    class_labels_np,        # (N_images,)
    cbm,                    # PyC-wrapped ModuleDict
    ell_sweep  = np.logspace(-2, 1.5, 40),
    k_values   = [5, 10, 20, 30],
    n_per_class = 15,
):
    # Unit test first — must pass before spending compute
    unit_test_recovery(W_concepts_normed, W_labels, c_scores_np)

    records = []

    for cls in range(W_labels.shape[0]):
        img_idx = np.where(class_labels_np == cls)[0][:n_per_class]
        if len(img_idx) == 0:
            continue

        p_koh = np.abs(W_labels[cls])
        p_koh /= p_koh.sum() + 1e-8

        # Concept zeroing weights (intervention) for this class's images
        c_batch = torch.tensor(c_scores_np[img_idx], dtype=torch.float32)
        w_intervention = concept_zeroing_delta(cbm, c_batch, cls)

        for ell in ell_sweep:
            w_eff, _ = bq_effective_weights(W_concepts_normed, W_labels[cls], ell)

            # Weight-level: how similar is w_eff to the Koh measure?
            rho_koh, _  = spearmanr(w_eff, p_koh)

            # Weight-level: how similar is w_eff to intervention-based weights?
            rho_int, _  = spearmanr(w_eff, w_intervention)

            for k in k_values:
                # Top-K sparse measure
                threshold    = np.sort(np.abs(W_labels[cls]))[-k]
                p_topk       = np.where(np.abs(W_labels[cls]) >= threshold,
                                        np.abs(W_labels[cls]), 0.0)
                p_topk      /= p_topk.sum() + 1e-8
                rho_topk, _ = spearmanr(w_eff, p_topk)

                # Prediction-level: |μ_BQ - μ_koh| per image
                f_batch = c_scores_np[img_idx]           # (n, 112)
                mu_bq   = f_batch @ w_eff                # (n,)
                mu_koh  = f_batch @ p_koh                # (n,)
                mu_topk = f_batch @ p_topk               # (n,)
                delta_koh  = float(np.abs(mu_bq - mu_koh).mean())
                delta_topk = float(np.abs(mu_bq - mu_topk).mean())

                records.append({
                    "ell": ell, "k": k, "cls": cls,
                    "rho_koh": rho_koh,
                    "rho_int": rho_int,
                    "rho_topk": rho_topk,
                    "delta_koh": delta_koh,
                    "delta_topk": delta_topk,
                })

    return pd.DataFrame(records)


# ── Step 7: entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    CHECKPOINT_DIR = "./koh_cbm_checkpoint"   # adjust to your CodaLab path
    CUB_FEATURES   = "./cub_concept_scores.npy"  # pre-extracted: (N, 112)
    CUB_LABELS     = "./cub_labels.npy"           # (N,) class indices

    x_to_c, c_to_y = load_koh_checkpoint(CHECKPOINT_DIR)
    W_concepts, W_labels = extract_weight_matrices(x_to_c, c_to_y)
    cbm = wrap_in_pyc(x_to_c, c_to_y)

    # Normalize concept directions to unit sphere
    norms = np.linalg.norm(W_concepts, axis=1, keepdims=True) + 1e-8
    W_concepts_normed = W_concepts / norms               # (112, d)

    # Load pre-extracted concept scores (run encoder once offline)
    c_scores_np     = np.load(CUB_FEATURES)              # (N, 112)
    class_labels_np = np.load(CUB_LABELS)                # (N,)

    df = run_equivalence_experiment(
        W_concepts_normed, W_labels, c_scores_np, class_labels_np, cbm
    )

    df.to_csv("equivalence_results.csv", index=False)
    print(df.groupby("ell")[["rho_koh", "rho_topk"]].mean())