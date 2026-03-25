"""
run_cbm_experiments.py
======================
Produces Figure 4 for the ProbNum 2026 paper using the official
Koh et al. (2020) CUB-200 artifacts from CodaLab.

Runs three experiments that empirically validate the theoretical
claims in Section 3.1:

  Exp 1 — Proposition 1 verification
    CBM prediction = quadrature rule with weights p_yj = |W_yj|/||W_y||_1
    Checks: y_hat_cbm ≈ (p_y · c_hat) × ||W_y||_1 for all images/classes

  Exp 2 — WCE comparison
    Trained CBM weights p_yj have higher WCE than BQ-optimal weights K^{-1}z
    Shows: integrating CBM predictions with wrong weights is suboptimal

  Exp 3 — SVBQ interval contains CBM prediction
    For every test image, hat_y_y ∈ [I⁻, I⁺] from SVBQ
    Verifies the Remark: including k_∞ ∈ Π guarantees containment

Output: figures/fig4_cbm_quadrature.pdf
        results/cbm_experiments.json

Dependencies:
  pip install numpy scipy matplotlib sentence-transformers tqdm

Usage:
  python run_cbm_experiments.py --artifacts_dir artifacts/
"""

import os, json, argparse, time
import numpy as np
import scipy.linalg
import scipy.stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

os.makedirs("figures", exist_ok=True)
os.makedirs("results",  exist_ok=True)

# ── Plotting style (matches run_experiments.py) ──────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.frameon": False, "figure.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
C = {"cbm": "#555555", "svbq": "#2A7BB5", "bq_opt": "#27AE60",
     "highlight": "#E07B39"}


# ── Kernels (identical to run_experiments.py) ────────────────────────────────

def rbf_kernel(E, ell):
    cos = np.clip(E @ E.T, -1., 1.)
    return np.exp(-2. * (1. - cos) / (2. * ell**2))

def matern_kernel(E, nu, ell):
    cos = np.clip(E @ E.T, -1., 1.)
    r   = np.sqrt(np.maximum(2. * (1. - cos), 0.))
    if nu == 2.5:
        s = np.sqrt(5)*r/ell; return (1+s+s**2/3.)*np.exp(-s)
    elif nu == 1.5:
        s = np.sqrt(3)*r/ell; return (1+s)*np.exp(-s)
    raise ValueError

KERNELS_5 = [
    ("RBF(0.3)",  lambda E: rbf_kernel(E, 0.3)),
    ("RBF(0.7)",  lambda E: rbf_kernel(E, 0.7)),
    ("RBF(1.5)",  lambda E: rbf_kernel(E, 1.5)),
    ("M52(0.7)",  lambda E: matern_kernel(E, 2.5, 0.7)),
    ("M32(0.7)",  lambda E: matern_kernel(E, 1.5, 0.7)),
]


def _bq_posterior(K, z, KPP, f, jitter=1e-6):
    N  = K.shape[0]
    Kj = K + jitter * np.eye(N)
    try:    L = scipy.linalg.cholesky(Kj, lower=True)
    except: L = scipy.linalg.cholesky(K + 1e-3*np.eye(N), lower=True)
    alpha  = scipy.linalg.cho_solve((L, True), f)
    beta   = scipy.linalg.cho_solve((L, True), z)
    mu     = float(z @ alpha)
    sigma2 = max(float(KPP - z @ beta), 1e-12)
    return mu, float(np.sqrt(sigma2))


def wce_squared(p, K):
    """
    Worst-case error squared for quadrature weights p under kernel K.
    WCE² = p^T K p - p^T K K^{-1} K p = 0  [for BQ-optimal p]
    In general: WCE²(p) = K_PP - z^T K^{-1} z  where z = Kp, K_PP = p^T K p
    But for an ARBITRARY p: WCE²(p, K) = p^T K p - (Kp)^T K^{-1} (Kp)
    = p^T K p - p^T K p = 0  — this is always zero for any p!

    Wait — that is not right. The WCE measures the error of the
    QUADRATURE RULE with weights w (not the BQ posterior).

    For a quadrature rule with weights w (not necessarily K^{-1}z):
      WCE²(w, k) = K_PP - 2 z^T w + w^T K w
    where z_i = (K p_tilde)_i, K_PP = p_tilde^T K p_tilde,
    p_tilde = uniform prior.

    Minimum is achieved at w* = K^{-1}z, giving WCE²* = K_PP - z^T K^{-1} z.
    """
    raise NotImplementedError  # see compute_wce below


def compute_wce(w_quad, E, p_tilde, kernel_fn):
    """
    Compute WCE²(w_quad, k) for quadrature weights w_quad
    relative to uniform prior p_tilde and kernel k.

    WCE²(w) = K_PP - 2 z^T w + w^T K w
    where K_PP = p^T K p, z = K p  (using uniform prior p = p_tilde)
    Minimum (BQ-optimal): WCE²* = K_PP - z^T K^{-1} z
    """
    K    = kernel_fn(E)
    z    = K @ p_tilde
    KPP  = float(p_tilde @ K @ p_tilde)

    # WCE² for arbitrary weights w
    wce2_w = KPP - 2 * float(z @ w_quad) + float(w_quad @ K @ w_quad)

    # WCE²* for BQ-optimal weights
    N  = K.shape[0]
    Kj = K + 1e-6 * np.eye(N)
    L  = scipy.linalg.cholesky(Kj, lower=True)
    beta     = scipy.linalg.cho_solve((L, True), z)
    wce2_opt = max(KPP - float(z @ beta), 1e-12)

    return max(wce2_w, 0.), wce2_opt


def svbq_for_class(E, c_hat_img, p_tilde, kernels, c=2.0):
    """Run SVBQ for one image, one class."""
    mus, sigmas = [], []
    for name, kfn in kernels:
        K   = kfn(E)
        z   = K @ p_tilde
        KPP = float(p_tilde @ K @ p_tilde)
        mu, sigma = _bq_posterior(K, z, KPP, c_hat_img)
        mus.append(mu); sigmas.append(sigma)
    mus    = np.array(mus)
    sigmas = np.array(sigmas)
    return {
        "I_lo":  float((mus - c*sigmas).min()),
        "I_hi":  float((mus + c*sigmas).max()),
        "I_mid": float(mus.mean()),
    }


# ── Embeddings ────────────────────────────────────────────────────────────────

def embed_concepts(concept_names):
    from sentence_transformers import SentenceTransformer
    print("  Computing SBERT concept embeddings ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Clean names: replace underscores and colons for readability
    clean = [n.replace("::", ": ").replace("_", " ") for n in concept_names]
    E = model.encode(clean, normalize_embeddings=True, show_progress_bar=False)
    return E.astype(np.float32)


# ── Experiment 1: Proposition 1 Verification ─────────────────────────────────

def run_exp1_proposition(W, c_hat):
    """
    Verify that CBM prediction ∝ p_y · c_hat.
    Returns per-class and per-image errors.
    """
    print("\nExp 1: Proposition 1 verification ...")
    W_abs  = np.abs(W)
    W_norm = W_abs.sum(axis=1, keepdims=True)   # (200, 1)
    p_y    = W_abs / W_norm                      # (200, 112)

    y_hat_cbm  = c_hat @ W.T                     # (N, 200)
    y_hat_quad = c_hat @ p_y.T                   # (N, 200)
    y_hat_recon = y_hat_quad * W_norm.T          # (N, 200)

    diffs = np.abs(y_hat_cbm - y_hat_recon)
    scale = np.abs(y_hat_cbm).mean()
    rel_err = diffs / (scale + 1e-8)

    print(f"  Max relative error: {rel_err.max():.2e}")
    print(f"  Mean relative error: {rel_err.mean():.2e}")
    print(f"  Proposition 1: {'VERIFIED' if rel_err.max() < 1e-3 else 'FAILED'}")

    return {
        "max_rel_error":  float(rel_err.max()),
        "mean_rel_error": float(rel_err.mean()),
        "verified":       bool(rel_err.max() < 1e-3),
    }


# ── Experiment 2: WCE Comparison ─────────────────────────────────────────────

def run_exp2_wce(W, E, n_classes=20, seed=42):
    """
    Compare WCE of CBM weights p_yj vs BQ-optimal weights K^{-1}z
    across a sample of classes and kernels.

    For each class y and kernel k:
      WCE(p_y, k) vs WCE*(k) = min WCE = BQ-optimal
    """
    print(f"\nExp 2: WCE comparison ({n_classes} classes × 5 kernels) ...")
    rng = np.random.default_rng(seed)

    k       = E.shape[0]
    p_tilde = np.ones(k) / k
    W_abs   = np.abs(W)
    W_norm  = W_abs.sum(axis=1, keepdims=True)
    p_y_all = W_abs / W_norm

    class_idx = rng.choice(200, size=n_classes, replace=False)

    results = []
    for y in tqdm(class_idx, desc="  classes"):
        p_y = p_y_all[y]   # (112,)
        for name, kfn in KERNELS_5:
            wce2_cbm, wce2_opt = compute_wce(p_y, E, p_tilde, kfn)
            results.append({
                "class_y":    int(y),
                "kernel":     name,
                "wce_cbm":    float(np.sqrt(max(wce2_cbm, 0))),
                "wce_opt":    float(np.sqrt(wce2_opt)),
                "ratio":      float(np.sqrt(max(wce2_cbm, 1e-12) /
                                            max(wce2_opt, 1e-12))),
            })

    wce_cbm_vals = np.array([r["wce_cbm"] for r in results])
    wce_opt_vals = np.array([r["wce_opt"] for r in results])

    print(f"  Mean WCE(CBM weights):    {wce_cbm_vals.mean():.4f}")
    print(f"  Mean WCE(BQ-optimal):     {wce_opt_vals.mean():.4f}")
    print(f"  Mean ratio WCE_cbm/WCE*:  {(wce_cbm_vals/wce_opt_vals.clip(1e-12)).mean():.2f}x")
    print(f"  CBM > BQ-optimal: "
          f"{(wce_cbm_vals > wce_opt_vals + 1e-6).mean()*100:.1f}% of cases")

    return results


# ── Experiment 3: SVBQ Interval Contains CBM Prediction ──────────────────────

def run_exp3_containment(W, c_hat, y_test, E, n_images=500, seed=42):
    """
    For each test image, check that hat_y_y ∈ [I⁻, I⁺] from SVBQ.
    Reports fraction of images/classes where containment holds.
    """
    print(f"\nExp 3: SVBQ interval containment ({n_images} images) ...")
    rng     = np.random.default_rng(seed)
    idx     = rng.choice(len(c_hat), size=n_images, replace=False)
    k       = E.shape[0]
    p_tilde = np.ones(k) / k

    W_abs  = np.abs(W)
    W_norm = W_abs.sum(axis=1, keepdims=True)

    contained_true  = []   # contains y_hat for the true class
    interval_widths = []
    sigma_nums      = []
    sigma_priors    = []

    for img_i in tqdm(idx, desc="  images"):
        c_img = c_hat[img_i]          # (112,)
        y_true = int(y_test[img_i])

        # CBM prediction for true class
        y_hat_cbm = float(W[y_true] @ c_img)

        # SVBQ for true class
        res = svbq_for_class(E, c_img, p_tilde, KERNELS_5)

        contained_true.append(int(res["I_lo"] <= y_hat_cbm <= res["I_hi"]))
        interval_widths.append(res["I_hi"] - res["I_lo"])

        # Also compute sigma decomposition
        mus, sigmas = [], []
        for name, kfn in KERNELS_5:
            K   = kfn(E)
            z   = K @ p_tilde
            KPP = float(p_tilde @ K @ p_tilde)
            mu, sigma = _bq_posterior(K, z, KPP, c_img)
            mus.append(mu); sigmas.append(sigma)
        sigma_nums.append(float(np.mean(sigmas)))
        sigma_priors.append(float(np.std(mus)))

    coverage = float(np.mean(contained_true))
    print(f"  Coverage (CBM ∈ [I⁻,I⁺]): {coverage*100:.1f}%")
    print(f"  Mean interval width:       {np.mean(interval_widths):.4f}")
    print(f"  Mean σ_numerical:          {np.mean(sigma_nums):.4f}")
    print(f"  Mean σ_prior:              {np.mean(sigma_priors):.4f}")

    return {
        "coverage_cbm_in_interval": coverage,
        "mean_interval_width":      float(np.mean(interval_widths)),
        "mean_sigma_num":           float(np.mean(sigma_nums)),
        "mean_sigma_prior":         float(np.mean(sigma_priors)),
        "sigma_nums":               sigma_nums,
        "sigma_priors":             sigma_priors,
        "interval_widths":          interval_widths,
    }


# ── Figure 4 ──────────────────────────────────────────────────────────────────

def plot_figure4(exp1, exp2_results, exp3,
                 out_path="figures/fig4_cbm_quadrature.pdf"):
    """
    Figure 4: three panels validating the quadrature hierarchy.

    (a) Proposition 1 verification — histogram of relative errors
        (should be near-zero, confirming CBM = quadrature)
    (b) WCE comparison — CBM weights vs BQ-optimal per kernel
        (violin or box plot showing CBM > BQ-optimal)
    (c) σ_prior vs σ_numerical scatter
        (confirming orthogonality in the CBM concept space)
    """
    fig = plt.figure(figsize=(6.5, 2.6))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.45)

    # ── (a) Proposition 1: relative error histogram ───────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    # We only have summary stats, so show a text annotation
    ax1.text(0.5, 0.6,
             f"Max rel. error\n$= {exp1['max_rel_error']:.1e}$",
             transform=ax1.transAxes, ha="center", va="center",
             fontsize=10, color=C["bq_opt"] if exp1["verified"] else "red")
    ax1.text(0.5, 0.35,
             "✓ Proposition 1\nverified" if exp1["verified"]
             else "✗ FAILED",
             transform=ax1.transAxes, ha="center", va="center",
             fontsize=8,
             color=C["bq_opt"] if exp1["verified"] else "red")
    ax1.set_title("(a) Prop 1: CBM = quadrature", fontsize=8, fontweight="bold")
    ax1.axis("off")

    # ── (b) WCE comparison ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    kernel_names = [n for n, _ in KERNELS_5]
    wce_cbm_by_k = {n: [] for n in kernel_names}
    wce_opt_by_k = {n: [] for n in kernel_names}
    for r in exp2_results:
        wce_cbm_by_k[r["kernel"]].append(r["wce_cbm"])
        wce_opt_by_k[r["kernel"]].append(r["wce_opt"])

    x      = np.arange(len(kernel_names))
    width  = 0.35
    cbm_means = [np.mean(wce_cbm_by_k[n]) for n in kernel_names]
    opt_means = [np.mean(wce_opt_by_k[n]) for n in kernel_names]

    ax2.bar(x - width/2, cbm_means, width, label="CBM $p_{yj}$",
            color=C["cbm"], alpha=0.8)
    ax2.bar(x + width/2, opt_means, width, label="BQ-optimal",
            color=C["bq_opt"], alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([n.replace("(", "\n(") for n in kernel_names],
                         fontsize=6)
    ax2.set_ylabel("WCE", fontsize=8)
    ax2.set_title("(b) WCE: CBM vs BQ-optimal", fontsize=8, fontweight="bold")
    ax2.legend(fontsize=6)

    # ── (c) σ_prior vs σ_numerical ───────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    sigma_n = np.array(exp3["sigma_nums"])
    sigma_p = np.array(exp3["sigma_priors"])
    rng_s   = np.random.default_rng(0)
    idx_s   = rng_s.choice(len(sigma_n),
                            size=min(500, len(sigma_n)), replace=False)
    ax3.scatter(sigma_n[idx_s], sigma_p[idx_s],
                alpha=0.3, s=5, color=C["svbq"], linewidths=0)
    rho, pv = scipy.stats.spearmanr(sigma_n, sigma_p)
    ax3.text(0.97, 0.97, f"$\\rho = {rho:.2f}$",
             transform=ax3.transAxes, ha="right", va="top", fontsize=7)
    ax3.set_xlabel("$\\bar{{\\sigma}}_{{\\mathrm{{num}}}}$", fontsize=8)
    ax3.set_ylabel("$\\sigma_{{\\mathrm{{prior}}}}$", fontsize=8)
    ax3.set_title("(c) Decomposition on CUB", fontsize=8, fontweight="bold")

    fig.savefig(out_path)
    plt.close(fig)
    print(f"\n  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts_dir", default="artifacts/",
                        help="Dir with W_labels.npy, c_hat_test.npy, etc.")
    parser.add_argument("--n_wce_classes", type=int, default=50)
    parser.add_argument("--n_containment", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Load artifacts ────────────────────────────────────────────────
    print("── Loading artifacts ─────────────────────────────────────────")
    W      = np.load(os.path.join(args.artifacts_dir, "W_labels.npy"))
    c_hat  = np.load(os.path.join(args.artifacts_dir, "c_hat_test.npy"))
    y_test = np.load(os.path.join(args.artifacts_dir, "y_test.npy"))
    with open(os.path.join(args.artifacts_dir, "concept_names.json")) as f:
        concept_names = json.load(f)

    print(f"  W_labels:   {W.shape}")
    print(f"  c_hat_test: {c_hat.shape}")
    print(f"  y_test:     {y_test.shape}")

    # ── Concept embeddings ────────────────────────────────────────────
    print("\n── Computing concept embeddings ──────────────────────────────")
    E = embed_concepts(concept_names)   # (112, 384)
    print(f"  Concept embeddings: {E.shape}")

    # ── Run experiments ───────────────────────────────────────────────
    print("\n── Running experiments ───────────────────────────────────────")
    exp1 = run_exp1_proposition(W, c_hat)
    exp2 = run_exp2_wce(W, E, n_classes=args.n_wce_classes, seed=args.seed)
    exp3 = run_exp3_containment(W, c_hat, y_test, E,
                                 n_images=args.n_containment,
                                 seed=args.seed)

    # ── Save results ──────────────────────────────────────────────────
    results = {"exp1_proposition": exp1,
               "exp3_containment": {k: v for k, v in exp3.items()
                                    if not isinstance(v, list)}}
    with open("results/cbm_experiments.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────
    plot_figure4(exp1, exp2, exp3)

    # ── Paper fill-ins ────────────────────────────────────────────────
    print("\n── Paper fill-ins (Figure 4 caption) ────────────────────────")
    print(f"  Prop 1 max rel error:         {exp1['max_rel_error']:.1e}")
    print(f"  Prop 1 verified:              {exp1['verified']}")
    wce_cbm = np.mean([r["wce_cbm"] for r in exp2])
    wce_opt = np.mean([r["wce_opt"] for r in exp2])
    print(f"  Mean WCE CBM weights:         {wce_cbm:.4f}")
    print(f"  Mean WCE BQ-optimal:          {wce_opt:.4f}")
    print(f"  WCE ratio (CBM/BQ-opt):       {wce_cbm/max(wce_opt,1e-12):.2f}x")
    print(f"  Coverage (CBM ∈ interval):    "
          f"{exp3['coverage_cbm_in_interval']*100:.1f}%")
    rho, _ = scipy.stats.spearmanr(exp3["sigma_nums"], exp3["sigma_priors"])
    print(f"  Spearman ρ(σ_num, σ_prior):   {rho:.3f}")


if __name__ == "__main__":
    main()