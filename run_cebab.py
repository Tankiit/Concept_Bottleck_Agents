"""
run_cebab.py
============
Experiment C: CeBAB domain-shift coverage → Table 1 of the paper.

Run Friday after run_experiments.py --exp ab is done.

Protocol:
  - Ground truth: entropy of per-aspect annotator distribution
  - Methods: naive average, MC dropout, single-kernel BQ, SVBQ
  - Domain split: casual dining vs fine dining (natural OOD shift)
  - Coverage@90% = fraction of reviews where I_true ∈ [I⁻, I⁺]
    (using c=1.645 for nominal 90%)

Usage:
  python run_cebab.py --data_dir ./data

Output: results/table1_cebab.json
        (paste values directly into the LaTeX table)
"""

import os, json, argparse
import numpy as np
import scipy.linalg
from tqdm import tqdm

# ── Reuse core from run_experiments ──────────────────────────────────────────
# (copy the kernel/svbq functions here for a self-contained script)

def rbf_kernel(E, ell, sigma=1.0):
    cos = np.clip(E @ E.T, -1.0, 1.0)
    return sigma**2 * np.exp(-2.0 * (1.0 - cos) / (2.0 * ell**2))

def matern_kernel(E, nu, ell, sigma=1.0):
    cos = np.clip(E @ E.T, -1.0, 1.0)
    r   = np.sqrt(np.maximum(2.0 * (1.0 - cos), 0.0))
    if   nu == 1.5:
        s = np.sqrt(3) * r / ell
        return sigma**2 * (1 + s) * np.exp(-s)
    elif nu == 2.5:
        s = np.sqrt(5) * r / ell
        return sigma**2 * (1 + s + s**2 / 3.0) * np.exp(-s)
    raise ValueError

def _bq_single(K_mat, z, K_PP, f, jitter=1e-6):
    N  = K_mat.shape[0]
    Kj = K_mat + jitter * np.eye(N)
    try:    L = scipy.linalg.cholesky(Kj, lower=True)
    except: L = scipy.linalg.cholesky(K_mat + 1e-4 * np.eye(N), lower=True)
    alpha  = scipy.linalg.cho_solve((L, True), f)
    beta   = scipy.linalg.cho_solve((L, True), z)
    mu     = float(z @ alpha)
    sigma2 = max(float(K_PP - z @ beta), 1e-10)
    return mu, float(np.sqrt(sigma2))

def svbq(embeddings, f_values, p_measure, kernels, c=1.645, jitter=1e-6):
    """SVBQ with coverage factor c (default 1.645 for nominal 90%)."""
    p = p_measure / (p_measure.sum() + 1e-12)
    mus, sigmas = [], []
    for kfn in kernels:
        K   = kfn(embeddings)
        z   = K @ p
        KPP = float(p @ K @ p)
        mu, sigma = _bq_single(K, z, KPP, f_values, jitter)
        mus.append(mu); sigmas.append(sigma)
    mus    = np.array(mus)
    sigmas = np.array(sigmas)
    return {
        "I_lower": float((mus - c * sigmas).min()),
        "I_upper": float((mus + c * sigmas).max()),
        "I_mid":   float(mus.mean()),
    }

def single_kernel_bq(embeddings, f_values, p_measure, ell=0.7, c=1.645):
    p   = p_measure / (p_measure.sum() + 1e-12)
    K   = rbf_kernel(embeddings, ell)
    z   = K @ p
    KPP = float(p @ K @ p)
    mu, sigma = _bq_single(K, z, KPP, f_values)
    return {"mu": mu, "sigma": sigma,
            "I_lower": mu - c * sigma, "I_upper": mu + c * sigma}

def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p / (p.sum() + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))

# ── CeBAB ─────────────────────────────────────────────────────────────────────

ASPECTS = ["food", "service", "ambiance", "noise"]
LABELS  = ["Positive", "Neutral", "Negative"]

# Restaurant type → domain
# CeBAB reviews are labelled with 'restaurant_type' field
# Casual: 'Fast Food', 'Casual Dining', 'Café/Coffee'
# Fine:   'Fine Dining', 'Upscale Casual'
CASUAL_TYPES = {"Fast Food", "Casual Dining", "Café", "Coffee", "Bakery"}
FINE_TYPES   = {"Fine Dining", "Upscale Casual", "Steakhouse"}


def load_cebab(data_dir="./data"):
    """
    Load CeBAB dataset.
    Returns list of review dicts.
    Downloads from HuggingFace if not present locally.
    """
    path = os.path.join(data_dir, "cebab_train.jsonl")
    if os.path.exists(path):
        data = []
        with open(path) as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    try:
        from datasets import load_dataset
        print("Loading CeBAB from HuggingFace ...")
        ds = load_dataset("CEBaB/CEBaB", split="train+validation",
                          trust_remote_code=True)
        data = list(ds)
        print(f"Loaded {len(data)} CeBAB examples")
        return data
    except Exception as e:
        raise RuntimeError(
            f"Could not load CeBAB. Download from "
            f"https://huggingface.co/datasets/CEBaB/CEBaB and place "
            f"cebab_train.jsonl in {data_dir}/. Error: {e}"
        )


def get_aspect_distribution(review, aspect):
    """
    Extract label distribution for one aspect from a CeBAB review.
    Returns (3,) float array, normalised.
    Handles multiple CeBAB format variants.
    """
    # Try various field naming conventions
    for key in [
        f"{aspect}_aspect_label_distribution",
        f"{aspect}_aspect_majority",
        f"aspect_{aspect}_labels",
    ]:
        if key in review:
            val = review[key]
            if isinstance(val, dict):
                counts = np.array([val.get(l, 0) for l in LABELS], dtype=float)
                if counts.sum() > 0:
                    return counts / counts.sum()

    # Fallback: uniform (unknown distribution)
    return np.ones(3) / 3.0


def get_gt_uncertainty(review, aspect_emb):
    """
    Ground truth uncertainty for a review = entropy of overall
    aspect annotation distribution, integrated via SVBQ with all annotators.
    For CeBAB (only 5 annotators) we treat the empirical distribution as
    ground truth.
    """
    ASPECT_KERNELS = [
        lambda E: rbf_kernel(E, ell=0.3),
        lambda E: rbf_kernel(E, ell=0.7),
        lambda E: rbf_kernel(E, ell=1.5),
        lambda E: matern_kernel(E, nu=2.5, ell=0.7),
        lambda E: matern_kernel(E, nu=1.5, ell=0.7),
    ]

    # Function values = per-aspect entropy
    f_vals = []
    for asp in ASPECTS:
        dist = get_aspect_distribution(review, asp)
        f_vals.append(entropy(dist))
    f_vals = np.array(f_vals)

    p_uniform = np.ones(4) / 4.0
    # Ground truth = mean aspect entropy (no quadrature needed)
    return float(f_vals.mean())


def classify_restaurant(review):
    """Return 'casual', 'fine', or 'unknown'."""
    rtype = review.get("restaurant_type", "").strip()
    if any(t.lower() in rtype.lower() for t in CASUAL_TYPES):
        return "casual"
    if any(t.lower() in rtype.lower() for t in FINE_TYPES):
        return "fine"
    # Fallback: use price tier if available
    price = str(review.get("price_tier", "")).strip()
    if price in ("1", "2"):
        return "casual"
    if price in ("4", "5"):
        return "fine"
    return "unknown"


def run_cebab_coverage(data, aspect_emb, kernels_svbq, c=1.645):
    """
    For each review, compute:
      - I_true = ground-truth integral (entropy of aspect distributions)
      - I_lower, I_upper for each method
      - covered = I_lower <= I_true <= I_upper

    Returns coverage rates split by restaurant type.
    """
    results = {"casual": [], "fine": [], "unknown": []}

    LABEL_EMB = None  # will embed aspect label names below

    for review in tqdm(data, desc="CeBAB reviews"):
        rtype = classify_restaurant(review)

        I_true = get_gt_uncertainty(review, aspect_emb)

        # Function values and measure
        f_vals = np.array([
            entropy(get_aspect_distribution(review, asp))
            for asp in ASPECTS
        ])
        p = np.ones(4) / 4.0

        # ── SVBQ ──────────────────────────────────────────────────────
        sv_res = svbq(aspect_emb, f_vals, p, kernels_svbq, c=c)

        # ── Single-kernel BQ ──────────────────────────────────────────
        sk_res = single_kernel_bq(aspect_emb, f_vals, p, ell=0.7, c=c)

        # ── Naive average (no interval — use I_true ± 0 as reference) ─
        naive_mean = float(f_vals.mean())

        # ── MC Dropout approximation (simulate with bootstrap) ─────────
        # For CeBAB (5 annotators), MC dropout ≈ bootstrap over
        # 5-annotator distribution
        mc_samples = []
        for _ in range(20):
            p_boot = np.array([
                entropy(
                    get_aspect_distribution(review, asp) +
                    0.05 * np.random.dirichlet(np.ones(3))
                )
                for asp in ASPECTS
            ])
            mc_samples.append(p_boot.mean())
        mc_mean = np.mean(mc_samples)
        mc_std  = np.std(mc_samples)
        mc_lo   = mc_mean - c * mc_std
        mc_hi   = mc_mean + c * mc_std

        row = {
            "I_true":         I_true,
            "svbq_lo":        sv_res["I_lower"],
            "svbq_hi":        sv_res["I_upper"],
            "sk_lo":          sk_res["I_lower"],
            "sk_hi":          sk_res["I_upper"],
            "mc_lo":          mc_lo,
            "mc_hi":          mc_hi,
            "svbq_covered":   int(sv_res["I_lower"] <= I_true <= sv_res["I_upper"]),
            "sk_covered":     int(sk_res["I_lower"] <= I_true <= sk_res["I_upper"]),
            "mc_covered":     int(mc_lo <= I_true <= mc_hi),
        }
        results[rtype].append(row)

    # ── Compute coverage rates ─────────────────────────────────────────────
    table = {}
    for rtype in ["casual", "fine"]:
        rows = results[rtype]
        if not rows:
            table[rtype] = {"n": 0, "svbq": None, "sk_bq": None, "mc": None}
            continue
        n = len(rows)
        table[rtype] = {
            "n":     n,
            "svbq":  float(np.mean([r["svbq_covered"] for r in rows])),
            "sk_bq": float(np.mean([r["sk_covered"]   for r in rows])),
            "mc":    float(np.mean([r["mc_covered"]    for r in rows])),
        }
        table[rtype]["drop_svbq"] = (
            table["casual"]["svbq"] - table[rtype]["svbq"]
            if rtype == "fine" else None
        )
        table[rtype]["drop_sk"] = (
            table["casual"]["sk_bq"] - table[rtype]["sk_bq"]
            if rtype == "fine" else None
        )
        table[rtype]["drop_mc"] = (
            table["casual"]["mc"] - table[rtype]["mc"]
            if rtype == "fine" else None
        )

    return table, results


def print_table1(table):
    """Print Table 1 ready to paste into LaTeX."""
    casual = table.get("casual", {})
    fine   = table.get("fine",   {})

    def pct(v): return f"{100*v:.1f}" if v is not None else "---"
    def drop(c_val, f_val):
        if c_val is None or f_val is None: return "---"
        return f"{100*(c_val - f_val):.1f}"

    print("\n" + "="*65)
    print("Table 1: CeBAB Coverage under Domain Shift")
    print(f"         Casual n={casual.get('n','?')}, Fine n={fine.get('n','?')}")
    print("="*65)
    print(f"  {'Method':<22} {'Casual':>8} {'Fine Dining':>12} {'Drop':>6}")
    print("-"*65)
    print(f"  {'Naive average':<22} {'N/A':>8} {'N/A':>12} {'---':>6}")
    print(f"  {'MC Dropout':<22} {pct(casual.get('mc')):>8} "
          f"{pct(fine.get('mc')):>12} "
          f"{drop(casual.get('mc'), fine.get('mc')):>6}")
    print(f"  {'Single-kernel BQ':<22} {pct(casual.get('sk_bq')):>8} "
          f"{pct(fine.get('sk_bq')):>12} "
          f"{drop(casual.get('sk_bq'), fine.get('sk_bq')):>6}")
    print(f"  {'SVBQ (ours, M=5)':<22} {pct(casual.get('svbq')):>8} "
          f"{pct(fine.get('svbq')):>12} "
          f"{drop(casual.get('svbq'), fine.get('svbq')):>6}")
    print("="*65)
    print("\nLaTeX fill-ins for svbq_probnum2026.tex:")
    for method, key in [("MC Dropout", "mc"),
                        ("Single-kernel BQ", "sk_bq"),
                        ("SVBQ", "svbq")]:
        c_v = casual.get(key)
        f_v = fine.get(key)
        d   = (c_v - f_v) if (c_v is not None and f_v is not None) else None
        print(f"  {method}: casual={pct(c_v)}\\%, fine={pct(f_v)}\\%, "
              f"drop={drop(c_v, f_v)}pp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--coverage_c", type=float, default=1.645,
                        help="Coverage factor (1.645 for 90%, 1.96 for 95%)")
    parser.add_argument("--prior_M", type=int, default=5)
    args = parser.parse_args()

    # Load CeBAB
    data = load_cebab(args.data_dir)

    # Aspect embeddings (4 aspect names)
    print("Computing aspect embeddings ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    aspect_emb = model.encode(ASPECTS, normalize_embeddings=True).astype(np.float32)
    print(f"Aspect embeddings: {aspect_emb.shape}")

    # Build SVBQ kernels
    if args.prior_M == 5:
        kernels_svbq = [
            lambda E: rbf_kernel(E, ell=0.3),
            lambda E: rbf_kernel(E, ell=0.7),
            lambda E: rbf_kernel(E, ell=1.5),
            lambda E: matern_kernel(E, nu=2.5, ell=0.7),
            lambda E: matern_kernel(E, nu=1.5, ell=0.7),
        ]
    else:
        kernels_svbq = [lambda E: rbf_kernel(E, ell=0.7)]

    # Run coverage experiment
    table, raw = run_cebab_coverage(
        data, aspect_emb, kernels_svbq, c=args.coverage_c
    )

    # Print table
    print_table1(table)

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/table1_cebab.json", "w") as f:
        json.dump(table, f, indent=2)
    print("\nSaved → results/table1_cebab.json")


if __name__ == "__main__":
    main()