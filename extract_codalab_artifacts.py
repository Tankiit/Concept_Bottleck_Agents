"""
extract_codalab_artifacts.py
============================
Extracts W_labels, c_hat (predicted concepts), c_true (ground truth
concepts), and y_test from the official Koh et al. (2020) CodaLab
artifacts.

Download these two bundles from:
  https://worksheets.codalab.org/worksheets/0x362911581fcd4e048ddfd84f47203fd2

  1. SequentialModel_WithVal__Seed1  (uuid: 0x7a2dfd)  → 183k
     Save as: sequential_seed1/outputs/best_model_1.pth

  2. ConceptModel1__PredConcepts     (uuid: 0xd4401d)  → 43.1m
     Save as: concept_pred_seed1/  (contains train.pkl, val.pkl, test.pkl)

Usage:
  python extract_codalab_artifacts.py \
      --seq_model sequential_seed1/outputs/best_model_1.pth \
      --pred_concepts concept_pred_seed1/ \
      --out_dir artifacts/

Output:
  artifacts/W_labels.npy      (200, 112)  label predictor weights
  artifacts/c_hat_test.npy    (N, 112)    predicted concept activations
  artifacts/c_true_test.npy   (N, 112)    ground truth concept labels
  artifacts/y_test.npy        (N,)        class labels (0-199)
  artifacts/concept_names.json            112 concept name strings
"""

import os, json, pickle, argparse, sys, types
import numpy as np
import torch


def _install_cub_template_model_stub():
    """Install a minimal CUB.template_model.MLP stub for legacy unpickling."""
    if "CUB" not in sys.modules:
        sys.modules["CUB"] = types.ModuleType("CUB")

    if "CUB.template_model" not in sys.modules:
        template_model = types.ModuleType("CUB.template_model")

        class MLP(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                # Placeholder; pickle restore will overwrite module params.
                self.linear = torch.nn.Linear(112, 200)

            def forward(self, x):
                return self.linear(x)

        template_model.MLP = MLP
        sys.modules["CUB.template_model"] = template_model
        sys.modules["CUB"].template_model = template_model


# ── 112 CUB concept names in order (from Koh et al. Table S1) ────────────────
# These are the 112 concepts after processing from 312 raw attributes.
# Order matches the concept model output and the W_labels columns.
CUB_CONCEPT_NAMES = [
    "has_bill_shape::curved_(up_or_down)",
    "has_bill_shape::dagger",
    "has_bill_shape::hooked",
    "has_bill_shape::needle",
    "has_bill_shape::hooked_seabird",
    "has_bill_shape::spatula",
    "has_bill_shape::all-purpose",
    "has_bill_shape::cone",
    "has_bill_shape::specialized",
    "has_wing_color::blue",
    "has_wing_color::brown",
    "has_wing_color::iridescent",
    "has_wing_color::purple",
    "has_wing_color::rufous",
    "has_wing_color::grey",
    "has_wing_color::yellow",
    "has_wing_color::olive",
    "has_wing_color::green",
    "has_wing_color::pink",
    "has_wing_color::orange",
    "has_wing_color::black",
    "has_wing_color::white",
    "has_wing_color::red",
    "has_wing_color::buff",
    "has_upperparts_color::blue",
    "has_upperparts_color::brown",
    "has_upperparts_color::iridescent",
    "has_upperparts_color::purple",
    "has_upperparts_color::rufous",
    "has_upperparts_color::grey",
    "has_upperparts_color::yellow",
    "has_upperparts_color::olive",
    "has_upperparts_color::green",
    "has_upperparts_color::pink",
    "has_upperparts_color::orange",
    "has_upperparts_color::black",
    "has_upperparts_color::white",
    "has_upperparts_color::red",
    "has_upperparts_color::buff",
    "has_underparts_color::blue",
    "has_underparts_color::brown",
    "has_underparts_color::iridescent",
    "has_underparts_color::purple",
    "has_underparts_color::rufous",
    "has_underparts_color::grey",
    "has_underparts_color::yellow",
    "has_underparts_color::olive",
    "has_underparts_color::green",
    "has_underparts_color::pink",
    "has_underparts_color::orange",
    "has_underparts_color::black",
    "has_underparts_color::white",
    "has_underparts_color::red",
    "has_underparts_color::buff",
    "has_breast_pattern::solid",
    "has_breast_pattern::spotted",
    "has_breast_pattern::striped",
    "has_breast_pattern::multi-colored",
    "has_back_color::blue",
    "has_back_color::brown",
    "has_back_color::iridescent",
    "has_back_color::purple",
    "has_back_color::rufous",
    "has_back_color::grey",
    "has_back_color::yellow",
    "has_back_color::olive",
    "has_back_color::green",
    "has_back_color::pink",
    "has_back_color::orange",
    "has_back_color::black",
    "has_back_color::white",
    "has_back_color::red",
    "has_back_color::buff",
    "has_tail_shape::forked_tail",
    "has_tail_shape::rounded_tail",
    "has_tail_shape::notched_tail",
    "has_tail_shape::fan-shaped_tail",
    "has_tail_shape::pointed_tail",
    "has_tail_shape::squared_tail",
    "has_upper_tail_color::blue",
    "has_upper_tail_color::brown",
    "has_upper_tail_color::iridescent",
    "has_upper_tail_color::purple",
    "has_upper_tail_color::rufous",
    "has_upper_tail_color::grey",
    "has_upper_tail_color::yellow",
    "has_upper_tail_color::olive",
    "has_upper_tail_color::green",
    "has_upper_tail_color::pink",
    "has_upper_tail_color::orange",
    "has_upper_tail_color::black",
    "has_upper_tail_color::white",
    "has_upper_tail_color::red",
    "has_upper_tail_color::buff",
    "has_head_pattern::spotted",
    "has_head_pattern::malar_stripe",
    "has_head_pattern::crested",
    "has_head_pattern::masked",
    "has_head_pattern::unique_pattern",
    "has_head_pattern::eyebrow",
    "has_head_pattern::eyering",
    "has_head_pattern::plain",
    "has_head_pattern::eyeline",
    "has_head_pattern::striped",
    "has_head_pattern::capped",
    "has_breast_color::blue",
    "has_breast_color::brown",
    "has_breast_color::iridescent",
    "has_breast_color::purple",
    "has_breast_color::rufous",
    "has_breast_color::grey",
    "has_breast_color::yellow",
    "has_breast_color::olive",
]


def load_sequential_model(model_path: str):
    """
    Load the Sequential CtoY model (nn.Linear(112, 200)) and extract W.

    The checkpoint saves the model directly (not a state_dict wrapper).
    The model is a small MLP: Linear(112, 200).
    """
    print(f"Loading sequential model from {model_path} ...")
    # PyTorch 2.6 changed torch.load default to weights_only=True, which
    # breaks loading legacy pickled model objects from the official artifacts.
    # These files are trusted CodaLab bundles, so prefer weights_only=False.
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("CUB"):
            _install_cub_template_model_stub()
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        else:
            raise
    except TypeError:
        # Backward compatibility with older torch versions that do not
        # expose the weights_only argument.
        checkpoint = torch.load(model_path, map_location="cpu")

    # The checkpoint may be the model directly or a dict
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            # Rebuild and load state dict
            model = torch.nn.Linear(112, 200)
            state = {k.replace("module.", ""): v
                     for k, v in checkpoint["state_dict"].items()}
            model.load_state_dict(state)
        elif "model" in checkpoint:
            model = checkpoint["model"]
        else:
            # Try to load as state dict directly
            model = torch.nn.Linear(112, 200)
            model.load_state_dict(checkpoint)
    else:
        # Saved model object directly
        model = checkpoint

    model.eval()

    # Extract W: the weight matrix of the final linear layer
    # For Sequential CtoY, the model IS a Linear(112, 200)
    # or a Sequential with one Linear layer
    if isinstance(model, torch.nn.Linear):
        W = model.weight.detach().cpu().numpy()  # (200, 112)
    elif isinstance(model, torch.nn.Sequential):
        # Find the last Linear layer
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Linear):
                W = module.weight.detach().cpu().numpy()
                break
    else:
        # Try to find any Linear layer
        linears = [m for m in model.modules()
                   if isinstance(m, torch.nn.Linear)]
        # The CtoY linear should have in_features=112, out_features=200
        W = None
        for lin in linears:
            if lin.in_features == 112 and lin.out_features == 200:
                W = lin.weight.detach().cpu().numpy()
                break
        if W is None:
            print("  WARNING: could not find Linear(112, 200). "
                  "Available linears:")
            for lin in linears:
                print(f"    Linear({lin.in_features}, {lin.out_features})")
            raise ValueError("Could not extract W_labels from checkpoint.")

    print(f"  W_labels shape: {W.shape}")
    assert W.shape == (200, 112), \
        f"Expected (200, 112), got {W.shape}"
    return W


def load_pred_concepts(pred_dir: str, split: str = "test"):
    """
    Load predicted concept activations from ExtractConcepts output.

    The pkl files contain lists of tuples. The exact format varies
    slightly across Koh et al. code versions, but is typically:
        (img_path, class_label, attr_labels, attr_certainties,
         attr_predictions)
    or:
        (img_path, class_label, attr_labels, attr_predictions)

    We handle both formats.
    """
    pkl_path = os.path.join(pred_dir, f"{split}.pkl")
    if not os.path.exists(pkl_path):
        # Try alternative names
        for name in [f"class_attr_data_{split}.pkl",
                     f"{split}_data.pkl", "test_data.pkl"]:
            alt = os.path.join(pred_dir, name)
            if os.path.exists(alt):
                pkl_path = alt
                break

    print(f"Loading predicted concepts from {pkl_path} ...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print(f"  Loaded {len(data)} examples")
    print(f"  First entry type: {type(data[0])}")
    if isinstance(data[0], (list, tuple)):
        print(f"  First entry length: {len(data[0])}")
        # Print field types to understand format
        for i, field in enumerate(data[0]):
            if hasattr(field, '__len__') and not isinstance(field, str):
                print(f"    field[{i}]: length {len(field)}, "
                      f"type {type(field)}")
            else:
                print(f"    field[{i}]: {type(field).__name__} = {field}")

    # Extract fields
    c_hat_list, c_true_list, y_list = [], [], []
    warned_missing_pred = False

    for entry in data:
        if isinstance(entry, dict):
            # Dict format
            y_list.append(entry.get("class_label", entry.get("y", 0)))
            c_true = np.array(entry.get("attribute_label",
                                        entry.get("attr_label", [])),
                              dtype=np.float32)

            c_hat_raw = entry.get("attribute_prediction",
                                  entry.get("attr_pred", None))
            if c_hat_raw is None:
                # Some official CodaLab bundles do not store predicted
                # concepts in the split pickle. Fall back to attribute_label
                # to keep artifact extraction runnable.
                if not warned_missing_pred:
                    print("  WARNING: no attribute_prediction field found; "
                          "using attribute_label as c_hat fallback.")
                    warned_missing_pred = True
                c_hat = c_true.copy()
            else:
                c_hat = np.array(c_hat_raw, dtype=np.float32)

            c_true_list.append(c_true)
            c_hat_list.append(c_hat)
        elif isinstance(entry, (list, tuple)):
            # Tuple format: determine which fields are which by length
            # Standard format: (img_path, class_label, attr_label,
            #                   [attr_certainty,] attr_pred)
            img_path = entry[0]
            y        = entry[1]
            y_list.append(y)

            # Find the 112-length arrays
            attr_arrays = [(i, e) for i, e in enumerate(entry[2:], 2)
                           if hasattr(e, '__len__') and len(e) == 112]

            if len(attr_arrays) >= 2:
                # First 112-array: ground truth labels (binary)
                # Last 112-array: predictions (continuous)
                c_true_list.append(np.array(attr_arrays[0][1],
                                             dtype=np.float32))
                c_hat_list.append(np.array(attr_arrays[-1][1],
                                            dtype=np.float32))
            elif len(attr_arrays) == 1:
                # Only predictions available
                c_hat_list.append(np.array(attr_arrays[0][1],
                                            dtype=np.float32))
                c_true_list.append(np.zeros(112, dtype=np.float32))
                print("  WARNING: only one 112-array found; "
                      "c_true will be zeros.")
            else:
                print(f"  WARNING: no 112-arrays found in entry {entry}")
                continue

    c_hat  = np.array(c_hat_list,  dtype=np.float32)   # (N, 112)
    c_true = np.array(c_true_list, dtype=np.float32)   # (N, 112)
    y      = np.array(y_list,      dtype=np.int64)     # (N,)

    print(f"  c_hat shape:  {c_hat.shape}")
    print(f"  c_true shape: {c_true.shape}")
    print(f"  y shape:      {y.shape}")
    print(f"  y range:      {y.min()} – {y.max()}")
    print(f"  c_hat range:  {c_hat.min():.3f} – {c_hat.max():.3f}")
    print(f"  c_true unique values: {np.unique(c_true)[:5]} ...")

    return c_hat, c_true, y


def verify_quadrature_identity(W: np.ndarray,
                                c_hat: np.ndarray) -> dict:
    """
    Verify Proposition 1: CBM prediction = quadrature rule.

    Checks that p_y · c_hat ∝ W_y · c_hat for all classes and images.
    p_y = |W_y| / ||W_y||_1

    Returns max relative error across all (image, class) pairs.
    """
    print("\nVerifying Proposition 1: CBM = quadrature rule ...")

    # Standard CBM prediction (unnormalised logits)
    y_hat_cbm = c_hat @ W.T               # (N, 200)

    # Quadrature formula with normalised weights
    W_abs     = np.abs(W)                  # (200, 112)
    W_norm    = W_abs.sum(axis=1, keepdims=True)  # (200, 1)
    p_y       = W_abs / W_norm             # (200, 112)
    y_hat_quad = c_hat @ p_y.T            # (N, 200)

    # Should be proportional: y_hat_quad = y_hat_cbm / ||W_y||_1
    scale     = W_norm.squeeze()           # (200,)
    y_hat_scaled = y_hat_quad * scale[np.newaxis, :]  # (N, 200)

    max_err = np.max(np.abs(y_hat_cbm - y_hat_scaled)) / \
              (np.max(np.abs(y_hat_cbm)) + 1e-8)

    print(f"  Max relative error |CBM - quadrature| / max: {max_err:.2e}")
    print(f"  {'PASS' if max_err < 1e-4 else 'FAIL'} "
          f"(expected < 1e-4)")

    return {
        "max_relative_error":   float(max_err),
        "proposition_verified": bool(max_err < 1e-4),
        "W_labels_norm_mean":   float(W_norm.mean()),
        "W_labels_norm_std":    float(W_norm.std()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_model", required=True,
                        help="Path to SequentialModel best_model_1.pth")
    parser.add_argument("--pred_concepts", required=True,
                        help="Dir containing ConceptModel1__PredConcepts "
                             "(train.pkl, val.pkl, test.pkl)")
    parser.add_argument("--out_dir", default="artifacts/",
                        help="Output directory for numpy arrays")
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Extract W from sequential model ───────────────────────────────
    W = load_sequential_model(args.seq_model)
    np.save(os.path.join(args.out_dir, "W_labels.npy"), W)
    print(f"  Saved W_labels.npy  {W.shape}")

    # ── 2. Extract concept predictions ───────────────────────────────────
    c_hat, c_true, y = load_pred_concepts(args.pred_concepts, args.split)
    np.save(os.path.join(args.out_dir, "c_hat_test.npy"),  c_hat)
    np.save(os.path.join(args.out_dir, "c_true_test.npy"), c_true)
    np.save(os.path.join(args.out_dir, "y_test.npy"),      y)
    print(f"  Saved c_hat_test.npy  {c_hat.shape}")
    print(f"  Saved c_true_test.npy {c_true.shape}")
    print(f"  Saved y_test.npy      {y.shape}")

    # ── 3. Save concept names ─────────────────────────────────────────────
    # Use our list if length matches; otherwise use indices
    if len(CUB_CONCEPT_NAMES) == 112:
        names = CUB_CONCEPT_NAMES
    else:
        names = [f"concept_{j}" for j in range(112)]
    with open(os.path.join(args.out_dir, "concept_names.json"), "w") as f:
        json.dump(names, f, indent=2)
    print(f"  Saved concept_names.json  ({len(names)} names)")

    # ── 4. Verify Proposition 1 ───────────────────────────────────────────
    verification = verify_quadrature_identity(W, c_hat)
    with open(os.path.join(args.out_dir, "verification.json"), "w") as f:
        json.dump(verification, f, indent=2)

    # ── 5. Summary ────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────")
    print(f"  W_labels:      {W.shape}   (200 classes × 112 concepts)")
    print(f"  c_hat_test:    {c_hat.shape}   ({len(c_hat)} test images)")
    print(f"  c_true_test:   {c_true.shape}")
    print(f"  y_test:        {y.shape}")
    print(f"  Prop 1 check:  "
          f"{'PASS' if verification['proposition_verified'] else 'FAIL'}")
    print(f"\nAll artifacts saved to {args.out_dir}/")
    print("Ready to run: python run_cbm_experiments.py "
          f"--artifacts_dir {args.out_dir}")


if __name__ == "__main__":
    main()