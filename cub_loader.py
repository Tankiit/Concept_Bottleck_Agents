"""
cub_loader.py
=============
Minimal, self-contained CUB-200-2011 dataset loader.

Reads the Koh et al. pkl files directly.
Handles the absolute-path issue by scanning the CUB images directory
once and building a filename → full_path mapping.

Each pkl entry is a dict with keys:
  img_path         : str   (absolute path from original cluster — needs remapping)
  class_label      : int   (0-indexed, 0–199)
  attribute_label  : list  (112 binary ints — already filtered to 112 concepts)
  attribute_certainty : list (112 ints, 1–4)

Usage:
  from cub_loader import CUBDataset, CONCEPT_NAMES, CONCEPT_GROUP_MAP
  ds = CUBDataset("train", cub_root="/path/to/CUB_200_2011")
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

# ── 112 CUB concept names (Koh et al. denoised subset, in pkl order) ─────────
# From CONCEPT_SEMANTICS[SELECTED_CONCEPTS] in the CEM cub.py loader.
# Order matches the attribute_label field in the pkl files.
CONCEPT_NAMES = [
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

# Semantic groups — used for kernel structure analysis
CONCEPT_GROUP_MAP = {
    "bill_shape":       list(range(0, 9)),
    "wing_color":       list(range(9, 24)),
    "upperparts_color": list(range(24, 39)),
    "underparts_color": list(range(39, 54)),
    "breast_pattern":   list(range(54, 58)),
    "back_color":       list(range(58, 73)),
    "tail_shape":       list(range(73, 79)),
    "upper_tail_color": list(range(79, 94)),
    "head_pattern":     list(range(94, 105)),
    "breast_color":     list(range(105, 112)),
}

N_CLASSES   = 200
N_CONCEPTS  = 112


# ── Path remapping ─────────────────────────────────────────────────────────────

_path_cache: dict[str, str] = {}

def _build_path_cache(cub_root: str) -> dict[str, str]:
    """Scan CUB images/ directory once and map filename → full path."""
    global _path_cache
    if _path_cache:
        return _path_cache
    images_dir = os.path.join(cub_root, "images")
    print(f"  Building CUB path cache from {images_dir} ...")
    for cls_dir in os.listdir(images_dir):
        cls_path = os.path.join(images_dir, cls_dir)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            _path_cache[fname] = os.path.join(cls_path, fname)
    print(f"  Cached {len(_path_cache)} image paths")
    return _path_cache


def _remap_path(old_path: str, cub_root: str) -> str:
    """Convert a cluster-absolute path to the local path."""
    cache = _build_path_cache(cub_root)
    fname = os.path.basename(old_path)
    return cache.get(fname, old_path)  # fallback: return original (will fail loudly)


# ── Transforms ────────────────────────────────────────────────────────────────

_TRAIN_TF = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_VAL_TF = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────

class CUBDataset(Dataset):
    """
    CUB-200-2011 dataset reading directly from Koh et al. pkl files.

    Returns: (img_tensor, concept_tensor, class_label)
      img_tensor     : (3, 224, 224) float32
      concept_tensor : (112,)        float32  binary concept labels
      class_label    : int           0–199
    """

    def __init__(self, split: str, cub_root: str):
        """
        Args:
            split:    "train", "val", or "test"
            cub_root: path to CUB_200_2011/ directory
                      (must contain images/ and class_attr_data_10/)
        """
        self.split    = split
        self.cub_root = cub_root
        self.is_train = (split == "train")

        pkl_path = os.path.join(cub_root, "class_attr_data_10", f"{split}.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"PKL not found: {pkl_path}\n"
                f"Expected layout: {cub_root}/class_attr_data_10/{{train,val,test}}.pkl"
            )

        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)

        # Build path cache once
        _build_path_cache(cub_root)

        # Process entries
        self.items: list[dict] = []
        n_missing = 0
        for entry in raw:
            img_path = _remap_path(entry["img_path"], cub_root)
            if not os.path.exists(img_path):
                n_missing += 1
                continue
            attrs = np.array(entry["attribute_label"], dtype=np.float32)
            if len(attrs) != N_CONCEPTS:
                # Unexpected size — skip rather than silently return wrong data
                continue
            self.items.append({
                "img_path":   img_path,
                "class_label": int(entry["class_label"]),
                "concepts":   attrs,
            })

        if n_missing > 0:
            print(f"  WARNING: {n_missing} images not found in {split} split "
                  f"(skipped). {len(self.items)} usable.")
        else:
            print(f"  {split}: {len(self.items)} images loaded OK")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        img  = Image.open(item["img_path"]).convert("RGB")
        img  = (_TRAIN_TF if self.is_train else _VAL_TF)(img)
        return img, torch.tensor(item["concepts"]), item["class_label"]

    def concept_weights(self) -> np.ndarray:
        """
        Per-concept positive class weight for weighted BCE.
        weight_j = (n_neg_j / n_pos_j), clipped to [0.1, 10].
        """
        attrs = np.stack([it["concepts"] for it in self.items])  # (N, 112)
        n_pos = attrs.sum(0).clip(1)
        n_neg = (1 - attrs).sum(0).clip(1)
        return np.clip(n_neg / n_pos, 0.1, 10.0).astype(np.float32)