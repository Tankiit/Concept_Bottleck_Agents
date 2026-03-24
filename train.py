# ============================================================
# train_koh_pyc.py
#
# Trains Koh et al. independent CBM on CUB-200-2011
# using the CEM CUB loader + PyC concept layer.
#
# Produces after training:
#   checkpoints/encoder_best.pth
#   checkpoints/predictor_best.pth
#   artifacts/W_concepts.npy      (112, d) — for BQ angular kernel
#   artifacts/W_labels.npy        (200, 112) — for BQ measure
#   artifacts/c_test.npy          (N_test, 112) — concept scores
#   artifacts/y_test.npy          (N_test,)
#   artifacts/concept_names.json  — ordered list of 112 concept strings
#   artifacts/concept_group_map.json — groups for BQ structure test
# ============================================================

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from torch_concepts.data.datasets import cub as cub

# Optional progress bars (no-op fallback if tqdm unavailable)
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):
        return x

# ── CEM CUB loader from torch_concepts ─────────────────────────────────────
# Use the built-in CUB dataset from torch_concepts package
from torch_concepts.data.datasets import cub as cub_module
from torch_concepts.data.datasets.cub import (
    SELECTED_CONCEPTS,
    CONCEPT_SEMANTICS,
    CONCEPT_GROUP_MAP,
    N_CLASSES,
)

# Patch the buggy CUBDataset.__init__ method
original_init = cub_module.CUBDataset.__init__

def patched_init(self, split='train', uncertain_concept_labels=False, root='./CUB200/',
                 path_transform=None, sample_transform=None, concept_transform=None,
                 label_transform=None, uncertainty_based_random_labels=False,
                 unc_map=None, selected_concepts=None, training_augment=True):
    # Call the original init with fixed concept handling
    if unc_map is None:
        unc_map = [{0: 0.5, 1: 0.5, 2: 0.5, 3: 0.75, 4: 1.0}, {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.75, 4: 1.0}]

    # Set basic attributes (provide default transforms if None)
    self.split = split
    self.uncertain_concept_labels = uncertain_concept_labels
    self.root = root
    self.path_transform = path_transform
    self.sample_transform = sample_transform if sample_transform is not None else identity_transform
    self.concept_transform = concept_transform if concept_transform is not None else identity_transform
    self.label_transform = label_transform if label_transform is not None else identity_transform
    self.training_augment = training_augment
    self.uncertainty_based_random_labels = uncertainty_based_random_labels
    self.unc_map = unc_map

    if selected_concepts is None:
        selected_concepts = list(range(len(SELECTED_CONCEPTS)))
    self.selected_concepts = selected_concepts

    # Fix the buggy concept names assignment
    import numpy as np
    self.concept_names = self.concept_attr_names = list(
        np.array(CONCEPT_SEMANTICS)[selected_concepts]
    )
    self.task_names = self.task_attr_names = cub_module.CLASS_NAMES

    # Continue with the rest of the original init
    import os
    base_dir = os.path.join(root, 'class_attr_data_10')
    self.pkl_file_path = os.path.join(base_dir, f'{split}.pkl')

    import pickle
    import numpy as np
    with open(self.pkl_file_path, 'rb') as f:
        self.data = pickle.load(f)

    # Process the data (apply path_transform once per item)
    processed_data = []
    if path_transform:
        for item in self.data:
            it = dict(item)
            it['img_path'] = path_transform(it['img_path'])
            processed_data.append(it)
        self.data = processed_data
    else:
        self.data = list(self.data)

    # Build and cache transforms once (avoid per-sample Compose construction)
    import torchvision.transforms as _T
    self._train_transform = _T.Compose([
        _T.RandomResizedCrop(_IMG_SIZE),
        _T.RandomHorizontalFlip(),
        _T.ColorJitter(brightness=0.2, contrast=0.2),
        _T.ToTensor(),
        _T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    self._eval_transform = _T.Compose([
        _T.Resize(_IMG_SIZE),
        _T.CenterCrop(_IMG_SIZE),
        _T.ToTensor(),
        _T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def identity_transform(x):
    """Default identity transform for dataset items."""
    return x

# Patch the concept_weights method as well
original_concept_weights = cub_module.CUBDataset.concept_weights

def patched_concept_weights(self):
    """Return concept weights, handling the index mismatch properly."""
    import numpy as np
    # Get the original weights (should be length 312)
    try:
        weights = original_concept_weights(self)
    except (IndexError, AttributeError):
        # If the original method fails, create a simple weight array
        # based on the full concept semantics
        imbalance_ratio = np.ones(len(cub_module.CONCEPT_SEMANTICS))
        weights = imbalance_ratio[self.selected_concepts]
    return weights

# Patch the __getitem__ method to handle the index mismatch
original_getitem = cub_module.CUBDataset.__getitem__

def patched_getitem(self, idx):
    """Get item, handling the concept index mismatch properly."""
    import numpy as np
    from PIL import Image
    import os

    # Get the raw data
    img_data = self.data[idx]

    # Load image
    img_path = img_data['img_path']
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        # Return a dummy black image if loading fails
        img = Image.new('RGB', (299, 299), (0, 0, 0))

    # Apply cached transforms
    if self.training_augment and self.split == 'train':
        transform = self._train_transform
    else:
        transform = self._eval_transform

    img = self.sample_transform(img)
    img = transform(img)

    # Get concepts - handle the index mismatch
    attr_label = np.array(img_data['attribute_label'])

    # The attr_label should have the same length as CONCEPT_SEMANTICS (312)
    # We need to select the ones we want using selected_concepts
    if len(attr_label) == len(cub_module.CONCEPT_SEMANTICS):
        # Full attribute vector, select the concepts we want
        concepts = attr_label[self.selected_concepts]
    elif len(attr_label) == len(self.selected_concepts):
        # Already the right size, use as-is
        concepts = attr_label
    else:
        # Unexpected size, take the first len(selected_concepts) elements
        concepts = attr_label[:len(self.selected_concepts)]

    concepts = torch.tensor(concepts, dtype=torch.float32)
    concepts = self.concept_transform(concepts)

    # Get class label
    class_label = img_data['class_label']
    class_label = self.label_transform(class_label)

    return img, concepts, class_label

# Apply the patches
cub_module.CUBDataset.__init__ = patched_init
cub_module.CUBDataset.concept_weights = patched_concept_weights
cub_module.CUBDataset.__getitem__ = patched_getitem

CUBDataset = cub_module.CUBDataset

# ── PyC (optional — falls back gracefully) ────────────────────────────────────
try:
    import torch_concepts as pyc
    PYC_AVAILABLE = True
    print("pytorch-concepts available — using LinearConceptLayer")
except ImportError:
    PYC_AVAILABLE = False
    print("pytorch-concepts not found — using annotated nn.Linear fallback")


# ─────────────────────────────────────────────────────────────────────────────
# Config — mirrors Koh et al. Table S1 / replication papers
# ─────────────────────────────────────────────────────────────────────────────

CFG = dict(
    cub_dir        = os.environ.get("CUB_DIR", "/Users/tanmoy/research/data/CUB_200_2011/CUB_200_2011"),
    num_workers    = 4,          # parallel image loading (safe on Mac with spawn)
    checkpoint_dir = "./checkpoints",
    artifact_dir   = "./artifacts",
    # Backbone/input
    backbone       = os.environ.get("BACKBONE", "resnet50"),  # "resnet50" or "inception_v3"
    input_size     = int(os.environ.get("INPUT_SIZE", 224)),  # 224 for ResNet, 299 for Inception
    # Stage 1: x → c
    # Phase A: freeze backbone, train heads only (fast warmup)
    stage1_warmup_epochs = 20,   # epochs with backbone frozen
    # Phase B: full fine-tune backbone + heads
    stage1_epochs  = 100,        # max total epochs (incl. warmup); early-stop kicks in
    stage1_lr      = 0.01,
    stage1_wd      = 4e-5,
    stage1_momentum= 0.9,
    stage1_patience= 5,          # ReduceLROnPlateau patience
    stage1_min_lr  = 1e-5,
    stage1_early_stop = 15,      # stop if val loss hasn't improved for N epochs
    # Stage 2: c → y
    stage2_epochs  = 100,
    stage2_lr      = 0.01,
    stage2_wd      = 4e-5,
    stage2_momentum= 0.9,
    # Shared
    batch_size     = 32,         # smaller batches → more MPS-friendly
    n_concepts     = 112,
    n_classes      = 200,
    device         = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
    aux_loss_weight= 0.4,       # InceptionV3 auxiliary classifier weight
    seed           = 42,
)

# Global image size used by dataset transforms; updated from CFG on import
_IMG_SIZE = CFG["input_size"]


# ─────────────────────────────────────────────────────────────────────────────
# Utility: get ordered concept names from the loader
# ─────────────────────────────────────────────────────────────────────────────

def get_concept_names() -> list[str]:
    """
    Returns the 112 concept name strings in the order they appear
    in the pkl attribute_label vectors.

    SELECTED_CONCEPTS[j] is the index into CONCEPT_SEMANTICS for concept j.
    This is the ground-truth ordering for W_concepts rows.
    """
    return [CONCEPT_SEMANTICS[i] for i in SELECTED_CONCEPTS]


# Global cache for filename to path mapping
_filename_to_path = {}

def build_cub_path_mapping(cub_root: str) -> dict:
    """
    Build a mapping from filenames to full paths for the CUB dataset.
    This is more efficient than searching for each file individually.
    """
    import os
    global _filename_to_path

    if _filename_to_path:
        return _filename_to_path

    images_dir = os.path.join(cub_root, "images")
    print(f"Building CUB filename mapping from {images_dir}...")

    for class_dir in os.listdir(images_dir):
        class_path = os.path.join(images_dir, class_dir)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                _filename_to_path[filename] = os.path.join(class_path, filename)

    print(f"Built mapping for {len(_filename_to_path)} images")
    return _filename_to_path


def fix_cub_image_path(old_path: str, cub_root: str) -> str:
    """
    Fix the image paths in the CUB dataset pickle files.
    The original paths point to a different system, so we need to
    extract just the filename and construct the correct local path.
    """
    import os
    # Build the mapping if not already done
    path_mapping = build_cub_path_mapping(cub_root)

    # Extract the filename from the old path
    filename = os.path.basename(old_path)

    # Look up the new path
    if filename in path_mapping:
        return path_mapping[filename]

    # If not found, return the original path (will error later, but gives info)
    print(f"Warning: Could not find image file: {filename}")
    return old_path


# ─────────────────────────────────────────────────────────────────────────────
# Concept head: PyC annotated layer OR plain nn.Linear with stored names
# ─────────────────────────────────────────────────────────────────────────────

class AnnotatedConceptHead(nn.Module):
    """
    Linear(d → 112) with concept name annotations stored as metadata.
    Uses PyC's LinearConceptLayer if available, otherwise plain nn.Linear.

    Either way, .weight is (112, d) — the W_concepts matrix for BQ.
    .concept_names gives the ordered list of 112 concept strings.
    """
    def __init__(self, in_features: int, concept_names: list[str]):
        super().__init__()
        self.concept_names = concept_names
        n = len(concept_names)

        # Use standard nn.Linear for now (PyC API has changed)
        self.linear = nn.Linear(in_features, n)
        # Store names as a non-parameter attribute
        self.linear.concept_names = concept_names

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, d)  →  logits: (B, 112)"""
        return self.linear(h)

    @property
    def weight(self) -> torch.Tensor:
        """(112, d) — concept direction vectors in feature space."""
        return self.linear.weight


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 model: InceptionV3 + AnnotatedConceptHead
# ─────────────────────────────────────────────────────────────────────────────

class ConceptEncoder(nn.Module):
    """
    Backbone (ResNet50 default, InceptionV3 optional) + annotated linear head.

    - ResNet50 (default): expects ~224 input, no aux branch.
    - InceptionV3: expects 299 input, returns aux branch during training.
    """
    def __init__(self, concept_names: list[str], backbone: str | None = None):
        super().__init__()
        bname = backbone or CFG.get("backbone", "resnet50")
        self._backbone_type = bname

        if bname == "inception_v3":
            inc = models.inception_v3(pretrained=True, aux_logits=True)
            d   = inc.fc.in_features  # 2048
            inc.fc           = nn.Identity()
            inc.AuxLogits.fc = nn.Identity()
            self.backbone    = inc
            self.aux_head    = nn.Linear(768, len(concept_names))
        elif bname == "resnet50":
            try:
                res = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            except Exception:
                res = models.resnet50(pretrained=True)
            d   = res.fc.in_features  # 2048
            res.fc         = nn.Identity()
            self.backbone  = res
            self.aux_head  = None
        else:
            raise ValueError(f"Unsupported backbone: {bname}")

        self.concept_head = AnnotatedConceptHead(d, concept_names)

    def forward(self, x: torch.Tensor):
        if self._backbone_type == "inception_v3":
            if self.training:
                out    = self.backbone(x)
                h_main = out.logits
                h_aux  = out.aux_logits
                c_main = self.concept_head(h_main)
                c_aux  = self.aux_head(h_aux) if self.aux_head is not None else None
                return c_main, c_aux
            else:
                h = self.backbone(x)
                return self.concept_head(h), None
        else:
            h = self.backbone(x)
            return self.concept_head(h), None

    def get_W_concepts(self) -> np.ndarray:
        return self.concept_head.weight.detach().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 model: Linear label predictor
# ─────────────────────────────────────────────────────────────────────────────

class LabelPredictor(nn.Module):
    """
    Single linear layer: concept scores → class logits.
    This is exactly Koh et al.'s c→y model.

    .weight: (200, 112) — W_labels for BQ measure.
    """
    def __init__(self, n_concepts=112, n_classes=200):
        super().__init__()
        self.fc = nn.Linear(n_concepts, n_classes)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """c: (B, 112) → logits: (B, 200)"""
        return self.fc(c)

    def get_W_labels(self) -> np.ndarray:
        """Returns W_labels: (200, 112) for BQ."""
        return self.fc.weight.detach().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 training
# ─────────────────────────────────────────────────────────────────────────────

def _set_backbone_trainable(encoder: "ConceptEncoder", trainable: bool):
    """Freeze or unfreeze InceptionV3 backbone weights."""
    for param in encoder.backbone.parameters():
        param.requires_grad = trainable


def train_stage1(
    encoder: "ConceptEncoder",
    train_loader: DataLoader,
    val_loader:   DataLoader,
    cfg:          dict,
) -> "ConceptEncoder":
    """
    Trains x→c on concept labels using weighted BCE.

    Two-phase training for speed:
      Phase A (warmup_epochs): backbone FROZEN  → only heads trained  (fast)
      Phase B (remaining):     backbone UNFROZEN → full fine-tuning    (thorough)

    Early stopping terminates training if val loss doesn't improve
    for `stage1_early_stop` consecutive epochs.
    """
    device = cfg["device"]
    encoder = encoder.to(device)
    warmup = cfg["stage1_warmup_epochs"]
    early_stop_patience = cfg["stage1_early_stop"]

    # Weighted BCE — addresses concept class imbalance
    train_ds = train_loader.dataset
    pos_weight = torch.tensor(
        train_ds.concept_weights(), dtype=torch.float32
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Enable AMP on CUDA for speed; MPS autocast kept conservative
    use_cuda_amp = (device == "cuda")

    # Phase A: freeze backbone, only optimise heads
    print(f"  [Phase A] Freezing backbone for {warmup} warmup epochs ...")
    _set_backbone_trainable(encoder, False)
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, encoder.parameters()),
        lr=cfg["stage1_lr"],
        momentum=cfg["stage1_momentum"],
        weight_decay=cfg["stage1_wd"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1,
        patience=cfg["stage1_patience"],
        min_lr=cfg["stage1_min_lr"],
    )

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    save_path    = os.path.join(cfg["checkpoint_dir"], "encoder_best.pth")
    best_loss    = float("inf")
    no_improve   = 0
    phase_b_done = False

    for epoch in range(cfg["stage1_epochs"]):

        # ── Switch to Phase B after warmup
        if epoch == warmup and not phase_b_done:
            print(f"  [Phase B] Unfreezing backbone for full fine-tuning ...")
            _set_backbone_trainable(encoder, True)
            optimizer = torch.optim.SGD(
                encoder.parameters(),
                lr=cfg["stage1_lr"] * 0.1,   # lower LR when unfreezing
                momentum=cfg["stage1_momentum"],
                weight_decay=cfg["stage1_wd"],
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1,
                patience=cfg["stage1_patience"],
                min_lr=cfg["stage1_min_lr"],
            )
            phase_b_done = True

        

        # ── Train
        encoder.train()
        train_loss = 0.0
        n_batches = 0
        iterator = tqdm(train_loader, desc=f"Train E{epoch+1}", leave=False)
        for imgs, concepts, _ in iterator:
            imgs = imgs.to(device, non_blocking=True)
            concepts = concepts.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if use_cuda_amp:
                from torch.cuda.amp import autocast, GradScaler
                scaler = locals().get("_scaler")
                if scaler is None:
                    scaler = GradScaler()
                    globals().update({"_scaler": scaler})
                with autocast():
                    c_logits, c_aux = encoder(imgs)
                    loss = criterion(c_logits, concepts)
                    if c_aux is not None:
                        loss = loss + cfg["aux_loss_weight"] * criterion(c_aux, concepts)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # FP32 or MPS without grad scaling
                c_logits, c_aux = encoder(imgs)
                loss = criterion(c_logits, concepts)
                if c_aux is not None:
                    loss = loss + cfg["aux_loss_weight"] * criterion(c_aux, concepts)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            n_batches  += 1
        train_loss /= max(n_batches, 1)

        # ── Validate
        encoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            viter = tqdm(val_loader, desc=f"Val   E{epoch+1}", leave=False)
            for imgs, concepts, _ in viter:
                imgs = imgs.to(device, non_blocking=True)
                concepts = concepts.to(device, non_blocking=True)
                c_logits, _ = encoder(imgs)
                val_loss   += criterion(c_logits, concepts).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        phase_tag  = "A" if epoch < warmup else "B"
        print(f"  Stage1 [{epoch+1:3d}/{cfg['stage1_epochs']}] Phase {phase_tag}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  lr={current_lr:.2e}")

        # ── Checkpoint + early stopping
        if val_loss < best_loss:
            best_loss  = val_loss
            no_improve = 0
            torch.save(encoder.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= early_stop_patience and epoch >= warmup:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(no improvement for {early_stop_patience} epochs)")
                break

    encoder.load_state_dict(torch.load(save_path, map_location=device))
    print(f"Stage 1 done. Best val loss: {best_loss:.4f}")
    return encoder


# ─────────────────────────────────────────────────────────────────────────────
# Extract concept scores (run encoder once, cache to disk)
# ─────────────────────────────────────────────────────────────────────────────

def extract_concept_scores(
    encoder:    ConceptEncoder,
    dataloader: DataLoader,
    device:     str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs the frozen encoder over an entire split.
    Returns:
        c_scores: (N, 112) float32 — sigmoid-activated ∈ [0,1]
        labels:   (N,)     int64
    """
    encoder.eval()
    all_c, all_y = [], []

    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Extract", leave=False)
        # Lightweight autocast during inference on accelerators
        use_cuda = device == "cuda"
        try:
            from torch.cuda.amp import autocast as _cuda_autocast
        except Exception:
            _cuda_autocast = None
        for imgs, concepts, labels in iterator:
            imgs = imgs.to(device, non_blocking=True)
            if use_cuda and _cuda_autocast is not None:
                with _cuda_autocast():
                    c_logits, _ = encoder(imgs)
            else:
                c_logits, _ = encoder(imgs)
            all_c.append(torch.sigmoid(c_logits).cpu())
            # Normalise label type
            if isinstance(labels, torch.Tensor):
                all_y.append(labels)
            else:
                all_y.append(torch.tensor(labels))

    return (
        torch.cat(all_c).numpy().astype(np.float32),
        torch.cat(all_y).numpy().astype(np.int64),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 training
# ─────────────────────────────────────────────────────────────────────────────

def train_stage2(
    predictor: LabelPredictor,
    c_train:   np.ndarray,
    y_train:   np.ndarray,
    c_val:     np.ndarray,
    y_val:     np.ndarray,
    cfg:       dict,
) -> LabelPredictor:
    """
    Trains c→y with frozen concept scores.
    Uses full dataset tensors (fast — only 112-dim inputs).
    """
    device = cfg["device"]
    predictor = predictor.to(device)

    c_tr = torch.tensor(c_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    c_vl = torch.tensor(c_val,   dtype=torch.float32).to(device)
    y_vl = torch.tensor(y_val,   dtype=torch.long).to(device)

    optimizer = torch.optim.SGD(
        predictor.parameters(),
        lr=cfg["stage2_lr"],
        momentum=cfg["stage2_momentum"],
        weight_decay=cfg["stage2_wd"],
    )
    criterion = nn.CrossEntropyLoss()
    # Use ReduceLROnPlateau on validation accuracy (mode=max)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1,
        patience=cfg.get("stage1_patience", 5),
        min_lr=cfg.get("stage1_min_lr", 1e-5),
    )

    save_path    = os.path.join(cfg["checkpoint_dir"], "predictor_best.pth")
    best_val_acc = 0.0

    for epoch in tqdm(range(cfg["stage2_epochs"]), desc="Stage2", leave=False):
        predictor.train()
        optimizer.zero_grad()
        loss = criterion(predictor(c_tr), y_tr)
        loss.backward()
        optimizer.step()

        predictor.eval()
        with torch.no_grad():
            val_acc = (predictor(c_vl).argmax(1) == y_vl).float().mean().item()
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(predictor.state_dict(), save_path)

    predictor.load_state_dict(torch.load(save_path, map_location=device))
    print(f"Stage 2 done. Best val accuracy: {best_val_acc:.4f}")
    return predictor


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_concept_accuracy(
    encoder: ConceptEncoder,
    loader:  DataLoader,
    device:  str,
) -> float:
    """Mean per-concept AUC proxy: average binary accuracy across 112 concepts."""
    encoder.eval()
    correct, total = 0, 0
    with torch.no_grad():
        iterator = tqdm(loader, desc="Concept Acc", leave=False)
        for imgs, concepts, _ in iterator:
            preds = torch.sigmoid(encoder(imgs.to(device, non_blocking=True))[0]) > 0.5
            correct += (preds.cpu() == concepts.bool()).sum().item()
            total   += concepts.numel()
    return correct / total


def evaluate_task_accuracy(
    predictor: LabelPredictor,
    c_scores:  np.ndarray,
    labels:    np.ndarray,
    device:    str,
) -> float:
    predictor.eval()
    c = torch.tensor(c_scores, dtype=torch.float32).to(device)
    y = torch.tensor(labels,   dtype=torch.long).to(device)
    with torch.no_grad():
        acc = (predictor(c).argmax(1) == y).float().mean().item()
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# Save BQ artifacts
# ─────────────────────────────────────────────────────────────────────────────

def save_bq_artifacts(
    encoder:   ConceptEncoder,
    predictor: LabelPredictor,
    c_test:    np.ndarray,
    y_test:    np.ndarray,
    cfg:       dict,
):
    """
    Saves the four arrays needed for the BQ equivalence experiment:

      W_concepts  (112, 2048) — concept direction vectors
                                rows = concept j's direction in InceptionV3 space
                                kernel:  k(w_i, w_j) = exp(-||ŵ_i - ŵ_j||² / 2ℓ²)

      W_labels    (200, 112)  — label predictor weights
                                W_labels[cls] = learned measure p for class cls
                                BQ with ℓ→∞ recovers Koh output exactly

      c_test      (N, 112)    — concept activations for test images
                                f values for BQ integration

      y_test      (N,)        — ground-truth class labels

    Also saves concept_names.json and concept_group_map.json
    for interpretability and BQ structure analysis.
    """
    os.makedirs(cfg["artifact_dir"], exist_ok=True)
    d = cfg["artifact_dir"]

    W_concepts = encoder.get_W_concepts()       # (112, 2048)
    W_labels   = predictor.get_W_labels()       # (200, 112)

    np.save(os.path.join(d, "W_concepts.npy"), W_concepts)
    np.save(os.path.join(d, "W_labels.npy"),   W_labels)
    np.save(os.path.join(d, "c_test.npy"),     c_test)
    np.save(os.path.join(d, "y_test.npy"),     y_test)

    concept_names = get_concept_names()
    with open(os.path.join(d, "concept_names.json"), "w") as f:
        json.dump(concept_names, f, indent=2)

    # Convert CONCEPT_GROUP_MAP to plain dict for JSON
    group_map = {k: list(v) for k, v in CONCEPT_GROUP_MAP.items()}
    with open(os.path.join(d, "concept_group_map.json"), "w") as f:
        json.dump(group_map, f, indent=2)

    print(f"\nBQ artifacts saved to {d}/")
    print(f"  W_concepts:  {W_concepts.shape}  (concept directions)")
    print(f"  W_labels:    {W_labels.shape}  (label predictor / BQ measure)")
    print(f"  c_test:      {c_test.shape}  (concept scores for experiments)")
    print(f"  concept_names: {len(concept_names)} concepts")
    print(f"  concept_groups: {len(group_map)} groups  ← BQ structure test")

    # Quick sanity: norms of concept directions
    norms = np.linalg.norm(W_concepts, axis=1)
    print(f"\n  W_concepts row norms: "
          f"mean={norms.mean():.3f}  std={norms.std():.3f}")
    print(f"  (Large variance here is expected — "
          f"angular kernel normalises these)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], help="Force compute device")
    parser.add_argument("--gpu", action="store_true", help="Shortcut for --device cuda")
    args = parser.parse_args()

    # Resolve device preference
    auto_device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    requested = None
    if args.device:
        requested = args.device
    elif args.gpu:
        requested = "cuda"
    if requested == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA but not available; falling back to auto device.")
        requested = None
    if requested == "mps" and not torch.backends.mps.is_available():
        print("Requested MPS but not available; falling back to auto device.")
        requested = None

    CFG["device"] = requested or auto_device

    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)

    # Backend perf knobs
    if CFG["device"] == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print(f"Backbone: {CFG['backbone']}  | Input size: {CFG['input_size']}  | Device: {CFG['device']}")

    # ── Concept names (112, ordered to match pkl attribute_label vectors)
    concept_names = get_concept_names()
    print(f"Loaded {len(concept_names)} concept names")
    print(f"First 3: {concept_names[:3]}")
    print(f"Groups:  {list(CONCEPT_GROUP_MAP.keys())[:5]} ...")

    # ── CUB data loaders (CEM loader handles transforms, path fixing, etc.)
    # Create a path transform function to fix the absolute paths in pickle files
    import functools
    path_transform = functools.partial(fix_cub_image_path, cub_root=CFG["cub_dir"])

    train_ds = CUBDataset(split="train", root=CFG["cub_dir"], path_transform=path_transform,
                          selected_concepts=SELECTED_CONCEPTS)
    val_ds   = CUBDataset(split="val",   root=CFG["cub_dir"], path_transform=path_transform,
                          selected_concepts=SELECTED_CONCEPTS)
    test_ds  = CUBDataset(split="test",  root=CFG["cub_dir"], path_transform=path_transform,
                          selected_concepts=SELECTED_CONCEPTS)

    # pin_memory only helps with CUDA; skip on MPS/CPU to avoid warnings
    _pin = CFG["device"] == "cuda"
    # persistent_workers keeps loader processes alive between epochs (big speedup)
    _persist = CFG["num_workers"] > 0
    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], shuffle=True,
        num_workers=CFG["num_workers"], pin_memory=_pin,
        persistent_workers=_persist,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=_pin,
        persistent_workers=_persist,
    )
    test_loader = DataLoader(
        test_ds, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=_pin,
        persistent_workers=_persist,
    )

    # ── Stage 1: train x → c
    print("\n=== Stage 1: concept encoder (x → c) ===")
    encoder  = ConceptEncoder(concept_names)
    encoder  = train_stage1(encoder, train_loader, val_loader, CFG)

    concept_acc = evaluate_concept_accuracy(encoder, val_loader, CFG["device"])
    print(f"Stage 1 val concept accuracy: {concept_acc:.4f}  "
          f"(Koh reports ~0.93 for independent)")

    # ── Extract and cache concept scores
    print("\nExtracting concept scores ...")
    c_train, y_train = extract_concept_scores(encoder, train_loader, CFG["device"])
    c_val,   y_val   = extract_concept_scores(encoder, val_loader,   CFG["device"])
    c_test,  y_test  = extract_concept_scores(encoder, test_loader,  CFG["device"])
    print(f"  train: {c_train.shape}, val: {c_val.shape}, test: {c_test.shape}")

    # ── Stage 2: train c → y (frozen concepts)
    print("\n=== Stage 2: label predictor (c → y) ===")
    predictor = LabelPredictor(CFG["n_concepts"], CFG["n_classes"])
    predictor = train_stage2(
        predictor, c_train, y_train, c_val, y_val, CFG
    )

    task_acc = evaluate_task_accuracy(
        predictor, c_test, y_test, CFG["device"]
    )
    print(f"Stage 2 test task accuracy:  {task_acc:.4f}  "
          f"(Koh reports ~72-75% for independent)")

    # ── Save BQ artifacts
    save_bq_artifacts(encoder, predictor, c_test, y_test, CFG)


if __name__ == "__main__":
    main()
