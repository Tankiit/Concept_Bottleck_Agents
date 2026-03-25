"""
train_koh_pyc.py
================
Trains Koh et al. independent CBM on CUB-200-2011.
Uses cub_loader.py - a minimal direct pkl reader that correctly
handles the cluster-absolute path issue.

Two-stage independent training:
  Stage 1: x -> c  (ResNet18/50 backbone + Linear concept head)
  Stage 2: c -> y  (frozen concept scores -> Linear label predictor)

Produces:
  checkpoints/encoder_best.pth
  checkpoints/predictor_best.pth
  artifacts/W_concepts.npy      (112, d)
  artifacts/W_labels.npy        (200, 112)
  artifacts/c_test.npy          (N_test, 112)
  artifacts/y_test.npy          (N_test,)
  artifacts/concept_names.json
  artifacts/concept_group_map.json

Usage:
  python train.py                              # ResNet18
  BACKBONE=resnet50 python train.py            # ResNet50
  python train.py --resume checkpoints/encoder_epoch_20.pth
  python train.py --stage2-only
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
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


CFG = dict(
    cub_root=os.environ.get(
        "CUB_DIR", "/Users/tanmoy/research/data/CUB_200_2011/CUB_200_2011"
    ),
    backbone=os.environ.get("BACKBONE", "resnet18"),
    checkpoint_dir="./checkpoints",
    artifact_dir="./artifacts",
    stage1_warmup=10,
    stage1_epochs=100,
    stage1_lr={"resnet18": 0.01, "resnet50": 0.005},
    stage1_wd=4e-5,
    stage1_momentum=0.9,
    stage1_patience=5,
    stage1_min_lr=1e-5,
    stage1_early_stop=10,
    stage2_epochs=200,
    stage2_lr=0.01,
    stage2_wd=4e-5,
    stage2_momentum=0.9,
    batch_size=64,
    n_concepts=N_CONCEPTS,
    n_classes=N_CLASSES,
    num_workers=4,
    aux_loss_weight=0.4,
    seed=42,
    save_freq=10,
    device=(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    ),
)


class ConceptEncoder(nn.Module):
    """ResNet backbone + linear concept head."""

    def __init__(self, backbone: str, n_concepts: int):
        super().__init__()
        self._bname = backbone
        if backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            d = base.fc.in_features
        elif backbone == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            d = base.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}.")
        base.fc = nn.Identity()
        self.backbone = base
        self.head = nn.Linear(d, n_concepts)

    def forward(self, x):
        return self.head(self.backbone(x)), None

    def freeze_backbone(self, freeze: bool = True):
        for p in self.backbone.parameters():
            p.requires_grad_(not freeze)

    def get_W_concepts(self) -> np.ndarray:
        return self.head.weight.detach().cpu().numpy()


class LabelPredictor(nn.Module):
    """Single linear layer c -> y."""

    def __init__(self, n_concepts: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(n_concepts, n_classes)

    def forward(self, c):
        return self.fc(c)

    def get_W_labels(self) -> np.ndarray:
        return self.fc.weight.detach().cpu().numpy()


def _make_sgd(model, lr, cfg):
    return torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, momentum=cfg["stage1_momentum"], weight_decay=cfg["stage1_wd"],
    )


def _make_scheduler(opt, cfg):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.1,
        patience=cfg["stage1_patience"], min_lr=cfg["stage1_min_lr"],
    )


def _save(model, opt, sched, epoch, tr, vl, path):
    key = "encoder_state_dict" if isinstance(model, ConceptEncoder) else "predictor_state_dict"
    torch.save({
        "epoch": epoch + 1,
        key: model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
        "train_loss": tr,
        "val_loss": vl,
    }, path)


def _load(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    key = "encoder_state_dict" if isinstance(model, ConceptEncoder) else "predictor_state_dict"
    model.load_state_dict(ckpt[key] if key in ckpt else ckpt)


def train_stage1(encoder, train_loader, val_loader, cfg):
    device = cfg["device"]
    encoder = encoder.to(device)
    warmup = cfg["stage1_warmup"]
    base_lr = cfg["stage1_lr"].get(encoder._bname, 0.01)

    pos_w = torch.tensor(
        train_loader.dataset.concept_weights(), dtype=torch.float32
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    encoder.freeze_backbone(True)
    optimizer = _make_sgd(encoder, base_lr, cfg)
    scheduler = _make_scheduler(optimizer, cfg)

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    save_path = os.path.join(cfg["checkpoint_dir"], "encoder_best.pth")
    best_loss = float("inf")
    no_improve = 0
    phase_b = False

    print(f"  Backbone: {encoder._bname}  | LR: {base_lr}  | Device: {device}")
    print(f"  Phase A: backbone frozen for {warmup} warmup epochs")

    for epoch in range(cfg["stage1_epochs"]):
        if epoch == warmup and not phase_b:
            print("  Phase B: unfreezing backbone (LR x 0.1)")
            encoder.freeze_backbone(False)
            optimizer = _make_sgd(encoder, base_lr * 0.1, cfg)
            scheduler = _make_scheduler(optimizer, cfg)
            phase_b = True

        encoder.train()
        tr_loss = 0.0
        for imgs, concepts, _ in tqdm(train_loader, desc=f"S1 E{epoch+1}", leave=False):
            imgs, concepts = imgs.to(device), concepts.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = encoder(imgs)
            loss = criterion(logits, concepts)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        encoder.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for imgs, concepts, _ in val_loader:
                imgs, concepts = imgs.to(device), concepts.to(device)
                logits, _ = encoder(imgs)
                vl_loss += criterion(logits, concepts).item()
        vl_loss /= len(val_loader)
        scheduler.step(vl_loss)

        phase = "A" if epoch < warmup else "B"
        lr = optimizer.param_groups[0]["lr"]
        print(f"  S1 [{epoch+1:3d}] {phase}  train={tr_loss:.4f}  val={vl_loss:.4f}  lr={lr:.1e}")

        if vl_loss < best_loss:
            best_loss = vl_loss
            no_improve = 0
            _save(encoder, optimizer, scheduler, epoch, tr_loss, vl_loss, save_path)
        else:
            no_improve += 1
            if no_improve >= cfg["stage1_early_stop"] and phase_b:
                print(f"  Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % cfg["save_freq"] == 0:
            ep_path = os.path.join(cfg["checkpoint_dir"], f"encoder_epoch_{epoch+1}.pth")
            _save(encoder, optimizer, scheduler, epoch, tr_loss, vl_loss, ep_path)

    _load(encoder, save_path, device)
    print(f"Stage 1 complete. Best val loss: {best_loss:.4f}")
    return encoder


def extract_scores(encoder, loader, device):
    """Run frozen encoder -> (N, 112) float32 concept activations."""
    encoder.eval()
    all_c, all_y = [], []
    with torch.no_grad():
        for imgs, _, labels in tqdm(loader, desc="Extract", leave=False):
            logits, _ = encoder(imgs.to(device))
            all_c.append(torch.sigmoid(logits).cpu())
            all_y.append(labels if isinstance(labels, torch.Tensor) else torch.tensor(labels))
    return (
        torch.cat(all_c).numpy().astype(np.float32),
        torch.cat(all_y).numpy().astype(np.int64),
    )


def train_stage2(predictor, c_train, y_train, c_val, y_val, cfg):
    device = cfg["device"]
    predictor = predictor.to(device)

    xtr = torch.tensor(c_train).to(device)
    ytr = torch.tensor(y_train, dtype=torch.long).to(device)
    xvl = torch.tensor(c_val).to(device)
    yvl = torch.tensor(y_val, dtype=torch.long).to(device)

    opt = torch.optim.SGD(
        predictor.parameters(), lr=cfg["stage2_lr"],
        momentum=cfg["stage2_momentum"], weight_decay=cfg["stage2_wd"],
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.1,
        patience=cfg["stage1_patience"], min_lr=cfg["stage1_min_lr"],
    )
    crit = nn.CrossEntropyLoss()
    save_path = os.path.join(cfg["checkpoint_dir"], "predictor_best.pth")
    best_val_acc = 0.0

    for epoch in range(cfg["stage2_epochs"]):
        predictor.train()
        opt.zero_grad()
        loss = crit(predictor(xtr), ytr)
        loss.backward()
        opt.step()

        predictor.eval()
        with torch.no_grad():
            val_acc = (predictor(xvl).argmax(1) == yvl).float().mean().item()
        sched.step(val_acc)

        if (epoch + 1) % 20 == 0:
            lr = opt.param_groups[0]["lr"]
            print(f"  S2 [{epoch+1:3d}]  loss={loss.item():.4f}  "
                  f"val_acc={val_acc:.4f}  best={best_val_acc:.4f}  lr={lr:.1e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            _save(predictor, opt, sched, epoch, loss.item(), val_acc, save_path)

        if (epoch + 1) % cfg["save_freq"] == 0:
            ep_path = os.path.join(cfg["checkpoint_dir"], f"predictor_epoch_{epoch+1}.pth")
            _save(predictor, opt, sched, epoch, loss.item(), val_acc, ep_path)

    _load(predictor, save_path, device)
    print(f"Stage 2 complete. Best val acc: {best_val_acc:.4f}")
    return predictor


def save_artifacts(encoder, predictor, c_test, y_test, cfg):
    d = cfg["artifact_dir"]
    os.makedirs(d, exist_ok=True)

    w_concepts = encoder.get_W_concepts()
    w_labels = predictor.get_W_labels()

    np.save(os.path.join(d, "W_concepts.npy"), w_concepts)
    np.save(os.path.join(d, "W_labels.npy"), w_labels)
    np.save(os.path.join(d, "c_test.npy"), c_test)
    np.save(os.path.join(d, "y_test.npy"), y_test)

    with open(os.path.join(d, "concept_names.json"), "w") as f:
        json.dump(CONCEPT_NAMES, f, indent=2)
    group_map = {k: list(v) for k, v in CONCEPT_GROUP_MAP.items()}
    with open(os.path.join(d, "concept_group_map.json"), "w") as f:
        json.dump(group_map, f, indent=2)

    print(f"\nArtifacts saved to {d}/")
    print(f"  W_concepts: {w_concepts.shape}")
    print(f"  W_labels:   {w_labels.shape}")
    print(f"  c_test:     {c_test.shape}")
    norms = np.linalg.norm(w_concepts, axis=1)
    print(f"  W_concepts norms: mean={norms.mean():.3f}  std={norms.std():.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str)
    parser.add_argument("--stage2-only", action="store_true")
    parser.add_argument("--backbone", default=None, choices=["resnet18", "resnet50"])
    args = parser.parse_args()

    if args.backbone:
        CFG["backbone"] = args.backbone

    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    print(f"Device:   {CFG['device']}")
    print(f"Backbone: {CFG['backbone']}")
    print(f"CUB root: {CFG['cub_root']}")

    print("\nLoading datasets ...")
    pin = CFG["device"] == "cuda"
    nw = CFG["num_workers"]
    kw = dict(num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))

    train_ds = CUBDataset("train", CFG["cub_root"])
    val_ds   = CUBDataset("val",   CFG["cub_root"])
    test_ds  = CUBDataset("test",  CFG["cub_root"])

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"], shuffle=False, **kw)

    batch = next(iter(train_loader))
    imgs, concepts, labels = batch
    assert imgs.shape == (CFG["batch_size"], 3, 224, 224), f"Unexpected: {imgs.shape}"
    assert concepts.shape[1] == N_CONCEPTS, f"Unexpected concept dim: {concepts.shape}"
    print(f"Sanity OK: imgs={imgs.shape}, concepts={concepts.shape}, labels={labels.shape}")
    assert imgs.abs().mean() > 0.01, "Images appear black -- check CUB_DIR!"
    print(f"Image mean abs: {imgs.abs().mean():.3f}  (>0.01 = OK)")

    encoder = ConceptEncoder(CFG["backbone"], CFG["n_concepts"])

    if not args.stage2_only:
        print("\n=== Stage 1: concept encoder ===")
        if args.resume:
            print(f"Resuming from {args.resume}")
            _load(encoder, args.resume, CFG["device"])
        encoder = train_stage1(encoder, train_loader, val_loader, CFG)
    else:
        enc_path = os.path.join(CFG["checkpoint_dir"], "encoder_best.pth")
        print(f"\nStage 2 only -- loading encoder from {enc_path}")
        _load(encoder, enc_path, CFG["device"])
    encoder = encoder.to(CFG["device"])

    print("\nExtracting concept scores ...")
    c_train, y_train = extract_scores(encoder, train_loader, CFG["device"])
    c_val,   y_val   = extract_scores(encoder, val_loader,   CFG["device"])
    c_test,  y_test  = extract_scores(encoder, test_loader,  CFG["device"])
    print(f"  train={c_train.shape}  val={c_val.shape}  test={c_test.shape}")

    concept_acc = (
        (c_val > 0.5).astype(float) == np.stack([it["concepts"] for it in val_ds.items])
    ).mean()
    print(f"  Val concept accuracy: {concept_acc:.4f}  (Koh independent ~0.93)")

    print("\n=== Stage 2: label predictor ===")
    predictor = LabelPredictor(CFG["n_concepts"], CFG["n_classes"])
    predictor = train_stage2(predictor, c_train, y_train, c_val, y_val, CFG)

    predictor.eval()
    with torch.no_grad():
        logits = predictor(torch.tensor(c_test).to(CFG["device"]))
        test_acc = (logits.argmax(1).cpu().numpy() == y_test).mean()
    print(f"\nTest task accuracy: {test_acc:.4f}  (Koh ~0.72-0.75; ResNet18 ~0.65-0.70)")

    save_artifacts(encoder, predictor, c_test, y_test, CFG)


if __name__ == "__main__":
    main()
