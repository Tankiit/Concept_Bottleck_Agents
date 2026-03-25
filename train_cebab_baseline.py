import os
import argparse
import time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

from text_cache_loader import load_text_dataset_cached


def pick_device(prefer: str | None = None) -> str:
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    if prefer == "mps" and torch.backends.mps.is_available():
        return "mps"
    # Auto
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_latent_cache(latent_cache_dir: str, batch_size: int = 16):
    """
    Load pre-computed embeddings from latent_cache directory.

    Expects files named like:
    - cebab_low_rank_rank128_svd_train_latents.npy
    - cebab_low_rank_rank128_svd_train_labels.npy
    - cebab_low_rank_rank128_svd_train_concepts.npy (optional)
    - cebab_low_rank_rank128_svd_train_is_unknown.npy (optional)
    And similar for val and test splits.
    """
    latent_cache_dir = Path(latent_cache_dir)

    # Find the CEBaB latent files
    pattern = "cebab_low_rank_rank128_svd"  # As seen in the cache directory

    def load_split(split: str):
        latents = np.load(latent_cache_dir / f"{pattern}_{split}_latents.npy")
        labels = np.load(latent_cache_dir / f"{pattern}_{split}_labels.npy")

        # Load optional files
        concepts_path = latent_cache_dir / f"{pattern}_{split}_concepts.npy"
        unknown_path = latent_cache_dir / f"{pattern}_{split}_is_unknown.npy"

        # Convert to tensors
        latents_tensor = torch.from_numpy(latents).float()
        labels_tensor = torch.from_numpy(labels).long()

        tensors = [latents_tensor, labels_tensor]

        # Load concepts and unknown flags if available
        if concepts_path.exists():
            concepts = np.load(concepts_path)
            tensors.append(torch.from_numpy(concepts).long())
        if unknown_path.exists():
            unknown = np.load(unknown_path)
            tensors.append(torch.from_numpy(unknown).float())

        return TensorDataset(*tensors)

    train_ds = load_split("train")
    val_ds = load_split("val")
    test_ds = load_split("test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Load metadata
    metadata_path = latent_cache_dir / f"{pattern}_metadata.json"
    metadata = {}
    if metadata_path.exists():
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)

    return train_loader, val_loader, test_loader, metadata


def to_device(batch, device: str, use_latent: bool = False):
    """
    Move batch to device.

    For latent mode: batch is (latents, labels, [concepts], [is_unknown])
    For text mode: batch is (input_ids, attention_mask, labels, [concepts], [is_unknown])
    """
    if use_latent:
        latents = batch[0].to(device, non_blocking=True)
        labels = batch[1].to(device, non_blocking=True)
        return latents, labels
    else:
        input_ids = batch[0].to(device, non_blocking=True)
        attention_mask = batch[1].to(device, non_blocking=True)
        labels = batch[2].to(device, non_blocking=True)
        return input_ids, attention_mask, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", type=str, default="/Users/tanmoy/research/Credal_Sets/Wasserstein/data_cache")
    p.add_argument("--latent_cache_dir", type=str, default="/Users/tanmoy/research/Credal_Sets/Wasserstein/latent_cache/bert_cebab",
                   help="Path to latent cache with pre-computed embeddings")
    p.add_argument("--use_latent_cache", action="store_true", help="Use pre-computed embeddings from latent_cache_dir")
    p.add_argument("--tokenizer", type=str, default="distilbert-base-uncased")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints_text")
    # Perf/PEFT options
    p.add_argument("--freeze_backbone", action="store_true", help="Freeze transformer and train only classifier head")
    p.add_argument("--train_last_k", type=int, default=0, help="Unfreeze last K transformer layers (overrides freeze when >0)")
    p.add_argument("--amp", action="store_true", help="Enable AMP (CUDA preferred; experimental on MPS)")
    # LoRA options
    p.add_argument("--lora", action="store_true", help="Enable LoRA adapters on attention proj layers")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_targets", type=str, default="qv", help="Which attention linears to adapt: qv|qkv|all")
    args = p.parse_args()

    prefer = args.device or ("cuda" if args.gpu else None)
    device = pick_device(prefer)

    # Load cached loaders (text or latent)
    if args.use_latent_cache:
        print(f"Loading pre-computed embeddings from: {args.latent_cache_dir}")
        tl, vl, te, meta = load_latent_cache(
            latent_cache_dir=args.latent_cache_dir,
            batch_size=args.batch_size,
        )
        latent_dim = 128  # Low-rank SVD rank
        print(f"Using latent cache: {meta.get('extraction_config', {}) if meta else 'No metadata'}")
    else:
        tl, vl, te, _, meta = load_text_dataset_cached(
            dataset_name="cebab",
            cache_dir=args.cache_dir,
            tokenizer_name=args.tokenizer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_workers=0,
        )
        print(f"Using text cache: {meta}")

    print(f"Device: {device}")
    train_size = len(tl.dataset)
    val_size = len(vl.dataset)
    test_size = len(te.dataset)
    print(f"Sizes: train={train_size}, val={val_size}, test={test_size}, batch_size={args.batch_size}")

    # Model: DistilBERT for text or simple MLP for latent embeddings
    if args.use_latent_cache:
        # Simple MLP classifier for pre-computed embeddings
        model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3)
        )
        print("Using MLP classifier on frozen embeddings")
    else:
        # Full DistilBERT for text
        from transformers import DistilBertForSequenceClassification
        try:
            model = DistilBertForSequenceClassification.from_pretrained(
                args.tokenizer, num_labels=3
            )
        except Exception:
            # Fallback to base model name if tokenizer is different
            model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=3
            )
        print("Using DistilBERT for text classification")
    model.to(device)

    # Skip LoRA and BERT-specific options when using latent cache
    if args.use_latent_cache:
        if args.lora or args.freeze_backbone or args.train_last_k > 0:
            print("Warning: --lora, --freeze_backbone, --train_last_k are ignored when using latent cache")
    else:
        # --- LoRA injection (manual, no external deps) ---
        class LoRALinear(nn.Module):
            def __init__(self, base: nn.Linear, r: int, alpha: float):
                super().__init__()
                self.base = base
                for p in self.base.parameters():
                    p.requires_grad = False
                in_f, out_f = base.in_features, base.out_features
                self.r = r
                self.scale = alpha / max(1, r)
                # A: in->r (init Kaiming), B: r->out (zeros so initial delta=0)
                self.lora_A = nn.Linear(in_f, r, bias=False)
                self.lora_B = nn.Linear(r, out_f, bias=False)
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B.weight)

            def forward(self, x):
                return self.base(x) + self.scale * self.lora_B(self.lora_A(x))

        def replace_module(parent, name: str, new_module: nn.Module):
            setattr(parent, name, new_module)

        def iter_parent_modules(model: nn.Module):
            for name, module in model.named_modules():
                for child_name, child in module.named_children():
                    yield module, child_name, child, f"{name}.{child_name}" if name else child_name

        import math
        lora_applied = 0
        if args.lora:
            targets = args.lora_targets.lower()
            def want(name: str) -> bool:
                if targets == "qv":
                    return name.endswith("q_lin") or name.endswith("v_lin")
                if targets == "qkv":
                    return name.endswith("q_lin") or name.endswith("k_lin") or name.endswith("v_lin")
                return name.endswith("q_lin") or name.endswith("k_lin") or name.endswith("v_lin") or name.endswith("out_lin")

            for parent, child_name, child, fqname in iter_parent_modules(model):
                if isinstance(child, nn.Linear) and want(child_name) and \
                   ("distilbert.transformer.layer" in fqname and ".attention." in fqname):
                    lora = LoRALinear(child, r=args.lora_rank, alpha=args.lora_alpha)
                    replace_module(parent, child_name, lora)
                    lora_applied += 1
            print(f"LoRA applied to {lora_applied} attention linears (rank={args.lora_rank}, alpha={args.lora_alpha}).")
            # Move new parameters to device
            model.to(device)

        # Optionally freeze backbone / unfreeze last K layers
        if args.freeze_backbone and args.train_last_k <= 0:
            for n, p in model.named_parameters():
                if not n.startswith("classifier"):
                    p.requires_grad = False
            print("Backbone frozen; training classifier head only.")
        if args.train_last_k > 0:
            # First freeze all except classifier
            for n, p in model.named_parameters():
                if not n.startswith("classifier"):
                    p.requires_grad = False
            # Then unfreeze last K transformer layers
            k = args.train_last_k
            layers = getattr(model.distilbert.transformer, "layer", [])
            if layers:
                for i in range(max(0, len(layers)-k), len(layers)):
                    for p in layers[i].parameters():
                        p.requires_grad = True
                print(f"Unfrozen last {k} transformer layers.")
            else:
                print("Warning: could not access transformer layers for partial unfreeze.")

        # If LoRA enabled, ensure only classifier and LoRA params are trainable unless overridden by train_last_k
        if args.lora and args.train_last_k == 0 and not args.freeze_backbone:
            for n, p in model.named_parameters():
                if not (n.startswith("classifier") or ".lora_" in n):
                    p.requires_grad = False
            print("Frozen backbone except classifier and LoRA params.")

    # Report trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100.0*trainable/total:.2f}%)")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-6
    )

    best_val_acc = 0.0
    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "cebab_distilbert_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        t0 = time.time()
        scaler = None
        use_amp = args.amp and (device == "cuda" or device == "mps")
        if use_amp and device == "cuda":
            from torch.cuda.amp import GradScaler, autocast
            scaler = GradScaler()
        elif use_amp and device == "mps":
            from torch.amp import autocast

        for batch in tqdm(tl, desc=f"Train E{epoch}", leave=False):
            batch_data = to_device(batch, device, use_latent=args.use_latent_cache)
            optimizer.zero_grad(set_to_none=True)

            if args.use_latent_cache:
                # Latent mode: batch_data is (latents, labels)
                latents, labels = batch_data
                logits = model(latents)
                loss = nn.functional.cross_entropy(logits, labels)
            else:
                # Text mode: batch_data is (input_ids, attention_mask, labels)
                input_ids, attention_mask, labels = batch_data

            if use_amp and device == "cuda":
                with autocast():
                    if args.use_latent_cache:
                        loss = nn.functional.cross_entropy(model(latents), labels)
                    else:
                        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = out.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            elif use_amp and device == "mps":
                # Experimental: MPS autocast may not accelerate all ops
                with autocast(device_type="mps", dtype=torch.float16):
                    if args.use_latent_cache:
                        loss = nn.functional.cross_entropy(model(latents), labels)
                    else:
                        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = out.loss
                loss.backward()
                optimizer.step()
            else:
                if args.use_latent_cache:
                    loss = nn.functional.cross_entropy(model(latents), labels)
                else:
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = out.loss
                loss.backward()
                optimizer.step()
            running_loss += float(loss.detach().cpu())
            n_batches += 1
        train_loss = running_loss / max(1, n_batches)
        t1 = time.time()
        train_time = t1 - t0

        # Validate
        model.eval()
        correct = 0
        total = 0
        vloss = 0.0
        with torch.no_grad():
            v0 = time.time()
            for batch in tqdm(vl, desc=f"Val   E{epoch}", leave=False):
                batch_data = to_device(batch, device, use_latent=args.use_latent_cache)

                if args.use_latent_cache:
                    latents, labels = batch_data
                    logits = model(latents)
                    loss = nn.functional.cross_entropy(logits, labels, reduction='sum')
                    vloss += float(loss)
                    preds = logits.argmax(dim=1)
                else:
                    input_ids, attention_mask, labels = batch_data
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    logits = out.logits
                    vloss += float(out.loss.detach().cpu() * labels.numel())  # Scale to match sum reduction
                    preds = logits.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.numel()
        val_loss = vloss / max(1, total)  # Average per sample
        val_acc = correct / max(1, total)
        scheduler.step(val_acc)

        v1 = time.time()
        val_time = v1 - v0
        tput = train_size / train_time if train_time > 0 else float('inf')
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}\n"
            f"  Timing: train {train_time:.1f}s ({tput:.1f} ex/s) | val {val_time:.1f}s"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model": model.state_dict(),
                "val_acc": best_val_acc,
                "epoch": epoch,
            }, best_path)

    print(f"Best val acc: {best_val_acc:.4f}")

    # Final test evaluation with best model
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(te, desc="Test", leave=False):
            batch_data = to_device(batch, device, use_latent=args.use_latent_cache)

            if args.use_latent_cache:
                latents, labels = batch_data
                logits = model(latents)
            else:
                input_ids, attention_mask, labels = batch_data
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    test_acc = correct / max(1, total)
    print(f"Test acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
