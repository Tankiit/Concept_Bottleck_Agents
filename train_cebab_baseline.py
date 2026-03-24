import os
import argparse
import time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def to_device(batch, device: str):
    # batch is a tuple from TensorDataset:
    # (input_ids, attention_mask, labels, [concept_labels], [is_unknown])
    input_ids = batch[0].to(device, non_blocking=True)
    attention_mask = batch[1].to(device, non_blocking=True)
    labels = batch[2].to(device, non_blocking=True)
    return input_ids, attention_mask, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", type=str, default="/Users/tanmoy/research/Credal_Sets/Wasserstein/data_cache")
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
    args = p.parse_args()

    prefer = args.device or ("cuda" if args.gpu else None)
    device = pick_device(prefer)

    # Load cached loaders
    tl, vl, te, _, meta = load_text_dataset_cached(
        dataset_name="cebab",
        cache_dir=args.cache_dir,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=0,
    )

    print(f"Using cache: {meta}")
    print(f"Device: {device}")
    train_size = len(tl.dataset)
    val_size = len(vl.dataset)
    test_size = len(te.dataset)
    print(f"Sizes: train={train_size}, val={val_size}, test={test_size}, batch_size={args.batch_size}")

    # Model: DistilBERT (3-way classification)
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

    # Report trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100.0*trainable/total:.2f}%)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
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
            input_ids, attention_mask, labels = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if use_amp and device == "cuda":
                with autocast():
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = out.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            elif use_amp and device == "mps":
                # Experimental: MPS autocast may not accelerate all ops
                with autocast(device_type="mps", dtype=torch.float16):
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = out.loss
                loss.backward()
                optimizer.step()
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
                input_ids, attention_mask, labels = to_device(batch, device)
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = out.logits
                vloss += float(out.loss.detach().cpu())
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
        val_loss = vloss / max(1, len(vl))
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
            input_ids, attention_mask, labels = to_device(batch, device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    test_acc = correct / max(1, total)
    print(f"Test acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
