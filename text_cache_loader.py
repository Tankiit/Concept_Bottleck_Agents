"""
Cached text dataset loader integrating external Wasserstein dataloader.

- Uses /Users/tanmoy/research/Credal_Sets/Wasserstein/dataloader.py for parsing
  CEBaB and other datasets consistently.
- Builds tokenized tensors once and writes them under a cache directory
  (e.g., /Users/tanmoy/research/Credal_Sets/Wasserstein/data_cache).
- Subsequent runs load pre-tokenized tensors directly for fast startup.

This module does not change training loops; it only provides ready-made
PyTorch DataLoaders with tensors.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Any

import torch
from torch.utils.data import TensorDataset, DataLoader


def _resolve_wasserstein_path(default: str | None = None) -> str:
    path = (
        os.environ.get("WASSERSTEIN_DIR")
        or default
        or "/Users/tanmoy/research/Credal_Sets/Wasserstein"
    )
    if path not in sys.path:
        sys.path.append(path)
    return path


def _build_cache_key(dataset: str, tokenizer_name: str, max_length: int) -> str:
    safe_tok = tokenizer_name.replace("/", "_")
    return f"{dataset}__{safe_tok}__L{max_length}"


def _cache_paths(cache_dir: str, key: str) -> Dict[str, Path]:
    base = Path(cache_dir)
    base.mkdir(parents=True, exist_ok=True)
    return {
        split: base / f"{key}_{split}.pt"
        for split in ("train", "val", "test")
    }


def _materialize_from_loader(loader: DataLoader) -> Dict[str, torch.Tensor]:
    inputs, masks, labels = [], [], []
    concepts, unknowns = [], []

    for batch in loader:
        inputs.append(batch["input_ids"])              # [B, L]
        masks.append(batch["attention_mask"])          # [B, L]
        labels.append(batch["labels"])                 # [B]
        if "concept_labels" in batch:
            concepts.append(batch["concept_labels"])   # [B, K]
        if "is_unknown" in batch:
            unknowns.append(batch["is_unknown"])       # [B, K]

    data = {
        "input_ids": torch.cat(inputs, dim=0),
        "attention_mask": torch.cat(masks, dim=0),
        "labels": torch.cat(labels, dim=0),
    }
    if concepts:
        data["concept_labels"] = torch.cat(concepts, dim=0)
    if unknowns:
        data["is_unknown"] = torch.cat(unknowns, dim=0)
    return data


def _tensor_ds_from_payload(payload: Dict[str, torch.Tensor]) -> TensorDataset:
    keys = ["input_ids", "attention_mask", "labels"]
    opt_keys = ["concept_labels", "is_unknown"]
    tensors = [payload[k] for k in keys]
    for k in opt_keys:
        if k in payload:
            tensors.append(payload[k])
    return TensorDataset(*tensors)


def load_text_dataset_cached(
    dataset_name: str,
    cache_dir: str = "/Users/tanmoy/research/Credal_Sets/Wasserstein/data_cache",
    tokenizer_name: str = "distilbert-base-uncased",
    max_length: int = 128,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Any, Dict]:
    """
    Load dataset via external Wasserstein dataloader, with persistent cache.

    - First run: builds tokenized tensors and saves under cache_dir.
    - Later runs: loads tensors directly and returns simple DataLoaders.
    """
    # Import external dataloader
    _resolve_wasserstein_path()
    from dataloader import (
        load_dataset_splits as wz_load_splits,
        DatasetConfig as WzConfig,
    )

    key = _build_cache_key(dataset_name, tokenizer_name, max_length)
    paths = _cache_paths(cache_dir, key)

    # If all caches exist, load and return fast loaders
    if all(p.exists() for p in paths.values()):
        payloads = {split: torch.load(path, map_location="cpu") for split, path in paths.items()}
        train_ds = _tensor_ds_from_payload(payloads["train"])
        val_ds = _tensor_ds_from_payload(payloads["val"])
        test_ds = _tensor_ds_from_payload(payloads["test"])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        metadata = {"cached": True, "cache_key": key, "cache_dir": cache_dir}
        return train_loader, val_loader, test_loader, None, metadata

    # Else, build from source loaders with deferred tokenization for speed
    use_batched_collate = dataset_name in ("snli", "chaosnli")
    cfg = WzConfig(
        label_type="ternary" if dataset_name == "cebab" else "binary",
        max_length=max_length,
        tokenizer_name=tokenizer_name,
        batch_size=batch_size,
        num_workers=0,  # keep collate picklable
        defer_tokenization=use_batched_collate,
    )
    train_loader, val_loader, test_loader, tokenizer, meta = wz_load_splits(dataset_name, config=cfg)

    # Materialize tensors and save
    payloads = {
        "train": _materialize_from_loader(train_loader),
        "val": _materialize_from_loader(val_loader),
        "test": _materialize_from_loader(test_loader),
    }
    for split, path in paths.items():
        torch.save(payloads[split], path)

    # Return loaders backed by tensor datasets
    train_ds = _tensor_ds_from_payload(payloads["train"])
    val_ds = _tensor_ds_from_payload(payloads["val"])
    test_ds = _tensor_ds_from_payload(payloads["test"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    meta = meta or {}
    meta.update({"cached": False, "cache_key": key, "cache_dir": cache_dir})
    return train_loader, val_loader, test_loader, tokenizer, meta


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=[
        "cebab", "sst2", "sst5", "imdb", "yelp", "hatexplain", "civil_comments",
        "goemotions", "chaosnli", "tid8", "snli"
    ])
    p.add_argument("--cache_dir", default="/Users/tanmoy/research/Credal_Sets/Wasserstein/data_cache")
    p.add_argument("--tokenizer", default="distilbert-base-uncased")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    args = p.parse_args()

    tl, vl, te, tok, meta = load_text_dataset_cached(
        args.dataset, cache_dir=args.cache_dir, tokenizer_name=args.tokenizer,
        max_length=args.max_length, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(f"Cached loaders ready: {meta}")
    for name, loader in (("train", tl), ("val", vl), ("test", te)):
        try:
            b = next(iter(loader))
            if isinstance(b, (list, tuple)):
                shapes = [tuple(getattr(t, 'shape', ())) for t in b]
                print(f"{name}: {shapes}")
            elif isinstance(b, dict):
                shapes = {k: tuple(getattr(v, 'shape', ())) for k, v in b.items()}
                print(f"{name}: {shapes}")
            else:
                print(f"{name}: batch type {type(b)}")
        except StopIteration:
            print(f"{name}: empty loader")
