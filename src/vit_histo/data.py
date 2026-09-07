"""Data pipeline for the breast-histopathology (IDC) patch dataset.

Returns (train_loader, val_loader, processor) for ViT fine-tuning.

Contract used by train.py:
    train_loader, val_loader, processor = build_dataloaders(cfg)
Each batch is a dict: {"pixel_values": (B,3,224,224) float, "labels": (B,) long}
— exactly what ViTForImageClassification expects.
"""

from __future__ import annotations

import torch
from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor


def _subsample(split, n: int | None, seed: int):
    """Shuffle then take the first `n` rows. If n is None, return the whole split."""
    if n is None:
        return split
    return split.shuffle(seed=seed).select(range(min(n, len(split))))



def _build_transforms(processor):
    """Return (train_tf, val_tf).
    Goal: 
    train_tf: flips + 90-degree rotation, then resize/tensor/normalize
    val_tf:   resize/tensor/normalize only (deterministic)
    Both use processor.image_mean / processor.image_std for normalization.
    """
    mean, std = processor.image_mean, processor.image_std
    size = (processor.size["height"], processor.size["width"])   # (224, 224)

    val_tf = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),         
        transforms.RandomVerticalFlip(),
        transforms.RandomChoice([                   
            transforms.RandomRotation((0, 0)),
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270)),
        ]),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_tf, val_tf


def _make_hf_transform(torch_tf, image_key: str, label_key: str):
    """Wrap a torchvision transform into a HF with_transform callable.

    Receives a batch dict (each column is a list) and returns
    {"pixel_values": [tensors...], "labels": [ints...]}.
    The .convert("RGB") is a no-op for RGB images but guards against any
    grayscale/RGBA patches that would otherwise yield the wrong channel count.
    """
    def _apply(batch):
        pixel_values = [torch_tf(img.convert("RGB")) for img in batch[image_key]]
        return {"pixel_values": pixel_values, "labels": batch[label_key]}
    return _apply


def _collate(examples):
    """Stack a list of {'pixel_values','labels'} dicts into batched tensors."""
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    labels = torch.tensor([e["labels"] for e in examples], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}


def build_dataloaders(cfg: DictConfig):
    """Return (train_loader, val_loader, test_loader, image_processor).

    The dataset ships with train / validation / test splits, so we load all
    three. Test is built here but MUST NOT be touched during tuning — it's
    only for the final Phase-2 evaluation.
    """
    # load
    ds = load_dataset(cfg.data.dataset_name)

    # subsample each split (test uses its own subset size)
    train_split = _subsample(ds["train"], cfg.data.train_subset, cfg.seed)
    val_split = _subsample(ds["validation"], cfg.data.val_subset, cfg.seed)
    test_split = _subsample(ds["test"], cfg.data.get("test_subset", None), cfg.seed)

    # processor
    processor = AutoImageProcessor.from_pretrained(cfg.model.name)

    # transforms (train augmented; val AND test deterministic)
    train_tf, val_tf = _build_transforms(processor)

    # attach transforms
    image_key, label_key = "image", "label"
    train_split = train_split.with_transform(
        _make_hf_transform(train_tf, image_key, label_key)
    )
    val_split = val_split.with_transform(
        _make_hf_transform(val_tf, image_key, label_key)
    )
    test_split = test_split.with_transform(
        _make_hf_transform(val_tf, image_key, label_key)   # deterministic, like val
    )

    # dataloaders
    train_loader = DataLoader(
        train_split,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_split,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=_collate,
    )
    test_loader = DataLoader(
        test_split,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=_collate,
    )

    return train_loader, val_loader, test_loader, processor