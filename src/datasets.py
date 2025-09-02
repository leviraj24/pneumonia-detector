# src/datasets.py
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def _build_transforms(img_size=224):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        # ImageNet mean/std (good for ImageNet-pretrained backbones)
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tfms, val_tfms

def get_dataloaders(
    data_dir="data",
    batch_size=32,
    num_workers=2,
    img_size=224,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
):
    """
    Expects:
        data/
          train/NORMAL, train/PNEUMONIA
          val/NORMAL,   val/PNEUMONIA
    """
    train_tfms, val_tfms = _build_transforms(img_size)

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfms)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=val_tfms)

    # Windows note: persistent_workers requires num_workers>0
    can_persist = persistent_workers and num_workers > 0

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=can_persist, prefetch_factor=prefetch_factor
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=can_persist, prefetch_factor=prefetch_factor
    )
    return train_loader, val_loader, train_ds.classes

def get_class_weights(dataset):
    """
    Compute inverse-frequency class weights for imbalance.
    """
    targets = [label for _, label in dataset.samples]
    counts = torch.bincount(torch.tensor(targets), minlength=len(dataset.classes))
    total = counts.sum().float()
    weights = total / (len(counts) * counts.float().clamp(min=1))
    return weights
