from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .common import ROOT, read_csv, resolve


def transform(cfg, train=False):
    p = cfg["preprocessing"]
    ops = ([T.RandomResizedCrop(p["crop"], scale=(0.75, 1.0), antialias=True),
            T.RandomHorizontalFlip(), T.RandomRotation(10),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15)] if train else
           [T.Resize(p["resize"], antialias=True), T.CenterCrop(p["crop"])])
    return T.Compose(ops + [T.ToTensor(), T.Normalize(p["mean"], p["std"])])


class ManifestDataset(Dataset):
    def __init__(self, cfg, split, train=False):
        self.rows = read_csv(ROOT / f"data/splits/{split}.csv")
        self.root = resolve(cfg["data"]["root"])
        self.transform = transform(cfg, train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["path"]) as image:
            tensor = self.transform(ImageOps.exif_transpose(image).convert("RGB"))
        return tensor, int(row["label"]), row["path"]


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def loader(cfg, split, batch_size=1, train=False):
    return DataLoader(ManifestDataset(cfg, split, train), batch_size=batch_size,
                      shuffle=train, num_workers=cfg["train"]["workers"],
                      worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(cfg["seed"]),
                      drop_last=False)


def loss_function(cfg, device):
    weights = None
    if cfg["train"]["class_weights"]:
        labels = [int(r["label"]) for r in read_csv(ROOT / "data/splits/train.csv")]
        counts = np.bincount(labels, minlength=len(cfg["classes"]))
        if (counts == 0).any():
            raise ValueError("Train split is missing a class")
        weights = torch.tensor(len(labels) / (len(counts) * counts), dtype=torch.float32, device=device)
    return torch.nn.CrossEntropyLoss(weight=weights)
