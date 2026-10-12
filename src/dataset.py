"""تحميل الصور وتحويلاتها (augmentation) من ملف splits.csv."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int, train: bool) -> T.Compose:
    normalize = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    if train:
        # لا نستخدم القلب الأفقي لأن اتجاه النص على البطاقة مهم،
        # ولا نضيف blur لأن الفئة قد تعتمد على الضبابية.
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.8, 1.25)),
            T.RandomRotation(8),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
            normalize,
        ])
    # التقييم: تغيير حجم كامل بدون قص حتى لا تضيع أجزاء من البطاقة.
    return T.Compose([T.Resize((img_size, img_size)), normalize])


def load_rows(splits_csv: str | Path, split: str | None = None) -> list[dict]:
    splits_csv = Path(splits_csv)
    rows = []
    with open(splits_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if split is not None and r["split"] != split:
                continue
            p = Path(r["path"])
            if not p.is_absolute():
                p = (splits_csv.parent / p).resolve()
            rows.append({"path": p, "label": r["label"], "sha256": r.get("sha256", "")})
    return rows


class ImageDataset(Dataset):
    def __init__(self, rows: list[dict], class_to_idx: dict[str, int], transform: T.Compose):
        self.rows = rows
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        img = Image.open(r["path"]).convert("RGB")
        return self.transform(img), self.class_to_idx[r["label"]]


def make_loader(rows, classes, img_size, train, batch_size=16, num_workers=2, shuffle=None) -> DataLoader:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    ds = ImageDataset(rows, class_to_idx, build_transforms(img_size, train))
    if shuffle is None:
        shuffle = train
    # drop_last يمنع خطأ BatchNorm مع دفعة أخيرة بحجم 1.
    drop_last = train and len(ds) > batch_size
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )
