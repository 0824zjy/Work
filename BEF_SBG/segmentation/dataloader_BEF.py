import os
import json
import random
from typing import List, Dict

from PIL import Image

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms


def _sobel_boundary_from_mask(mask_tensor: torch.Tensor, thresh: float = 0.0):
    """
    Input:
        mask_tensor: [1,H,W], value range [0,1]
    Output:
        boundary: [1,H,W], value range [0,1]
    """
    assert mask_tensor.dim() == 3 and mask_tensor.size(0) == 1

    kx = torch.tensor(
        [[-1., 0., 1.],
         [-2., 0., 2.],
         [-1., 0., 1.]],
        dtype=mask_tensor.dtype,
        device=mask_tensor.device,
    ).view(1, 1, 3, 3)

    ky = torch.tensor(
        [[-1., -2., -1.],
         [0., 0., 0.],
         [1., 2., 1.]],
        dtype=mask_tensor.dtype,
        device=mask_tensor.device,
    ).view(1, 1, 3, 3)

    gx = F.conv2d(mask_tensor.unsqueeze(0), kx, padding=1)
    gy = F.conv2d(mask_tensor.unsqueeze(0), ky, padding=1)

    mag = torch.sqrt(gx ** 2 + gy ** 2).squeeze(0)
    mag = mag / (mag.max() + 1e-8)

    if thresh > 0:
        mag = (mag > thresh).float()

    return mag


def read_jsonl(path: str) -> List[Dict]:
    data_items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            item = json.loads(s)
            data_items.append(item)
    return data_items


class WeightedBEFDataset(data.Dataset):
    """
    Weighted segmentation dataset for final BGDNet training.

    Each jsonl line:
    {
      "image": image_path,
      "mask": mask_path,
      "weight": 1.0,
      "source": "real" or "bef_sbg"
    }

    Return:
      image:    [3,H,W]
      gt:       [1,H,W]
      boundary: [1,H,W]
      weight:   scalar tensor
      source:   string
    """

    def __init__(
        self,
        weighted_jsonl: str,
        trainsize: int = 352,
        augmentation: bool = False,
    ):
        self.weighted_jsonl = weighted_jsonl
        self.trainsize = trainsize
        self.augmentation = augmentation

        self.items = read_jsonl(weighted_jsonl)

        if len(self.items) == 0:
            raise RuntimeError(f"No training samples found in {weighted_jsonl}")

        self.items = [
            x for x in self.items
            if os.path.exists(x.get("image", "")) and os.path.exists(x.get("mask", ""))
        ]

        if len(self.items) == 0:
            raise RuntimeError(f"No valid image-mask pairs found in {weighted_jsonl}")

        self.size = len(self.items)

        self.real_count = sum(1 for x in self.items if x.get("source", "") == "real")
        self.synthetic_count = sum(1 for x in self.items if x.get("source", "") != "real")

        print("[WeightedBEFDataset]")
        print(f"  jsonl           = {weighted_jsonl}")
        print(f"  total           = {self.size}")
        print(f"  real            = {self.real_count}")
        print(f"  synthetic       = {self.synthetic_count}")
        print(f"  augmentation    = {self.augmentation}")

        if self.augmentation:
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ])

            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ])

            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return self.size

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def binary_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("L")

    def __getitem__(self, index):
        item = self.items[index]

        image_path = item["image"]
        mask_path = item["mask"]
        weight = float(item.get("weight", 1.0))
        source = item.get("source", "unknown")

        image = self.rgb_loader(image_path)
        gt = self.binary_loader(mask_path)

        seed = np.random.randint(1998)

        random.seed(seed)
        torch.manual_seed(seed)
        image = self.img_transform(image)

        random.seed(seed)
        torch.manual_seed(seed)
        gt = self.gt_transform(gt)

        gt = torch.clamp(gt, 0.0, 1.0)

        with torch.no_grad():
            boundary = _sobel_boundary_from_mask(gt)

        weight_tensor = torch.tensor(weight, dtype=torch.float32)

        return image, gt, boundary, weight_tensor, source


def get_weighted_loader(
    weighted_jsonl: str,
    batchsize: int,
    trainsize: int,
    shuffle: bool = True,
    num_workers: int = 8,
    pin_memory: bool = True,
    augmentation: bool = False,
    drop_last: bool = True,
):
    dataset = WeightedBEFDataset(
        weighted_jsonl=weighted_jsonl,
        trainsize=trainsize,
        augmentation=augmentation,
    )

    loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return loader
