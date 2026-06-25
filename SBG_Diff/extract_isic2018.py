#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deterministic nested splits for ISIC2018-style naming:

Images:
  /path/train/Images/ISIC_0000000.jpg

Masks:
  /path/train/Masks/ISIC_0000000_segmentation.png

This script:
- Extracts IDs from Images as "ISIC_0000000"
- Checks corresponding mask exists as "{ID}_segmentation.<ext>" (ext can be png/jpg/tif...)
- Produces nested splits (e.g., 5% and 10% where 10% includes 5%)
- Writes IDs (one per line) to txt files

Example:
python /data/zjy_work/ISIC2018/extract_isic2018.py --img_dir /data/zjy_work/ISIC2018/train/Images --mask_dir /data/zjy_work/ISIC2018/train/Masks --out_dir /data/zjy_work/ISIC2018/splits --seed 2026 --pcts 5 10 20 --sort_output

"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_files_by_stem(dir_path: Path, exts: Set[str]) -> Dict[str, Path]:
    """
    Return dict: stem -> Path for files in dir_path with suffix in exts.
    If duplicate stems exist with different extensions, the last one wins.
    """
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    out: Dict[str, Path] = {}
    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            out[p.stem] = p
    return out


def write_ids(path: Path, ids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _id in ids:
            f.write(_id + "\n")


def make_nested_splits(ids: List[str], seed: int, pcts: List[int]) -> List[Tuple[int, List[str]]]:
    if not ids:
        raise ValueError("No valid IDs found.")
    pcts = sorted(set(pcts))
    if any(p <= 0 or p >= 100 for p in pcts):
        raise ValueError("Each pct must be in (0, 100).")

    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    splits: List[Tuple[int, List[str]]] = []
    for pct in pcts:
        k = int(round(n * (pct / 100.0)))
        k = max(1, min(k, n))
        splits.append((pct, shuffled[:k]))

    # sanity: nested
    prev = set()
    for pct, sub in splits:
        cur = set(sub)
        if prev and not prev.issubset(cur):
            raise RuntimeError(f"Nested split violated at {pct}%.")
        prev = cur

    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True, type=str)
    ap.add_argument("--mask_dir", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--seed", default=2026, type=int)
    ap.add_argument("--pcts", nargs="+", default=[5, 10], type=int)
    ap.add_argument("--prefix", default="train", type=str)
    ap.add_argument("--sort_output", action="store_true", help="Sort IDs inside output files")
    ap.add_argument(
        "--mask_suffix",
        default="_segmentation",
        type=str,
        help="Mask filename suffix appended to ID stem (default: _segmentation)",
    )
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)

    images = list_files_by_stem(img_dir, IMG_EXTS)
    masks = list_files_by_stem(mask_dir, MASK_EXTS)

    # masks stems look like: ISIC_0000000_segmentation
    # map base_id -> mask_path by stripping suffix
    base_to_mask: Dict[str, Path] = {}
    suffix = args.mask_suffix
    for stem, p in masks.items():
        if stem.endswith(suffix):
            base_id = stem[: -len(suffix)]
            base_to_mask[base_id] = p

    img_ids = sorted(images.keys())
    valid_ids = [i for i in img_ids if i in base_to_mask]

    missing_mask = [i for i in img_ids if i not in base_to_mask]
    # (Optional) masks without corresponding image
    masks_without_image = [bid for bid in base_to_mask.keys() if bid not in images]

    print(f"[INFO] Images found: {len(img_ids)}")
    print(f"[INFO] Masks found : {len(masks)} (with suffix '{suffix}' recognized: {len(base_to_mask)})")
    print(f"[INFO] Valid pairs : {len(valid_ids)}")

    if missing_mask:
        print(f"[WARN] Images missing masks: {len(missing_mask)} (showing up to 10): {missing_mask[:10]}")
    if masks_without_image:
        print(f"[WARN] Masks missing images: {len(masks_without_image)} (showing up to 10): {masks_without_image[:10]}")

    # Deterministic nested splits
    splits = make_nested_splits(valid_ids, seed=args.seed, pcts=args.pcts)

    for pct, ids in splits:
        out_path = out_dir / f"{args.prefix}_{pct}p.txt"
        to_write = sorted(ids) if args.sort_output else ids
        write_ids(out_path, to_write)
        print(f"[OK] Wrote {pct}% ({len(ids)} ids) -> {out_path}")

    print("[DONE] Now you can use these txt lists to load subsets deterministically.")
    print("Note: Each line is base ID like 'ISIC_0000000' (no extension, no _segmentation).")


if __name__ == "__main__":
    main()
