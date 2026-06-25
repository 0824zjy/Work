#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python /data/zjy_work/scripts/build_data_txt.py --syn_root /data/zjy_work/ISIC2018/syn_from_trainmask_cn_decoder --syn_ratio 1.0 
import os
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Set

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def read_lines(txt: Path) -> List[str]:
    out = []
    with txt.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out

def write_lines(txt: Path, lines: List[str]) -> None:
    txt.parent.mkdir(parents=True, exist_ok=True)
    with txt.open("w", encoding="utf-8") as f:
        for s in lines:
            f.write(s + "\n")

def list_paired_stems(images_dir: Path, masks_dir: Path) -> List[str]:
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"masks_dir not found: {masks_dir}")

    img_stems: Set[str] = set()
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            img_stems.add(p.stem)

    msk_stems: Set[str] = set()
    for p in masks_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            msk_stems.add(p.stem)

    valid = sorted(img_stems & msk_stems)
    if len(valid) == 0:
        raise RuntimeError("No valid paired (image,mask) stems found. Check filenames and dirs.")
    return valid

def sample_syn_stems(all_syn_stems: List[str], n_need: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    if n_need <= 0:
        return []
    if n_need >= len(all_syn_stems):
        # 不够就全取（也可改成循环采样，但这里保持简单可控）
        return all_syn_stems[:]
    stems = all_syn_stems[:]
    rng.shuffle(stems)
    return stems[:n_need]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_split_dir", type=str, default="/data/zjy_work/ISIC2018/splits",
                    help="包含 train_5p.txt/train_10p.txt/train_20p.txt 的目录")
    ap.add_argument("--splits", nargs="+", default=["5p", "10p", "20p"],
                    help="要处理的 split 名称，对应 train_{split}.txt")
    ap.add_argument("--syn_root", type=str, required=True,
                    help="合成数据根目录（包含 images/ 与 masks/）")
    ap.add_argument("--out_dir", type=str, default="/data/zjy_work/data_txt",
                    help="输出txt目录")
    ap.add_argument("--syn_ratio", type=float, default=1.0,
                    help="合成样本数 = 真实样本数 * syn_ratio (默认1.0)")
    ap.add_argument("--seed", type=int, default=2026)

    args = ap.parse_args()

    real_split_dir = Path(args.real_split_dir)
    syn_root = Path(args.syn_root)
    out_dir = Path(args.out_dir)

    images_dir = syn_root / "images"
    masks_dir = syn_root / "masks"
    all_syn_stems = list_paired_stems(images_dir, masks_dir)
    print(f"[INFO] total synthetic paired samples: {len(all_syn_stems)} from {syn_root}")

    for sp in args.splits:
        real_txt = real_split_dir / f"train_{sp}.txt"
        if not real_txt.exists():
            raise FileNotFoundError(f"Real split not found: {real_txt}")

        real_ids = read_lines(real_txt)
        n_real = len(real_ids)
        n_syn = int(round(n_real * args.syn_ratio))
        syn_stems = sample_syn_stems(all_syn_stems, n_syn, seed=args.seed + hash(sp) % 100000)

        # 输出：real 与 syn 分开存（训练时分别传给 train_list1 / train_list2）
        out_real = out_dir / f"train_{sp}_real.txt"
        out_syn = out_dir / f"train_{sp}_syn.txt"

        write_lines(out_real, real_ids)
        write_lines(out_syn, syn_stems)

        print(f"[OK] {sp}: real={n_real} -> {out_real}")
        print(f"[OK] {sp}: syn ={len(syn_stems)} (ratio={args.syn_ratio}) -> {out_syn}")

    if args.copy_test_txt is not None:
        src = Path(args.copy_test_txt)
        if not src.exists():
            raise FileNotFoundError(f"test txt not found: {src}")
        dst = out_dir / "test.txt"
        write_lines(dst, read_lines(src))
        print(f"[OK] copied test txt -> {dst}")

    print("[DONE] all txts saved to", out_dir)

if __name__ == "__main__":
    main()
