import cv2
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0


def sobel_edge(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy + 1e-6)
    edge = edge / (edge.max() + 1e-6)
    return np.clip(edge, 0.0, 1.0)


def hard_boundary_from_mask(mask, kernel_size=3):
    mask_bin = (mask > 0.5).astype(np.float32)

    k = int(kernel_size)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1

    kernel = np.ones((k, k), np.uint8)
    mask_u8 = (mask_bin * 255).astype(np.uint8)

    dilation = cv2.dilate(mask_u8, kernel, iterations=1).astype(np.float32) / 255.0
    erosion = cv2.erode(mask_u8, kernel, iterations=1).astype(np.float32) / 255.0

    boundary = np.clip(dilation - erosion, 0.0, 1.0)
    return boundary


def soft_boundary_prior_from_mask(mask, radius=12, tau=4.0, kernel_size=3):
    hard = hard_boundary_from_mask(mask, kernel_size=kernel_size)

    soft = hard.copy()
    prev = hard.copy()

    for r in range(1, radius + 1):
        k = 2 * r + 1
        kernel = np.ones((k, k), np.uint8)

        hard_u8 = (hard * 255).astype(np.uint8)
        dilated = cv2.dilate(hard_u8, kernel, iterations=1).astype(np.float32) / 255.0

        shell = np.clip(dilated - prev, 0.0, 1.0)
        weight = np.exp(-float(r) / float(tau))

        soft = np.maximum(soft, shell * weight)
        prev = dilated

    return np.clip(soft, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--radius", type=int, default=12)
    parser.add_argument("--tau", type=float, default=4.0)
    parser.add_argument("--kernel_size", type=int, default=3)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)

    image_files = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ])

    rows = []

    for img_path in image_files:
        mask_path = mask_dir / img_path.name

        if not mask_path.exists():
            print(f"[WARN] missing mask for {img_path.name}")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] failed to read image: {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        edge = sobel_edge(gray)

        mask = read_gray(mask_path)
        if mask.shape != edge.shape:
            mask = cv2.resize(mask, (edge.shape[1], edge.shape[0]), interpolation=cv2.INTER_NEAREST)

        prior = soft_boundary_prior_from_mask(
            mask,
            radius=args.radius,
            tau=args.tau,
            kernel_size=args.kernel_size,
        )

        outside = 1.0 - prior

        ober = float((edge * outside).mean())
        bcr = float((edge * prior).sum() / ((edge * outside).sum() + 1e-6))

        rows.append({
            "filename": img_path.name,
            "OBER": ober,
            "BCR": bcr,
        })

    df = pd.DataFrame(rows)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"[OK] saved per-image metrics to {out_csv}")

    if len(df) > 0:
        print("[Mean]")
        print(df[["OBER", "BCR"]].mean())
        print("[Std]")
        print(df[["OBER", "BCR"]].std())
    else:
        print("[WARN] no valid samples found")


if __name__ == "__main__":
    main()
