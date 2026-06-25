import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


# =========================
# Config (edit here)
# =========================
IMAGES_DIR = Path("/data/zjy_work/ISIC2018/ISIC2018_Task1-2_Training_Input/images")
MASKS_DIR  = Path("/data/zjy_work/ISIC2018/ISIC2018_Task1-2_Training_Input/masks")
OUT_DIR    = Path("/data/zjy_work/ISIC2018/ISIC2018_Task1-2_Training_Input/overlay_results")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Overlay color (R, G, B) and transparency (0~255)
OVERLAY_COLOR = (255, 0, 0)   # red
OVERLAY_ALPHA = 110          # 0 fully transparent, 255 fully opaque

# If mask size differs from image: "resize" or "skip" or "error"
SIZE_MISMATCH_POLICY = "resize"  # options: resize / skip / error

# Binarization threshold for mask (0~255). Pixels > threshold are treated as foreground.
MASK_THRESHOLD = 0

# Output naming
OUT_SUFFIX = "_mask_overlay"  # output: <orig_stem> + OUT_SUFFIX + ".png"
# =========================


def load_mask_as_binary(mask_path: Path) -> np.ndarray:
    """
    Load mask image and return a boolean array: True = foreground.
    Supports grayscale/RGB/PNG with alpha.
    """
    m = Image.open(mask_path)

    # If has alpha channel, prefer alpha as mask; else convert to L.
    if m.mode in ("RGBA", "LA"):
        alpha = m.split()[-1]
        m_gray = alpha.convert("L")
    else:
        m_gray = m.convert("L")

    m_np = np.array(m_gray, dtype=np.uint8)
    fg = m_np > MASK_THRESHOLD
    return fg


def overlay_mask_on_image(img_path: Path, mask_path: Path, out_path: Path):
    img = Image.open(img_path).convert("RGBA")
    w, h = img.size

    # Load mask
    mask_img = Image.open(mask_path)
    if mask_img.size != (w, h):
        if SIZE_MISMATCH_POLICY == "resize":
            mask_img = mask_img.resize((w, h), resample=Image.NEAREST)
        elif SIZE_MISMATCH_POLICY == "skip":
            return False, f"skip size mismatch: img={img.size}, mask={mask_img.size}"
        else:
            raise ValueError(f"Size mismatch: img={img.size}, mask={mask_img.size}")

    # Convert mask to binary foreground
    # (re-open from resized mask_img to keep it consistent)
    # Use same logic as load_mask_as_binary but on in-memory image
    m = mask_img
    if m.mode in ("RGBA", "LA"):
        alpha = m.split()[-1]
        m_gray = alpha.convert("L")
    else:
        m_gray = m.convert("L")
    m_np = np.array(m_gray, dtype=np.uint8)
    fg = m_np > MASK_THRESHOLD  # bool

    # Build per-pixel alpha mask (0..255)
    alpha_np = np.zeros((h, w), dtype=np.uint8)
    alpha_np[fg] = np.uint8(OVERLAY_ALPHA)

    # Build overlay RGBA image
    overlay_np = np.zeros((h, w, 4), dtype=np.uint8)
    overlay_np[..., 0] = OVERLAY_COLOR[0]
    overlay_np[..., 1] = OVERLAY_COLOR[1]
    overlay_np[..., 2] = OVERLAY_COLOR[2]
    overlay_np[..., 3] = alpha_np

    overlay = Image.fromarray(overlay_np, mode="RGBA")

    # Composite: img + overlay (overlay alpha controls blend)
    out = Image.alpha_composite(img, overlay)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path, format="PNG")
    return True, "ok"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS])

    missing_masks = 0
    processed = 0
    skipped = 0

    for img_path in tqdm(image_files, desc="Overlaying masks", unit="img"):
        stem = img_path.stem  # e.g., ISIC_0000000
        mask_path = MASKS_DIR / f"{stem}_segmentation.png"
        if not mask_path.exists():
            missing_masks += 1
            continue

        out_name = f"{stem}{OUT_SUFFIX}.png"
        out_path = OUT_DIR / out_name

        ok, msg = overlay_mask_on_image(img_path, mask_path, out_path)
        if ok:
            processed += 1
        else:
            skipped += 1

    print("\nDone.")
    print(f"Images found     : {len(image_files)}")
    print(f"Processed        : {processed}")
    print(f"Skipped          : {skipped}")
    print(f"Missing masks    : {missing_masks}")
    print(f"Output directory : {OUT_DIR}")


if __name__ == "__main__":
    main()
