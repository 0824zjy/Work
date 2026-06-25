import os
import re
import json
import shutil
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def is_image_file(p: Path):
    return p.suffix.lower() in IMG_EXTS


def is_mask_like_path(p: Path):
    """
    Exclude ISIC official segmentation / ground truth masks.
    This is important because /data/zjy_work/ISIC2018/train may contain both images and masks.
    """
    s = str(p).lower()
    name = p.name.lower()

    mask_keywords = [
        "segmentation",
        "_seg",
        "mask",
        "masks",
        "label",
        "labels",
        "groundtruth",
        "ground_truth",
        "gt",
    ]

    for k in mask_keywords:
        if k in s or k in name:
            return True

    return False


def list_images_recursive(root, exclude_masks=False, max_images=0):
    root = Path(root)
    paths = []

    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if not is_image_file(p):
            continue
        if exclude_masks and is_mask_like_path(p):
            continue
        paths.append(p)

    if max_images is not None and int(max_images) > 0:
        paths = paths[: int(max_images)]

    return paths


def list_images_nonrecursive(root):
    root = Path(root)
    if not root.exists():
        return []
    return [p for p in sorted(root.iterdir()) if p.is_file() and is_image_file(p)]


def resolve_generated_image_dir(result_dir):
    result_dir = Path(result_dir)

    candidates = [
        result_dir / "images",
        result_dir / "image",
        result_dir / "imgs",
        result_dir / "samples",
    ]

    for c in candidates:
        if c.is_dir():
            return c

    return result_dir


def parse_syn_id(path: Path):
    """
    Expected generated filename:
        id-000000_s-00_idx-0.png
        id-000001_s-01_idx-0.png
    """
    m = re.search(r"id-(\d+)", path.name)
    if m is None:
        return None
    return int(m.group(1))


def read_prompt_targets(prompt_json):
    targets = []

    if prompt_json is None or not os.path.isfile(prompt_json):
        return targets

    with open(prompt_json, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if "target" in item:
                targets.append(item["target"])
            elif "image" in item:
                targets.append(item["image"])
            elif "jpg" in item:
                targets.append(item["jpg"])

    return targets


def build_filename_index(real_paths):
    """
    Build index by filename and stem for resolving target images.
    """
    index = {}
    for p in real_paths:
        index[p.name] = p
        index[p.stem] = p
    return index


def resolve_real_path(path_str, real_index):
    p = Path(path_str)

    if p.is_file():
        return p

    if p.name in real_index:
        return real_index[p.name]

    if p.stem in real_index:
        return real_index[p.stem]

    return None


def build_pairs(syn_paths, prompt_json, real_paths):
    """
    For CLIP-I and LPIPS, pair each generated image with its corresponding real image.

    Preferred:
      generated id -> prompt_json row -> target path

    Fallback:
      generated id -> sorted real image list
    """
    pairs = []

    prompt_targets = read_prompt_targets(prompt_json)
    real_index = build_filename_index(real_paths)

    for i, sp in enumerate(syn_paths):
        idx = parse_syn_id(sp)

        rp = None

        if idx is not None and idx < len(prompt_targets):
            rp = resolve_real_path(prompt_targets[idx], real_index)

        if rp is None:
            if idx is None:
                idx = i
            if len(real_paths) > 0:
                idx = min(idx, len(real_paths) - 1)
                rp = real_paths[idx]

        if rp is not None and Path(rp).is_file():
            pairs.append((sp, Path(rp)))

    return pairs


def safe_image_name(p: Path):
    h = hashlib.md5(str(p).encode("utf-8")).hexdigest()[:10]
    return f"{p.stem}_{h}.png"


def prepare_resized_dir(paths, dst_dir, size=299):
    dst_dir = Path(dst_dir)

    if dst_dir.exists():
        shutil.rmtree(dst_dir)

    dst_dir.mkdir(parents=True, exist_ok=True)

    n = 0

    for p in tqdm(paths, desc=f"resize -> {dst_dir.name}"):
        try:
            img = Image.open(p).convert("RGB")
            img = img.resize((size, size), Image.BICUBIC)
            img.save(dst_dir / safe_image_name(p))
            n += 1
        except Exception as e:
            print(f"[WARN] skip {p}: {e}")

    return n


def compute_fid_kid(real_dir, syn_dir, device, batch_size):
    from torch_fidelity import calculate_metrics

    real_paths = list_images_nonrecursive(real_dir)
    syn_paths = list_images_nonrecursive(syn_dir)

    n_real = len(real_paths)
    n_syn = len(syn_paths)

    if n_real < 2 or n_syn < 2:
        raise RuntimeError(f"FID/KID needs at least 2 images. real={n_real}, syn={n_syn}")

    kid_subset_size = min(100, n_real, n_syn)
    kid_subset_size = max(2, kid_subset_size)

    print(f"[INFO] FID/KID real={n_real}, syn={n_syn}, kid_subset_size={kid_subset_size}")

    metrics = calculate_metrics(
        input1=str(real_dir),
        input2=str(syn_dir),
        cuda=device.startswith("cuda"),
        isc=False,
        fid=True,
        kid=True,
        kid_subset_size=kid_subset_size,
        kid_subsets=100,
        batch_size=batch_size,
        verbose=True,
    )

    fid = float(metrics.get("frechet_inception_distance", np.nan))
    kid = float(metrics.get("kernel_inception_distance_mean", np.nan))

    return fid, kid


def pil_to_tensor_01(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1)
    return ten


def pil_to_tensor_m11(path, size=None):
    ten = pil_to_tensor_01(path, size=size)
    return ten * 2.0 - 1.0


@torch.no_grad()
def extract_clip_features(paths, device, batch_size, clip_model_name, clip_pretrained):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name,
        pretrained=clip_pretrained,
    )
    model = model.to(device)
    model.eval()

    feats = []

    for i in tqdm(range(0, len(paths), batch_size), desc="CLIP features"):
        batch_paths = paths[i:i + batch_size]
        imgs = []

        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(preprocess(img))

        imgs = torch.stack(imgs, dim=0).to(device)

        feat = model.encode_image(imgs)
        feat = F.normalize(feat.float(), dim=-1)

        feats.append(feat.cpu())

    return torch.cat(feats, dim=0)


def compute_clip_i(pairs, device, batch_size, clip_model_name, clip_pretrained):
    if len(pairs) < 1:
        return np.nan

    syn_paths = [p[0] for p in pairs]
    real_paths = [p[1] for p in pairs]

    syn_feat = extract_clip_features(
        syn_paths,
        device=device,
        batch_size=batch_size,
        clip_model_name=clip_model_name,
        clip_pretrained=clip_pretrained,
    )

    real_feat = extract_clip_features(
        real_paths,
        device=device,
        batch_size=batch_size,
        clip_model_name=clip_model_name,
        clip_pretrained=clip_pretrained,
    )

    sims = (syn_feat * real_feat).sum(dim=1)
    return float(sims.mean().item())


def compute_cmmd(real_paths, syn_paths, device, batch_size, clip_model_name, clip_pretrained):
    """
    CMMD proxy: MMD in CLIP image feature space.
    Lower is better.
    """
    if len(real_paths) < 2 or len(syn_paths) < 2:
        return np.nan

    real_feat = extract_clip_features(
        real_paths,
        device=device,
        batch_size=batch_size,
        clip_model_name=clip_model_name,
        clip_pretrained=clip_pretrained,
    )

    syn_feat = extract_clip_features(
        syn_paths,
        device=device,
        batch_size=batch_size,
        clip_model_name=clip_model_name,
        clip_pretrained=clip_pretrained,
    )

    x = real_feat.float()
    y = syn_feat.float()

    n = x.shape[0]
    m = y.shape[0]

    z = torch.cat([x, y], dim=0)

    max_for_median = min(2000, z.shape[0])
    if z.shape[0] > max_for_median:
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(z.shape[0], generator=g)[:max_for_median]
        z_med = z[idx]
    else:
        z_med = z

    dists = torch.pdist(z_med, p=2).pow(2)
    dists = dists[dists > 0]

    if dists.numel() == 0:
        sigma2 = torch.tensor(1.0)
    else:
        sigma2 = torch.median(dists)

    sigma2 = torch.clamp(sigma2, min=1e-6)
    gamma = 1.0 / (2.0 * sigma2)

    xx = torch.cdist(x, x, p=2).pow(2)
    yy = torch.cdist(y, y, p=2).pow(2)
    xy = torch.cdist(x, y, p=2).pow(2)

    kxx = torch.exp(-gamma * xx)
    kyy = torch.exp(-gamma * yy)
    kxy = torch.exp(-gamma * xy)

    kxx = (kxx.sum() - torch.diag(kxx).sum()) / (n * (n - 1))
    kyy = (kyy.sum() - torch.diag(kyy).sum()) / (m * (m - 1))
    kxy = kxy.mean()

    mmd = kxx + kyy - 2.0 * kxy
    mmd = torch.clamp(mmd, min=0.0)

    return float(mmd.item())


@torch.no_grad()
def compute_lpips(pairs, device, batch_size, size):
    if len(pairs) < 1:
        return np.nan

    import lpips

    loss_fn = lpips.LPIPS(net="alex").to(device)
    loss_fn.eval()

    scores = []

    for i in tqdm(range(0, len(pairs), batch_size), desc="LPIPS"):
        batch = pairs[i:i + batch_size]

        syn_imgs = []
        real_imgs = []

        for sp, rp in batch:
            syn_imgs.append(pil_to_tensor_m11(sp, size=size))
            real_imgs.append(pil_to_tensor_m11(rp, size=size))

        syn_imgs = torch.stack(syn_imgs, dim=0).to(device)
        real_imgs = torch.stack(real_imgs, dim=0).to(device)

        score = loss_fn(syn_imgs, real_imgs)
        score = score.view(-1).detach().cpu()

        scores.append(score)

    scores = torch.cat(scores, dim=0)
    return float(scores.mean().item())


@torch.no_grad()
def compute_mos_proxy(syn_paths, device, batch_size, size, mos_model):
    """
    MOS is not true human MOS. This is automatic no-reference IQA proxy.
    If pyiqa/model weights are unavailable, return NA.
    """
    if mos_model.lower() in ["none", "na", "disable", "disabled"]:
        return np.nan

    try:
        import pyiqa
    except Exception as e:
        print(f"[WARN] pyiqa unavailable, MOS=NA. Error: {e}")
        return np.nan

    try:
        metric = pyiqa.create_metric(mos_model, device=device)
        metric.eval()
    except Exception as e:
        print(f"[WARN] cannot create MOS model {mos_model}, MOS=NA. Error: {e}")
        return np.nan

    scores = []

    for i in tqdm(range(0, len(syn_paths), batch_size), desc=f"MOS proxy {mos_model}"):
        batch_paths = syn_paths[i:i + batch_size]
        imgs = []

        for p in batch_paths:
            imgs.append(pil_to_tensor_01(p, size=size))

        imgs = torch.stack(imgs, dim=0).to(device)

        try:
            score = metric(imgs)
            score = score.view(-1).detach().cpu()
            scores.append(score)
        except Exception as e:
            print(f"[WARN] MOS failed on batch {i}: {e}")

    if len(scores) == 0:
        return np.nan

    scores = torch.cat(scores, dim=0)
    return float(scores.mean().item())


def fmt(x):
    if x is None:
        return "NA"
    try:
        if np.isnan(float(x)):
            return "NA"
    except Exception:
        return "NA"
    return f"{float(x):.6f}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--real_root", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--prompt_json", type=str, default="")
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fid_size", type=int, default=299)
    parser.add_argument("--pair_size", type=int, default=256)

    parser.add_argument("--max_real_images", type=int, default=0)
    parser.add_argument("--max_syn_images", type=int, default=0)

    parser.add_argument("--clip_model_name", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--mos_model", type=str, default="musiq")

    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    method = args.method
    real_root = Path(args.real_root)
    result_dir = Path(args.result_dir)
    work_dir = Path(args.work_dir)

    image_dir = resolve_generated_image_dir(result_dir)

    if not real_root.exists():
        raise FileNotFoundError(f"real_root not found: {real_root}")

    if not image_dir.exists():
        raise FileNotFoundError(f"generated image dir not found: {image_dir}")

    print("============================================================")
    print(f"[METHOD] {method}")
    print(f"[REAL_ROOT] {real_root}")
    print(f"[RESULT_DIR] {result_dir}")
    print(f"[IMAGE_DIR] {image_dir}")
    print(f"[PROMPT_JSON] {args.prompt_json}")
    print("============================================================")

    real_paths = list_images_recursive(
        real_root,
        exclude_masks=True,
        max_images=args.max_real_images,
    )

    syn_paths = list_images_recursive(
        image_dir,
        exclude_masks=False,
        max_images=args.max_syn_images,
    )

    print(f"[INFO] real images found: {len(real_paths)}")
    print(f"[INFO] synthetic images found: {len(syn_paths)}")

    if len(real_paths) < 2:
        raise RuntimeError(f"too few real images found from {real_root}")

    if len(syn_paths) < 2:
        raise RuntimeError(f"too few synthetic images found from {image_dir}")

    work_dir.mkdir(parents=True, exist_ok=True)

    real_fid_dir = work_dir / method / "real_299"
    syn_fid_dir = work_dir / method / "syn_299"

    prepare_resized_dir(real_paths, real_fid_dir, size=args.fid_size)
    prepare_resized_dir(syn_paths, syn_fid_dir, size=args.fid_size)

    fid = np.nan
    kid = np.nan
    clip_i = np.nan
    lpips_score = np.nan
    cmmd = np.nan
    mos = np.nan

    try:
        fid, kid = compute_fid_kid(
            real_fid_dir,
            syn_fid_dir,
            device=device,
            batch_size=args.batch_size,
        )
    except Exception as e:
        print(f"[WARN] FID/KID failed: {e}")

    pairs = build_pairs(
        syn_paths=syn_paths,
        prompt_json=args.prompt_json,
        real_paths=real_paths,
    )

    print(f"[INFO] paired images for CLIP-I/LPIPS: {len(pairs)}")

    try:
        clip_i = compute_clip_i(
            pairs=pairs,
            device=device,
            batch_size=args.batch_size,
            clip_model_name=args.clip_model_name,
            clip_pretrained=args.clip_pretrained,
        )
    except Exception as e:
        print(f"[WARN] CLIP-I failed: {e}")

    try:
        lpips_score = compute_lpips(
            pairs=pairs,
            device=device,
            batch_size=max(1, min(args.batch_size, 16)),
            size=args.pair_size,
        )
    except Exception as e:
        print(f"[WARN] LPIPS failed: {e}")

    try:
        cmmd = compute_cmmd(
            real_paths=real_paths,
            syn_paths=syn_paths,
            device=device,
            batch_size=args.batch_size,
            clip_model_name=args.clip_model_name,
            clip_pretrained=args.clip_pretrained,
        )
    except Exception as e:
        print(f"[WARN] CMMD failed: {e}")

    try:
        mos = compute_mos_proxy(
            syn_paths=syn_paths,
            device=device,
            batch_size=max(1, min(args.batch_size, 16)),
            size=args.pair_size,
            mos_model=args.mos_model,
        )
    except Exception as e:
        print(f"[WARN] MOS failed: {e}")

    row = {
        "method": method,
        "real_root": str(real_root),
        "result_dir": str(result_dir),
        "image_dir": str(image_dir),
        "num_real": len(real_paths),
        "num_syn": len(syn_paths),
        "num_pairs": len(pairs),
        "FID_down": fmt(fid),
        "KID_down": fmt(kid),
        "CLIP-I_up": fmt(clip_i),
        "LPIPS_down": fmt(lpips_score),
        "CMMD_down": fmt(cmmd),
        "MOS_up": fmt(mos),
    }

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([row])

    if out_csv.exists():
        old = pd.read_csv(out_csv)
        old = old[old["method"] != method]
        df = pd.concat([old, df], ignore_index=True)

    df.to_csv(out_csv, index=False)

    print("============================================================")
    print("[RESULT]")
    for k, v in row.items():
        print(f"{k}: {v}")
    print(f"[SAVED] {out_csv}")
    print("============================================================")


if __name__ == "__main__":
    main()
