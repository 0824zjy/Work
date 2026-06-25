import os
import json
import argparse
from typing import List, Optional


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
MASK_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def read_id_list(list_txt: str) -> List[str]:
    ids = []
    with open(list_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.append(s)
    return ids


def find_existing_file(dir_path: str, stem: str, exts: List[str]) -> Optional[str]:
    for ext in exts:
        p = os.path.join(dir_path, stem + ext)
        if os.path.exists(p):
            return p
    return None


def find_mask(mask_dir: str, stem: str) -> Optional[str]:
    p = find_existing_file(mask_dir, stem + "_segmentation", MASK_EXTS)
    if p is not None:
        return p

    p = find_existing_file(mask_dir, stem, MASK_EXTS)
    if p is not None:
        return p

    return None


def write_jsonl(f, item):
    f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--real_image_dir", type=str, required=True)
    parser.add_argument("--real_mask_dir", type=str, required=True)
    parser.add_argument("--real_list_txt", type=str, required=True)
    parser.add_argument("--gen_jsonl", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, required=True)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    real_ids = read_id_list(args.real_list_txt)

    real_count = 0
    real_missing_image = 0
    real_missing_mask = 0

    gen_count = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as out_f:
        for stem in real_ids:
            image_path = find_existing_file(args.real_image_dir, stem, IMG_EXTS)
            mask_path = find_mask(args.real_mask_dir, stem)

            if image_path is None:
                real_missing_image += 1
                continue

            if mask_path is None:
                real_missing_mask += 1
                continue

            write_jsonl(out_f, {
                "image": image_path,
                "mask": mask_path,
                "weight": 1.0,
                "source": "real",
            })
            real_count += 1

        if os.path.exists(args.gen_jsonl):
            with open(args.gen_jsonl, "r", encoding="utf-8") as gf:
                for line in gf:
                    line = line.strip()
                    if not line:
                        continue

                    item = json.loads(line)

                    image_path = item.get("image", None)
                    mask_path = item.get("mask", None)
                    weight = float(item.get("weight", 1.0))

                    if image_path is None or mask_path is None:
                        continue

                    if not os.path.exists(image_path) or not os.path.exists(mask_path):
                        continue

                    write_jsonl(out_f, {
                        "image": image_path,
                        "mask": mask_path,
                        "weight": weight,
                        "source": item.get("source", "bef_sbg"),
                    })
                    gen_count += 1
        else:
            print(f"[WARN] gen_jsonl does not exist: {args.gen_jsonl}")

    total_count = real_count + gen_count

    print("[DONE] weighted segmentation dataset jsonl generated.")
    print(f"  out_jsonl           = {args.out_jsonl}")
    print(f"  real_count          = {real_count}")
    print(f"  gen_count           = {gen_count}")
    print(f"  total_count         = {total_count}")
    print(f"  real_missing_image  = {real_missing_image}")
    print(f"  real_missing_mask   = {real_missing_mask}")


if __name__ == "__main__":
    main()
