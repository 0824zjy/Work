import os
import json
import shutil
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_json", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.prompt_json, "r") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    for i, item in enumerate(lines):
        src = item["target"]
        ext = os.path.splitext(src)[1]
        dst = out_dir / f"real_{i:06d}{ext}"
        shutil.copy(src, dst)

    print(f"[OK] copied {len(lines)} real images to {out_dir}")


if __name__ == "__main__":
    main()
