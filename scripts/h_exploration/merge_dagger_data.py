import argparse
import json
import random
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_pt", type=str, required=True)
    p.add_argument("--dagger_pt", type=str, required=True)
    p.add_argument("--out_pt", type=str, required=True)
    p.add_argument(
        "--dagger_ratio",
        type=float,
        default=1.0,
        help="Desired dagger blocks per teacher block. 1.0 means equal counts.",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(int(args.seed))

    teacher = torch.load(args.teacher_pt, weights_only=False)
    dagger = torch.load(args.dagger_pt, weights_only=False)

    if not isinstance(teacher, list) or not isinstance(dagger, list):
        raise ValueError("Expected both teacher and dagger data to be lists of blocks.")

    teacher_n = len(teacher)
    dagger_n = len(dagger)
    if teacher_n == 0 or dagger_n == 0:
        raise ValueError("Teacher or dagger dataset is empty.")

    if args.dagger_ratio <= 0:
        raise ValueError("--dagger_ratio must be > 0")

    target_dagger = int(round(float(args.dagger_ratio) * teacher_n))
    if target_dagger > dagger_n:
        target_dagger = dagger_n

    sampled_dagger = random.sample(dagger, target_dagger)

    merged = teacher + sampled_dagger

    out_path = Path(args.out_pt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, out_path)

    meta = {
        "teacher_pt": args.teacher_pt,
        "dagger_pt": args.dagger_pt,
        "out_pt": str(out_path),
        "teacher_blocks": teacher_n,
        "dagger_blocks": dagger_n,
        "dagger_ratio_requested": float(args.dagger_ratio),
        "dagger_blocks_used": target_dagger,
        "merged_blocks": len(merged),
        "seed": int(args.seed),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
