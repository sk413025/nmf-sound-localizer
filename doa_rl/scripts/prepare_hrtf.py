import argparse
from doa_rl.env.hrtf_loader import load_pth_to_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True, help="Path to .pth H file or device IRs (pth assumed)")
    ap.add_argument("--type", type=str, default="device")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    # Minimal: convert existing .pth H into .npz with keys H, angles
    load_pth_to_npz(args.src, args.out, transpose_to_df=True)
    print(f"Saved H to {args.out}")


if __name__ == "__main__":
    main()

