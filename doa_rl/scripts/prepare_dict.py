import argparse
import glob
import os
import numpy as np
import librosa

from doa_rl.env import stft_mag
from doa_rl.features import train_nmf_is


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", type=str, required=True)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--n_fft", type=int, default=512)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_iter", type=int, default=200)
    ap.add_argument("--out", type=str, default="data/W_usm_is.npz")
    args = ap.parse_args()

    wavs = sorted(glob.glob(os.path.join(args.wav_dir, "*.wav")))
    Ys = []
    for p in wavs:
        y, _ = librosa.load(p, sr=args.sr, mono=True)
        Y = stft_mag(y, fs=args.sr, n_fft=args.n_fft)
        Ys.append(Y)
    W = train_nmf_is(Ys, K=args.K, n_iter=args.n_iter)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, W=W)
    print("Saved W:", args.out, W.shape)


if __name__ == "__main__":
    main()

