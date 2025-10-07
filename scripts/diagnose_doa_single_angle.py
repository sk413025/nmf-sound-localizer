#!/usr/bin/env python3
"""
Diagnose DOA internals for a single angle and compare with debug logic.

Steps:
 1) Load one X/Y pair for a chosen angle and compute STFT magnitude (band-limited).
 2) Compute single-angle transfer function H_d via log-space (geometric) pooling.
 3) Learn W_from_X and U_from_X from X_data using sklearn NMF (same as debug).
 4) Build DOA localizer with D=1 (single angle), load W and H_d, construct A.
 5) Compare three reconstructions and stats:
      - Y_hat_debug = diag(H_d) @ W @ U_from_X         (pure debug path)
      - Y_hat_doa_gt = A @ X_gt  (X_gt = U_from_X)      (DOA with ground-truth X)
      - Y_hat_doa_est = A @ X_est (X_est from factorize) (DOA estimated X)
    Print MSE/correlation/scale ratios for each, plus A/X/Y stats.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from scipy import signal
from sklearn.decomposition import NMF as SKNMF
import logging

# Ensure project root on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nmf_localizer.config import NMFConfig
from nmf_localizer.core.localizer import NMFSoundLocalizer


def corr2(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    sa, sb = np.std(a), np.std(b)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(0.0 if np.isnan(c) else c)


def main():
    ap = argparse.ArgumentParser(description='Diagnose DOA internals for a single angle vs debug path')
    ap.add_argument('--x-root', required=True, help='Path to X (original) root with angle_*/')
    ap.add_argument('--y-root', required=True, help='Path to Y (box) root with angle_*/')
    ap.add_argument('--angle', type=int, default=90, help='Angle degree to test, e.g. 90')
    ap.add_argument('--n-components', type=int, default=15, help='NMF components for W_from_X')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('diagnose')

    cfg = NMFConfig(
        sample_rate=16000,
        n_fft=2048,
        hop_length=512,
        freq_min=500.0,
        freq_max=1500.0,
        beta=0.0,
        device='cpu'
    )

    # Load one pair from the chosen angle
    angle_dir = f'angle_{args.angle}'
    x_files = sorted((Path(args.x_root) / angle_dir).glob('*.npy'))
    y_files = sorted((Path(args.y_root) / angle_dir).glob('*.npy'))
    if not x_files or not y_files:
        raise FileNotFoundError(f'No files found under {angle_dir}')

    x = np.load(x_files[0]).astype(np.float32).flatten()
    y = np.load(y_files[0]).astype(np.float32).flatten()
    L = min(len(x), len(y))
    x = x[:L]
    y = y[:L]
    logger.info(f'Loaded angle {args.angle} pair: length {L}')

    # STFT consistent with config, band-limit
    f, t, Xst = signal.stft(x, fs=cfg.sample_rate, nperseg=cfg.n_fft,
                            noverlap=cfg.n_fft - cfg.hop_length, window=cfg.window)
    _, _, Yst = signal.stft(y, fs=cfg.sample_rate, nperseg=cfg.n_fft,
                            noverlap=cfg.n_fft - cfg.hop_length, window=cfg.window)
    Xmag = np.abs(Xst)
    Ymag = np.abs(Yst)
    mask = (f >= cfg.freq_min) & (f <= cfg.freq_max)
    X_data = Xmag[mask, :]
    Y_data = Ymag[mask, :]
    F, N = X_data.shape
    logger.info(f'Band-limited shapes: X={X_data.shape}, Y={Y_data.shape}')

    # Single-angle H via log-space (geometric mean)
    eps = 1e-12
    H_d = np.exp(np.mean(np.log(Y_data + eps) - np.log(X_data + eps), axis=1)).reshape(-1, 1)  # (F,1)
    logger.info(f'H_d stats: mean={H_d.mean():.4e}, std={H_d.std():.4e}, min={H_d.min():.4e}, max={H_d.max():.4e}')

    # Learn W_from_X, U_from_X from X_data (debug behavior)
    nmf_x = SKNMF(n_components=args.n_components, init='nndsvd', max_iter=1000, random_state=42)
    H_time = nmf_x.fit_transform(X_data.T)  # (N,K)
    W_from_X = nmf_x.components_.T         # (F,K)
    U_from_X = H_time.T                    # (K,N)
    logger.info(f'W_from_X stats: mean={W_from_X.mean():.4e}, std={W_from_X.std():.4e}')

    # Debug reconstruction
    Y_hat_debug = (H_d * (W_from_X @ U_from_X))
    corr_debug = corr2(Y_data, Y_hat_debug)
    scale_dbg = Y_hat_debug.mean() / (Y_data.mean() + eps)
    logger.info(f'DEBUG: corr={corr_debug:.4f}, scale={scale_dbg:.4f}, MSE={np.mean((Y_data - Y_hat_debug)**2):.3e}')

    # DOA localizer with D=1 (single angle)
    localizer = NMFSoundLocalizer(cfg)
    localizer.load_source_dictionary(torch.from_numpy(W_from_X).float())
    localizer.load_transfer_functions(torch.from_numpy(H_d).float(), angles=torch.tensor([float(args.angle)]))

    # A stats
    A = localizer.A.detach().cpu().numpy()
    logger.info(f'A stats: shape={A.shape}, mean={A.mean():.4e}, std={A.std():.4e}, min={A.min():.4e}, max={A.max():.4e}')

    # DOA with ground-truth X (X_gt = U_from_X since D=1)
    X_gt = U_from_X
    Y_hat_doa_gt = A @ X_gt
    corr_doa_gt = corr2(Y_data, Y_hat_doa_gt)
    logger.info(f'DOA-GT: corr={corr_doa_gt:.4f}, MSE={np.mean((Y_data - Y_hat_doa_gt)**2):.3e}')

    # Compare debug vs DOA-GT
    mse_dbg_vs_gt = np.mean((Y_hat_debug - Y_hat_doa_gt)**2)
    corr_dbg_vs_gt = corr2(Y_hat_debug, Y_hat_doa_gt)
    logger.info(f'DEBUG vs DOA-GT: corr={corr_dbg_vs_gt:.6f}, MSE={mse_dbg_vs_gt:.3e}')

    # DOA estimated X via factorize
    Y_tensor = torch.from_numpy(Y_data).float()
    X_est_t, info = localizer.factorize(Y_tensor)
    X_est = X_est_t.detach().cpu().numpy()
    Y_hat_doa_est = A @ X_est
    corr_est = corr2(Y_data, Y_hat_doa_est)
    logger.info(f'DOA-EST: corr={corr_est:.4f}, MSE={np.mean((Y_data - Y_hat_doa_est)**2):.3e}, n_iter={info["n_iter"]}')

    # X stats
    logger.info(f'X_gt stats: mean={X_gt.mean():.4e}, std={X_gt.std():.4e}, min={X_gt.min():.4e}, max={X_gt.max():.4e}')
    logger.info(f'X_est stats: mean={X_est.mean():.4e}, std={X_est.std():.4e}, min={X_est.min():.4e}, max={X_est.max():.4e}')

    # Final check
    if corr_dbg_vs_gt > 0.9999 and mse_dbg_vs_gt < 1e-10:
        print('✅ Single-angle DOA (with GT X) perfectly matches debug reconstruction.')
    else:
        print('⚠ Single-angle DOA (with GT X) differs from debug reconstruction — check inputs.')

if __name__ == '__main__':
    raise SystemExit(main())

