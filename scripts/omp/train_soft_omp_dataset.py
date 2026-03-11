#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt

from nmf_localizer.core.transfer_functions import TransferFunctionProcessor
from nmf_localizer.config.defaults import NMFConfig
from doa_rl.data import DoADataset, create_dataloader
from doa_rl.omp.soft_omp import build_dictionary, TrainableRoutedSoftOMP


def ensure_same_F(H: torch.Tensor, W: torch.Tensor):
    if H.shape[0] != W.shape[0]:
        raise ValueError(f"F mismatch: H.F={H.shape[0]} vs W.F={W.shape[0]} — align freq band and n_fft")


def save_run(out_dir: Path, losses, curves_id, curves_ood, A_group, doa_pred_idx, fig):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=150)
    np.savez(out_dir / "metrics.npz",
             losses=np.array(losses),
             curves_id=np.array(curves_id),
             curves_ood=np.array(curves_ood),
             A_group=np.array(A_group),
             doa_pred_idx=int(doa_pred_idx))
    # minimal code state
    code_files = [Path(__file__), Path('doa_rl/omp/soft_omp.py')]
    files = []
    for p in code_files:
        try:
            import hashlib
            h = hashlib.sha256(Path(p).read_bytes()).hexdigest()
            files.append({'path': str(p), 'sha256': h})
        except Exception:
            pass
    (out_dir / 'code_state.json').write_text(json.dumps({'files': files}, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Train/eval Routed Soft-OMP on real dataset with H/W")
    ap.add_argument('--dataset_root', type=str, required=True, help='Root with angle_XX/*.npy')
    ap.add_argument('--angles', type=int, nargs='*', default=None, help='Angles to include, e.g., 0 15 30')
    ap.add_argument('--H_path', type=str, default='h_matrix_normalized_original_to_box.pth')
    ap.add_argument('--W_path', type=str, default='legacy/assets/doa_normalized_config_c_corrected/models/usm.pth')
    ap.add_argument('--freq_min', type=float, default=300.0)
    ap.add_argument('--freq_max', type=float, default=3000.0)
    ap.add_argument('--steps', type=int, default=6)
    ap.add_argument('--top_e', type=int, default=2)
    ap.add_argument('--L', type=int, default=2)
    ap.add_argument('--iters', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--out_dir', type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load H
    tfp = TransferFunctionProcessor(NMFConfig(freq_min=args.freq_min, freq_max=args.freq_max, device=str(device)))
    H_all, angles_tensor, meta = tfp.load_transfer_functions(args.H_path)
    freqs = None
    try:
        d = torch.load(args.H_path, weights_only=False)
        if isinstance(d, dict) and 'freqs' in d:
            freqs = np.array(d['freqs'])
    except Exception:
        pass
    if freqs is not None:
        H_proc, _ = tfp.apply_frequency_limit(H_all, freqs, args.freq_min, args.freq_max)
    else:
        H_proc = H_all
    H = H_proc.float().to(device)

    # Subset angles if requested
    if args.angles:
        mask = [int(a) in set(map(int, args.angles)) for a in angles_tensor.tolist()]
        idx = torch.tensor([i for i, m in enumerate(mask) if m], dtype=torch.long)
        if idx.numel() == 0:
            raise ValueError("No matching angles found in H for the requested subset")
        H = H[:, idx]
        angles_subset = angles_tensor[idx]
    else:
        angles_subset = angles_tensor

    E = H.shape[1]

    # Load W
    Wd = torch.load(args.W_path, map_location='cpu', weights_only=False)
    if isinstance(Wd, dict) and 'W' in Wd:
        W = Wd['W'].float()
    elif isinstance(Wd, torch.Tensor):
        W = Wd.float()
    else:
        raise ValueError(f"Unrecognized W format in {args.W_path}")
    W = W.to(device)

    ensure_same_F(H, W)
    F, M = W.shape[0], W.shape[1]
    print(f"Loaded H: {tuple(H.shape)}, W: {tuple(W.shape)} (F={F}, E={E}, M={M})")

    # Build D
    D, idx2 = build_dictionary(W, H)
    D = D.to(device)

    # Dataset
    if args.angles is None:
        angles_list = [int(a) for a in angles_subset.tolist()]
    else:
        angles_list = sorted(set(int(a) for a in args.angles))
    ds = DoADataset(args.dataset_root, angles=angles_list, fs=NMFConfig.sample_rate, n_fft=NMFConfig.n_fft,
                    freq_min=args.freq_min, freq_max=args.freq_max)
    dl = create_dataloader(ds, batch_size=1, shuffle=True)

    # Model
    model = TrainableRoutedSoftOMP(F=F, E=E, M=M, steps=args.steps, top_e=args.top_e, L=args.L,
                                   tau_e=0.5, tau_a=0.2, eta=0.5, routing='gumbel').to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3)

    # Train loop (mini): use mean across time as y for simplicity
    losses = []
    it = 0
    for batch in dl:
        Y = batch['Y'][0].to(device)  # (F, N)
        if Y.shape[0] != F:
            raise ValueError(f"STFT grid mismatch: Y.F={Y.shape[0]} vs H/W.F={F} — adjust freq_min/max, n_fft")
        y = Y.mean(dim=1)
        rec_loss = 0.0
        x_hat, r_curve = model(y, D, train_mode=True)
        y_hat = D @ x_hat
        rec_loss = torch.nn.functional.mse_loss(y_hat, y)
        opt.zero_grad(); rec_loss.backward(); opt.step()
        losses.append(float(rec_loss.item()))
        it += 1
        if it >= args.iters:
            break
        if it % max(5, args.iters // 5) == 0:
            print(f"[iter {it}/{args.iters}] rec={rec_loss.item():.4f}")

    # Eval ID/OOD curves using the same D (placeholder for OOD by shuffling W)
    def eval_curves(steps: int, n_eval: int = 20):
        curves = np.zeros(steps)
        n = 0
        for batch in dl:
            Y = batch['Y'][0].to(device)
            if Y.shape[0] != F:
                raise ValueError(f"STFT grid mismatch: Y.F={Y.shape[0]} vs H/W.F={F}")
            y = Y.mean(dim=1)
            _, r_curve = model(y, D, train_mode=False)
            curves += np.array(r_curve)
            n += 1
            if n >= n_eval:
                break
        return curves / max(n, 1)

    curves_id = eval_curves(args.steps, n_eval=min(12, len(ds)))
    curves_ood = eval_curves(args.steps, n_eval=min(12, len(ds)))  # placeholder same D

    # A_group and predicted group for a single example
    with torch.no_grad():
        for batch in dl:
            Y = batch['Y'][0].to(device)
            if Y.shape[0] != F:
                raise ValueError(f"STFT grid mismatch: Y.F={Y.shape[0]} vs H/W.F={F}")
            y = Y.mean(dim=1)
            x_hat, _ = model(y, D, train_mode=False)
            A_group = x_hat.view(E, M).abs().sum(dim=1).detach().cpu().numpy()
            doa_pred_idx = int(np.argmax(A_group))
            break

    # Plot
    fig, axs = plt.subplots(1, 4, figsize=(16, 3))
    axs[0].plot(np.arange(1, len(losses) + 1), losses); axs[0].set_title("Training loss"); axs[0].set_xlabel("iter"); axs[0].set_ylabel("loss")
    axs[1].plot(np.arange(1, args.steps + 1), curves_id, label="ID"); axs[1].set_title("Residual vs step (ID)"); axs[1].legend()
    axs[2].plot(np.arange(1, args.steps + 1), curves_ood, label="OOD"); axs[2].set_title("Residual vs step (OOD)"); axs[2].legend()
    axs[3].bar(np.arange(len(A_group)), A_group); axs[3].axvline(doa_pred_idx, color='r', linestyle='--', label=f'pred={doa_pred_idx}')
    axs[3].set_title("Group energy A_group"); axs[3].legend()

    # Save run (also include predicted angle in a small JSON sidecar)
    save_run(out_dir, losses, curves_id, curves_ood, A_group, doa_pred_idx, fig)
    try:
        doa_pred_angle = float(angles_subset[doa_pred_idx].item())
    except Exception:
        doa_pred_angle = None
    (Path(args.out_dir) / 'summary.json').write_text(json.dumps({
        'F': int(F), 'E': int(E), 'M': int(M),
        'doa_pred_idx': int(doa_pred_idx),
        'doa_pred_angle': doa_pred_angle,
        'angles_subset': [float(a) for a in angles_subset.tolist()],
    }, indent=2))
    print(f"Saved outputs to {out_dir}")


if __name__ == '__main__':
    main()
