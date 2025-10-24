#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

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


def write_json(path: Path, obj: Dict[str, Any]):
    path.write_text(json.dumps(obj, indent=2))


def save_code_state(out_dir: Path):
    files = []
    for p in [Path(__file__), Path('doa_rl/omp/soft_omp.py')]:
        try:
            import hashlib
            h = hashlib.sha256(Path(p).read_bytes()).hexdigest()
            files.append({'path': str(p), 'sha256': h})
        except Exception:
            pass
    (out_dir / 'code_state.json').write_text(json.dumps({'files': files}, indent=2))


def plot_confusion(cm: np.ndarray, labels: List[int], out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title('DoA Confusion Matrix (counts)')
    ax.set_xlabel('Pred angle (deg)')
    ax.set_ylabel('GT angle (deg)')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_hist(errors: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(errors, bins=20, color='tab:orange', edgecolor='black')
    ax.set_title('Absolute Angle Error (deg)')
    ax.set_xlabel('|error| (deg)')
    ax.set_ylabel('count')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Evaluate Routed Soft-OMP DoA accuracy on real dataset')
    ap.add_argument('--dataset_root', type=str, required=True)
    ap.add_argument('--H_path', type=str, default='h_matrix_normalized_original_to_box.pth')
    ap.add_argument('--W_path', type=str, default='doa_normalized_config_c_corrected/models/usm.pth')
    ap.add_argument('--angles', type=int, nargs='*', default=None, help='Angles to evaluate; if omitted, use all angles present in dataset that exist in H')
    ap.add_argument('--freq_min', type=float, default=300.0)
    ap.add_argument('--freq_max', type=float, default=3000.0)
    ap.add_argument('--steps', type=int, default=6)
    ap.add_argument('--top_e', type=int, default=2)
    ap.add_argument('--L', type=int, default=2)
    ap.add_argument('--n_eval', type=int, default=200)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--out_dir', type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f'Device: {device}')

    # Load H
    tfp = TransferFunctionProcessor(NMFConfig(freq_min=args.freq_min, freq_max=args.freq_max, device=str(device)))
    H_all, angles_tensor, meta = tfp.load_transfer_functions(args.H_path)
    freqs = None
    try:
        d = torch.load(args.H_path, weights_only=False)
        if isinstance(d, dict) and 'freqs' in d:
            import numpy as _np
            freqs = _np.array(d['freqs'])
    except Exception:
        pass
    if freqs is not None:
        H_proc, _ = tfp.apply_frequency_limit(H_all, freqs, args.freq_min, args.freq_max)
    else:
        H_proc = H_all
    H = H_proc.float().to(device)

    # Determine requested angles: use provided, otherwise derive from dataset folders
    angle_list_all = [int(a) for a in (angles_tensor.cpu().numpy().tolist() if not isinstance(angles_tensor, list) else angles_tensor)]
    angle_to_idx_all = {int(a): i for i, a in enumerate(angle_list_all)}

    # If not provided, derive from dataset root
    if args.angles is None or len(args.angles) == 0:
        # scan dataset_root for angle_XX directories
        ds_angles = []
        for p in sorted(Path(args.dataset_root).glob('angle_*')):
            try:
                a = int(str(p.name).split('_')[-1])
                ds_angles.append(a)
            except Exception:
                continue
        if not ds_angles:
            raise ValueError('No angle_* folders found in dataset_root; specify --angles explicitly')
        req_angles = sorted(ds_angles)
    else:
        req_angles = [int(a) for a in args.angles]

    # Validate against H angles with no-fallback (dataset angles must be subset of H)
    if not set(req_angles).issubset(set(angle_list_all)):
        missing = sorted(set(req_angles) - set(angle_list_all))
        raise ValueError(f"Requested angles not in H: {missing}")

    idxs = torch.tensor([angle_to_idx_all[a] for a in req_angles], dtype=torch.long)
    H = H[:, idxs]
    angles_subset = torch.tensor(req_angles, dtype=torch.float32)

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
    F, E, M = H.shape[0], H.shape[1], W.shape[1]
    print(f"Loaded H: {tuple(H.shape)}, W: {tuple(W.shape)} (F={F}, E={E}, M={M})")

    # Build dictionary
    D, idx2 = build_dictionary(W, H)
    D = D.to(device)

    # Dataset and strict angle set equality check
    ds = DoADataset(args.dataset_root, angles=req_angles, fs=NMFConfig.sample_rate, n_fft=NMFConfig.n_fft,
                    freq_min=args.freq_min, freq_max=args.freq_max)
    # Verify dataset angle set equals requested/H subset
    ds_angles_present = sorted({int(a) for _, a, _ in ds.index}) if hasattr(ds, 'index') else req_angles
    if sorted(req_angles) != ds_angles_present:
        raise ValueError(f"Dataset angles {ds_angles_present} differ from requested/H subset {sorted(req_angles)}")

    dl = create_dataloader(ds, batch_size=1, shuffle=False)

    # Model (no training; hard eval path only)
    model = TrainableRoutedSoftOMP(F=F, E=E, M=M, steps=args.steps, top_e=args.top_e, L=args.L,
                                   tau_e=0.5, tau_a=0.2, eta=0.5, routing='gumbel').to(device)
    model.eval()

    # Prepare mappings
    angle_to_group = {int(a): i for i, a in enumerate(req_angles)}

    # Evaluation loop
    preds_path = out_dir / 'predictions.jsonl'
    n_samples = 0
    n_correct = 0
    n_within5 = 0
    n_within10 = 0
    errors: List[float] = []
    cm = np.zeros((E, E), dtype=int)
    # Residual curves aggregation (overall and per-angle)
    steps = int(args.steps)
    curve_sum = np.zeros(steps, dtype=float)
    per_angle_sum: Dict[int, np.ndarray] = {a: np.zeros(steps, dtype=float) for a in req_angles}
    per_angle_count: Dict[int, int] = {a: 0 for a in req_angles}

    with preds_path.open('w') as fw:
        for batch in dl:
            if n_samples >= args.n_eval:
                break
            Y = batch['Y'][0].to(device)  # (F, N)
            if Y.shape[0] != F:
                raise ValueError(f"STFT grid mismatch: Y.F={Y.shape[0]} vs H/W.F={F}")
            y = Y.mean(dim=1)
            with torch.no_grad():
                x_hat, r_curve = model(y, D, train_mode=False)
                A_group = x_hat.view(E, M).abs().sum(dim=1)
                pred_idx = int(torch.argmax(A_group).item())
            gt_angle = float(batch['angle_deg'][0])
            if int(gt_angle) not in angle_to_group:
                raise ValueError(f"GT angle {gt_angle} not in requested set {req_angles}")
            gt_idx = int(angle_to_group[int(gt_angle)])
            pred_angle = float(angles_subset[pred_idx].item())
            err = abs(pred_angle - gt_angle)
            # angle wrap-around minimal error
            err = min(err, 360.0 - err)
            correct = int(pred_idx == gt_idx)
            within5 = int(err <= 5.0)
            within10 = int(err <= 10.0)
            n_samples += 1
            n_correct += correct
            n_within5 += within5
            n_within10 += within10
            errors.append(err)
            cm[gt_idx, pred_idx] += 1
            # accumulate residual curves
            rc = np.array(r_curve, dtype=float)
            if rc.shape[0] != steps:
                raise ValueError(f"Unexpected r_curve length {rc.shape[0]} vs steps={steps}")
            curve_sum += rc
            a_int = int(gt_angle)
            per_angle_sum[a_int] += rc
            per_angle_count[a_int] += 1
            rec = {
                'path': batch['path'][0],
                'angle_gt': gt_angle,
                'angle_gt_idx': gt_idx,
                'pred_angle': pred_angle,
                'pred_idx': pred_idx,
                'err_abs_deg': err,
                'correct_top1': bool(correct),
                'within_5deg': bool(within5),
                'within_10deg': bool(within10),
            }
            fw.write(json.dumps(rec) + '\n')

    errors_np = np.array(errors, dtype=float) if errors else np.zeros((0,), dtype=float)
    top1_acc = float(n_correct) / max(n_samples, 1)
    acc_5 = float(n_within5) / max(n_samples, 1)
    acc_10 = float(n_within10) / max(n_samples, 1)
    mae = float(errors_np.mean()) if n_samples > 0 else None
    med = float(np.median(errors_np)) if n_samples > 0 else None
    p95 = float(np.percentile(errors_np, 95)) if n_samples > 0 else None

    metrics = {
        'n_samples': n_samples,
        'angles': req_angles,
        'F': int(F), 'E': int(E), 'M': int(M),
        'top1_acc': top1_acc,
        'acc_within_5deg': acc_5,
        'acc_within_10deg': acc_10,
        'mae_deg': mae,
        'median_abs_err_deg': med,
        'p95_abs_err_deg': p95,
    }
    write_json(out_dir / 'doa_metrics.json', metrics)

    # Plots
    plot_confusion(cm, req_angles, out_dir / 'confusion_matrix.png')
    if n_samples > 0:
        plot_error_hist(errors_np, out_dir / 'error_hist.png')

    # Curves plot (restore curves.png expected by users)
    if n_samples > 0:
        mean_curve = curve_sum / float(n_samples)
        fig, axs = plt.subplots(1, 2, figsize=(12, 3))
        axs[0].plot(np.arange(1, steps + 1), mean_curve, label='mean over all samples')
        axs[0].set_title('Residual vs step (Eval)')
        axs[0].set_xlabel('step'); axs[0].set_ylabel('||r||'); axs[0].legend()
        # per-angle curves
        for a in req_angles:
            c = per_angle_sum[a] / max(per_angle_count[a], 1)
            axs[1].plot(np.arange(1, steps + 1), c, label=str(a))
        axs[1].set_title('Residual vs step per angle')
        axs[1].set_xlabel('step'); axs[1].set_ylabel('||r||'); axs[1].legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / 'curves.png', dpi=150)
        plt.close(fig)

    # Save code state
    save_code_state(out_dir)
    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to {out_dir}")


if __name__ == '__main__':
    main()
