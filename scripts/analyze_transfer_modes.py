#!/usr/bin/env python3
"""
Transfer mode analysis for OMP Transformer Speech260 run (Phase 1).

Given a run directory (with code_state.json from commit 06bf65de), this script:
  1) Reloads the LDV transfer matrix H(f, θ) and angles.
  2) Builds a frequency×angle matrix H_log = log(|H|).
  3) Computes its SVD: H_log = U Σ V^T.
  4) Saves transfer_modes.npz under the run directory.
  5) Produces basic visualizations:
       - Singular values vs mode index.
       - Top-R frequency modes u_r(f) (300–3000 Hz).
       - Top-R angular modes v_r(θ).

This is a pure post-hoc analysis; it does not modify or retrain any models.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_run_state(run_dir: Path):
    code_state_path = run_dir / "code_state.json"
    if not code_state_path.is_file():
        raise SystemExit(f"Missing code_state.json in run_dir: {code_state_path}")
    with code_state_path.open("r") as f:
        code_state = json.load(f)
    return code_state


def build_freq_axis(F: int, fs: float = 16000.0, n_fft: float = 2048.0, f_min: float = 300.0, f_max: float = 3000.0):
    """
    Reconstruct the frequency axis corresponding to H(f,θ).

    We follow the same STFT configuration as in the main script:
      fs=16000, n_fft=2048, band [300, 3000] Hz, resulting in F=346 bins.
    """
    df = fs / n_fft
    k_start = int(np.ceil(f_min / df))
    freqs = (k_start + np.arange(F)) * df
    # Clamp to requested band (should already match F)
    mask = (freqs >= f_min) & (freqs <= f_max)
    freqs = freqs[mask]
    if freqs.shape[0] != F:
        # If there's a mismatch, fall back to a simple linspace over [300,3000]
        freqs = np.linspace(f_min, f_max, F)
    return freqs


def analyze_transfer_modes(run_dir: Path, device: str = "cpu", top_r: int = 6):
    code_state = load_run_state(run_dir)
    args_dict = code_state.get("args", {})

    h_path = args_dict.get("h_path")
    w_path = args_dict.get("w_path")
    if not h_path or not w_path:
        raise SystemExit("code_state.args is missing h_path or w_path.")

    # Reuse existing loader for H and angles
    import importlib

    omp_mod = importlib.import_module("scripts.omp-transformer-ldv")
    H, W, angles = omp_mod.load_raw_ldv_matrices(h_path, w_path, device="cpu")

    # H is (F, E) with non-negative real entries
    H_np = H.cpu().numpy()
    F, E = H_np.shape

    # Build frequency axis (Hz)
    freqs = build_freq_axis(F)
    angles_deg = np.array(angles, dtype=float)

    # Log-magnitude matrix for SVD (avoid log(0))
    eps = 1e-8
    H_log = np.log(np.clip(np.abs(H_np), eps, None))

    # Center each frequency row to remove global bias (optional but helps numerics)
    H_log_centered = H_log - H_log.mean(axis=1, keepdims=True)

    # SVD: H_log_centered = U Σ V^T
    U, S, Vt = np.linalg.svd(H_log_centered, full_matrices=False)
    V = Vt.T  # (E, E)

    # DOA encoding capacity metrics:
    #   - energy_r: normalized σ_r^2 (total log|H| energy per mode)
    #   - var_v_r: variance of v_r(θ) over DOA (DOA sensitivity)
    #   - doa_cap_r: energy_r * var_v_r (per-mode contribution to DOA encoding)
    #   - cumulative curves for both energy and doa_cap
    sigma_sq = S**2
    energy_r = sigma_sq / (sigma_sq.sum() + 1e-12)  # (E,)
    var_v_r = V.var(axis=0)  # variance over angles for each mode r
    doa_cap_r = energy_r * var_v_r
    doa_cap_r_norm = doa_cap_r / (doa_cap_r.sum() + 1e-12)
    cum_energy = np.cumsum(energy_r)
    cum_doa_cap = np.cumsum(doa_cap_r_norm)

    # Save transfer modes + capacity metrics to NPZ
    out_npz = run_dir / "transfer_modes.npz"
    np.savez(
        out_npz,
        U=U,
        S=S,
        V=V,
        freqs=freqs,
        angles_deg=angles_deg,
        H_log=H_log,
        H_log_centered=H_log_centered,
        energy_r=energy_r,
        var_v_r=var_v_r,
        doa_cap_r=doa_cap_r,
        doa_cap_r_norm=doa_cap_r_norm,
        cum_energy=cum_energy,
        cum_doa_cap=cum_doa_cap,
    )
    print(f"[tm] Saved transfer modes to: {out_npz}")

    # Visualizations
    # 1) Singular values and cumulative DOA encoding capacity
    r_idx = np.arange(1, E + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(r_idx, S, marker="o", label="σ_r (log scale)")
    ax2 = ax.twinx()
    ax2.plot(r_idx, cum_energy, "g-", marker="s", label="cum energy")
    ax2.plot(r_idx, cum_doa_cap, "r-", marker="^", label="cum DOA capacity")
    ax.set_xlabel("Mode index r")
    ax.set_ylabel("Singular value σ_r")
    ax2.set_ylabel("Cumulative fraction")
    ax.set_title("Transfer modes: singular values and DOA encoding capacity")
    ax.grid(True, alpha=0.3)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="lower right")
    fig.tight_layout()
    sv_path = run_dir / "transfer_modes_singular_values_and_capacity.png"
    fig.savefig(sv_path, dpi=200)
    plt.close(fig)
    print(f"[tm] Saved singular values / DOA capacity plot: {sv_path}")

    # 2) Top-R frequency modes u_r(f)
    R = min(top_r, E)
    fig, axes = plt.subplots(R, 1, figsize=(8, 2.0 * R), sharex=True)
    if R == 1:
        axes = [axes]
    for r in range(R):
        ax = axes[r]
        ax.plot(freqs, U[:, r])
        ax.set_ylabel(f"u_{r+1}(f)")
        ax.grid(True, alpha=0.3)
        if r == 0:
            ax.set_title("Top frequency modes u_r(f) (log|H| SVD)")
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.tight_layout()
    u_path = run_dir / "transfer_modes_u_top.png"
    fig.savefig(u_path, dpi=200)
    plt.close(fig)
    print(f"[tm] Saved top frequency modes plot: {u_path}")

    # 3) Top-R angular modes v_r(θ)
    fig, axes = plt.subplots(R, 1, figsize=(8, 2.0 * R), sharex=True)
    if R == 1:
        axes = [axes]
    for r in range(R):
        ax = axes[r]
        ax.plot(angles_deg, V[:, r], marker="o")
        ax.set_ylabel(f"v_{r+1}(θ)")
        ax.grid(True, alpha=0.3)
        if r == 0:
            ax.set_title("Top angular modes v_r(θ) (log|H| SVD)")
    axes[-1].set_xlabel("Angle (deg)")
    fig.tight_layout()
    v_path = run_dir / "transfer_modes_v_top.png"
    fig.savefig(v_path, dpi=200)
    plt.close(fig)
    print(f"[tm] Saved top angular modes plot: {v_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Transfer mode (SVD) analysis for OMP Transformer Speech260 run."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to training run directory (must contain code_state.json).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for loading tensors: cpu|mps|cuda (default: cpu).",
    )
    parser.add_argument(
        "--top_r",
        type=int,
        default=6,
        help="Number of leading modes to visualize (default: 6).",
    )

    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"run_dir does not exist: {run_dir}")

    analyze_transfer_modes(run_dir, device=args.device, top_r=args.top_r)


if __name__ == "__main__":
    main()
