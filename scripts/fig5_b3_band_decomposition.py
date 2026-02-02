#!/usr/bin/env python3
"""
Equivalence-preserving frequency-band decomposition for Figure 5 panel B3.

Spec: docs/fig5_b3_band_decomposition_spec.md

This script produces two band decompositions for a single sample:
1) Physics baseline decomposition equivalent to modal_routing_val.npz:g_energy_expert (sign-aligned).
2) Model-native QK routing score decomposition equivalent to modal_routing_val.npz:scores_expert
   via Integrated Gradients (IG) attribution over input frequency bins.

All outputs are written under --out_dir (default: results/fig5_b3_band_decomp_<timestamp>/).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: torch. Activate the project environment (e.g., conda env trl-training)."
    ) from exc

# Avoid Matplotlib writing to a non-writable default config directory (common on shared machines).
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))

import matplotlib.pyplot as plt

# Ensure repository root is importable when invoked as `python scripts/...`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import NA_matplotlib_guild as na_style


@dataclass(frozen=True)
class Band:
    name: str
    min_hz: float
    max_hz: float
    include_upper: bool


BANDS: tuple[Band, ...] = (
    Band("300-500", 300.0, 500.0, False),
    Band("500-1000", 500.0, 1000.0, False),
    Band("1000-2000", 1000.0, 2000.0, False),
    Band("2000-3000", 2000.0, 3000.0, True),  # inclusive upper to cover 3000 exactly
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Equivalence-preserving band decomposition for Fig5 B3 (physics + QK IG)."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="results/omp_transformer_speech260_trainval_split_full_20251115_082341",
        help="Training run directory containing dictionary.npz, modal_routing_val.npz, model_best.pth, code_state.json.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: results/fig5_b3_band_decomp_<timestamp>/).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Torch device for IG computation (default: cpu).",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=2048,
        help="Number of IG steps (default: 2048).",
    )
    parser.add_argument(
        "--ig_baseline",
        type=str,
        default="mean_val",
        choices=["mean_val", "zero"],
        help="IG baseline spectrum: mean_val (unit-normalized mean of Y_val) or zero (all zeros).",
    )
    parser.add_argument(
        "--ig_method",
        type=str,
        default="trapezoid",
        choices=["simpson", "trapezoid"],
        help="Numerical integration rule for IG over alpha in [0,1] (default: trapezoid).",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=None,
        help="Optional fixed sample index into modal_routing_val.npz:Y_val. If omitted, auto-select representative sample.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def now_compact() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_git_head(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return None


def safe_git_dirty(repo_root: Path) -> bool | None:
    try:
        out = subprocess.check_output(["git", "-C", str(repo_root), "status", "--porcelain=v1"], text=True)
        return bool(out.strip())
    except Exception:
        return None


def write_code_state(
    out_dir: Path,
    *,
    repo_root: Path,
    run_dir: Path,
    args: argparse.Namespace,
    dataset_fingerprint: str | None,
) -> None:
    code_files = [
        repo_root / "scripts" / "fig5_b3_band_decomposition.py",
        repo_root / "NA_matplotlib_guild.py",
        repo_root / "scripts" / "omp-transformer-ldv.py",
        repo_root / "doa_rl" / "data.py",
        repo_root / "docs" / "fig5_b3_band_decomposition_spec.md",
        repo_root / "docs" / "fig5_b3_band_decomposition_agent_prompt.md",
    ]
    input_files = [
        run_dir / "dictionary.npz",
        run_dir / "modal_routing_val.npz",
        run_dir / "model_best.pth",
        run_dir / "code_state.json",
    ]

    def collect(paths: list[Path]) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        for p in paths:
            if not p.is_file():
                continue
            try:
                rel = str(p.relative_to(repo_root))
            except ValueError:
                rel = str(p)
            entries.append({"path": rel, "sha256": sha256_file(p)})
        return entries

    payload = {
        "timestamp": iso_now(),
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", None),
        "mplotlib_configdir": os.environ.get("MPLCONFIGDIR", None),
        "git_head": safe_git_head(repo_root),
        "git_dirty": safe_git_dirty(repo_root),
        "run_dir": str(run_dir),
        "dataset_fingerprint": dataset_fingerprint,
        "args": vars(args),
        "code_files": collect(code_files),
        "input_files": collect(input_files),
    }
    write_json(out_dir / "code_state.json", payload)


class Logger:
    def __init__(self, log_path: Path):
        self._fp = log_path.open("w", encoding="utf-8")

    def close(self) -> None:
        self._fp.close()

    def log(self, msg: str) -> None:
        print(msg, flush=True)
        self._fp.write(msg + "\n")
        self._fp.flush()


def load_npz(path: Path) -> dict[str, Any]:
    return dict(np.load(path, allow_pickle=True))


def freqs_from_F(F: int) -> np.ndarray:
    return np.linspace(300.0, 3000.0, F)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise SystemExit("Requested --device mps, but torch.backends.mps is not available. Use --device cpu.")
        return torch.device("mps")
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("Requested --device cuda, but CUDA is not available. Use --device cpu.")
        return torch.device("cuda")
    return torch.device("cpu")


def build_band_masks(freqs_hz: np.ndarray) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for band in BANDS:
        if band.include_upper:
            mask = (freqs_hz >= band.min_hz) & (freqs_hz <= band.max_hz)
        else:
            mask = (freqs_hz >= band.min_hz) & (freqs_hz < band.max_hz)
        if not np.any(mask):
            raise ValueError(f"No bins for band {band.name} ({band.min_hz}-{band.max_hz} Hz).")
        masks.append(mask)

    # Guardrail: masks must be disjoint and cover all bins exactly once.
    covered = np.zeros_like(freqs_hz, dtype=np.int32)
    for mask in masks:
        covered += mask.astype(np.int32)
    if covered.min() != 1 or covered.max() != 1:
        bad = np.where(covered != 1)[0]
        raise ValueError(
            "Band partition error: bins must be covered exactly once. "
            f"Found {bad.size} bins with coverage != 1."
        )
    return masks


def find_representative_case(routing: dict[str, Any], angles_deg: np.ndarray) -> tuple[int, int, int]:
    scores_expert = np.asarray(routing["scores_expert"])
    g_energy_expert = np.asarray(routing["g_energy_expert"])
    labels = np.asarray(routing["labels"]).astype(int)

    qk_pred = scores_expert.argmax(axis=1)
    g_pred = g_energy_expert.argmax(axis=1)

    angle_145_idx = int(np.argmin(np.abs(angles_deg - 145.0)))
    angle_60_idx = int(np.argmin(np.abs(angles_deg - 60.0)))
    if float(angles_deg[angle_145_idx]) != 145.0:
        raise ValueError(f"Angle 145° not found in angles_deg; nearest={angles_deg[angle_145_idx]}.")
    if float(angles_deg[angle_60_idx]) != 60.0:
        raise ValueError(f"Angle 60° not found in angles_deg; nearest={angles_deg[angle_60_idx]}.")

    for idx in range(labels.shape[0]):
        if labels[idx] == angle_145_idx and g_pred[idx] == angle_60_idx and qk_pred[idx] == angle_145_idx:
            return idx, angle_145_idx, angle_60_idx

    qk_correct = qk_pred == labels
    g_wrong = g_pred != labels
    qk_fix = qk_correct & g_wrong
    if np.any(qk_fix):
        idx = int(np.where(qk_fix)[0][0])
        return idx, int(labels[idx]), int(g_pred[idx])

    return 0, int(labels[0]), int(g_pred[0])


def physics_band_decomposition(
    D: np.ndarray, y: np.ndarray, band_masks: list[np.ndarray], E: int, M: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Equivalent decomposition of g_energy_expert.

    Returns:
      g_energy_full: (E,)
      g_energy_band: (K,E) sign-aligned contributions that sum to g_energy_full
      g_energy_recon: (E,) = sum_k g_energy_band[k]
    """
    D64 = D.astype(np.float64, copy=False)
    y64 = y.astype(np.float64, copy=False)

    g_full = D64.T @ y64  # (P,)
    g_full_em = g_full.reshape(E, M)
    g_energy_full = np.abs(g_full_em).sum(axis=1)  # (E,)

    sign_full = np.sign(g_full)  # (P,)
    K = len(band_masks)
    g_energy_band = np.zeros((K, E), dtype=np.float64)
    for k, mask in enumerate(band_masks):
        g_k = D64[mask, :].T @ y64[mask]  # (P,)
        c_k = sign_full * g_k  # (P,) sign-aligned so that sum_k c_k == |g_full|
        g_energy_band[k] = c_k.reshape(E, M).sum(axis=1)  # (E,)

    g_energy_recon = g_energy_band.sum(axis=0)
    return g_energy_full, g_energy_band, g_energy_recon


def build_model_from_run(run_dir: Path, device: torch.device) -> torch.nn.Module:
    code_state_path = run_dir / "code_state.json"
    ckpt_path = run_dir / "model_best.pth"
    if not code_state_path.is_file():
        raise FileNotFoundError(f"Missing code_state.json: {code_state_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing model_best.pth: {ckpt_path}")

    code_state = json.loads(code_state_path.read_text())
    args_dict = code_state.get("args", {})

    import importlib

    omp_mod = importlib.import_module("scripts.omp-transformer-ldv")

    # Build model with the same hyperparameters as training.
    # Mirror scripts/eval_omp_transformer_split.py:build_model_from_args but keep it local here.
    angles = np.load(run_dir / "dictionary.npz", allow_pickle=True)["angles"]
    F = int(np.load(run_dir / "dictionary.npz", allow_pickle=True)["D"].shape[0])
    E = int(len(angles))
    M = int(args_dict.get("n_atoms", 8))

    d_model = int(args_dict.get("d_model", 64))
    nhead = int(args_dict.get("nhead", 2))
    nlayers = int(args_dict.get("nlayers", 1))

    if d_model % max(nhead, 1) != 0:
        for candidate in [2, 4, 8, 1]:
            if d_model % candidate == 0:
                nhead = candidate
                break
        else:
            nhead = 1

    model = omp_mod.FullTransformerRoutedSoftOMP(
        F=F,
        E=E,
        M=M,
        d=d_model,
        nhead=nhead,
        nlayers=nlayers,
        steps=int(args_dict.get("steps", 2)),
        top_e=int(args_dict.get("top_e", 2)),
        L=int(args_dict.get("top_l", 2)),
        tau_e=0.5,
        tau_a=0.2,
        eta=0.5,
        routing="gumbel",
        routing_mode=args_dict.get("routing_mode", "qk"),
        hybrid_alpha=float(args_dict.get("hybrid_alpha", 0.5)),
        routing_e=args_dict.get("routing_activation_e", "gumbel"),
        routing_a=args_dict.get("routing_activation_a", "gumbel"),
        score_norm_mode=args_dict.get("score_norm", "none"),
        score_reg_weight=float(args_dict.get("score_reg_weight", 0.0)),
        expert_agg=args_dict.get("expert_agg", "l2"),
    ).to(device)

    model.no_type_bias = bool(args_dict.get("no_type_bias", False))
    model.encoder_identity = bool(args_dict.get("encoder_identity", False))
    model.single_gate_expert = bool(args_dict.get("single_gate_expert", False))
    model.score_center_atoms = bool(args_dict.get("score_center_atoms", False))
    model.score_center_expert = bool(args_dict.get("score_center_expert", False))
    model.d_can_attend_r = bool(args_dict.get("d_can_attend_r", False))
    model.use_hard_gumbel = bool(args_dict.get("use_hard_gumbel", False))

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def model_scores_expert_step0(model: torch.nn.Module, y: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    # Use train_mode=True because the model stores step-0 routing signals in last_outputs there.
    # scores_expert is computed before any stochastic routing; it is deterministic under model.eval().
    _x, _r = model(y, D, train_mode=True)
    if getattr(model, "last_outputs", None) is None or "scores_expert" not in model.last_outputs:
        raise RuntimeError("Model did not populate last_outputs['scores_expert'].")
    return model.last_outputs["scores_expert"]


def ig_band_decomposition(
    model: torch.nn.Module,
    y: np.ndarray,
    y0: np.ndarray,
    D: np.ndarray,
    band_masks: list[np.ndarray],
    device: torch.device,
    n_steps: int,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrated Gradients (IG) band decomposition of model-native scores_expert.

    Returns:
      scores_full: (E,)
      scores_base: (E,) = f(y0)
      scores_band: (K,E) = IG band attributions (sum of per-bin IG per band)
      scores_recon: (E,) = scores_base + sum_k scores_band[k]
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if method not in ("simpson", "trapezoid"):
        raise ValueError(f"Invalid IG method: {method}")
    if method == "simpson" and (n_steps % 2) != 0:
        raise ValueError("Simpson's rule requires n_steps to be even.")

    D_t = torch.from_numpy(D.astype(np.float32, copy=False)).to(device)
    y_t = torch.from_numpy(y.astype(np.float32, copy=False)).to(device)
    y0_t = torch.from_numpy(y0.astype(np.float32, copy=False)).to(device)
    delta = y_t - y0_t

    # Speed: we only need step-0 scores, so run a single step.
    orig_steps = getattr(model, "steps", None)
    if orig_steps is not None:
        model.steps = 1

    with torch.no_grad():
        scores_base = model_scores_expert_step0(model, y0_t, D_t).detach().cpu().numpy().astype(np.float64)
        scores_full = model_scores_expert_step0(model, y_t, D_t).detach().cpu().numpy().astype(np.float64)

    E = int(scores_full.shape[0])
    K = len(band_masks)
    band_indices = [np.where(mask)[0] for mask in band_masks]

    scores_band = np.zeros((K, E), dtype=np.float64)

    for t in range(0, n_steps + 1):
        alpha = float(t) / float(n_steps)
        if method == "trapezoid":
            weight = 0.5 if t in (0, n_steps) else 1.0
        else:
            # Simpson weights: 1,4,2,4,...,2,4,1
            if t in (0, n_steps):
                weight = 1.0
            elif (t % 2) == 1:
                weight = 4.0
            else:
                weight = 2.0
        y_alpha = (y0_t + alpha * delta).detach().clone().requires_grad_(True)

        se = model_scores_expert_step0(model, y_alpha, D_t)  # (E,)
        if se.shape[0] != E:
            raise RuntimeError(f"Unexpected scores_expert shape: {tuple(se.shape)} (expected ({E},)).")

        for e in range(E):
            retain = e != E - 1
            grad = torch.autograd.grad(se[e], y_alpha, retain_graph=retain, create_graph=False)[0]
            if grad is None:
                raise RuntimeError("autograd.grad returned None; IG cannot proceed.")
            for k, idx in enumerate(band_indices):
                scores_band[k, e] += weight * (delta[idx] * grad[idx]).sum().detach().cpu().item()

    if method == "trapezoid":
        scores_band /= float(n_steps)
    else:
        # Simpson: multiply by h/3 with h=1/n_steps
        scores_band /= float(3 * n_steps)
    scores_recon = scores_base + scores_band.sum(axis=0)

    if orig_steps is not None:
        model.steps = orig_steps

    return scores_full, scores_base, scores_band, scores_recon


def plot_band_lines(
    out_path: Path,
    title: str,
    angles_deg: np.ndarray,
    full: np.ndarray,
    base: np.ndarray | None,
    band: np.ndarray,
    recon: np.ndarray,
    ylabel: str,
    band_labels: list[str],
    subtitle: str,
) -> None:
    na_style.set_nature_rcparams(base_fontsize=7)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif", "Computer Modern Roman"],
            "mathtext.fontset": "stix",
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
        }
    )

    fig = plt.figure(figsize=(na_style.mm_to_in(90), na_style.mm_to_in(60)))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.88)

    ax.grid(True, alpha=0.2, linestyle="--", color="gray")
    ax.set_axisbelow(True)

    ax.plot(angles_deg, full, color="black", linewidth=1.2, alpha=0.9, label="Full (target)")
    ax.plot(angles_deg, recon, color="black", linewidth=1.0, alpha=0.7, linestyle="--", label="Reconstruction")
    if base is not None:
        ax.plot(angles_deg, base, color="slategray", linewidth=0.9, alpha=0.8, linestyle=":", label="Baseline (y0)")

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for k in range(band.shape[0]):
        ax.plot(
            angles_deg,
            band[k],
            color=colors[k % len(colors)],
            linewidth=1.0,
            alpha=0.85,
            label=f"Band {band_labels[k]}",
        )

    ax.set_title(title + "\n" + subtitle)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(float(angles_deg[0]), float(angles_deg[-1]))

    tick_indices = [0, 9, 18, 27, 36]
    tick_indices = [i for i in tick_indices if i < len(angles_deg)]
    ax.set_xticks([angles_deg[i] for i in tick_indices])
    ax.set_xticklabels([f"{int(angles_deg[i])}°" for i in tick_indices])

    ax.legend(frameon=False, loc="upper right", handlelength=1.2, ncol=2)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def minmax_norm(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    denom = x_max - x_min
    if denom <= eps:
        return np.zeros_like(x, dtype=np.float64)
    return (x - x_min) / (denom + eps)


def plot_polar_full_and_bands(
    out_path: Path,
    *,
    angles_deg: np.ndarray,
    label_deg: float,
    physics_full: np.ndarray,
    qk_full: np.ndarray,
    physics_band: np.ndarray,
    qk_band: np.ndarray,
    band_labels: list[str],
) -> None:
    """
    Generate a compact polar (radar-style) comparison between:
      - full-band targets (physics_full, qk_full)
      - per-band contributions (physics_band[k], qk_band[k])

    Note:
    - Band contributions can be signed (physics sign-aligned; QK IG attribution).
    - For polar visualization we min-max normalize each curve to [0,1].
    """
    na_style.set_nature_rcparams(base_fontsize=6)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif", "Computer Modern Roman"],
            "mathtext.fontset": "stix",
            "axes.titlesize": 6,
            "legend.fontsize": 5,
        }
    )

    angles_rad = np.deg2rad(angles_deg)
    true_angle_rad = float(np.deg2rad(label_deg))

    fig = plt.figure(figsize=(na_style.mm_to_in(170), na_style.mm_to_in(110)))
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.06, top=0.92, wspace=0.35, hspace=0.45)

    panels: list[tuple[str, np.ndarray, np.ndarray]] = [("Full band (target)", physics_full, qk_full)]
    for k, name in enumerate(band_labels):
        panels.append((f"Band {name}", physics_band[k], qk_band[k]))

    for i, (title, phys, qk) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 3, i, projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1.1)

        phys_n = minmax_norm(phys)
        qk_n = minmax_norm(qk)

        ax.plot(angles_rad, phys_n, "o-", color="coral", linewidth=0.8, markersize=1.6, alpha=0.7, label="OMP")
        ax.fill_between(angles_rad, 0.0, phys_n, color="coral", alpha=0.18)

        ax.plot(angles_rad, qk_n, "s-", color="darkgreen", linewidth=0.9, markersize=1.6, alpha=0.9, label="QK")
        ax.fill_between(angles_rad, 0.0, qk_n, color="darkgreen", alpha=0.22)

        ax.plot([true_angle_rad, true_angle_rad], [0.0, 1.0], color="lime", linewidth=1.0, linestyle="--", alpha=0.9)

        ax.set_title(title)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        if i == 1:
            ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.15, 1.15), handlelength=1.0)

    # Hide the unused 6th subplot.
    ax_unused = fig.add_subplot(2, 3, 6)
    ax_unused.axis("off")

    fig.suptitle(
        "Fig5 B3 polar comparison (min-max normalized per curve)\n"
        "Bands show: Physics sign-aligned contribution; QK IG attribution (Δ from baseline)",
        fontsize=7,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"run_dir does not exist: {run_dir}")

    out_dir = Path(args.out_dir) if args.out_dir is not None else Path("results") / f"fig5_b3_band_decomp_{now_compact()}"
    ensure_dir(out_dir)

    logger = Logger(out_dir / "run.log")
    try:
        logger.log("=== Fig5 B3 Band Decomposition (Equivalence-Preserving) ===")
        logger.log(f"run_dir: {run_dir}")
        logger.log(f"out_dir: {out_dir}")
        logger.log(f"device: {args.device}")
        logger.log(f"n_steps (IG): {args.n_steps}")
        logger.log(f"ig_baseline: {args.ig_baseline}")
        logger.log(f"ig_method: {args.ig_method}")
        logger.log(f"cwd: {Path.cwd()}")
        logger.log(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")

        dict_data = load_npz(run_dir / "dictionary.npz")
        routing = load_npz(run_dir / "modal_routing_val.npz")
        code_state = json.loads((run_dir / "code_state.json").read_text()) if (run_dir / "code_state.json").is_file() else {}

        D = np.asarray(dict_data["D"])
        angles_deg = np.asarray(dict_data["angles"]).astype(np.float64)

        Y_all = np.asarray(routing["Y_val"])
        labels = np.asarray(routing["labels"]).astype(int)

        F, P = D.shape
        E = int(angles_deg.shape[0])
        if P % E != 0:
            raise ValueError(f"Dictionary shape mismatch: P={P} not divisible by E={E}.")
        M = P // E
        if M != 8:
            logger.log(f"[warn] Expected M=8 atoms per expert, got M={M}. Proceeding but verify assumptions.")

        if Y_all.shape[1] != F:
            raise ValueError(f"F mismatch: Y.F={Y_all.shape[1]} vs D.F={F}")
        if labels.shape[0] != Y_all.shape[0]:
            raise ValueError(f"N mismatch: labels.N={labels.shape[0]} vs Y.N={Y_all.shape[0]}")

        freqs_hz = freqs_from_F(F)
        band_masks = build_band_masks(freqs_hz)
        band_indices = [np.where(m)[0].tolist() for m in band_masks]

        # Choose sample
        if args.sample_idx is not None:
            sample_idx = int(args.sample_idx)
            if not (0 <= sample_idx < Y_all.shape[0]):
                raise ValueError(f"sample_idx out of range: {sample_idx} (N={Y_all.shape[0]})")
            true_expert = int(labels[sample_idx])
            wrong_expert = int(np.asarray(routing["g_energy_expert"])[sample_idx].argmax())
        else:
            sample_idx, true_expert, wrong_expert = find_representative_case(routing, angles_deg)

        y = Y_all[sample_idx]
        label_idx = int(labels[sample_idx])
        label_deg = float(angles_deg[label_idx])

        logger.log(f"sample_idx: {sample_idx}")
        logger.log(f"label_idx: {label_idx} (label_deg={label_deg:.1f})")
        logger.log(f"true_expert (selected): {true_expert} ({angles_deg[true_expert]:.1f} deg)")
        logger.log(f"wrong_expert (physics argmax): {wrong_expert} ({angles_deg[wrong_expert]:.1f} deg)")

        write_json(
            out_dir / "band_defs.json",
            {
                "freqs_hz": freqs_hz.tolist(),
                "bands": [
                    {
                        "name": band.name,
                        "min_hz": band.min_hz,
                        "max_hz": band.max_hz,
                        "include_upper": band.include_upper,
                        "bin_indices": idx,
                    }
                    for band, idx in zip(BANDS, band_indices, strict=True)
                ],
            },
        )

        # Physics decomposition (equivalent to g_energy_expert definition)
        g_energy_full, g_energy_band, g_energy_recon = physics_band_decomposition(D, y, band_masks, E=E, M=M)
        phys_err = np.max(np.abs(g_energy_full - g_energy_recon))
        logger.log(f"[physics] recon max_abs_error: {phys_err:.3e}")

        # Optional sanity: compare to stored g_energy_expert for this sample.
        stored_g = np.asarray(routing["g_energy_expert"])[sample_idx].astype(np.float64)
        stored_phys_err = np.max(np.abs(stored_g - g_energy_full))
        logger.log(f"[physics] vs stored modal_routing_val.npz:g_energy_expert max_abs_error: {stored_phys_err:.3e}")

        # Model-native IG decomposition (equivalent to scores_expert definition)
        device = resolve_device(args.device)
        model = build_model_from_run(run_dir, device=device)
        if args.ig_baseline == "zero":
            y0 = np.zeros_like(y)
        else:
            y0 = Y_all.mean(axis=0).astype(np.float64)
            y0 = y0 / (np.linalg.norm(y0) + 1e-12)
        scores_full, scores_base, scores_band, scores_recon = ig_band_decomposition(
            model=model,
            y=y,
            y0=y0,
            D=D,
            band_masks=band_masks,
            device=device,
            n_steps=int(args.n_steps),
            method=str(args.ig_method),
        )
        ig_err = np.max(np.abs(scores_full - scores_recon))
        logger.log(f"[qk_ig] recon max_abs_error: {ig_err:.3e}")

        stored_se = np.asarray(routing["scores_expert"])[sample_idx].astype(np.float64)
        stored_se_err = np.max(np.abs(stored_se - scores_full))
        logger.log(f"[qk_ig] vs stored modal_routing_val.npz:scores_expert max_abs_error: {stored_se_err:.3e}")

        # Acceptance checks (spec)
        qk_tol = None
        if args.ig_baseline == "mean_val" and args.ig_method == "trapezoid":
            if int(args.n_steps) >= 2048:
                qk_tol = 1e-3
            elif int(args.n_steps) >= 1024:
                qk_tol = 2e-3
        dataset_fingerprint = code_state.get("dataset_fingerprint", None)
        checks = {
            "run_dir": str(run_dir),
            "out_dir": str(out_dir),
            "sample_idx": int(sample_idx),
            "label_idx": int(label_idx),
            "label_deg": float(label_deg),
            "physics_recon_max_abs_error": float(phys_err),
            "physics_recon_tolerance": 1e-6,
            "qk_ig_recon_max_abs_error": float(ig_err),
            "qk_ig_recon_tolerance": qk_tol,
            "qk_ig_n_steps": int(args.n_steps),
            "qk_ig_method": str(args.ig_method),
            "qk_ig_baseline": str(args.ig_baseline),
            "physics_vs_stored_max_abs_error": float(stored_phys_err),
            "qk_vs_stored_max_abs_error": float(stored_se_err),
            "dataset_fingerprint": dataset_fingerprint,
            "git_head": code_state.get("git_head", None),
            "git_dirty": code_state.get("git_dirty", None),
        }
        write_json(out_dir / "checks.json", checks)

        write_code_state(
            out_dir,
            repo_root=_REPO_ROOT,
            run_dir=run_dir,
            args=args,
            dataset_fingerprint=dataset_fingerprint,
        )

        np.savez(
            out_dir / "decomposition.npz",
            angles_deg=angles_deg.astype(np.float64),
            freqs_hz=freqs_hz.astype(np.float64),
            sample_idx=int(sample_idx),
            label_idx=int(label_idx),
            label_deg=float(label_deg),
            g_energy_full=g_energy_full,
            g_energy_band=g_energy_band,
            g_energy_recon=g_energy_recon,
            scores_expert_full=scores_full,
            scores_expert_base=scores_base,
            scores_expert_band=scores_band,
            scores_expert_recon=scores_recon,
        )

        band_labels = [b.name for b in BANDS]

        plot_band_lines(
            out_path=out_dir / "Fig5_B3_BAND_DECOMP_PHYSICS.pdf",
            title="Fig5 B3 Physics band decomposition",
            angles_deg=angles_deg,
            full=g_energy_full,
            base=None,
            band=g_energy_band,
            recon=g_energy_recon,
            ylabel="g_energy_expert (equivalent)",
            band_labels=band_labels,
            subtitle="Sign-aligned decomposition (Σ bands = full)",
        )

        plot_band_lines(
            out_path=out_dir / "Fig5_B3_BAND_DECOMP_QK_IG.pdf",
            title="Fig5 B3 QK band decomposition",
            angles_deg=angles_deg,
            full=scores_full,
            base=scores_base,
            band=scores_band,
            recon=scores_recon,
            ylabel="scores_expert (model-native, equivalent)",
            band_labels=band_labels,
            subtitle="Integrated Gradients attribution (base + Σ bands = full)",
        )

        plot_polar_full_and_bands(
            out_path=out_dir / "Fig5_B3_POLAR_FULL_AND_BANDS.pdf",
            angles_deg=angles_deg,
            label_deg=label_deg,
            physics_full=g_energy_full,
            qk_full=scores_full,
            physics_band=g_energy_band,
            qk_band=scores_band,
            band_labels=band_labels,
        )

        logger.log(f"Wrote outputs to: {out_dir}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
