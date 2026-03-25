#!/usr/bin/env python3
"""
Audit the provenance and low-rank sensitivity of the canonical 37-angle H matrix.

Outputs are written under results/h_matrix_provenance_audit_<timestamp>/ and
include an exact rebuild of the canonical H matrix plus a small ablation table
covering stage, pooling, and STFT option changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import signal

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nmf_localizer.config.defaults import NMFConfig
from nmf_localizer.core.stft_unified_processor import STFTUnifiedProcessor
from nmf_localizer.core.transfer_functions import TransferFunctionProcessor

DEFAULT_RUN_DIR = REPO_ROOT / "results" / "omp_transformer_speech260_trainval_split_full_20251115_082341"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"


@dataclass(frozen=True)
class VariantSpec:
    stage: str
    original_root: Path
    box_root: Path
    pooling: str
    stft_mode: str
    detrend: bool | str
    boundary: str | None
    padded: bool


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sort_angle_dirs(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("angle_")],
        key=lambda path: int(path.name.split("_")[1]),
    )


def _load_numpy_h(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["H"], dtype=np.float64)


def _load_torch_h(path: Path) -> np.ndarray:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return np.asarray(data["H"].cpu().numpy(), dtype=np.float64)


def _load_torch_angles(path: Path) -> np.ndarray:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return np.asarray(data["angles"].cpu().numpy(), dtype=np.float64)


def _compare_arrays(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    diff = np.abs(reference - candidate)
    return {
        "shape_equal": bool(reference.shape == candidate.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "allclose_atol_1e-6": bool(np.allclose(reference, candidate, atol=1e-6, rtol=1e-6)),
        "allclose_atol_1e-4": bool(np.allclose(reference, candidate, atol=1e-4, rtol=1e-4)),
    }


def _singular_metrics(matrix: np.ndarray) -> tuple[float, int]:
    singular_values = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    energy = singular_values**2
    energy = energy / energy.sum()
    cumulative = np.cumsum(energy)
    top3 = float(cumulative[min(2, len(cumulative) - 1)])
    r90 = int(np.searchsorted(cumulative, 0.9) + 1)
    return top3, r90


def _low_rank_metrics(h_matrix: np.ndarray) -> dict[str, Any]:
    log_h = np.log(np.clip(h_matrix, 1e-12, None))
    direct_top3, direct_r90 = _singular_metrics(h_matrix)
    center_top3, center_r90 = _singular_metrics(h_matrix - h_matrix.mean(axis=1, keepdims=True))
    log_top3, log_r90 = _singular_metrics(log_h)
    log_center_top3, log_center_r90 = _singular_metrics(log_h - log_h.mean(axis=1, keepdims=True))
    return {
        "direct_top3": direct_top3,
        "direct_r90": direct_r90,
        "center_top3": center_top3,
        "center_r90": center_r90,
        "log_top3": log_top3,
        "log_r90": log_r90,
        "log_center_top3": log_center_top3,
        "log_center_r90": log_center_r90,
        "h_min": float(h_matrix.min()),
        "h_max": float(h_matrix.max()),
        "h_mean": float(h_matrix.mean()),
        "h_std": float(h_matrix.std()),
    }


def _compute_manual_variant(config: NMFConfig, spec: VariantSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_columns: list[np.ndarray] = []
    freqs_filtered: np.ndarray | None = None
    angles: list[float] = []

    box_dirs = {int(path.name.split("_")[1]): path for path in _sort_angle_dirs(spec.box_root)}

    for original_dir in _sort_angle_dirs(spec.original_root):
        angle_deg = int(original_dir.name.split("_")[1])
        box_dir = box_dirs[angle_deg]
        original_files = sorted(original_dir.glob("*.npy"))[: config.n_files_per_angle]
        box_files = sorted(box_dir.glob("*.npy"))[: config.n_files_per_angle]

        per_file: list[np.ndarray] = []
        for original_file, box_file in zip(original_files, box_files):
            x = np.load(original_file).astype(np.float64).flatten()
            y = np.load(box_file).astype(np.float64).flatten()
            length = min(len(x), len(y))
            x = x[:length]
            y = y[:length]

            freqs, _, x_stft = signal.stft(
                x,
                fs=config.sample_rate,
                window=config.window,
                nperseg=config.n_fft,
                noverlap=config.n_fft - config.hop_length,
                nfft=config.n_fft,
                detrend=spec.detrend,
                boundary=spec.boundary,
                padded=spec.padded,
                return_onesided=True,
            )
            freqs_y, _, y_stft = signal.stft(
                y,
                fs=config.sample_rate,
                window=config.window,
                nperseg=config.n_fft,
                noverlap=config.n_fft - config.hop_length,
                nfft=config.n_fft,
                detrend=spec.detrend,
                boundary=spec.boundary,
                padded=spec.padded,
                return_onesided=True,
            )
            if not np.allclose(freqs, freqs_y):
                raise ValueError(f"Frequency mismatch for angle {angle_deg}")

            h_stft = np.abs(y_stft / (x_stft + 1e-12))
            if spec.pooling == "linear":
                pooled = h_stft.mean(axis=1)
            elif spec.pooling == "geometric":
                pooled = np.exp(np.mean(np.log(h_stft + 1e-12), axis=1))
            else:
                raise ValueError(f"Unsupported pooling: {spec.pooling}")
            per_file.append(pooled)

        per_file_array = np.stack(per_file, axis=0)
        if spec.pooling == "linear":
            h_angle = per_file_array.mean(axis=0)
        else:
            h_angle = np.exp(np.mean(np.log(np.clip(per_file_array, 1e-12, None)), axis=0))

        freq_mask = (freqs >= config.freq_min) & (freqs <= config.freq_max)
        if freqs_filtered is None:
            freqs_filtered = freqs[freq_mask]
        h_columns.append(h_angle[freq_mask])
        angles.append(float(angle_deg))

    if freqs_filtered is None:
        raise RuntimeError("No frequency bins were produced for the manual audit variant.")

    return np.stack(h_columns, axis=1), np.asarray(angles, dtype=np.float64), freqs_filtered


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, output_dir: Path, comparisons: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    rows_by_key = {(row["stage"], row["pooling"], row["stft_mode"]): row for row in rows}
    canonical_row = rows_by_key[("sync_vad", "geometric", "official")]
    normalized_row = rows_by_key[("sync_vad_normalized", "geometric", "official")]
    linear_row = rows_by_key[("sync_vad", "linear", "official")]
    dataset_like_row = rows_by_key[("sync_vad", "geometric", "dataset_like")]

    lines = [
        "# H Matrix Provenance Audit",
        "",
        f"- Output directory: `{output_dir}`",
        f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Canonical Rebuild Match",
        "",
        "| Target | Max abs diff | Mean abs diff | allclose(1e-6) |",
        "| --- | ---: | ---: | --- |",
    ]
    for label, result in comparisons.items():
        lines.append(
            f"| {label} | {result['max_abs_diff']:.3e} | {result['mean_abs_diff']:.3e} | "
            f"{'yes' if result['allclose_atol_1e-6'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Key Findings",
            "",
            (
                f"- Canonical `sync_vad + geometric + official` rebuild preserves strong low-rank structure in "
                f"`|H|`: top-3 cumulative energy `{canonical_row['direct_top3']:.3f}`, `r90={canonical_row['direct_r90']}`."
            ),
            (
                f"- The same canonical rebuild becomes much less low-rank after row-wise centering: "
                f"centered `|H|` top-3 `{canonical_row['center_top3']:.3f}`, `r90={canonical_row['center_r90']}`."
            ),
            (
                f"- Switching only to `linear` pooling is the largest upstream change tested: centered `|H|` shifts "
                f"from top-3 `{canonical_row['center_top3']:.3f}`, `r90={canonical_row['center_r90']}` to "
                f"`{linear_row['center_top3']:.3f}`, `r90={linear_row['center_r90']}`."
            ),
            (
                f"- Switching only to `sync_vad_normalized` does not reproduce the canonical artifact and weakens the "
                f"canonical direct `|H|` concentration slightly: top-3 `{normalized_row['direct_top3']:.3f}`, "
                f"`r90={normalized_row['direct_r90']}`."
            ),
            (
                f"- Switching only STFT boundary/detrend settings leaves the canonical metrics nearly unchanged: "
                f"centered `|H|` top-3 `{dataset_like_row['center_top3']:.3f}`, `r90={dataset_like_row['center_r90']}`."
            ),
            "",
            "## Interpretation",
            "",
            "- Upstream provenance: the current canonical 37-angle `H` used by the primary speech260 run is reproduced from the paired `sync_vad` white-noise roots, not from the normalized roots.",
            "- Downstream analysis: for Fig. 2 style SVD, row-wise centering is the most destructive preprocessing step; log alone concentrates energy, but log plus centering spreads it back out.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit canonical H-matrix provenance and low-rank sensitivity.")
    parser.add_argument(
        "--original-root",
        type=Path,
        default=Path("/Users/sbplab/LDV-data-processed/white_noise_original_data_no_edge_sync_vad"),
        help="Canonical Original/X root used to rebuild the 37-angle H matrix.",
    )
    parser.add_argument(
        "--box-root",
        type=Path,
        default=Path("/Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad"),
        help="Canonical Box/Y root used to rebuild the 37-angle H matrix.",
    )
    parser.add_argument(
        "--normalized-original-root",
        type=Path,
        default=Path("/Users/sbplab/LDV-data-processed/white_noise_original_data_no_edge_sync_vad_normalized"),
        help="Normalized Original/X root for the sensitivity audit.",
    )
    parser.add_argument(
        "--normalized-box-root",
        type=Path,
        default=Path("/Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized"),
        help="Normalized Box/Y root for the sensitivity audit.",
    )
    parser.add_argument(
        "--canonical-h-path",
        type=Path,
        default=Path("/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth"),
        help="Canonical external H artifact to compare against.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Primary speech260 run directory containing preprocessing.pth and dictionary.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to results/h_matrix_provenance_audit_<timestamp>/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (DEFAULT_RESULTS_ROOT / f"h_matrix_provenance_audit_{_timestamp()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = NMFConfig(
        sample_rate=16000,
        n_fft=2048,
        hop_length=512,
        freq_min=300.0,
        freq_max=3000.0,
        n_files_per_angle=3,
    )

    processor = STFTUnifiedProcessor(config)
    tf_processor = TransferFunctionProcessor(config)

    rebuilt_h, rebuilt_angles, _angle_folders, rebuilt_meta = processor.estimate_transfer_functions_stft(
        args.original_root,
        args.box_root,
        method="stft_unified",
        time_pooling="geometric",
    )
    rebuilt_processed, processing_info = tf_processor.process_transfer_functions(
        rebuilt_h,
        rebuilt_angles,
        freqs=np.asarray(rebuilt_meta["freqs"]),
    )
    rebuilt_save_path = output_dir / "h_rebuilt_sync_vad_geometric.pth"
    tf_processor.save_transfer_functions(
        rebuilt_processed,
        rebuilt_angles,
        rebuilt_save_path,
        metadata={
            **rebuilt_meta,
            **processing_info,
            "original_root": str(args.original_root),
            "box_root": str(args.box_root),
            "description": "Canonical 37-angle H rebuilt from white-noise sync_vad roots for provenance audit.",
        },
    )

    rebuilt_np = np.asarray(rebuilt_processed.cpu().numpy(), dtype=np.float64)
    canonical_np = _load_torch_h(args.canonical_h_path)
    preprocessing_np = _load_torch_h(args.run_dir / "preprocessing.pth")
    dictionary_np = _load_numpy_h(args.run_dir / "dictionary.npz")

    rebuilt_angles_np = np.asarray(rebuilt_angles.cpu().numpy(), dtype=np.float64)
    canonical_angles_np = _load_torch_angles(args.canonical_h_path)
    preprocessing_angles_np = _load_torch_angles(args.run_dir / "preprocessing.pth")
    dictionary_angles_np = np.asarray(np.load(args.run_dir / "dictionary.npz", allow_pickle=True)["angles"], dtype=np.float64)

    comparisons = {
        "canonical_h_path": {
            **_compare_arrays(rebuilt_np, canonical_np),
            "angles_equal": bool(np.array_equal(rebuilt_angles_np, canonical_angles_np)),
        },
        "run_preprocessing_pth": {
            **_compare_arrays(rebuilt_np, preprocessing_np),
            "angles_equal": bool(np.array_equal(rebuilt_angles_np, preprocessing_angles_np)),
        },
        "run_dictionary_npz": {
            **_compare_arrays(rebuilt_np, dictionary_np),
            "angles_equal": bool(np.array_equal(rebuilt_angles_np, dictionary_angles_np)),
        },
        "canonical_vs_run_preprocessing": {
            **_compare_arrays(canonical_np, preprocessing_np),
            "angles_equal": bool(np.array_equal(canonical_angles_np, preprocessing_angles_np)),
        },
        "canonical_vs_run_dictionary": {
            **_compare_arrays(canonical_np, dictionary_np),
            "angles_equal": bool(np.array_equal(canonical_angles_np, dictionary_angles_np)),
        },
    }

    variants = [
        VariantSpec(
            stage="sync_vad",
            original_root=args.original_root,
            box_root=args.box_root,
            pooling="geometric",
            stft_mode="official",
            detrend=False,
            boundary="zeros",
            padded=True,
        ),
        VariantSpec(
            stage="sync_vad",
            original_root=args.original_root,
            box_root=args.box_root,
            pooling="linear",
            stft_mode="official",
            detrend=False,
            boundary="zeros",
            padded=True,
        ),
        VariantSpec(
            stage="sync_vad",
            original_root=args.original_root,
            box_root=args.box_root,
            pooling="geometric",
            stft_mode="dataset_like",
            detrend="constant",
            boundary=None,
            padded=False,
        ),
        VariantSpec(
            stage="sync_vad",
            original_root=args.original_root,
            box_root=args.box_root,
            pooling="linear",
            stft_mode="dataset_like",
            detrend="constant",
            boundary=None,
            padded=False,
        ),
        VariantSpec(
            stage="sync_vad_normalized",
            original_root=args.normalized_original_root,
            box_root=args.normalized_box_root,
            pooling="geometric",
            stft_mode="official",
            detrend=False,
            boundary="zeros",
            padded=True,
        ),
        VariantSpec(
            stage="sync_vad_normalized",
            original_root=args.normalized_original_root,
            box_root=args.normalized_box_root,
            pooling="linear",
            stft_mode="official",
            detrend=False,
            boundary="zeros",
            padded=True,
        ),
        VariantSpec(
            stage="sync_vad_normalized",
            original_root=args.normalized_original_root,
            box_root=args.normalized_box_root,
            pooling="geometric",
            stft_mode="dataset_like",
            detrend="constant",
            boundary=None,
            padded=False,
        ),
        VariantSpec(
            stage="sync_vad_normalized",
            original_root=args.normalized_original_root,
            box_root=args.normalized_box_root,
            pooling="linear",
            stft_mode="dataset_like",
            detrend="constant",
            boundary=None,
            padded=False,
        ),
    ]

    low_rank_rows: list[dict[str, Any]] = []
    for spec in variants:
        h_matrix, angles_np, freqs_np = _compute_manual_variant(config, spec)
        row = {
            "stage": spec.stage,
            "pooling": spec.pooling,
            "stft_mode": spec.stft_mode,
            "detrend": str(spec.detrend),
            "boundary": "None" if spec.boundary is None else spec.boundary,
            "padded": spec.padded,
            "freq_bins": int(len(freqs_np)),
            "freq_lo_hz": float(freqs_np[0]),
            "freq_hi_hz": float(freqs_np[-1]),
            "angles_count": int(len(angles_np)),
            "angles_match_canonical": bool(np.array_equal(angles_np, canonical_angles_np)),
        }
        row.update(_low_rank_metrics(h_matrix))
        if spec.stage == "sync_vad" and spec.pooling == "geometric" and spec.stft_mode == "official":
            row.update({
                "max_abs_diff_vs_canonical": comparisons["canonical_h_path"]["max_abs_diff"],
                "mean_abs_diff_vs_canonical": comparisons["canonical_h_path"]["mean_abs_diff"],
                "allclose_vs_canonical_1e6": comparisons["canonical_h_path"]["allclose_atol_1e-6"],
            })
        else:
            compare = _compare_arrays(canonical_np, h_matrix)
            row.update({
                "max_abs_diff_vs_canonical": compare["max_abs_diff"],
                "mean_abs_diff_vs_canonical": compare["mean_abs_diff"],
                "allclose_vs_canonical_1e6": compare["allclose_atol_1e-6"],
            })
        low_rank_rows.append(row)

    rebuild_match_path = output_dir / "rebuild_match.json"
    rebuild_match_path.write_text(
        json.dumps(
            {
                "canonical_roots": {
                    "original_root": str(args.original_root),
                    "box_root": str(args.box_root),
                },
                "normalized_roots": {
                    "original_root": str(args.normalized_original_root),
                    "box_root": str(args.normalized_box_root),
                },
                "canonical_h_path": str(args.canonical_h_path),
                "run_dir": str(args.run_dir),
                "comparisons": comparisons,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    csv_path = output_dir / "low_rank_sensitivity.csv"
    _write_csv(csv_path, low_rank_rows)

    markdown_path = output_dir / "low_rank_sensitivity.md"
    _write_markdown(markdown_path, output_dir, comparisons, low_rank_rows)

    print(f"[h-audit] Output directory: {output_dir}")
    print(f"[h-audit] Rebuilt H saved to: {rebuilt_save_path}")
    print(f"[h-audit] Match report: {rebuild_match_path}")
    print(f"[h-audit] Low-rank table: {csv_path}")
    print(f"[h-audit] Summary: {markdown_path}")


if __name__ == "__main__":
    main()
