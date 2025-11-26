#!/usr/bin/env python3
"""Generate angle-range domain-randomized shards with shared W/H."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

from doa_rl.domain_randomization import AngleRange, AngleRangeShardGenerator, GenerationConfig

DEFAULT_DATA_ROOT = Path("/Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized")
DEFAULT_H_PATH = Path("/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth")
DEFAULT_W_PATH = Path("doa_normalized_config_c_corrected/models/usm.pth")
DEFAULT_OUTPUT_ROOT = Path("results/angle_range_shards")


def _default_ranges(include_overlap: bool) -> Dict[str, AngleRange]:
    ranges = {
        "full": AngleRange("full", 0, 180),
        "low": AngleRange("low", 0, 60),
        "mid": AngleRange("mid", 65, 115),
        "high": AngleRange("high", 120, 180),
        # Targeted ranges to oversample weak/strong angles (5 deg step assumed)
        "weak_0_50": AngleRange("weak_0_50", 0, 50),
        "weak_140_170": AngleRange("weak_140_170", 140, 170),
        "strong_edges": AngleRange("strong_edges", 170, 180),
    }
    if include_overlap:
        ranges["overlap_30_90"] = AngleRange("overlap_30_90", 30, 90)
        ranges["overlap_90_150"] = AngleRange("overlap_90_150", 90, 150)
    return ranges


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate shared-dictionary shards for angle ranges.")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Input dataset root (white noise box data).")
    ap.add_argument("--h-path", type=Path, default=DEFAULT_H_PATH, help="Path to H matrix (.pth).")
    ap.add_argument("--w-path", type=Path, default=DEFAULT_W_PATH, help="Path to W matrix (.pth).")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Where to write shard outputs.")
    ap.add_argument("--clips-per-angle", type=int, default=3, help="Clips to use per angle (no fallback).")
    ap.add_argument("--k", type=int, default=6, help="OMP steps (DTMin trajectory length).")
    ap.add_argument("--m", type=int, default=50, help="Atoms per expert. Use 50 (full W) to preserve dictionary fidelity; 8 for compressed (legacy).")
    ap.add_argument("--reduction-seed", type=int, default=200, help="Seed for kmeans atom reduction (only if m < full W atoms).")
    ap.add_argument("--projection-seed", type=int, default=200, help="Seed for projection layers.")
    ap.add_argument("--include-overlap", action="store_true", help="Also generate overlap shards (30–90, 90–150).")
    ap.add_argument(
        "--range-names",
        type=str,
        default="full,low,mid,high",
        help="Comma-separated list of ranges to generate (names from default set).",
    )
    ap.add_argument(
        "--max-samples-per-shard",
        type=int,
        default=0,
        help="Optional cap on samples per shard (for smoke tests). 0 = no cap.",
    )
    ap.add_argument("--orig-sample-rate", type=int, default=48000, help="Original waveform sample rate.")
    ap.add_argument("--target-sample-rate", type=int, default=16000, help="Resampled STFT sample rate.")
    ap.add_argument("--n-fft", type=int, default=2048, help="FFT size.")
    ap.add_argument("--freq-min", type=float, default=300.0, help="Minimum frequency (Hz).")
    ap.add_argument("--freq-max", type=float, default=3000.0, help="Maximum frequency (Hz).")
    ap.add_argument("--d-model", type=int, default=128, help="Projection dimension for embeddings.")
    ap.add_argument("--early-stop-eps", type=float, default=0.0, help="Relative residual energy threshold for early stopping (0 disables).")
    ap.add_argument("--min-steps", type=int, default=1, help="Minimum OMP steps before early stop can trigger.")
    ap.add_argument("--early-stop-resid-ratio", type=float, default=0.0, help="Early stop when residual energy <= ratio * initial residual (0 disables).")
    ap.add_argument("--reduction-mode", type=str, default="kmeans", choices=["kmeans", "svd"], help="Atom reduction method when m < full.")
    ap.add_argument("--no-rtg", action="store_true", help="Disable RTG projection in embeddings (use only residual + step).")
    # === Soft-OMP parameters ===
    ap.add_argument("--soft-omp", action="store_true", help="Enable soft-OMP with Boltzmann sampling for trajectory diversity.")
    ap.add_argument("--temperature", type=float, default=1.0, help="Soft-OMP temperature (0=greedy, higher=more random).")
    ap.add_argument(
        "--temperatures",
        type=str,
        default=None,
        help="Comma-separated list of temperatures for multi-policy data collection (e.g., '0.1,0.5,1.0,2.0,5.0').",
    )
    ap.add_argument("--samples-per-temp", type=int, default=1, help="Number of trajectory samples per temperature per signal.")
    ap.add_argument(
        "--rtg-mode",
        type=str,
        default="return",
        choices=["return", "temperature", "entropy"],
        help="RTG computation mode: 'return' (cumulative reward), 'temperature' (1/τ), 'entropy' (selection confidence).",
    )
    ap.add_argument("--soft-omp-seed", type=int, default=None, help="Seed for soft-OMP sampling (None = random).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("generate_angle_range_shards")

    ranges = _default_ranges(args.include_overlap)
    requested = [name.strip() for name in args.range_names.split(",") if name.strip()]
    missing = [name for name in requested if name not in ranges]
    if missing:
        raise ValueError(f"Unknown range names: {missing}. Available: {sorted(ranges.keys())}")
    selected_ranges: List[AngleRange] = [ranges[name] for name in requested]

    # Parse temperatures if provided
    temperatures = None
    if args.temperatures:
        temperatures = tuple(float(t.strip()) for t in args.temperatures.split(",") if t.strip())

    config = GenerationConfig(
        data_root=args.data_root,
        w_path=args.w_path,
        h_path=args.h_path,
        output_root=args.output_root,
        clips_per_angle=args.clips_per_angle,
        k=args.k,
        m=args.m,
        reduction_seed=args.reduction_seed,
        projection_seed=args.projection_seed,
        orig_sample_rate=args.orig_sample_rate,
        target_sample_rate=args.target_sample_rate,
        n_fft=args.n_fft,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        d_model=args.d_model,
        normalize_w=True,
        normalize_d=True,
        early_stop_eps=args.early_stop_eps,
        min_steps=args.min_steps,
        early_stop_resid_ratio=args.early_stop_resid_ratio,
        reduction_mode=args.reduction_mode,
        use_rtg=not args.no_rtg,
        # Soft-OMP parameters
        soft_omp=args.soft_omp,
        temperature=args.temperature,
        temperatures=temperatures,
        samples_per_temp=args.samples_per_temp,
        rtg_mode=args.rtg_mode,
        soft_omp_seed=args.soft_omp_seed,
    )

    max_samples = args.max_samples_per_shard if args.max_samples_per_shard > 0 else None
    generator = AngleRangeShardGenerator(config)

    for angle_range in selected_ranges:
        logger.info("Generating shard %s (%s)", angle_range.name, angle_range.angles())
        shard_dir = generator.generate_shard(angle_range, max_samples=max_samples)
        logger.info("Shard %s written to %s", angle_range.name, shard_dir)


if __name__ == "__main__":
    main()
