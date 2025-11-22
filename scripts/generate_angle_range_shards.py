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
    ap.add_argument("--k", type=int, default=6, help="OMP steps.")
    ap.add_argument("--m", type=int, default=8, help="Atoms per expert after reduction.")
    ap.add_argument("--reduction-seed", type=int, default=200, help="Seed for kmeans atom reduction.")
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
    )

    max_samples = args.max_samples_per_shard if args.max_samples_per_shard > 0 else None
    generator = AngleRangeShardGenerator(config)

    for angle_range in selected_ranges:
        logger.info("Generating shard %s (%s)", angle_range.name, angle_range.angles())
        shard_dir = generator.generate_shard(angle_range, max_samples=max_samples)
        logger.info("Shard %s written to %s", angle_range.name, shard_dir)


if __name__ == "__main__":
    main()
