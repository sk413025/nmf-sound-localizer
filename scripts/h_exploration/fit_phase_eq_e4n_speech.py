import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from scripts.h_exploration.dataset_lag import DoALagDataset
from scripts.h_exploration.run_rtgomp_e4h_paper_eval import (
    P2Quantile,
    RunningStats,
    apply_pairing_mode,
    design_bandpass,
    gcc_phat_core,
    gcc_phat_pick_peak,
    generate_subset_manifest,
    load_wav_float32,
    next_pow2,
    validate_manifest,
)

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_or_make_manifest(
    *,
    dataset: DoALagDataset,
    mic_root: Path,
    ldv_root: Path,
    num_pairs: int,
    manifest_path: Path,
) -> Dict[str, object]:
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = generate_subset_manifest(
            dataset,
            str(mic_root),
            str(ldv_root),
            int(num_pairs),
            manifest_path,
        )
        logger.info("Wrote subset manifest: %s", manifest_path)

    missing_files, md5_mismatches = validate_manifest(dataset, manifest)
    if missing_files > 0:
        raise SystemExit(f"Missing files in manifest: {missing_files}")
    if md5_mismatches > 0:
        raise SystemExit(f"MD5 mismatches in manifest: {md5_mismatches}")
    return manifest


def _inband_mask(freqs_hz: np.ndarray, *, f_lo: float, f_hi: float) -> np.ndarray:
    return (freqs_hz >= float(f_lo)) & (freqs_hz <= float(f_hi))


def _unit_phase(z: np.ndarray, *, eps: float) -> np.ndarray:
    return z / (np.abs(z) + float(eps))


def fit_phase_eq(
    *,
    dataset: DoALagDataset,
    num_pairs: int,
    mic_root: Path,
    ldv_root: Path,
    mode: str,
    fs: int,
    hop_length: int,
    n_fft: int,
    tw: int,
    max_lag: int,
    search_radius_frames: int,
    freq_min: float,
    freq_max: float,
    out_dir: Path,
    seed: int,
) -> Tuple[Path, Dict[str, object]]:
    """
    Fit a per-frequency unit-magnitude complex phase equalizer (all-pass) for LDV.

    Output is written under out_dir/phase_eq/phase_eq.npz and a summary JSON.
    """
    if num_pairs <= 0:
        raise SystemExit(f"Invalid num_pairs={num_pairs}")

    bp_b, bp_a = design_bandpass(int(fs), f_lo=float(freq_min), f_hi=float(freq_max))

    segment_len_samples = (int(tw) - 1) * int(hop_length) + int(n_fft)
    nfft_fit = next_pow2(4 * int(segment_len_samples))
    if int(nfft_fit) % int(n_fft) != 0:
        raise SystemExit(f"nfft_fit must be a multiple of n_fft (exact bin alignment). nfft_fit={nfft_fit}, n_fft={n_fft}")
    stride = int(nfft_fit) // int(n_fft)

    freqs_fit_hz = np.fft.rfftfreq(int(nfft_fit), d=1.0 / float(fs))
    band_sel_fit = _inband_mask(freqs_fit_hz, f_lo=float(freq_min), f_hi=float(freq_max))
    n_fit_bins = int(freqs_fit_hz.size)

    sum_w = np.zeros(n_fit_bins, dtype=np.float64)
    sum_wu = np.zeros(n_fit_bins, dtype=np.complex128)

    psr_q50 = P2Quantile(0.5)
    psr_q90 = P2Quantile(0.9)
    boundary_rate = RunningStats()

    num_windows_total = 0
    num_windows_used = 0
    num_segment_oob = 0

    for clip_idx in range(int(num_pairs)):
        mic_path, ldv_path = dataset.clips[clip_idx]

        item = dataset[clip_idx]
        ldv_stft = item["ldv_stft"]
        T_total = int(ldv_stft.shape[0])

        Lag_Min = -int(max_lag)
        Lag_Max = int(max_lag)
        start_limit = Lag_Max
        end_limit = T_total - int(tw) + Lag_Min
        if end_limit <= start_limit:
            continue

        valid_starts = list(range(start_limit, end_limit, int(tw)))
        if len(valid_starts) == 0:
            continue

        mic_wav = load_wav_float32(Path(mic_path), expected_fs=int(fs))
        ldv_wav = load_wav_float32(Path(ldv_path), expected_fs=int(fs))

        # Match scipy.signal.stft boundary='zeros' by pre-padding n_fft//2.
        pad_left = int(n_fft) // 2
        max_seg_end = int(max(valid_starts)) * int(hop_length) + int(segment_len_samples)
        mic_pad_right = max(pad_left, int(max_seg_end) - (pad_left + int(mic_wav.size)))
        ldv_pad_right = max(pad_left, int(max_seg_end) - (pad_left + int(ldv_wav.size)))
        mic_wav_pad = np.pad(mic_wav, (pad_left, mic_pad_right), mode="constant")
        ldv_wav_pad = np.pad(ldv_wav, (pad_left, ldv_pad_right), mode="constant")

        search_radius_samples = int(search_radius_frames) * int(hop_length)

        for start_t in valid_starts:
            num_windows_total += 1

            seg_start = int(start_t) * int(hop_length)
            seg_end = int(seg_start) + int(segment_len_samples)
            if seg_start < 0 or seg_end > int(mic_wav_pad.size) or seg_end > int(ldv_wav_pad.size) or seg_end <= seg_start:
                num_segment_oob += 1
                continue

            x_seg = mic_wav_pad[seg_start:seg_end]
            y_seg = ldv_wav_pad[seg_start:seg_end]
            if x_seg.size != y_seg.size:
                raise RuntimeError(f"Segment length mismatch: {x_seg.size} vs {y_seg.size}")

            lags, abs_cc, freqs_hz, cross_power = gcc_phat_core(
                x_seg,
                y_seg,
                fs=int(fs),
                b=bp_b,
                a=bp_a,
                nfft=int(nfft_fit),
            )
            tau_hat_s, _tau_hat_ms, psr, bhit = gcc_phat_pick_peak(
                lags,
                abs_cc,
                fs=int(fs),
                tau_center_samples=0,
                search_radius_samples=int(search_radius_samples),
            )
            if tau_hat_s is None or not np.isfinite(float(tau_hat_s)):
                continue

            num_windows_used += 1
            if psr is not None and np.isfinite(float(psr)):
                psr_q50.update(float(psr))
                psr_q90.update(float(psr))
            if bhit is not None:
                boundary_rate.update(1.0 if bool(bhit) else 0.0)

            tau_sec = float(tau_hat_s) / float(fs)
            delay_rot = np.exp(1j * 2.0 * np.pi * freqs_hz * tau_sec)
            u = _unit_phase(cross_power.astype(np.complex128, copy=False), eps=1e-12) * delay_rot
            w = np.abs(cross_power).astype(np.float64, copy=False)

            # Deterministic weighted circular mean over windows (per frequency).
            sum_w[band_sel_fit] += w[band_sel_fit]
            sum_wu[band_sel_fit] += (w * u)[band_sel_fit]

        if (clip_idx + 1) % 10 == 0 or (clip_idx + 1) == int(num_pairs):
            logger.info("Processed clip %d / %d", clip_idx + 1, int(num_pairs))

    # Compute weighted circular mean direction per frequency.
    defined = (sum_w > 0.0) & band_sel_fit & np.isfinite(sum_w)
    H_hat = np.zeros(n_fit_bins, dtype=np.complex128)
    H_hat[defined] = sum_wu[defined] / sum_w[defined]

    mag = np.abs(H_hat)
    mag_ok = defined & np.isfinite(mag) & (mag > 0.0)
    unit = np.ones(n_fit_bins, dtype=np.complex128)
    unit[mag_ok] = H_hat[mag_ok] / mag[mag_ok]

    phase_eq_rfft = np.ones(n_fit_bins, dtype=np.complex64)
    phase_eq_rfft[mag_ok] = np.conj(unit[mag_ok]).astype(np.complex64, copy=False)
    defined_mask_rfft = mag_ok.astype(bool, copy=False)

    # Exact bin mapping to STFT grid (must not interpolate).
    n_stft_bins = int(n_fft) // 2 + 1
    stft_ids = stride * np.arange(n_stft_bins, dtype=np.int64)
    if int(stft_ids[-1]) >= int(phase_eq_rfft.size):
        raise RuntimeError("STFT->rFFT mapping overflow; check nfft_fit/n_fft stride.")

    phase_eq_stft = phase_eq_rfft[stft_ids].astype(np.complex64, copy=False)
    defined_mask_stft = defined_mask_rfft[stft_ids].astype(bool, copy=False)
    freqs_stft_hz = np.fft.rfftfreq(int(n_fft), d=1.0 / float(fs)).astype(np.float64, copy=False)

    # Metrics / sanity checks.
    any_nan_or_inf = bool(
        (not np.isfinite(np.abs(phase_eq_rfft)).all())
        or (not np.isfinite(np.abs(phase_eq_stft)).all())
        or (not np.isfinite(sum_w).all())
        or (not np.isfinite(sum_wu.real).all())
        or (not np.isfinite(sum_wu.imag).all())
    )
    unit_mag_err = float(np.max(np.abs(np.abs(phase_eq_rfft.astype(np.complex128)) - 1.0)))
    inband_stft = _inband_mask(freqs_stft_hz, f_lo=float(freq_min), f_hi=float(freq_max))
    inband_defined_fraction_stft = (
        float(np.mean(defined_mask_stft[inband_stft])) if int(np.sum(inband_stft)) > 0 else None
    )

    fit_summary = {
        "config": {
            "mic_root": str(mic_root),
            "ldv_root": str(ldv_root),
            "mode": str(mode),
            "num_pairs": int(num_pairs),
            "pairing_mode": "paired",
            "require_wav_only": True,
            "fs": int(fs),
            "hop_length": int(hop_length),
            "n_fft": int(n_fft),
            "freq_min": float(freq_min),
            "freq_max": float(freq_max),
            "max_lag": int(max_lag),
            "tw": int(tw),
            "search_radius_frames": int(search_radius_frames),
            "search_radius_samples": int(search_radius_frames) * int(hop_length),
            "nfft_fit": int(nfft_fit),
            "stft_stride_from_rfft": int(stride),
            "seed": int(seed),
        },
        "stats": {
            "num_windows_total": int(num_windows_total),
            "num_windows_used": int(num_windows_used),
            "num_segment_oob": int(num_segment_oob),
            "fraction_defined_tau": float(num_windows_used / num_windows_total) if num_windows_total > 0 else None,
            "boundary_hit_rate": boundary_rate.mean(),
            "gcc_psr_p50": psr_q50.result(),
            "gcc_psr_p90": psr_q90.result(),
            "inband_defined_fraction_stft": inband_defined_fraction_stft,
            "phase_eq_unit_mag_max_err": unit_mag_err,
            "any_nan_or_inf": any_nan_or_inf,
        },
    }

    phase_eq_dir = out_dir / "phase_eq"
    _ensure_dir(phase_eq_dir)
    npz_path = phase_eq_dir / "phase_eq.npz"
    np.savez(
        npz_path,
        freqs_hz=freqs_stft_hz,
        phase_eq_stft=phase_eq_stft,
        defined_mask_stft=defined_mask_stft,
        nfft_fit=np.asarray(int(nfft_fit), dtype=np.int64),
        phase_eq_rfft=phase_eq_rfft,
        defined_mask_rfft=defined_mask_rfft,
    )
    (phase_eq_dir / "phase_eq_fit_summary.json").write_text(json.dumps(fit_summary, indent=2), encoding="utf-8")
    logger.info("Wrote phase_eq: %s", npz_path)

    return npz_path, fit_summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mic_root", type=str, required=True)
    p.add_argument("--ldv_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--subset_manifest", type=str, default=None)
    p.add_argument("--mode", type=str, choices=["scale_check_subset", "full_dataset", "smoke"], required=True)
    p.add_argument("--num_pairs", type=int, default=None)
    p.add_argument(
        "--pairing_mode",
        type=str,
        default="paired",
        choices=["paired", "mispair_shift1"],
    )
    p.add_argument("--require_wav_only", type=int, default=0)
    p.add_argument("--fs", type=int, default=16000)
    p.add_argument("--hop_length", type=int, default=160)
    p.add_argument("--n_fft", type=int, default=2048)
    p.add_argument("--freq_min", type=float, default=300.0)
    p.add_argument("--freq_max", type=float, default=3000.0)
    p.add_argument("--max_lag", type=int, default=50)
    p.add_argument("--tw", type=int, default=32)
    p.add_argument("--search_radius_frames", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if str(args.pairing_mode) != "paired":
        raise SystemExit("E4n FIT stage must use pairing_mode=paired (calibration requires correct pairing).")

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)
    _ensure_dir(out_dir / "summary")

    mic_root = Path(args.mic_root)
    ldv_root = Path(args.ldv_root)
    if not mic_root.exists():
        raise SystemExit(f"Missing mic_root: {mic_root}")
    if not ldv_root.exists():
        raise SystemExit(f"Missing ldv_root: {ldv_root}")

    dataset = DoALagDataset(
        str(mic_root),
        str(ldv_root),
        angle=None,
        hop_length=int(args.hop_length),
        fs=int(args.fs),
        n_fft=int(args.n_fft),
    )
    apply_pairing_mode(dataset, str(args.pairing_mode))

    if args.mode == "scale_check_subset":
        expected_pairs = 48
        if args.num_pairs is not None and int(args.num_pairs) != expected_pairs:
            raise SystemExit(f"scale_check_subset requires num_pairs={expected_pairs}")
        num_pairs = expected_pairs
    elif args.mode == "full_dataset":
        expected_pairs = len(dataset)
        if args.num_pairs is not None and int(args.num_pairs) != expected_pairs:
            raise SystemExit("full_dataset requires num_pairs=len(dataset)")
        num_pairs = expected_pairs
    else:
        num_pairs = 3 if args.num_pairs is None else int(args.num_pairs)
        if num_pairs <= 0:
            raise SystemExit("smoke mode requires num_pairs > 0")
        if num_pairs > len(dataset):
            raise SystemExit(f"smoke num_pairs={num_pairs} exceeds dataset length {len(dataset)}")

    if int(args.require_wav_only) == 1:
        for idx in range(int(num_pairs)):
            mic_path, ldv_path = dataset.clips[idx]
            if not str(mic_path).endswith(".wav") or not str(ldv_path).endswith(".wav"):
                raise SystemExit(
                    "require_wav_only=1 but found non-wav path in dataset subset: "
                    f"{mic_path} , {ldv_path}"
                )

    manifest_path = Path(args.subset_manifest) if args.subset_manifest else (out_dir / "subset_manifest.json")
    _load_or_make_manifest(
        dataset=dataset,
        mic_root=mic_root,
        ldv_root=ldv_root,
        num_pairs=int(num_pairs),
        manifest_path=manifest_path,
    )

    phase_eq_path, fit_summary = fit_phase_eq(
        dataset=dataset,
        num_pairs=int(num_pairs),
        mic_root=mic_root,
        ldv_root=ldv_root,
        mode=str(args.mode),
        fs=int(args.fs),
        hop_length=int(args.hop_length),
        n_fft=int(args.n_fft),
        tw=int(args.tw),
        max_lag=int(args.max_lag),
        search_radius_frames=int(args.search_radius_frames),
        freq_min=float(args.freq_min),
        freq_max=float(args.freq_max),
        out_dir=out_dir,
        seed=int(args.seed),
    )

    # Also write a lightweight pointer JSON for convenience.
    (out_dir / "summary" / "phase_eq_pointer.json").write_text(
        json.dumps({"phase_eq_path": str(phase_eq_path)}, indent=2),
        encoding="utf-8",
    )
    logger.info("Done. phase_eq=%s", phase_eq_path)
    logger.info("Fit stats: %s", json.dumps(fit_summary["stats"], indent=2))


if __name__ == "__main__":
    main()
