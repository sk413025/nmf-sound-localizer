#!/usr/bin/env python3
"""
Audit E4o-Speech result directories for internal consistency.

What this checks (no re-running required):
  1) Subset manifests match across conditions (same fingerprint + file list).
  2) Stored summary numbers are reproducible from the raw JSONL using the exact
     same P2 quantile estimator and filtering logic.
  3) Compute-matched capture metrics are invariant across phase_eq sources
     (expected for unit-magnitude phase rotations).

Example:
  PYTHONPATH=. python scripts/h_exploration/audit_e4o_speech_results.py \\
    --lambda_c 3e-4 \\
    --run_dirs results/e4o_speech_scale_none results/e4o_speech_scale_perwin results/e4o_speech_scale_agg
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from scripts.h_exploration.run_rtgomp_e4h_paper_eval import P2Quantile


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _almost_equal(a: Optional[float], b: Optional[float], *, atol: float = 1e-12) -> bool:
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= float(atol)


@dataclass(frozen=True)
class SubsampleP2Metrics:
    num_windows: int
    num_defined: int
    num_phase_defined: int
    psr_p50: Optional[float]
    r2_p50: Optional[float]
    spread_p50: Optional[float]


def _recompute_subsample_p2_metrics(
    jsonl_path: Path,
    *,
    lambda_c: float,
    coarse_source: str,
) -> SubsampleP2Metrics:
    num_windows = 0
    num_defined = 0
    num_phase_defined = 0

    psr_q50 = P2Quantile(0.5)
    r2_q50 = P2Quantile(0.5)
    spread_q50 = P2Quantile(0.5)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("coarse_source") != coarse_source:
                continue
            if float(row.get("lambda_c")) != float(lambda_c):
                continue

            num_windows += 1

            tau_ms = row.get("gcc_phat_tau_hat_ms")
            if tau_ms is not None and math.isfinite(float(tau_ms)):
                num_defined += 1

            psr = row.get("gcc_phat_psr")
            if psr is not None and math.isfinite(float(psr)):
                psr_q50.update(float(psr))

            spread = row.get("tau_band_spread_ms")
            if spread is not None and math.isfinite(float(spread)):
                spread_q50.update(float(spread))

            # Match the summary logic: update phase-slope quantiles only when
            # phase_slope_tau_hat_ms is defined AND undefined_reason is None.
            if row.get("phase_slope_tau_hat_ms") is not None and row.get("undefined_reason") is None:
                r2 = row.get("phase_slope_r2")
                if r2 is not None and math.isfinite(float(r2)):
                    num_phase_defined += 1
                    r2_q50.update(float(r2))

    return SubsampleP2Metrics(
        num_windows=num_windows,
        num_defined=num_defined,
        num_phase_defined=num_phase_defined,
        psr_p50=psr_q50.result(),
        r2_p50=r2_q50.result() if num_phase_defined > 0 else None,
        spread_p50=spread_q50.result(),
    )


def _find_lambda_row(rows: List[Dict[str, Any]], lambda_c: float) -> Dict[str, Any]:
    for row in rows:
        if float(row.get("lambda_c")) == float(lambda_c):
            return row
    raise SystemExit(f"lambda_c={lambda_c} not found in summary rows.")


def _print_kv(title: str, items: Iterable[Tuple[str, Any]]) -> None:
    print(f"\n== {title} ==")
    for k, v in items:
        print(f"- {k}: {v}")


def _compare_manifests(manifests: List[Tuple[Path, Dict[str, Any]]]) -> None:
    if not manifests:
        return
    base_path, base = manifests[0]
    base_fp = base.get("fingerprint_md5")
    base_hashes = base.get("file_hashes")
    for path, m in manifests[1:]:
        if m.get("fingerprint_md5") != base_fp:
            raise SystemExit(
                "subset_manifest fingerprint mismatch:\n"
                f"- {base_path}: {base_fp}\n"
                f"- {path}: {m.get('fingerprint_md5')}"
            )
        if m.get("file_hashes") != base_hashes:
            raise SystemExit(f"subset_manifest file_hashes mismatch between {base_path} and {path}.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lambda_c", type=float, required=True)
    p.add_argument(
        "--run_dirs",
        type=str,
        nargs="+",
        required=True,
        help="E4o run directories (e.g., results/e4o_speech_scale_none ..._perwin ..._agg).",
    )
    p.add_argument("--atol", type=float, default=1e-12, help="Absolute tolerance for float comparisons.")
    args = p.parse_args()

    lambda_c = float(args.lambda_c)
    run_dirs = [Path(x) for x in args.run_dirs]
    for d in run_dirs:
        if not d.is_dir():
            raise SystemExit(f"Not a directory: {d}")

    manifests: List[Tuple[Path, Dict[str, Any]]] = []
    capture_by_dir: Dict[Path, Dict[str, float]] = {}

    # 1) Manifest equality across dirs.
    for d in run_dirs:
        mpath = d / "subset_manifest.json"
        if not mpath.exists():
            raise SystemExit(f"Missing subset_manifest.json: {mpath}")
        manifests.append((mpath, _load_json(mpath)))
    _compare_manifests(manifests)
    _print_kv(
        "Subset Manifest (Shared)",
        [
            ("fingerprint_md5", manifests[0][1].get("fingerprint_md5")),
            ("num_pairs", manifests[0][1].get("num_pairs")),
            ("num_files", manifests[0][1].get("num_files")),
            ("selection", manifests[0][1].get("selection")),
        ],
    )

    # 2) Recompute P2 metrics from JSONL and compare to stored summary.
    for d in run_dirs:
        jsonl_path = d / "subsample_delay_diagnostics.jsonl"
        summary_path = d / "summary" / "subsample_delay_diagnostics_summary.json"
        if not jsonl_path.exists():
            raise SystemExit(f"Missing subsample_delay_diagnostics.jsonl: {jsonl_path}")
        if not summary_path.exists():
            raise SystemExit(f"Missing summary/subsample_delay_diagnostics_summary.json: {summary_path}")
        summary = _load_json(summary_path)
        row = _find_lambda_row(summary["rows"], lambda_c)

        for source in ("dt", "omp"):
            recomputed = _recompute_subsample_p2_metrics(jsonl_path, lambda_c=lambda_c, coarse_source=source)
            stored = row[source]

            ok = True
            ok &= int(stored["num_windows"]) == int(recomputed.num_windows)
            ok &= int(stored["num_defined"]) == int(recomputed.num_defined)
            # phase_slope_fraction_defined is derived from counts; validate it via recomputed count.
            stored_phase_frac = stored.get("phase_slope_fraction_defined")
            recomputed_phase_frac = (
                float(recomputed.num_phase_defined / recomputed.num_windows) if recomputed.num_windows > 0 else None
            )
            ok &= _almost_equal(stored_phase_frac, recomputed_phase_frac, atol=float(args.atol))

            ok &= _almost_equal(stored.get("psr_p50"), recomputed.psr_p50, atol=float(args.atol))
            ok &= _almost_equal(stored.get("tau_band_spread_ms_p50"), recomputed.spread_p50, atol=float(args.atol))
            ok &= _almost_equal(stored.get("phase_slope_r2_p50"), recomputed.r2_p50, atol=float(args.atol))

            status = "PASS" if ok else "FAIL"
            _print_kv(
                f"{d.name} :: subsample summary check ({source}, lambda_c={lambda_c}) [{status}]",
                [
                    ("stored.num_windows", stored.get("num_windows")),
                    ("recomputed.num_windows", recomputed.num_windows),
                    ("stored.psr_p50", stored.get("psr_p50")),
                    ("recomputed.psr_p50", recomputed.psr_p50),
                    ("stored.spread_p50", stored.get("tau_band_spread_ms_p50")),
                    ("recomputed.spread_p50", recomputed.spread_p50),
                    ("stored.r2_p50", stored.get("phase_slope_r2_p50")),
                    ("recomputed.r2_p50", recomputed.r2_p50),
                    ("stored.phase_frac", stored_phase_frac),
                    ("recomputed.phase_frac", recomputed_phase_frac),
                ],
            )
            if not ok:
                raise SystemExit(f"Summary mismatch detected for {d} ({source}).")

        # Capture summary (expected invariant to unit phase rotations).
        cap_path = d / "summary" / "compute_matched_summary.json"
        if cap_path.exists():
            cap = _load_json(cap_path)
            cap_row = _find_lambda_row(cap["rows"], lambda_c)
            capture_by_dir[d] = {
                "dt_capture_mean": float(cap_row["dt_capture_mean"]),
                "omp_capture_mean": float(cap_row["omp_capture_mean"]),
                "k_selected_mean": float(cap_row["k_selected_mean"]),
            }

    # 3) Capture invariance across dirs (if available).
    if capture_by_dir:
        capture_atol = max(float(args.atol), 1e-8)
        keys = ("dt_capture_mean", "omp_capture_mean", "k_selected_mean")
        base_dir = next(iter(capture_by_dir))
        base = capture_by_dir[base_dir]
        for d, metrics in capture_by_dir.items():
            for k in keys:
                if not _almost_equal(metrics[k], base[k], atol=capture_atol):
                    raise SystemExit(
                        "Compute-matched capture mismatch (should be invariant to unit phase rotations):\n"
                        f"- base={base_dir.name} {k}={base[k]}\n"
                        f"- curr={d.name} {k}={metrics[k]}"
                    )
        _print_kv(
            f"Compute-Matched Capture Invariance (lambda_c={lambda_c}, atol={capture_atol}) [PASS]",
            [(k, base[k]) for k in keys],
        )

    print("\nAll checks PASSED.")


if __name__ == "__main__":
    main()
