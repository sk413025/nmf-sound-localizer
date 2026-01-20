import argparse
import json
import math
from pathlib import Path
from typing import Any


def _rankdata(values: list[float]) -> list[float]:
    """
    Average ranks for ties, 1-based ranks.
    """
    indexed = list(enumerate(values))
    indexed.sort(key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearsonr(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx <= 0.0 or vy <= 0.0:
        return None
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / math.sqrt(vx * vy)


def spearman_rho(x: list[float], y: list[float]) -> float | None:
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearsonr(rx, ry)


def _get_summary(entry: dict[str, Any]) -> dict[str, Any]:
    summary = entry.get("summary")
    if isinstance(summary, dict):
        return summary
    return {}


def _as_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda_grid", type=str, required=True, help="Path to eval/lambda_grid.json")
    ap.add_argument(
        "--rho_threshold",
        type=float,
        default=-0.6,
        help="Recommended teacher monotonicity threshold: rho(lambda_c, steps_used_mean) <= this.",
    )
    ap.add_argument(
        "--min_action_change",
        type=float,
        default=0.05,
        help="Recommended student sensitivity threshold: max(action_change_rate_vs_ref) >= this.",
    )
    ap.add_argument(
        "--min_kl",
        type=float,
        default=0.0,
        help="Recommended student logit shift threshold: max(logits_kl_mean_vs_ref) > this.",
    )
    ap.add_argument("--out_json", type=str, default=None, help="Optional path to write acceptance_check.json")
    args = ap.parse_args()

    grid_path = Path(args.lambda_grid)
    payload = json.loads(grid_path.read_text())
    grid = payload.get("grid", [])
    if not isinstance(grid, list) or len(grid) == 0:
        raise SystemExit("lambda_grid.json missing non-empty 'grid' list")

    lambda_cs: list[float] = []
    steps_means: list[float] = []
    capture_means: list[float] = []
    action_changes: list[float] = []
    kls: list[float] = []

    per_entry = []
    for entry in grid:
        if not isinstance(entry, dict):
            continue
        c = _as_float(entry.get("lambda_c"))
        rtg0 = _as_float(entry.get("rtg0"))
        summary = _get_summary(entry)

        steps = _as_float(summary.get("steps_used_mean"))
        cap = _as_float(summary.get("final_capture_mean"))
        acr = _as_float(summary.get("action_change_rate_vs_ref"))
        kl = _as_float(summary.get("logits_kl_mean_vs_ref"))

        if c is None:
            continue

        if steps is not None:
            lambda_cs.append(c)
            steps_means.append(steps)

        if cap is not None:
            capture_means.append(cap)
        if acr is not None:
            action_changes.append(acr)
        if kl is not None:
            kls.append(kl)

        per_entry.append(
            {
                "lambda_c": c,
                "rtg0": rtg0,
                "steps_used_mean": steps,
                "final_capture_mean": cap,
                "action_change_rate_vs_ref": acr,
                "logits_kl_mean_vs_ref": kl,
            }
        )

    rho = spearman_rho(lambda_cs, steps_means) if len(lambda_cs) == len(steps_means) else None
    max_acr = max(action_changes) if action_changes else None
    max_kl = max(kls) if kls else None

    steps_min = min(steps_means) if steps_means else None
    steps_max = max(steps_means) if steps_means else None
    cap_min = min(capture_means) if capture_means else None
    cap_max = max(capture_means) if capture_means else None

    checks = {
        "teacher_monotonicity": {
            "value": rho,
            "threshold": float(args.rho_threshold),
            "pass": (rho is not None and rho <= float(args.rho_threshold)),
        },
        "teacher_non_degenerate_steps": {
            "value": {"min": steps_min, "max": steps_max},
            "pass": (steps_min is not None and steps_max is not None and steps_max > steps_min),
        },
        "student_action_change": {
            "value": max_acr,
            "threshold": float(args.min_action_change),
            "pass": (max_acr is not None and max_acr >= float(args.min_action_change)),
        },
        "student_logits_kl": {
            "value": max_kl,
            "threshold": float(args.min_kl),
            "pass": (max_kl is not None and max_kl > float(args.min_kl)),
        },
        "tradeoff_non_degenerate": {
            "value": {"steps_range": None, "capture_range": None},
            "pass": False,
        },
    }

    if steps_min is not None and steps_max is not None:
        steps_range = steps_max - steps_min
    else:
        steps_range = None
    if cap_min is not None and cap_max is not None:
        cap_range = cap_max - cap_min
    else:
        cap_range = None

    checks["tradeoff_non_degenerate"]["value"] = {"steps_range": steps_range, "capture_range": cap_range}
    checks["tradeoff_non_degenerate"]["pass"] = (
        steps_range is not None
        and cap_range is not None
        and (steps_range > 0.0)
        and (cap_range > 0.0)
    )

    overall_pass = all(v["pass"] for v in checks.values())

    result = {
        "overall_pass": overall_pass,
        "lambda_grid_path": str(grid_path),
        "checks": checks,
        "per_entry_summary": per_entry,
        "notes": [
            "This checker only validates the eval JSON structure and summary statistics; full reproducibility requires rerunning commands with the same manifest and seeds.",
            "If only stopping changes but early actions do not, action_change_rate may be small; consider adding STOP token, variable-horizon eval, or lookahead teacher to induce action-level differences.",
        ],
    }

    print(json.dumps(result, indent=2))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

