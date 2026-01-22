# Spec: E4f‑STOP — DTmin vs OMP K‑Sweep for Stop‑Action Head

This spec defines the **stop‑action head** branch for E4f when the checkpoint has
`head_out = M_lags + 1`. It evaluates DTmin vs OMP capture over K‑sweep while
respecting STOP actions.

This spec is evaluation‑only. **Do not train or change model code** during the run.

Dependencies:
- `docs/rtgomp_complexity_cost_E4f_omp_dtmin_random_spec.md`
- `docs/rtgomp_complexity_cost_E4f_omp_dtmin_random_plan.md`

---

## 0) Preconditions (MUST)

- The stop‑action K‑sweep evaluator **must exist** before running:
  - `scripts/eval_energy_capture_stop_action_k_sweep.py`
- If the script is missing, **STOP** and record the missing prerequisite.

---

## 1) Fixed Inputs (MUST)

Checkpoint:
- `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth`

Subset manifest (copy into the run directory):
- Source: `results/rtgomp_lambda_cost_E4d_teacher_forced_mask_fix_20260121_113357/subset_manifest.json`
- Required: `fingerprint_md5 == 668135f8f6f7baaf99dffeef4cbb1a21`
- Selection: first 3 clip pairs in dataset order (all angles)

Params (fixed):
- `hop_length=160`, `max_lag=50`, `Tw=32`, `max_k=16`, `gain=100.0`
- `rtg_dim=2`

---

## 2) Checkpoint Compatibility Preflight (MUST)

Verify the checkpoint is **stop‑action** and consistent with `max_lag`:

```bash
python - <<'PY'
import torch

ckpt = "results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth"
max_lag = 50
expected_mlags = max_lag * 2 + 1

state = torch.load(ckpt, map_location="cpu")
head_out = state["head.2.weight"].shape[0]
state_mlags = state["state_embed.weight"].shape[1]

if state_mlags != expected_mlags:
    raise SystemExit(
        f"Checkpoint state_embed M_lags mismatch. Expected {expected_mlags}, got {state_mlags}."
    )
if head_out != expected_mlags + 1:
    raise SystemExit(
        f"Checkpoint head_out mismatch. Expected {expected_mlags + 1}, got {head_out}."
    )
print(f"Checkpoint OK (stop‑action): state_embed M_lags={state_mlags}, head_out={head_out}.")
PY
```

---

## 3) Run Directory Setup

```bash
set -euo pipefail
export PYTHONPATH=.

RUN_DIR="results/rtgomp_lambda_cost_E4f_omp_dtmin_random_stop_action_<timestamp>"
mkdir -p "$RUN_DIR"
cp -f "results/rtgomp_lambda_cost_E4d_teacher_forced_mask_fix_20260121_113357/subset_manifest.json" \
  "$RUN_DIR/subset_manifest.json"
MANIFEST="$RUN_DIR/subset_manifest.json"
export MANIFEST
```

---

## 4) A) OMP vs Random (K‑sweep)

Run **Step A** from the main E4f spec unchanged, but use the stop‑action run directory:
- `docs/rtgomp_complexity_cost_E4f_omp_dtmin_random_spec.md` → Section “A) OMP vs Random (K‑sweep)”

---

## 5) B) DTmin vs OMP (K‑sweep, stop‑action)

This uses the stop‑action evaluator. **Required behavior**:

- `action_dim = M_lags + 1`, `stop_id = M_lags`
- For each frequency:
  - If STOP is chosen at step `k`, no new atom is added.
  - For subsequent steps, the residual **does not change** and capture stays constant.
  - Capture for each K in `{1,2,4,8,16}` equals the capture achieved by `min(K, stop_step)` atoms.
- OMP baseline remains the standard OMP (no STOP), computed for the exact K steps.

Run with locks + tee:

```bash
OUT_DIR="$RUN_DIR/B_dt_vs_omp_stop_action"
mkdir -p "$OUT_DIR"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "ERROR: lock exists ($LOCKDIR). Another run is using OUT_DIR=$OUT_DIR" >&2
  exit 1
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

CKPT="results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth"
MIC_ROOT=$(python - <<'PY'
import json, os
manifest = json.load(open(os.environ["MANIFEST"]))
print(manifest["mic_root"])
PY
)
LDV_ROOT=$(python - <<'PY'
import json, os
manifest = json.load(open(os.environ["MANIFEST"]))
print(manifest["ldv_root"])
PY
)

conda run -n trl-training python -u scripts/eval_energy_capture_stop_action_k_sweep.py \
  --mic_root "$MIC_ROOT" --ldv_root "$LDV_ROOT" \
  --ckpt_path "$CKPT" --out_dir "$OUT_DIR" \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --rtg_dim 2 --num_clips 3 --all_angles --use_stop_action \
  2>&1 | tee -a "$OUT_DIR/run.log"
```

Summarize `eval_stats.pt` (same schema as standard E4f):

```bash
python - <<'PY'
import json
import torch
from pathlib import Path

stats_path = Path("results/rtgomp_lambda_cost_E4f_omp_dtmin_random_stop_action_<timestamp>/B_dt_vs_omp_stop_action/eval_stats.pt")
stats = torch.load(stats_path, map_location="cpu")
dt = stats["DT"]
omp = stats["OMP"]

ks = [1, 2, 4, 8, 16]
rows = []
for k in ks:
    idx = k - 1
    rows.append({
        "k": k,
        "dt": float(dt[:, idx].mean().item()),
        "omp": float(omp[:, idx].mean().item()),
        "efficiency": float((dt[:, idx] / (omp[:, idx] + 1e-6)).mean().item()),
    })

out = {"source_eval_stats": str(stats_path), "rows": rows}
out_path = Path("results/rtgomp_lambda_cost_E4f_omp_dtmin_random_stop_action_<timestamp>/summary/dt_vs_omp_k_sweep.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"Wrote {out_path}")
PY
```

---

## 6) C) Combined Summary (OMP vs DTmin vs Random)

Run **Step C** from the main E4f spec unchanged, but use the stop‑action run directory:
- `docs/rtgomp_complexity_cost_E4f_omp_dtmin_random_spec.md` → Section “C) Combined Summary”

---

## 7) Acceptance Criteria

Use the **same acceptance criteria** as E4f:
- `docs/rtgomp_complexity_cost_E4f_omp_dtmin_random_spec.md` → Section “Acceptance Criteria (E4f)”

If STOP actions materially reduce DT capture, this should surface in the ratios and **fail** the acceptance thresholds.

