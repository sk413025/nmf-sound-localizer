# Spec: E4f — OMP vs DTmin vs Random Energy Capture Baseline

This spec defines the E4f evaluation-only baseline in the E4-series context. It produces a K-sweep comparison of OMP, DTmin, and Random capture on the **same real-data subset** and checkpoint.

Dependencies:
- `docs/rtgomp_complexity_cost_E4f_omp_dtmin_random_plan.md`
- `docs/rtgomp_complexity_cost_E4c_extended_validation_ABC_spec.md` (data + subset requirements)
- `docs/rtgomp_complexity_cost_E4e_teacher_forced_semantics_fix_spec.md` (checkpoint + params)

---

## 0) Execution Robustness (MUST)

- Run steps sequentially (no parallel runs).
- Use per-step directory locks.
- Always append logs with `tee -a`.

Lock rule:
```bash
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "ERROR: lock exists ($LOCKDIR). Another run is using OUT_DIR=$OUT_DIR" >&2
  exit 1
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT
```

---

## 1) Fixed Inputs (MUST match E4d/E4e context)

Checkpoint:
- `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth`

Subset manifest (copy into the run directory):
- Source: `results/rtgomp_lambda_cost_E4d_teacher_forced_mask_fix_20260121_113357/subset_manifest.json`
- Required: `fingerprint_md5 == 668135f8f6f7baaf99dffeef4cbb1a21`
- Selection: first 3 clip pairs in dataset order (all angles)

Data roots:
- `mic_root` and `ldv_root` from the subset manifest
- If roots are missing, FAIL fast and document the missing prerequisite.

Params (fixed):
- `hop_length=160`, `max_lag=50`, `Tw=32`, `max_k=16`, `gain=100.0`
- `rtg_dim=2`

---

## 2) Subset Consistency Check (MUST)

Verify the dataset order matches the manifest (first 3 pairs). This prevents silent drift if file ordering changes.

```bash
export PYTHONPATH=.
python - <<'PY'
import json
from scripts.h_exploration.dataset_lag import DoALagDataset

manifest = json.load(open("results/rtgomp_lambda_cost_E4d_teacher_forced_mask_fix_20260121_113357/subset_manifest.json"))
mic_root = manifest["mic_root"]
ldv_root = manifest["ldv_root"]
dataset = DoALagDataset(mic_root, ldv_root, angle=None, hop_length=160)
expected = [entry["path"] for entry in manifest["file_hashes"]]

actual = []
for idx in range(manifest["num_pairs"]):
    mic_path, ldv_path = dataset.clips[idx]
    actual.extend([str(mic_path), str(ldv_path)])

if expected != actual:
    raise SystemExit(f"Subset mismatch. Expected: {expected} Actual: {actual}")
print("Subset match OK.")
PY
```

If this fails, stop and fix the selection; do not continue with an implicit subset.

---

## 3) Checkpoint Compatibility Preflight (MUST)

Fail fast if the checkpoint output head is incompatible with the evaluation model. The current eval script
(`scripts/eval_energy_capture_generic.py`) uses `M_lags = 2*max_lag + 1` and does **not** include a stop-action
head. If the checkpoint was trained with a stop-action output (i.e., `head_out = M_lags + 1`), **this evaluation
cannot proceed** without a compatible eval script or a matching checkpoint.

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
        f"Checkpoint state_embed M_lags mismatch. Expected {expected_mlags}, got {state_mlags}. "
        "Use a checkpoint trained with matching max_lag."
    )
if head_out != expected_mlags:
    raise SystemExit(
        f"Checkpoint head_out mismatch. Expected {expected_mlags}, got {head_out}. "
        "This indicates a stop-action head or different config. Use a compatible eval script or checkpoint."
    )
print(f"Checkpoint OK: state_embed M_lags={state_mlags}, head_out={head_out}.")
PY
```

---

## 4) Compatibility Branches (MUST)

Based on the preflight result:

- **Branch A (standard, supported):** `head_out == M_lags`
  - Proceed with steps 5–7 using `scripts/eval_energy_capture_generic.py`.
- **Branch B (stop-action head):** `head_out == M_lags + 1`
  - Follow the stop-action K‑sweep spec:
    - `docs/rtgomp_complexity_cost_E4f_omp_dtmin_random_stop_action_k_sweep_spec.md`
  - If the stop-action evaluator is not available, **STOP** and record the missing prerequisite.

---

## 5) Run Directory Setup

```bash
set -euo pipefail
export PYTHONPATH=.

RUN_DIR="results/rtgomp_lambda_cost_E4f_omp_dtmin_random_<timestamp>"
mkdir -p "$RUN_DIR"
cp -f "results/rtgomp_lambda_cost_E4d_teacher_forced_mask_fix_20260121_113357/subset_manifest.json" \
  "$RUN_DIR/subset_manifest.json"
MANIFEST="$RUN_DIR/subset_manifest.json"
export MANIFEST
```

---

## 6) A) OMP vs Random (K-sweep)

```bash
OUT_DIR="$RUN_DIR/A_verify_omp_superiority"
mkdir -p "$OUT_DIR"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "ERROR: lock exists ($LOCKDIR). Another run is using OUT_DIR=$OUT_DIR" >&2
  exit 1
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

MIC_ROOT=$(python - <<'PY'
import json
import os
manifest = json.load(open(os.environ["MANIFEST"]))
print(manifest["mic_root"])
PY
)
LDV_ROOT=$(python - <<'PY'
import json
import os
manifest = json.load(open(os.environ["MANIFEST"]))
print(manifest["ldv_root"])
PY
)

conda run -n trl-training python -u verify_omp_superiority.py \
  --mic_root "$MIC_ROOT" --ldv_root "$LDV_ROOT" --all_angles \
  2>&1 | tee -a "$OUT_DIR/run.log"
```

Parse the log into JSON:
```bash
python - <<'PY'
import json
import re
from pathlib import Path

log_path = Path("results/rtgomp_lambda_cost_E4f_omp_dtmin_random_<timestamp>/A_verify_omp_superiority/run.log")
rows = []
for line in log_path.read_text().splitlines():
    if re.search(r"^INFO:__main__:\\d+\\s+\\|", line):
        parts = line.split("|")
        k = int(parts[0].split(":")[-1].strip())
        omp = float(parts[1].strip())
        rnd = float(parts[2].strip())
        gap = float(parts[3].strip())
        ratio = float(parts[4].strip().replace("x", ""))
        rows.append({"k": k, "omp": omp, "random": rnd, "gap": gap, "ratio": ratio})

out = {
    "source_log": str(log_path),
    "rows": rows,
}

out_path = Path("results/rtgomp_lambda_cost_E4f_omp_dtmin_random_<timestamp>/summary/omp_vs_random_k_sweep.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"Wrote {out_path}")
PY
```

---

## 7) B) DTmin vs OMP (K-sweep)

```bash
OUT_DIR="$RUN_DIR/B_dt_vs_omp"
mkdir -p "$OUT_DIR"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "ERROR: lock exists ($LOCKDIR). Another run is using OUT_DIR=$OUT_DIR" >&2
  exit 1
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

CKPT="results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth"
MIC_ROOT=$(python - <<'PY'
import json
import os
manifest = json.load(open(os.environ["MANIFEST"]))
print(manifest["mic_root"])
PY
)
LDV_ROOT=$(python - <<'PY'
import json
import os
manifest = json.load(open(os.environ["MANIFEST"]))
print(manifest["ldv_root"])
PY
)

conda run -n trl-training python -u scripts/eval_energy_capture_generic.py \
  --mic_root "$MIC_ROOT" --ldv_root "$LDV_ROOT" \
  --ckpt_path "$CKPT" --out_dir "$OUT_DIR" \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --rtg_dim 2 --num_clips 3 --all_angles \
  2>&1 | tee -a "$OUT_DIR/run.log"
```

Summarize `eval_stats.pt`:
```bash
python - <<'PY'
import json
import torch
from pathlib import Path

stats_path = Path("results/rtgomp_lambda_cost_E4f_omp_dtmin_random_<timestamp>/B_dt_vs_omp/eval_stats.pt")
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
out_path = Path("results/rtgomp_lambda_cost_E4f_omp_dtmin_random_<timestamp>/summary/dt_vs_omp_k_sweep.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"Wrote {out_path}")
PY
```

---

## 8) C) Combined Summary (OMP vs DTmin vs Random)

```bash
python - <<'PY'
import json
from pathlib import Path

base = Path("results/rtgomp_lambda_cost_E4f_omp_dtmin_random_<timestamp>/summary")
omp_rows = {r["k"]: r for r in json.loads((base / "omp_vs_random_k_sweep.json").read_text())["rows"]}
dt_rows = {r["k"]: r for r in json.loads((base / "dt_vs_omp_k_sweep.json").read_text())["rows"]}

rows = []
for k in sorted(omp_rows.keys()):
    row = {"k": k}
    row["omp"] = omp_rows[k]["omp"]
    row["random"] = omp_rows[k]["random"]
    row["dt"] = dt_rows[k]["dt"]
    row["dt_over_omp"] = dt_rows[k]["efficiency"]
    row["dt_minus_random"] = row["dt"] - row["random"]
    rows.append(row)

out = {"rows": rows}
out_path = base / "omp_dtmin_random_k_sweep.json"
out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"Wrote {out_path}")
PY
```

---

## 9) Acceptance Criteria (E4f)

From `summary/omp_vs_random_k_sweep.json` and `summary/dt_vs_omp_k_sweep.json`:

PASS if all:
- OMP > Random for all K; `gap(K=16) >= 0.05`.
- DTmin capture values are finite and in `[0, 1]`.
- `abs(DT - OMP) <= 0.02` at `K=16`.
- `DT/OMP >= 0.80` at `K=8`.
- DTmin >= Random for all K.
FAIL otherwise. If FAIL, describe the most likely cause (data mismatch, checkpoint mismatch, or evaluation bug).

---

## 10) Troubleshooting (Fail-Fast Guidance)

Use short probes to confirm assumptions before running long evaluations.

Common failure modes:

1) **Checkpoint mismatch (head_out vs M_lags)**
   - Symptom: `Model Load Failed` with size mismatch in `head.2.*`.
   - Cause: checkpoint trained with a stop-action head (`head_out = M_lags + 1`) while `eval_energy_capture_generic.py`
     expects `head_out = M_lags`.
   - Action: **STOP** and record the mismatch. Do not change code. Use a compatible checkpoint or update the spec
     with an explicit stop-action‑aware evaluator.

2) **Subset mismatch**
   - Symptom: subset consistency check fails (expected vs actual paths differ).
   - Cause: dataset ordering drift or manifest mismatch.
   - Action: **STOP**. Regenerate or re‑validate the manifest before any evaluation.

3) **Data roots missing**
   - Symptom: `mic_root` or `ldv_root` does not exist.
   - Cause: data not mounted or wrong root in manifest.
   - Action: **STOP** and document missing prerequisite; do not substitute paths.
