# Acceptance Report: E4g — Full (Non-Smoke) Evaluation on a Larger Real Subset

## 1) Executive Summary

- Run: `results/rtgomp_lambda_cost_E4g_full_eval_20260122_041059/`
- Outcome: `PASS`
- Purpose: Validate E4-series lambda-cost controllability and baselines beyond the smoke subset (`num_pairs=3`).

## 2) Setup (REQUIRED)

- Env: `trl-training`
- Device(s):
  - A (OMP vs Random): `cpu` (script default)
  - B_free/B_teacher_forced/C_free: `mps` for B runs; `cpu` for C_free due to MPS→CPU stall BECAUSE repeated `.cpu()` calls hang in MPS sync
  - Baseline: `cpu` (explicit `--device cpu`)
- Checkpoint: `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth`
- Subset manifest: `results/rtgomp_lambda_cost_E4g_full_eval_20260122_041059/subset_manifest.json`
  - `num_pairs = 6`
  - `fingerprint_md5 = 5fdabd9e79ac6f1b8d513e07c95798b1`
- Data roots (from manifest):
  - `mic_root = /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC`
  - `ldv_root = /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV`
- Fixed params:
  - `hop_length=160`, `max_lag=50`, `Tw=32`, `max_k=16`, `gain=100.0`
  - `rtg_dim=2`, `use_stop_action=true`
- Lambda grid:
  - `lambda_c_values = 1e-4,3e-4,1e-3,3e-3,1e-2`
- E4e prerequisite:
  - Confirmed `teacher_forced` semantics and `steps_used_mean` indexing match E4e spec: `YES`

## 3) Exact Commands (REQUIRED)

```bash
# A) OMP vs Random
export PYTHONPATH=.
RUN_DIR="results/rtgomp_lambda_cost_E4g_full_eval_20260122_041059"
NUM_PAIRS=6
LAMBDA_LIST="1e-4,3e-4,1e-3,3e-3,1e-2"
SEED_OMP_RANDOM=0

OUT_DIR="$RUN_DIR/A_verify_omp_superiority"
mkdir -p "$OUT_DIR"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "ERROR: lock exists ($LOCKDIR). Another run is using OUT_DIR=$OUT_DIR" >&2
  exit 1
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

MIC_ROOT=$(python - <<'PY'
import json, os
from pathlib import Path
manifest = json.loads(Path(os.environ["RUN_DIR"], "subset_manifest.json").read_text())
print(manifest["mic_root"])
PY
)
LDV_ROOT=$(python - <<'PY'
import json, os
from pathlib import Path
manifest = json.loads(Path(os.environ["RUN_DIR"], "subset_manifest.json").read_text())
print(manifest["ldv_root"])
PY
)

conda run -n trl-training python -u verify_omp_superiority.py \
  --mic_root "$MIC_ROOT" --ldv_root "$LDV_ROOT" --all_angles \
  --num_clips "$NUM_PAIRS" --random_trials 5 --stride 64 --seed "$SEED_OMP_RANDOM" \
  2>&1 | tee -a "$OUT_DIR/run.log"

conda run -n trl-training python - <<'PY'
import json
import re
from pathlib import Path
import os

run_dir = Path(os.environ["RUN_DIR"])
log_path = run_dir / "A_verify_omp_superiority" / "run.log"
rows = []
pattern = re.compile(r"^INFO:__main__:\\s*\\d+\\s+\\|")
for line in log_path.read_text(encoding="utf-8").splitlines():
    if pattern.search(line):
        parts = line.split("|")
        k = int(parts[0].split(":")[-1].strip())
        omp = float(parts[1].strip())
        rnd = float(parts[2].strip())
        gap = float(parts[3].strip())
        ratio = float(parts[4].strip().replace("x", ""))
        rows.append({"k": k, "omp": omp, "random": rnd, "gap": gap, "ratio": ratio})

out = {"source_log": str(log_path), "rows": rows}
out_path = run_dir / "summary" / "omp_vs_random_k_sweep.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print("Wrote", out_path)
PY
```

```bash
# B) Free rollout
export PYTHONPATH=.
RUN_DIR="results/rtgomp_lambda_cost_E4g_full_eval_20260122_041059"
CKPT="results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth"
LAMBDA_LIST="1e-4,3e-4,1e-3,3e-3,1e-2"

OUT_DIR="$RUN_DIR/B_free"
mkdir -p "$OUT_DIR"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "ERROR: lock exists ($LOCKDIR). Another run is using OUT_DIR=$OUT_DIR" >&2
  exit 1
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

MIC_ROOT=$(python - <<'PY'
import json, os
from pathlib import Path
manifest = json.loads(Path(os.environ["RUN_DIR"], "subset_manifest.json").read_text())
print(manifest["mic_root"])
PY
)
LDV_ROOT=$(python - <<'PY'
import json, os
from pathlib import Path
manifest = json.loads(Path(os.environ["RUN_DIR"], "subset_manifest.json").read_text())
print(manifest["ldv_root"])
PY
)

conda run -n trl-training python -u scripts/h_exploration/run_lambda_override_grid_eval.py \
  --mic_root "$MIC_ROOT" --ldv_root "$LDV_ROOT" \
  --ckpt_path "$CKPT" --subset_manifest "$RUN_DIR/subset_manifest.json" \
  --out_dir "$OUT_DIR" \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --rtg_dim 2 --use_stop_action \
  --rollout_mode free --teacher_min_k 1 \
  --lambda_c_values "$LAMBDA_LIST" 2>&1 | tee -a "$OUT_DIR/run.log"
```

```bash
# B) Teacher-forced rollout
export PYTHONPATH=.
RUN_DIR="results/rtgomp_lambda_cost_E4g_full_eval_20260122_041059"
CKPT="results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth"
LAMBDA_LIST="1e-4,3e-4,1e-3,3e-3,1e-2"

OUT_DIR="$RUN_DIR/B_teacher_forced"
mkdir -p "$OUT_DIR"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "ERROR: lock exists ($LOCKDIR). Another run is using OUT_DIR=$OUT_DIR" >&2
  exit 1
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

conda run -n trl-training python -u scripts/h_exploration/run_lambda_override_grid_eval.py \
  --mic_root "$MIC_ROOT" --ldv_root "$LDV_ROOT" \
  --ckpt_path "$CKPT" --subset_manifest "$RUN_DIR/subset_manifest.json" \
  --out_dir "$OUT_DIR" \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --rtg_dim 2 --use_stop_action \
  --rollout_mode teacher_forced --teacher_min_k 1 \
  --lambda_c_values "$LAMBDA_LIST" 2>&1 | tee -a "$OUT_DIR/run.log"
```

```bash
# C) Free rollout + acceptance check (CPU due to MPS stall)
export PYTHONPATH=.
RUN_DIR="results/rtgomp_lambda_cost_E4g_full_eval_20260122_041059"
CKPT="results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth"
LAMBDA_LIST="1e-4,3e-4,1e-3,3e-3,1e-2"

OUT_DIR="$RUN_DIR/C_free"
mkdir -p "$OUT_DIR"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "ERROR: lock exists ($LOCKDIR). Another run is using OUT_DIR=$OUT_DIR" >&2
  exit 1
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

conda run --no-capture-output -n trl-training python -u scripts/h_exploration/run_lambda_override_grid_eval.py \
  --mic_root "$MIC_ROOT" --ldv_root "$LDV_ROOT" \
  --ckpt_path "$CKPT" --subset_manifest "$RUN_DIR/subset_manifest.json" \
  --out_dir "$OUT_DIR" \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --rtg_dim 2 --use_stop_action \
  --rollout_mode free --teacher_min_k 1 \
  --lambda_c_values "$LAMBDA_LIST" \
  --device cpu 2>&1 | tee -a "$OUT_DIR/run.log"

conda run --no-capture-output -n trl-training python -u scripts/h_exploration/check_rtgomp_acceptance.py \
  --lambda_grid "$OUT_DIR/lambda_grid.json" \
  --out_json "$OUT_DIR/acceptance_check.json" 2>&1 | tee -a "$OUT_DIR/run.log"
```

```bash
# Baseline: DTmin vs OMP stop-action (CPU due to MPS stall)
export PYTHONPATH=.
RUN_DIR="results/rtgomp_lambda_cost_E4g_full_eval_20260122_041059"
CKPT="results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth"

OUT_DIR="$RUN_DIR/baseline_dt_vs_omp_stop_action"
mkdir -p "$OUT_DIR"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "ERROR: lock exists ($LOCKDIR). Another run is using OUT_DIR=$OUT_DIR" >&2
  exit 1
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

conda run --no-capture-output -n trl-training python -u scripts/eval_energy_capture_stop_action_k_sweep.py \
  --mic_root "$MIC_ROOT" --ldv_root "$LDV_ROOT" \
  --ckpt_path "$CKPT" --out_dir "$OUT_DIR" \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --rtg_dim 2 --num_clips "$NUM_PAIRS" --all_angles --use_stop_action \
  --device cpu 2>&1 | tee -a "$OUT_DIR/run.log"
```

## 4) Results (REQUIRED)

### 4.1 A) OMP vs Random (larger subset)

Artifacts:
- `results/rtgomp_lambda_cost_E4g_full_eval_20260122_041059/A_verify_omp_superiority/run.log`
- `results/rtgomp_lambda_cost_E4g_full_eval_20260122_041059/summary/omp_vs_random_k_sweep.json`

Table summary:
- K=1: OMP=0.9202 Random=0.1964 Gap=0.7238
- K=2: OMP=0.9600 Random=0.3728 Gap=0.5872
- K=4: OMP=0.9791 Random=0.5806 Gap=0.3985
- K=8: OMP=0.9924 Random=0.8168 Gap=0.1756
- K=16: OMP=0.9991 Random=0.8968 Gap=0.1023

Decision (A):
- PASS (all gaps > 0 and K=16 gap >= 0.05)

### 4.2 B) Free vs Teacher-Forced (E4e semantics)

Artifacts:
- `results/rtgomp_lambda_cost_E4g_full_eval_20260122_041059/B_free/lambda_grid.json`
- `results/rtgomp_lambda_cost_E4g_full_eval_20260122_041059/B_teacher_forced/lambda_grid.json`

Report:
- Free: spearman(lambda_c, steps_used_mean) ≈ -0.7
- Teacher-forced: spearman(lambda_c, steps_used_mean) ≈ -0.9
- Low-penalty consistency at lambda_c=min:
  - capture_free = 0.9965577632508626
  - capture_teacher_forced = 0.9836383049158546
  - Δcapture = 0.012919458335008072

Decision (B):
- PASS (both spearman <= -0.6)

### 4.3 C) Lambda-grid acceptance (free rollout)

Artifacts:
- `results/rtgomp_lambda_cost_E4g_full_eval_20260122_041059/C_free/lambda_grid.json`
- `results/rtgomp_lambda_cost_E4g_full_eval_20260122_041059/C_free/acceptance_check.json`

Key numeric checks:
- spearman(lambda_c, steps_used_mean) = -0.9 (target <= -0.6)
- steps_range = 0.39037398373983834 (target >= 0.10)
- capture_range = 0.0021573198791441373 (target >= 0.001)
- max(action_change_rate_vs_ref) = 0.37889833197516115 (target >= 0.05)
- max(logits_kl_mean_vs_ref) = 3.6113493877526404 (target > 0)

Decision (C):
- PASS

### 4.4 Baseline: OMP vs DTmin vs Random (K-sweep; stop-action)

Artifacts:
- `results/rtgomp_lambda_cost_E4g_full_eval_20260122_041059/baseline_dt_vs_omp_stop_action/eval_stats.pt`
- `results/rtgomp_lambda_cost_E4g_full_eval_20260122_041059/summary/dt_vs_omp_k_sweep.json`
- `results/rtgomp_lambda_cost_E4g_full_eval_20260122_041059/summary/omp_dtmin_random_k_sweep.json`

Key rows (dt_vs_omp):
- K=1: DT=0.3033, OMP=0.2922, efficiency=2.2617
- K=2: DT=0.1671, OMP=0.3231, efficiency=0.7476
- K=4: DT=0.1107, OMP=0.2748, efficiency=0.5592
- K=8: DT=0.0672, OMP=0.1925, efficiency=0.7842
- K=16: DT=0.0488, OMP=0.1826, efficiency=0.5717

DT - Random (from omp_dtmin_random_k_sweep):
- K=1: +0.1069
- K=2: -0.2057
- K=4: -0.4699
- K=8: -0.7496
- K=16: -0.8480

## 5) Acceptance Decision (REQUIRED)

- A: PASS
- B: PASS
- C: PASS
- Overall (E4g): PASS

## 6) Interpretation (REQUIRED; causal language)

- A interpretation:
  - OMP beats Random BECAUSE greedy least-squares selection aligns with the highest-correlation lags, increasing captured energy deterministically.
- B interpretation:
  - B passes THEREFORE STOP remains controllable under teacher residual evolution; steps_used_mean decreases with higher lambda_c, indicating the policy responds to cost scaling.
- C interpretation:
  - Negative spearman implies controllability BECAUSE higher penalty reduces remaining-steps preference, and the observed steps/capture ranges show non-degenerate tradeoffs.
- Baseline interpretation:
  - DT - Random becomes negative at higher K BECAUSE the stop-action policy truncates early and sacrifices capture compared to random continuation; this implies STOP incentives dominate at larger K without matching OMP’s deterministic gains.

## 7) Next Steps (REQUIRED)

- Re-run E4g baseline and C_free on MPS only after removing MPS→CPU sync in the inner loop (or keep CPU as the stable default).
- Consider reducing MPS↔CPU transfers by precomputing CPU tensors per block to prevent stalls.
- If full-dataset eval is needed, scale NUM_PAIRS upward and re-run with CPU to avoid MPS stalls.

