# Acceptance Report: E4f — OMP vs DTmin vs Random Energy Capture Baseline (Stop‑Action)

## 1) Executive Summary

- Run: `results/rtgomp_lambda_cost_E4f_omp_dtmin_random_stop_action_20260122_003006/`
- Outcome: `FAIL`
- Key claim: E4-series checkpoint has a quantified OMP vs DTmin vs Random energy-capture baseline on the validated subset.

## 2) Setup (REQUIRED)

- Env: `trl-training`
- Device: `mps`
- Checkpoint: `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth`
- Subset manifest: `results/rtgomp_lambda_cost_E4f_omp_dtmin_random_stop_action_20260122_003006/subset_manifest.json`
  - `fingerprint_md5 = 668135f8f6f7baaf99dffeef4cbb1a21`
- Params: `hop_length=160`, `max_lag=50`, `Tw=32`, `max_k=16`, `gain=100.0`, `rtg_dim=2`
- Data roots: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC` / `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV` (from manifest)
- Checkpoint compatibility:
  - `expected_mlags = 2*max_lag + 1 = 101`
  - `state_embed_mlags = 101`
  - `head_out = 102`
  - Compatibility: `PASS`
  - Branch: `B (head_out == M_lags + 1)`

## 3) Exact Commands (REQUIRED)

Subset consistency check:
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

Checkpoint compatibility preflight:
```bash
python - <<'PY'
import torch
ckpt = "results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth"
max_lag = 50
expected_mlags = max_lag * 2 + 1
state = torch.load(ckpt, map_location="cpu")
print("expected_mlags", expected_mlags)
print("state_embed_mlags", state["state_embed.weight"].shape[1])
print("head_out", state["head.2.weight"].shape[0])
PY
```

A) OMP vs Random (K-sweep):
```bash
export PYTHONPATH=.
RUN_DIR="results/rtgomp_lambda_cost_E4f_omp_dtmin_random_stop_action_20260122_003006"
OUT_DIR="$RUN_DIR/A_verify_omp_superiority"
mkdir -p "$OUT_DIR"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "ERROR: lock exists ($LOCKDIR). Another run is using OUT_DIR=$OUT_DIR" >&2
  exit 1
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

MIC_ROOT="/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC"
LDV_ROOT="/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV"

conda run -n trl-training python -u verify_omp_superiority.py   --mic_root "$MIC_ROOT" --ldv_root "$LDV_ROOT" --all_angles   2>&1 | tee -a "$OUT_DIR/run.log"
```

B) DTmin vs OMP (K-sweep, stop-action):
```bash
export PYTHONPATH=.
RUN_DIR="results/rtgomp_lambda_cost_E4f_omp_dtmin_random_stop_action_20260122_003006"
OUT_DIR="$RUN_DIR/B_dt_vs_omp_stop_action"
mkdir -p "$OUT_DIR"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "ERROR: lock exists ($LOCKDIR). Another run is using OUT_DIR=$OUT_DIR" >&2
  exit 1
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

CKPT="results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth"
MIC_ROOT="/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC"
LDV_ROOT="/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV"

conda run -n trl-training python -u scripts/eval_energy_capture_stop_action_k_sweep.py   --mic_root "$MIC_ROOT" --ldv_root "$LDV_ROOT"   --ckpt_path "$CKPT" --out_dir "$OUT_DIR"   --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0   --rtg_dim 2 --num_clips 3 --all_angles --use_stop_action   2>&1 | tee -a "$OUT_DIR/run.log"
```

## 4) Results (REQUIRED)

### 4.1 A) OMP vs Random (K-sweep)

From `summary/omp_vs_random_k_sweep.json`:
- K=1: OMP=`0.9140` Random=`0.1709` Gap=`0.7431`
- K=2: OMP=`0.9565` Random=`0.3249` Gap=`0.6316`
- K=4: OMP=`0.9767` Random=`0.5504` Gap=`0.4263`
- K=8: OMP=`0.9914` Random=`0.7864` Gap=`0.2050`
- K=16: OMP=`0.9989` Random=`0.8881` Gap=`0.1109`

### 4.2 B) DTmin vs OMP (K-sweep, stop-action)

From `summary/dt_vs_omp_k_sweep.json`:
- K=1: DT=`0.2132` OMP=`0.2167` Eff=`1.1137`
- K=2: DT=`0.3534` OMP=`0.3555` Eff=`1.0532`
- K=4: DT=`0.5544` OMP=`0.5474` Eff=`1.0453`
- K=8: DT=`0.7307` OMP=`0.8037` Eff=`0.9173`
- K=16: DT=`0.7582` OMP=`0.9861` Eff=`0.7690`

### 4.3 C) Combined Table (OMP vs DTmin vs Random)

From `summary/omp_dtmin_random_k_sweep.json`:
- K=1: OMP=0.9140, DT=0.2132, Random=0.1709
- K=2: OMP=0.9565, DT=0.3534, Random=0.3249
- K=4: OMP=0.9767, DT=0.5544, Random=0.5504
- K=8: OMP=0.9914, DT=0.7307, Random=0.7864
- K=16: OMP=0.9989, DT=0.7582, Random=0.8881

## 5) Acceptance Decision (REQUIRED)

- OMP > Random for all K: `PASS`
- K=16 gap >= 0.05: `PASS`
- DTmin in [0,1] (finite): `PASS`
- K=16 |DT-OMP| <= 0.02: `FAIL`
- K=8 DT/OMP >= 0.80: `PASS`
- DTmin >= Random for all K: `FAIL`
- Overall: `FAIL`
- Failure class (if FAIL): `model underperforms baseline`

## 6) Interpretation (REQUIRED; causal language)

- OMP exceeds Random for all K BECAUSE greedy correlation selection plus least-squares projection aligns atoms with the target energy, whereas Random does not.
- DTmin falls below Random at higher K DUE TO early STOP behavior reducing the number of active atoms, which limits capture at larger K.
- The K=16 DT vs OMP gap fails BECAUSE stop-action truncation prevents the model from matching the oracle’s full K-step reconstruction; THEREFORE the acceptance criteria are not met in this stop-action branch.
