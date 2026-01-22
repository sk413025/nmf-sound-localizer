# Acceptance Report Template: E4f — OMP vs DTmin vs Random Energy Capture Baseline

Use this template for E4f. Fill all placeholders. Keep everything in English.

---

# Acceptance Report: E4f — OMP vs DTmin vs Random Energy Capture Baseline

## 1) Executive Summary

- Run: `results/<run_name>/`
- Outcome: `<PASS/FAIL>`
- Key claim: E4-series checkpoint has a quantified OMP vs DTmin vs Random energy-capture baseline on the validated subset.

## 2) Setup (REQUIRED)

- Env: `trl-training`
- Device: `<mps/cpu/etc>`
- Checkpoint: `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth`
- Subset manifest: `results/<run_name>/subset_manifest.json`
  - `fingerprint_md5 = 668135f8f6f7baaf99dffeef4cbb1a21`
- Params: `hop_length=160`, `max_lag=50`, `Tw=32`, `max_k=16`, `gain=100.0`, `rtg_dim=2`
- Data roots: `<mic_root>` / `<ldv_root>` (from manifest)
- Checkpoint compatibility:
  - `expected_mlags = 2*max_lag + 1 = 101`
  - `state_embed_mlags = <...>`
  - `head_out = <...>`
  - Compatibility: `<PASS/FAIL>`
  - Branch: `<A: head_out==M_lags / B: head_out==M_lags+1>`

## 3) Exact Commands (REQUIRED)

Paste the exact commands executed from the E4f spec, including:
- Subset consistency check
- Checkpoint compatibility preflight
- A/B/C steps
- Any parsing scripts

## 4) Results (REQUIRED)

### 4.1 A) OMP vs Random (K-sweep)

From `summary/omp_vs_random_k_sweep.json`:
- K=1: OMP=`<...>` Random=`<...>` Gap=`<...>`
- K=2: OMP=`<...>` Random=`<...>` Gap=`<...>`
- K=4: OMP=`<...>` Random=`<...>` Gap=`<...>`
- K=8: OMP=`<...>` Random=`<...>` Gap=`<...>`
- K=16: OMP=`<...>` Random=`<...>` Gap=`<...>`

### 4.2 B) DTmin vs OMP (K-sweep)

From `summary/dt_vs_omp_k_sweep.json`:
- K=1: DT=`<...>` OMP=`<...>` Eff=`<...>`
- K=2: DT=`<...>` OMP=`<...>` Eff=`<...>`
- K=4: DT=`<...>` OMP=`<...>` Eff=`<...>`
- K=8: DT=`<...>` OMP=`<...>` Eff=`<...>`
- K=16: DT=`<...>` OMP=`<...>` Eff=`<...>`

### 4.3 C) Combined Table (OMP vs DTmin vs Random)

From `summary/omp_dtmin_random_k_sweep.json`:
- Provide a compact table or list for K in `{1,2,4,8,16}`.

## 5) Acceptance Decision (REQUIRED)

- OMP > Random for all K: `<PASS/FAIL>`
- K=16 gap >= 0.05: `<PASS/FAIL>`
- DTmin in [0,1] (finite): `<PASS/FAIL>`
- K=16 |DT-OMP| <= 0.02: `<PASS/FAIL>`
- K=8 DT/OMP >= 0.80: `<PASS/FAIL>`
- DTmin >= Random for all K: `<PASS/FAIL>`
- Overall: `<PASS/FAIL>`
- Failure class (if FAIL): `<checkpoint mismatch / subset mismatch / data root missing / eval bug / other>`

## 6) Interpretation (REQUIRED; causal language)

- Explain why OMP exceeds Random BECAUSE greedy correlation selection plus least-squares projection is a coherent pursuit method.
- Explain whether DTmin matches or exceeds OMP at low K BECAUSE of non-greedy planning, or why it lags DUE TO limited capacity.
- If any failure occurs, state the likely cause (data mismatch, checkpoint mismatch, or evaluation bug) and propose the minimal fix.
