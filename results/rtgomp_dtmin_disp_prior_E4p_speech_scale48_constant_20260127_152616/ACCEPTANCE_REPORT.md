# Acceptance Report: E4p-Speech — Dispersion-Prior Frequency Conditioning (Phase-Slope Subbands → Lag Prior)

Run date: 2026-01-27

Experiment ID: E4p-Speech

Decision: **FAIL** (mechanism degrades DT capture severely; conditioning ablation is negligible)

---

## 1) Background / Motivation / Purpose / Expected

- Background:
  - E4o-Speech showed `freq_idx` embedding ablations have negligible impact on DT capture, consistent with DT state being close to an OMP sufficient statistic.
- Motivation:
  - We want a **physics-derived** conditioning signal that is not a degenerate `freq_idx` ID, and that can relate to dispersion-like frequency dependence.
- Purpose:
  - Test whether a **phase-slope (group delay) estimate** can be converted into a **frequency-dependent lag prior** that DTmin uses meaningfully (normal vs shuffle), without breaking compute control.
- Expected:
  - `normal` should outperform `shuffle` on the physics-consistent error metric `E_tau` (frames), and DT capture should remain close to baseline.

---

## 2) Setup (Reproducibility-Required)

- Conda env: `trl-training`
- Device: cpu
- Data roots:
  - MIC: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC`
  - LDV: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV`
- DT checkpoint:
  - `results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth`
- Fixed params:
  - fs=16000, n_fft=2048, hop=160, band=[300,3000]Hz, max_lag=50, tw=32, max_k=16, gain=100, rtg_dim=2
- Lambda grid: `1e-5,3e-5,1e-4,2e-4,3e-4`
- Subset:
  - scale_check_subset: 48 pairs
  - subset fingerprint_md5: `739a181c331f347614090fffe6f4b491`

Code provenance:
- Each run includes `code_state.json` with:
  - `git_head`: `1ebe2f55252be0a0e126cabe41b0e59c4c0b1461`
  - `dirty`: true
  - SHA256 for executed files (notably `scripts/h_exploration/run_rtgomp_e4h_paper_eval.py`)

---

## 3) Exact Commands Executed

Smoke (1 pair, prior normal):
- `results/rtgomp_dtmin_disp_prior_E4p_speech_smoke_normal_20260127_150723/run.log`

Functional suite (scale_check_subset=48):
- Baseline (no prior):
  - `results/rtgomp_dtmin_disp_prior_E4p_speech_scale48_none_20260127_150754/run.log`
- Prior normal:
  - `results/rtgomp_dtmin_disp_prior_E4p_speech_scale48_normal_20260127_151605/run.log`
- Prior shuffle:
  - `results/rtgomp_dtmin_disp_prior_E4p_speech_scale48_shuffle_20260127_152042/run.log`
- Prior constant:
  - `results/rtgomp_dtmin_disp_prior_E4p_speech_scale48_constant_20260127_152616/run.log`

---

## 4) Artifacts (Paths)

Smoke:
- `results/rtgomp_dtmin_disp_prior_E4p_speech_smoke_normal_20260127_150723/summary/dispersion_prior_summary.json`

Functional:
- `results/rtgomp_dtmin_disp_prior_E4p_speech_scale48_none_20260127_150754/summary/compute_matched_summary.json`
- `results/rtgomp_dtmin_disp_prior_E4p_speech_scale48_normal_20260127_151605/summary/dispersion_prior_summary.json`
- `results/rtgomp_dtmin_disp_prior_E4p_speech_scale48_shuffle_20260127_152042/summary/dispersion_prior_summary.json`
- `results/rtgomp_dtmin_disp_prior_E4p_speech_scale48_constant_20260127_152616/summary/dispersion_prior_summary.json`
- Each run also contains `forced_k_summary.json`, `rtg_controllability_summary.json`, `freq_cond_audit_summary.json`, `subset_manifest.json`, `run.log`, `code_state.json`.

---

## 5) Results (Key Numbers @ lambda_c=3e-4)

Definitions:
- `DT_cm`: compute-matched DT capture mean (from `compute_matched_summary.json`)
- `E_tau`: `dt_first_lag_abs_err_vs_tau_physical.mean` in frames (from `dispersion_prior_summary.json`)

| Mode | DT_cm | k_selected_mean | E_tau (frames) |
|---|---:|---:|---:|
| none | 0.994506 | 11.948 | n/a |
| prior normal | 0.740311 | 6.459 | 0.7714 |
| prior shuffle | 0.740720 | 6.466 | 0.7739 |
| prior constant | 0.740636 | 6.463 | 0.7730 |

Conditioning delta:
- `E_tau_shuffle - E_tau_normal = 0.00244` frames (**negligible**)

Compute control:
- Spearman(lambda_c, k_selected_mean) is ~ -1.0 for all modes (control remains monotone).

Acceptance:
- FAIL because:
  - DT_cm collapses from ~0.995 → ~0.740 when the prior is enabled (severe degradation),
  - and the conditioning ablation (`shuffle`) changes `E_tau` by only ~0.002 frames (inconclusive).

---

## 6) Log Interpretation (Required)

1) `dispersion_prior_summary.json`:
- The phase-slope fits produce τ values near **sub-frame magnitude** (example clip: ~ -0.5 to -0.66 frames).
- Subband τ differences are small, so shuffling τ(f) across bins barely changes the assigned τ distribution.

2) `compute_matched_summary.json`:
- DT capture drops sharply when the prior is enabled.
- k_selected_mean drops from ~12 → ~6.5, indicating earlier stopping / reduced selected lags.

---

## 7) Physical / Mathematical Analysis (Required)

First principles:
- System model: `Y(f)=H(f)X(f)`
- Cross-power: `C(f)=X*(f)Y(f)≈|X(f)|^2 H(f)` so `arg C(f)` contains the phase of `H(f)`.
- Phase slope in a band estimates **group delay** (a single average delay per band).

Why this fails for DTmin/OMP lag selection:
- DTmin approximates an OMP-like greedy policy in a **sparse delay dictionary** where multiple lags can be selected to fit multipath structure (`y ≈ Σ_m a_f[m] x_f[t-m]`).
- A single-band group delay is not the same object as the **set of multipath delays** OMP chooses.
- Therefore, a strong Gaussian prior centered at the band group delay penalizes many lags that OMP would legitimately use.
- This implies DT’s effective action space collapses to a narrow neighborhood near τ, which reduces k_selected and reduces capture.

Why shuffle barely changes `E_tau`:
- The fitted τ(f) values are close across subbands (sub-frame range), so permuting them does not strongly alter the prior’s center in practice.
- Additionally, DT’s state (`|D^H r|`) remains the dominant signal, so the prior does not induce a large, frequency-specific deviation relative to the already small τ range.

---

## 8) Cross-Experiment Analysis (Required; ≥3 Results commits)

- E4o-Speech (`1ebe2f5`): `freq_idx` embedding ablations are negligible BECAUSE DT state is close to sufficient for OMP-like lag selection.
- E4o-Speech (`7a5f017`): per-frequency lag variability exists, but aggregation reduces spread, suggesting that naive per-freq behavior can be unstable.
- E4n-Speech (`5ff97bf`): dispersion calibration attempts can fail, indicating that phase/dispersion phenomena are sensitive to the chosen physical estimator.

Pattern recognition:
- Conditioning must be tied to the **same physical object** the policy is optimizing.
- Group delay is a valid physical quantity, but it does not directly encode the sparse multipath support used by OMP/DTmin, therefore it can harm reconstruction.

---

## 9) Extracted Principles (Required)

- Design principle:
  - Condition DT on **multipath-support-relevant** structure (shared delay support across frequency), not on a single band group delay.
- Hypothesis formation:
  - If we center/normalize the prior so it does not change stop calibration, and if τ(f) contains meaningful variation, then shuffle should measurably degrade.
- Resource allocation:
  - Invest in a prior that is **relative** among lags (logit bias with zero-mean) rather than an absolute penalty that changes the stop-vs-lag scale.
- Risk mitigation:
  - Always measure k_selected and capture together; a prior can silently change compute by making stop more attractive.

---

## 10) Next Steps (Concrete)

1) Center the prior per frequency so it does not shift the overall lag-logit scale vs STOP:
   - e.g., `prior <- prior - mean(prior)` (per frequency), or apply the same offset to the STOP logit.
2) Reduce prior strength:
   - smaller `beta`, larger `sigma_frames`, and/or clamp the minimum penalty to avoid `-O(10^2..10^3)` logit shifts.
3) Replace group-delay prior with a multipath-support prior:
   - derive a small candidate lag set from cross-correlation peaks / aggregated OMP selections across frequency, then bias DT toward that set.

---

## 11) Reproduction Instructions (Required)

```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/exp-interspeech-GRU2:$PYTHONPATH
export MPLCONFIGDIR=/tmp/mpl

# Re-run one mode (example: prior normal, scale_check_subset)
conda run -n trl-training PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python -u scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \\\n+  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \\\n+  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \\\n+  --ckpt_path results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth \\\n+  --out_dir results/rtgomp_dtmin_disp_prior_E4p_speech_rerun_normal_<TIMESTAMP> \\\n+  --mode scale_check_subset --device cpu --require_wav_only 1 \\\n+  --lambda_c_values 1e-5,3e-5,1e-4,2e-4,3e-4 \\\n+  --fs 16000 --hop_length 160 --n_fft 2048 --freq_min 300 --freq_max 3000 \\\n+  --max_lag 50 --tw 32 --max_k 16 --gain 100 --rtg_dim 2 \\\n+  --disp_prior_mode phase_slope_subbands --disp_prior_cond_mode normal \\\n+  --disp_prior_num_subbands 3 --disp_prior_min_bins 64 --disp_prior_sigma_frames 2.0 --disp_prior_beta 2.0 \\\n+  --write_delay_diagnostics 1\n+\n+# Verification\n+# - Check: summary/dispersion_prior_summary.json exists\n+# - Compare DT_cm at lambda_c=3e-4 with the table above\n+```

