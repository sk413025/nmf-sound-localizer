# Acceptance Report (Template): E4p-Speech — Dispersion-Prior Frequency Conditioning

Run date: <YYYY-MM-DD>

Experiment ID: E4p-Speech

This report MUST be completed for the Results commit. Use causal language: BECAUSE / DUE TO / THEREFORE / THIS IMPLIES.

---

## 1) Background / Motivation / Purpose / Expected

- Background:
- Motivation:
- Purpose (question answered):
- Expected outcome (with rationale):

---

## 2) Setup (Reproducibility-Required)

- Conda env: `trl-training`
- Device: cpu (recommended)
- Code version:
  - `git rev-parse HEAD`: <hash>
  - Dirty: <yes/no>
- Data roots:
  - MIC: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC`
  - LDV: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV`
- Subset manifest:
  - Path: `results/<run>/subset_manifest.json`
  - fingerprint_md5: <hash>
- DT checkpoint:
  - `results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth`
- Fixed params:
  - fs=16000, n_fft=2048, hop=160, band=[300,3000]Hz, max_lag=50, tw=32, max_k=16, gain=100, rtg_dim=2
- Lambda grid: `1e-5,3e-5,1e-4,2e-4,3e-4`

---

## 3) Exact Commands (Copy-Paste)

Include the exact commands executed (Smoke + Functional). Example:

```bash
conda run -n trl-training PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python -u scripts/h_exploration/run_rtgomp_e4h_paper_eval.py ...
```

---

## 4) Artifacts (Paths)

List all `results/<run_name>/` directories included in the commit:
- Smoke:
  - `results/<...>/run.log`
  - `results/<...>/summary/dispersion_prior_summary.json`
- Functional:
  - `results/<...>/summary/compute_matched_summary.json`
  - `results/<...>/summary/forced_k_summary.json`
  - `results/<...>/summary/rtg_controllability_summary.json`
  - `results/<...>/summary/freq_cond_audit_summary.json`
  - `results/<...>/summary/dispersion_prior_summary.json` (prior enabled)
  - `results/<...>/delay_diagnostics.jsonl` (if enabled)
  - `results/<...>/code_state.json`

---

## 5) Results (Numbers)

Primary metric (physics-consistent):
- `E_tau = dt_first_lag_abs_err_vs_tau_physical.mean` (frames) at `lambda_c=3e-4`

Provide a table at `lambda_c=3e-4`:

| Mode | DT_cm (capture) | k_selected_mean | E_tau (frames) |
|---|---:|---:|---:|
| none | | | n/a |
| prior normal | | | |
| prior shuffle | | | |
| prior constant | | | |

Also report:
- Spearman(lambda, k_selected) for each run (compute control)

Acceptance decision:
- PASS / PASS_WITH_WARNINGS / FAIL:
- Justification:

---

## 6) Log Interpretation (Required)

Read `summary/dispersion_prior_summary.json`:
- Explain what the phase-slope fits imply (tau_ms, r2, rmse) and whether they are stable.
- Explain why `E_tau_shuffle > E_tau_normal` occurs (or why it does not), in causal terms.

Read `summary/compute_matched_summary.json`:
- Explain whether capture is saturated and what that implies about sensitivity of this metric.

---

## 7) Physical / Mathematical Analysis (Required)

Start from first principles:
- System model: `Y(f)=H(f)X(f)`
- Cross-power: `C(f)=X*(f)Y(f)≈|X(f)|^2 H(f)`
- Phase slope → group delay: `τ = -(1/2π) dφ/df` (linear fit per subband)
- DT decision state: `s ≈ |D^H r|`

Explain:
- Why a lag prior derived from τ(f) is a physically meaningful “frequency conditioning”.
- Why `freq_idx` embedding can be ignored BECAUSE the DT state is (nearly) sufficient.
- Why the new mechanism should be sensitive to shuffling τ(f) across frequencies.

---

## 8) Cross-Experiment Analysis (Required; ≥3 Results commits)

Reference at least 3 prior Results commits (hashes) and compare patterns.

Example scaffold:
- Pattern recognition:
  - Commit <hash1> (E4o): freq_idx ablations negligible BECAUSE state is sufficient.
  - Commit <hash2> (E4n/E4m): dispersion diagnostics show band-dependent delays DUE TO phase structure.
  - Commit <hash3> (E4p): physics prior makes conditioning measurable THEREFORE the correct conditioning object is τ(f), not f.

Success factors:
- What works BECAUSE of physics constraints?

Failure modes:
- What fails DUE TO metric saturation or identifiability limits?

---

## 9) Extracted Principles (Required)

Convert observations into design rules:
- Design principle(s):
- Hypothesis formation:
- Resource allocation:
- Risk mitigation:
- Success amplification:

---

## 10) Meta-Reflection (Required)

- Methodology assessment:
- Documentation quality:
- Time/resource efficiency:
- Knowledge gaps:

---

## 11) Reproduction Instructions (Required)

```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=/path/to/project:$PYTHONPATH
export MPLCONFIGDIR=/tmp/mpl

# Re-run the functional suite (scale_check_subset)
<commands>

# Verification
# - Check: summary/dispersion_prior_summary.json exists
# - Check: E_tau_shuffle - E_tau_normal >= threshold (spec)
```

