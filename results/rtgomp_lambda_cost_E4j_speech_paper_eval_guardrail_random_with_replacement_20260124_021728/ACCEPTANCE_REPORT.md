# Acceptance Report: E4j-Speech -- Guardrail Diagnostic (Random with replacement)

## 1) Executive Summary

- Run (guardrail diagnostic): results/rtgomp_lambda_cost_E4j_speech_paper_eval_guardrail_random_with_replacement_20260124_021728/
- Outcome: PASS_WITH_WARNINGS (expected diagnostic behavior)
- Purpose:
  - Confirm the evaluator reports duplicate selections and capture out-of-range events when Random sampling allows duplicates.
- Dataset domain statement:
  - This run uses the speech WAV dataset only (no .npy files in the manifest): YES

## 2) Setup

- Conda env: trl-training
- Python: 3.11.13
- Device: cpu
- MPLCONFIGDIR: /tmp/mpl

- code_state.json: results/rtgomp_lambda_cost_E4j_speech_paper_eval_guardrail_random_with_replacement_20260124_021728/code_state.json

- Subset manifest: results/rtgomp_lambda_cost_E4j_speech_paper_eval_guardrail_random_with_replacement_20260124_021728/subset_manifest.json
  - num_pairs = 3
  - fingerprint_md5 = 4b0059419ff37ef2d3496302841b4ba2
  - all paths end with .wav: YES

## 3) Exact Commands

See:
- results/rtgomp_lambda_cost_E4j_speech_paper_eval_guardrail_random_with_replacement_20260124_021728/run.log

## 4) Results (Expected Guardrail Signals)

From summary/compute_matched_summary.json integrity:
- random_sampling = with_replacement
- random_duplicate_rate = 0.07198
- random_duplicate_count = 5977
- num_capture_out_of_range_total = 651

Decision:
- PASS_WITH_WARNINGS because duplicate_rate > 0 and out-of-range capture events are expected in this diagnostic setting.
- This is useful BECAUSE it proves the evaluator is not silently clamping or hiding ill-conditioning; THEREFORE the paper baseline (without replacement) guardrails are meaningful.

## 5) Reproduction

1) Environment:
- source ~/.zshrc
- conda activate trl-training
- export PYTHONPATH=.
- export MPLCONFIGDIR=/tmp/mpl

2) Execution:
- run the exact command in run.log

3) Verification:
- Confirm random_duplicate_rate > 0 and num_capture_out_of_range_total > 0.
