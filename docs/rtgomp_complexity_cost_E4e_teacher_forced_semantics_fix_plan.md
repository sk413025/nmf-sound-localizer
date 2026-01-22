# Plan: E4e — Teacher-Forced Semantics + Step Indexing Fix (Make B Meaningful)

E4d fixed teacher-forced duplicate lag selection by updating the teacher mask, which largely removed the catastrophic capture collapse. However, the B (free vs teacher-forced) check is still not a clean measure of “student STOP behavior under teacher residual evolution” because the current `teacher_forced` implementation mixes multiple semantics (teacher penalty-stop vs student STOP) and uses ambiguous step indexing.

This plan defines E4e: a minimal semantic cleanup so `teacher_forced` becomes a meaningful comparator to free rollout.

---

## Background

Executed artifacts:
- E4c extval A/B/C: `results/rtgomp_lambda_cost_E4c_extval_ABC_20260121_093247/` (commit `d913cbf`)
  - A and C pass; B teacher-forced was invalid (capture collapse).
- E4d mask fix + B rerun: `results/rtgomp_lambda_cost_E4d_teacher_forced_mask_fix_20260121_113357/` (commit `d6005ab`)
  - Teacher-forced monotonicity is corrected, but the check still fails the pre-registered capture threshold by a small margin.

This suggests the remaining issue is not “model is broken” but “teacher-forced mode is not measuring what we intended”.

---

## Problem Statement

We intend `teacher_forced` to mean:
- Teacher controls residual evolution (greedy lag choice + projection),
- Student controls STOP decisions (when to terminate),
so that we can compare student STOP behavior under a stable state distribution.

The current implementation instead embeds a teacher penalty-based stop mask and does not fully align:
- which residual is used for capture,
- what “steps_used_mean” counts (0-based vs number of selected atoms),
- and whether STOP is truly controlled by the student in teacher-forced mode.

---

## Hypotheses (Root Causes)

1) Step counting ambiguity: reported steps may be 0-based indices rather than “number of selected atoms”, causing acceptance thresholds and interpretation to drift.\n2) Mixed semantics: teacher penalty-stop and student STOP are entangled; therefore teacher-forced is not a clean “student STOP under teacher residual evolution” measurement.\n
---

## Minimal Fix (E4e)

In `scripts/h_exploration/run_lambda_override_grid_eval.py`, redefine `teacher_forced` mode as:

1) Teacher always selects lags greedily with uniqueness masking (already fixed in E4d).\n2) Teacher updates residuals ONLY for frequencies that have not been stopped by the student.\n3) The only termination criterion per frequency is student STOP (or reaching `max_k`).\n4) `steps_used_mean` is defined as the number of selected atoms (1..K), not 0-based indices.\n5) Keep the teacher penalty-stop rule (if needed) only as an auxiliary diagnostic field (e.g., `teacher_stop_step_mean`), not as the primary stop behavior.\n
This keeps the change scoped to evaluation semantics and avoids changing training or model weights.

---

## Experiment Design (E4e)

Rerun ONLY B (free + teacher-forced) on the same checkpoint and subset:
- checkpoint: `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth`\n- subset fingerprint: `668135f8f6f7baaf99dffeef4cbb1a21`\n- lambda grid: `1e-4,3e-4,1e-3,3e-3,1e-2`\n
Output:
- `results/rtgomp_lambda_cost_E4e_teacher_forced_semantics_fix_<timestamp>/`\n
---

## Acceptance Targets (E4e)

PASS if all:
- Teacher-forced at low penalty is high-capture and comparable to free:\n  - at `lambda_c=1e-4`, `capture_teacher_forced >= 0.98` AND `capture_free - capture_teacher_forced <= 0.02`\n- Teacher-forced STOP is controllable by lambda (non-degenerate):\n  - `spearman(lambda_c, steps_used_mean_teacher_forced) <= -0.6`\n  - `steps_range_teacher_forced >= 0.10`\n- Semantics correctness:\n  - In teacher-forced mode, capture and steps change when student STOP changes (not only when teacher penalty-stop changes).\n
If any fail, E4e is FAIL and must be analyzed causally.

