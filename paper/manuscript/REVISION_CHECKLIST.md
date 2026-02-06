# Manuscript Revision Checklist (AGENTS.md + Nature Communications Style)

Scope: `paper/manuscript/manuscript.md` — Results, Discussion, Methods  
Goal: Bring the manuscript into compliance with `AGENTS.md` (Manuscript Style + Equation/Derivation Policy) while preserving derivations (do not delete equations; moving/reorganizing is allowed).

How to use:
- Execute items top-to-bottom (P0 → P1 → P2).
- After completing an item, change `- [ ]` to `- [x]` and record the exact touched location(s) (line numbers) in the “Evidence” sub-bullet.

---

## P0 — Must Fix (AGENTS / reviewer blockers)

### R0-1 Results: Phenomenon-first paragraph openings
- [x] Rewrite Results opening sentences to start with the observed phenomenon, not setup/process.
  - Target: `manuscript.md` Results Fig.1 first paragraph (currently begins with “We first demonstrate… / In our setup…”).
  - Change: replace process/setup lead with phenomenon lead (e.g., “Single-point vibroacoustic spectra form direction-specific fingerprints…”).
  - Keep setup details as a short parenthetical + Methods pointer only.
  - Evidence: `paper/manuscript/manuscript.md:26`

### R0-2 Results: De-proceduralize subheaders (####)
- [x] Replace procedural subheaders with declarative discovery-style statements.
  - Targets: the `####` headers under Fig.2 (e.g., “Physics model → …”, “Discretization → …”, “... via SVD”).
  - Change: make each subheader read like a claim/phenomenon (not a method step).
  - Evidence: `paper/manuscript/manuscript.md:39,51,79`

### R0-3 Unify the definition ladder for Y / power / features (symbol consistency)
- [x] Make `Y` consistent across Results and Methods by separating:
  - Physical complex response: `Y(ω;θ)=V(x_L,y_L,ω)` (complex velocity response).
  - Observable power statistic: define a new symbol (e.g., `S(ω;θ)`), and make features depend on `S`, not directly on `Y`.
  - Estimator: in Methods, define `P_band[k]` as an estimator of `S(ω_k;θ)` (use a hat notation such as `Ŝ`).
- [x] Remove/replace any “identify |Y|^2 := P_band” phrasing with an explicit estimation relationship + assumptions.
- [x] Add bridge sentences (AGENTS requirement): each time symbols shift (Y → S → y), state:
  1) physical meaning, 2) why introduced here, 3) assumptions.
  - Evidence: `paper/manuscript/manuscript.md:21,42,55,199,209,215`

### R0-4 Results: Enforce “Hide the Method” boundary (remove method details)
- [x] In Results, remove/compact statements that describe implementation/protocol details:
  - Atom estimation specifics (calibration averaging/standardization specifics) → refer to Methods.
  - Attention internals (query/key/softmax mechanics, architecture specs) → refer to Methods.
- [x] Keep only Tier-A interpretive equations needed to explain the phenomenon; move the rest to Methods.
  - Evidence: `paper/manuscript/manuscript.md:59,115`

### R0-5 Results: Every key equation gets interpretation + figure evidence
- [x] After every standalone key equation in Results, add 1–2 sentences:
  - Physical interpretation (what it means in the physics/measurement context).
  - Evidence pointer to specific figure panels (e.g., Fig. 2a–c, Fig. 5b).
  - Evidence: `paper/manuscript/manuscript.md:40,49,59,65,75,86,105,125`

### R0-6 Contact-loading claim: align claim with evidence placeholders
- [x] In Results, soften any hard conclusion about contact-loading unless a concrete panel/metric is cited.
- [x] Add/extend placeholders in Methods so Results can point to a named result location (e.g., `{TBD_CONTACT_LOADING_PANEL_REF}`) without fabricating evidence.
  - Evidence: `paper/manuscript/manuscript.md:28,189`

### M0-1 Methods: Remove parameter/config tables (space discipline)
- [x] Delete the standalone `### Parameters` section/table and inline only key numeric settings at first use (Experimental setup, Signal processing pipeline, Training).
  - Evidence: `paper/manuscript/manuscript.md:182,198,293`

### M0-2 Methods: Replace hyperparameter bullet dumps with prose
- [x] Replace `Loss/Optimization/Protocol` bullets with a single compact paragraph and keep full detail in released configs.
  - Evidence: `paper/manuscript/manuscript.md:293`

### M0-3 Housekeeping: Move placeholder tracker out of the manuscript
- [x] Remove internal placeholder TODO blocks/comments from `manuscript.md`; keep a tracker in this checklist instead so the exported DOCX contains only submission content.
  - Evidence: `paper/manuscript/manuscript.md:321`

---

## P1 — Strongly Recommended (Nature Comm clarity / polish)

### D1-1 Discussion: Increase “active figure” anchoring
- [x] Add at least one explicit figure anchor per Discussion paragraph (mechanism paragraphs should cite Fig.1/2/5/4 as appropriate).
  - Evidence: `paper/manuscript/manuscript.md:167,169,171,173,175`

### D1-2 Discussion: Bridge sentence for α_m(θ) and modal language
- [x] When using symbols like `α_m(θ)`, add a short bridge sentence tying it to the earlier modal/separable form and stating assumptions.
  - Evidence: `paper/manuscript/manuscript.md:169`

### M1-1 Methods: Rewrite OMP step list into journal-style mathematical definition
- [x] Replace numbered step list with a compact recursive definition (support set update, LS refit, residual update), preserving full reproducibility.
  - Evidence: `paper/manuscript/manuscript.md:238`

### M1-2 Methods: Add a “Definition ladder” mini-section
- [x] Add a short paragraph/list that explicitly links:
  `v[n]` (velocity waveform) → `V[k,t]` (STFT) → `P[k] / P_band[k]` (power estimate) → `S(ω;θ)` (defined observable) → `y, \tilde y` (features) → `H` (dictionary).
  - Include units/physical meaning where relevant.
  - Evidence: `paper/manuscript/manuscript.md:199`

---

## P2 — Optional (further strengthen the IRDM narrative)

### R2-1 Results: Tighten Fig.2 derivation density (without deleting equations)
- [x] Keep only the minimal Tier-A set in Results; move extended derivation steps to Methods (“Derivation details and assumptions”) with a pointer.
  - Evidence: `paper/manuscript/manuscript.md:40,254`

### P2-2 Final consistency sweep
- [x] Ensure Results contain no STFT params, dataset split rules, training recipes, or architecture specs beyond “see Methods”.
- [x] Ensure no placeholders appear inside LaTeX math environments.
- [x] Ensure symbol continuity for `Y/S/P/y/tilde y/H/UΣV^T/z/A/x/K`.
  - Evidence: `paper/manuscript/manuscript.md:21,199`

---

## Acceptance Criteria (Done when all true)
- [x] Results paragraphs are phenomenon-first and evidence-driven.
- [x] Results headers (including `####`) read as discoveries, not procedure steps.
- [x] `Y` vs power/feature definitions are consistent and assumption-annotated.
- [x] Results contain only Tier-A interpretive equations; procedural details live in Methods.
- [x] Discussion follows the synthesis inverted funnel with figure-anchored mechanisms.
- [x] Methods reads as a technical manual with a clean definition ladder and journal-style algorithm descriptions.

---

## Internal-only — Placeholder tracker (do not include in submission exports)
- Abstract: {TBD_SNR_MIN_DB}.
- Results (Fig. 1): {TBD_WN_EVAL_CLIPS_PER_ANGLE}, {TBD_SIM_WITHIN_MEAN}, {TBD_SIM_WITHIN_SD}, {TBD_SIM_BETWEEN_MEAN}, {TBD_SIM_BETWEEN_SD}.
- Results (Fig. 2): {TBD_SVD_R}, {TBD_SVD_ENERGY_PCT}, {TBD_SVD_INFERENCE_POLICY}.
- Results (Fig. 4): {TBD_SPEECH_N_CLIPS_TOTAL}, {TBD_SPEECH_CLIPS_PER_ANGLE}, {TBD_SPEECH_TOP1_ACC_PCT}, {TBD_SPEECH_MAE_DEG}, {TBD_SPEECH_P95_DEG}, {TBD_N_INDEP_RUNS}, {TBD_ACC_SNR10_PCT}, {TBD_ACC_SNR5_PCT}, {TBD_ACC_SNR0_PCT}, {TBD_ACC_ABL_NO_TRANSFORMER_PCT}, {TBD_ACC_ABL_FIXED_HEURISTIC_PCT}, {TBD_ACC_ABL_DENSE_ROUTING_PCT}.
- Results (Fig. 5): {TBD_DIAGONAL_CONC_FACTOR}.
- Results (Fig. 6): {TBD_RMSE_RANGE_DEG}, {TBD_RMSE_OMP_COMPLEX_DEG}.
- Experimental setup: {TBD_RADIUS_M}, {TBD_N_ANGLES}, {TBD_ANGLE_RANGE_DEG}, {TBD_ANGLE_STEP_DEG}, {TBD_ANGLE_ZERO_REFERENCE}, {TBD_ANGLE_SIGN_CONVENTION}, {TBD_LDV_MODEL}, {TBD_LDV_SPOT_LOCATION}, {TBD_OBJECT_MOUNTING_CONDITION}, {TBD_TRIAL_DURATION_S}.
- Controls: {TBD_CONTACT_SENSOR_TYPE}, {TBD_CONTACT_SENSOR_MASS_G}, {TBD_CONTACT_SENSOR_LOCATION}, {TBD_CONTACT_SENSOR_ATTACHMENT}, {TBD_CONTACT_LOADING_RESULT_SUMMARY}, {TBD_CONTACT_LOADING_PANEL_REF}.
- Calibration/splits: {TBD_WN_CALIB_CLIPS_PER_ANGLE}, {TBD_WN_EVAL_CLIPS_PER_ANGLE}, {TBD_WN_SPLIT_RULE}, {TBD_SPEECH_SPLIT_RULE}, {TBD_SPLIT_TRAIN}, {TBD_SPLIT_VAL}, {TBD_SPLIT_TEST}.
- STFT/features: {TBD_FS_HZ}, {TBD_NFFT}, {TBD_STFT_WIN_SAMPLES}, {TBD_STFT_HOP_SAMPLES}, {TBD_FREQ_MIN_HZ}, {TBD_FREQ_MAX_HZ}, {TBD_F_BINS}, {TBD_EPS}, {TBD_ZSCORE_EPS}.
- Model/training: {TBD_K_STAGES}, {TBD_D_MODEL}, {TBD_N_HEADS}, {TBD_N_LAYERS}, {TBD_LOGIT_CONSTRUCTION}, {TBD_TRAIN_LOSS_DESCRIPTION}, {TBD_OPTIMIZER}, {TBD_LR}, {TBD_WEIGHT_DECAY}, {TBD_BATCH_SIZE}, {TBD_EPOCHS}, {TBD_RANDOM_SEED}.
- Ablations: {TBD_ABL_NO_TRANSFORMER_DEF}, {TBD_ABL_FIXED_HEURISTIC_DEF}, {TBD_ABL_DENSE_ROUTING_DEF}.
- Statistics: {TBD_N_REPEATS}, {TBD_REPLICATE_DEFINITION}.
- Availability/admin: {TBD_DATA_AVAILABILITY_URL}, {TBD_CODE_AVAILABILITY_URL}, {TBD_GRANT_INFO}.
