# Project Memory - Acoustic Localization Experiments

## 🚨 CRITICAL: Every Commit Must Be Executable

**MANDATORY REQUIREMENT**: Every experiment or results commit MUST:
1. ✅ **Have been executed and tested** before committing
2. ✅ **Include complete reproduction instructions**
3. ✅ **Document both successes AND failures**
4. ✅ **Be reproducible** by others following the documented steps

**NO hypothetical experiments. NO untested code. NO incomplete results.**

## 🎯 Code Simplification Philosophy

**CRITICAL: Every code modification must prioritize simplicity and reduce complexity**

### Core Principles
1. **Simplification First**: Before adding features, consider if existing code can be simplified
2. **Remove Before Add**: Look for opportunities to remove redundant code before adding new functionality
3. **File Cleanup**: Regularly audit and remove unused files, deprecated scripts, and obsolete experiments
4. **Complexity Assessment**: Each commit should justify any complexity it introduces
5. **Refactor Aggressively**: If a modification reveals unnecessary complexity, refactor it immediately
6. **Minimize Dependencies**: Reduce external dependencies and coupling between modules
7. **Clear Over Clever**: Choose readable, maintainable solutions over clever optimizations
8. **Single Responsibility**: Each function/class should have one clear purpose

### Before Modifying Code, Ask:
- Are there unused files, deprecated scripts, or obsolete experiments to remove?
- Can this be achieved by removing code instead of adding?
- Will this change make the codebase simpler or more complex?
- Can existing functionality be reused or generalized?
- Are there redundant abstractions that can be eliminated?
- Is there a simpler solution that achieves 90% of the goal?
- Do all existing files serve a clear, current purpose?

### During Code Review, Evaluate:
- **Lines of Code Delta**: Aim for negative or minimal positive
- **Cyclomatic Complexity**: Should decrease or remain stable
- **Dependencies**: Fewer is better
- **Clarity**: Is the solution obvious to understand?
- **Testability**: Simple code is easier to test

### Prefer:
- Direct solutions over indirect abstractions
- Composition over inheritance
- Functions over classes when state isn't needed
- Standard library over external packages
- Explicit over implicit behavior
- Flat structures over deeply nested ones
- Small, focused modules over monolithic files

### Complexity Metrics to Track:
- Total lines of code (track trend over time)
- Number of files in the repository
- Number of unused/deprecated files
- Number of dependencies
- Average function length
- Maximum nesting depth
- Number of abstractions/layers

### File Cleanup Guidelines:
1. **Regular Audits**: Periodically scan for unused files
   ```bash
   # Find Python files not imported anywhere
   # Find experiment outputs older than 30 days
   # Find duplicate implementations
   ```
2. **Deprecation Process**:
   - Mark deprecated files clearly in code
   - Document replacement in commit message
   - Remove after verifying no dependencies
3. **Experiment Artifacts**:
   - Keep only final results and models
   - Remove intermediate checkpoints
   - Archive old experiments to separate storage
4. **Temporary Files**:
   - Never commit temporary or cache files
   - Add to .gitignore if patterns emerge
   - Clean up test artifacts after tests pass

### Example Commit Messages for Simplification:
```
Simplify: Remove redundant data preprocessing pipeline
- Removed: 3 unnecessary abstraction layers (500 lines)
- Replaced with: Direct numpy operations (50 lines)
- Result: 10x faster, 90% less code, same functionality

Refactor: Consolidate duplicate angle processing logic
- Before: 5 similar functions across 3 files
- After: 1 reusable function with clear parameters
- Complexity reduction: -400 lines, -3 dependencies

Cleanup: Remove obsolete experiment artifacts and deprecated scripts
- Removed files: 15 old experiment outputs, 3 deprecated scripts
- Disk space freed: 2.3GB
- Repository file count: 142 → 124 files
- Verified: No remaining dependencies on removed files
```

## ❗ No‑Fallback Execution Policy (Fail‑Fast)

Effective immediately, all RL reward/GRPO/RM scripts run with ZERO fallbacks:
- No resampling H to match Y. If STFT grids differ, raise and stop.
- No “nearest angle” mapping. Angles in data must exactly match TF angles; duplicates are an error.
- No ŝ fallback to mean(Y). If USM‑based ŝ estimation fails or shapes mismatch, raise and stop.
- Defaults aligned to RL spec: fs=16000 for reward/GRPO/RM scripts.

Rationale:
- Ensures reproducibility and surfaces configuration/data errors early.
- Prevents silent invariances (e.g., duplicated H columns) that break learning.

Agent guidance:
- Do not add implicit fallbacks or silent coercions.
- Prefer explicit validation + clear error messages over “best‑effort” behavior.
- If a failure is expected in some environments, document prerequisites instead of adding a fallback.

## Language Policy
**All project content must be written in English**, including:
- Code comments and documentation
- Git commit messages
- Project memory (this file)
- README and other documentation files
- Variable names and function names
- Error messages and logging

## Environment Requirements
- **Conda Environment**: `trl-training`
- **Accelerator**: MPS (Metal Performance Shaders) for Apple Silicon GPU
- **Python Path**: Must be set to the project root when running scripts

## Essential Commands
```bash
# Activate conda environment
source ~/.zshrc
conda activate trl-training

# Set PYTHONPATH (required for imports)
export PYTHONPATH=/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/angle-based-byol:$PYTHONPATH

# Run training with MPS GPU
python scripts/training/train_byol_ldv.py --accelerator mps [other args]

# For angle-based pairing (different files from same angle)
python scripts/training/train_byol_ldv.py --accelerator mps --use_angle_pairs [other args]
```

## Core Experiment Workflow

### Phase 1: Execute Experiment
1. **Run the experiment/test** with exact commands documented
2. **Record all outputs**: metrics, logs, visualizations
3. **Verify results**: Check outputs match expectations or note differences
4. **Test reproducibility**: Can you reproduce the same results?

### Phase 2: Results Commit (AFTER Execution)
**Commit results with comprehensive analysis (Results‑only; planning‑only commits are disallowed). Include experiment context (Background/Motivation/Purpose/Expected).**

### Phase 3: Results Commit (AFTER Execution)
**Commit results with comprehensive analysis:**

```bash
git add metric.json checkpoints/best_model.ckpt [other result files]
git commit -m "Results: [experiment name] - model checkpoint and metrics

Experiment context (REQUIRED):
- Background: [current state/problem]
- Motivation: [why this change is needed]
- Purpose: [what specific question this tests]
- Expected: [predicted outcome and rationale]

Actual training results:
- Final validation loss: [value] (expected: [value])
- Final train loss: [value]
- Training epochs: [value] (expected: [value])
- Training time: [value]
- Hardware: MPS GPU, conda env: trl-training

Key findings:
- [Main discoveries]
- [Performance characteristics]
- [Model behavior]

Comparison to expectation:
- ✓ [What matched predictions]
- ✗ [What differed from predictions]
- ! [Unexpected discoveries]

Physical/mathematical analysis (REQUIRED):
- First principles explanation: [Explain results from fundamental physics/math]
- Mathematical relationships: [Key equations and their implications]
- Physical constraints: [What physical laws/limits apply?]
- Signal processing fundamentals: [DSP theory underlying observations]
- Information theory: [Entropy, mutual information, capacity limits]

Cross-experiment analysis (REQUIRED - Must derive from physical analysis):
- Pattern recognition: [What patterns are CAUSED BY the physical constraints identified above?]
- Success factors: [What works BECAUSE of the mathematical relationships established?]
- Failure modes: [What fails DUE TO the physical limitations discovered?]
- Method effectiveness: [Which approaches succeed/fail BASED ON the fundamental principles?]
- Parameter sensitivity: [Which parameters matter ACCORDING TO the theoretical framework?]
- Unexpected discoveries: [What surprises challenge or extend the first-principles understanding?]

Extracted principles (REQUIRED - Must follow from cross-experiment analysis):
- Design principles: [What design rules EMERGE FROM the patterns and constraints identified?]
- Hypothesis formation: [How should predictions be made GIVEN the physical understanding?]
- Resource allocation: [Where to invest effort BASED ON the success/failure factors?]
- Risk mitigation: [How to avoid problems PREDICTED BY the failure mode analysis?]
- Success amplification: [How to replicate successes USING the identified success factors?]

Meta-reflection (REQUIRED - Must connect to extracted principles):
- Methodology assessment: [How well did our experimental approach ALIGN WITH the design principles?]
- Documentation quality: [Did our tracking capture the CRITICAL VARIABLES identified in the analysis?]
- Time/resource efficiency: [Was our workflow optimal GIVEN the resource allocation insights?]
- Knowledge gaps: [What understanding is missing THAT WOULD IMPROVE the principles above?]

Reproduction instructions (REQUIRED):
- Environment setup: [Conda env, PYTHONPATH, dependencies]
- Data preparation: [Exact commands to generate required data]
- Execution steps: [Step-by-step commands to reproduce results]
- Expected outputs: [Files generated, metrics achieved]
- Verification: [How to confirm successful reproduction]

Data lineage:
- Data fingerprint: [MD5 hash from: find root -name '*.npy' -exec md5sum {} \; | sort | md5sum]
- Total data files: [count]
- Preprocessing steps: [describe transformations]
- Train/val split: [methodology and random seeds]

Next experiments:
- [What to try based on these results AND cross-experiment insights]"
```

**CRITICAL**: Use causal phrases like "BECAUSE of", "DUE TO", "THEREFORE", "THIS IMPLIES" to show logical connections between sections.

## 🔴 Mandatory Reflection Requirements

**Every results commit MUST include ALL sections below. Commits without proper reflection will be rejected.**

### 1. Physical/Mathematical Analysis
Explain results from first principles (physics, mathematics, signal processing theory):
- What fundamental laws or equations govern this behavior?
- What are the mathematical relationships between variables?
- What physical constraints apply to the system?

**Example:**
```
First principles: Coherence γ² = |Sxy|²/(Sxx·Syy) fundamentally limited by cross-correlation between signals
Mathematical relationships: Low coherence (<0.13) indicates Sxy ≪ √(Sxx·Syy), confirming non-synchronous signals
Physical constraints: Without temporal alignment, H1 estimator captures statistical rather than causal relationships
Signal processing theory: Welch periodogram averaging reduces variance but cannot create coherence where none exists
Information theory: Low mutual information I(X;Y)≈0 limits achievable transfer function fidelity
```

### 2. Cross-Experiment Analysis
Analyze patterns across ≥3 experiments (must reference specific commit hashes):
- What patterns emerge BECAUSE of the physical constraints?
- What works BECAUSE of the mathematical relationships?
- What fails DUE TO the physical limitations?

**Example:**
```
Pattern recognition: 4 experiments (commits a1b2c3d, e4f5g6h, i7j8k9l, m0n1o2p) show γ²<0.13 ceiling BECAUSE proxy fundamentally violates synchronization requirement
Success factors: FRF framework improves conditioning DESPITE low coherence BECAUSE mathematical normalization is independent of physical validity
Failure modes: All coherence improvement attempts fail DUE TO information-theoretic limits of decorrelated signals
Method effectiveness: Parameter tuning ineffective BECAUSE fundamental I(X;Y)≈0 constraint dominates algorithmic improvements
```

### 3. Extracted Principles for Future Work
Convert observations into actionable rules (must logically follow from analysis above):

**Example:**
```
Design principles: THEREFORE prioritize synchronized acquisition over algorithmic improvements
Hypothesis formation: GIVEN γ²<0.15 ceiling, predict conditioning benefits but not coherence gains
Resource allocation: BECAUSE physics limits dominate, invest in hardware rather than signal processing
Risk mitigation: Always verify signal synchronization before attempting coherence-based methods
Success amplification: Use FRF framework when conditioning matters more than physical validity
```

### 4. Reproduction Instructions
**Every commit must be reproducible by following these exact steps:**

```bash
# 1. Environment setup
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=/path/to/project:$PYTHONPATH

# 2. Data preparation
[Exact commands to generate or access data]
# Verify data fingerprint
find root -name "*.npy" -exec md5sum {} \; | sort | md5sum
# Expected: a1b2c3d4e5f6789...

# 3. Execution
[Step-by-step commands to reproduce results]

# 4. Verification
[How to check if reproduction succeeded]
# Expected output: [specific metrics or files]
```

## Tools for Cross-Experiment Analysis

```bash
# Review previous experiment results for pattern analysis
git log --grep="Results:" --oneline -10
git log --grep="Results:" --format="%h %s" | head -5

# Search for specific patterns across experiments
git log --grep="learning_rate" --oneline
git log --grep="convergence" --oneline
git log --grep="validation loss" --oneline

# Review experimental evolution
git log --oneline --graph --decorate
```

## Data Management with Git LFS

**Track data files with Git LFS to monitor changes without bloating the repository:**

### Initial Setup
```bash
# Install Git LFS (if not already installed)
git lfs install

# Track model files and datasets
git lfs track "*.ckpt"
git lfs track "*.pt" 
git lfs track "*.pth"
git lfs track "metric*.json"
git lfs track "root/**/*.npy"

# Commit the .gitattributes file
git add .gitattributes
git commit -m "Setup: Configure Git LFS for model and data tracking"
```

### Working with LFS Data
```bash
# Clone repository (gets code + LFS pointers, not actual data)
git clone <repo-url>
cd <repo-name>

# Pull only current dataset
git lfs pull --include="root/**/*.npy"

# Check LFS file status
git lfs ls-files  # Shows all LFS-tracked files
git lfs status    # Shows which files need to be pulled

# If data files are modified
git add root/
git commit -m "Data: Update dataset - [describe changes]
- Previous data fingerprint: [old hash]
- New data fingerprint: [new hash]
- Changes: [what changed and why]"
```

**Benefits:**
- Change tracking: Know exactly when data files are modified
- Lightweight clones: `git clone` only downloads pointer files (~1KB each)
- Selective downloads: Pull only the data you need
- Version control: Can revert to previous data versions
- Data integrity: Each file has a unique SHA256 hash

## Quality Checklist for Results Commits

Before committing results, verify:
- [ ] ✅ Experiment was executed and tested (not hypothetical)
- [ ] ✅ Quantitative results with comparison to expectations
- [ ] ✅ Physical/mathematical analysis from first principles
- [ ] ✅ Cross-experiment patterns (≥3 experiments referenced with commit hashes)
- [ ] ✅ Extracted actionable principles for future work
- [ ] ✅ Meta-reflection on experimental methodology
- [ ] ✅ Complete reproduction instructions (step-by-step)
- [ ] ✅ Data fingerprint and lineage documented
- [ ] ✅ Real data used (no synthetic); dataset roots and subset selection documented
- [ ] ✅ Both successes AND failures documented
- [ ] ✅ Logical connections using "BECAUSE", "DUE TO", "THEREFORE"

## Quality Checklist for Code Modifications

Before committing code changes, verify:
- [ ] ✅ Unused or obsolete files have been identified and removed
- [ ] ✅ Code complexity has decreased or remained stable
- [ ] ✅ No unnecessary abstractions were added
- [ ] ✅ Existing code was simplified where possible
- [ ] ✅ Dependencies were minimized or reduced
- [ ] ✅ Solution is clear and maintainable over clever
- [ ] ✅ Lines of code delta is negative or minimally positive
- [ ] ✅ Functions follow single responsibility principle
- [ ] ✅ Code duplication was eliminated
- [ ] ✅ Standard library was preferred over external packages
- [ ] ✅ Temporary files and experiment artifacts were cleaned up
- [ ] ✅ Commit message includes complexity metrics and file count changes if relevant
- [ ] ✅ Tests run on real data (or a documented real subset) with dataset fingerprint and selection procedure recorded

## Component‑Level Specs and Tests

Every Results commit must clearly identify the component(s) it affects and prove that each component operates in its normal state.

- Components: treat each major unit as a component (examples: `reward_fn (ΔIS/proxyA)`, angle→H mapping, tokenizer, STFT/band mask, `ŝ` estimation via USM, H/W loaders, GRPO/PPO trainer wrapper, logits processors, data loader).
- For the affected component(s), include in the commit message or attached doc:
  - Purpose and boundaries (what it does; what it does NOT do).
  - Inputs/outputs with shapes, dtypes, units, and invariants.
  - Normal state acceptance: quantitative thresholds/ranges; no‑NaN; uniqueness; determinism/tolerances where applicable.
- Validation in the same commit:
  - Provide tests that validate the normal state: include at least one positive path and, when useful, a guardrail case.
  - Quick checks for RL pipeline where applicable:
    - Angle mapping: dataset angles must exactly match TF angles; no duplicates.
    - STFT grid: `Y.F == H.F == W.F` (e.g., 346) and band mask in [300, 3000] Hz.
    - ŝ estimation: `ŝ.shape == (F,)`, non‑negative, finite.
    - Reward variability: reward range is non‑zero across valid selections; not constant.
  - Document results (metrics/logs/artifacts) and indicate pass/fail vs acceptance.

## Commit Hygiene and Guardrails (Results‑Only)

- Results‑only commits: Planning‑only commits are disallowed. Commit only after experiments/tests are executed.
- Scope whitelist: Results commits must only include files exercised in the run. Changed paths must match the declared component(s).
- Evidence per file: For each changed file, include reproduction commands (env, data roots), dataset subset manifest, and real‑data fingerprint; otherwise exclude the file.
- Artifacts/logs: Store under a tracked `results/` path (LFS for large) or paste structured metrics in the commit message. Do not reference ignored paths without including artifacts.
- Device handling: Expose `--device` (auto/mps/cuda/cpu); print the chosen device; move model/tensors explicitly. Ensure index tensors match device (MPS quirk).
- Subset manifest + fingerprint: Commit a JSON manifest of selected files/angles/seeds and the MD5 fingerprint for the exact subset used.
- Cross‑component isolation: Do not modify multiple components in one results commit unless all are tested. Otherwise split into separate results commits.
- Infra/config changes: If required to run (e.g., increase `n_positions`), document rationale and acceptance in the results commit and verify with targeted checks.
- History correction: If a wrong‑scope commit was made and not pushed, reset immediately. If pushed, clean via interactive rebase or a single corrective revert.

## Real‑Data Testing Requirement

All experiments and tests MUST use real data. Synthetic data is not allowed for validation.

- Use the real datasets defined in commit c96860bff90f442996bf5fa87fe5f4e41774723c (short: c96860b):
  - USM training root (Original, normalized):
    - `/Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original/white_noise_original_data_no_edge_sync_vad_normalized`
  - Test root (Box, normalized):
    - `/Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original/white_noise_box_data_no_edge_sync_vad_normalized`
- You may subsample from the real datasets for speed (e.g., select a small set of angles/clips), but MUST document:
  - Exact selection procedure (angles, clip ids, counts) and random seed(s) if applicable
  - Data fingerprint checksum of all files used (see Data lineage)
  - STFT/grid parameters used to produce `Y`
- Never replace real datasets with synthetic stand‑ins for “quick tests”. If environment constraints prevent accessing the real datasets, the run should fail fast and document the missing prerequisite.
- Results and acceptance criteria must be computed on real data (or a documented real subset), not on synthetic inputs.

## Examples: Good vs Bad Commits

### ❌ Bad Commit (no executed results)
```
Fix: Update inference.py
```

### ❌ Bad Results Commit
```
Cross-experiment analysis and learning:
- This experiment worked well
- The model converged faster than expected
- Results look good
```

### ✅ Good Results Commit
```
Cross-experiment analysis and learning:
- Pattern recognition: 3/4 experiments (commits 5a7f2c1, 9b8d3e4, 2c6f1a8) with lr>1e-3 converged in <15 epochs vs 25+ for lr<1e-4 BECAUSE higher learning rates allow faster escape from suboptimal local minima in the BYOL objective landscape
- Success factors: Angle-based pairing consistently shows 40-60% better embedding separation than frame-based (commits 7b8d9e2, 4a3c5f1) BECAUSE spatial information is preserved across different recordings whereas temporal segments emphasize acoustic transients
- Failure modes: All experiments with RMS normalization (commits 1d4e7f2, 8c3a6b9) failed to distinguish adjacent angles DUE TO removal of amplitude information that encodes distance-dependent attenuation
- Method effectiveness: FRF features provide 2x better acoustic localization than raw waveforms BASED ON the principle that frequency response captures angle-dependent acoustic transfer functions
- Parameter sensitivity: Batch size 16-32 optimal; 64+ causes training instability ACCORDING TO the variance-bias tradeoff in contrastive learning

Extracted principles for future experiments:
- Design principles: THEREFORE start with lr=1e-3, use angle-based pairing for spatial tasks, avoid RMS normalization for localization
- Hypothesis formation: GIVEN the 40-60% separation improvement pattern, predict new angle-based methods will show similar gains
- Resource allocation: BECAUSE data quality (angle labels) matters more than model complexity, invest in better angle annotations before architecture search
- Risk mitigation: Always ablate RMS normalization; monitor train/val loss ratio to detect overfitting early
- Success amplification: FRF+angle pairing is the winning combination - prioritize this approach in production systems
```

## Project Structure Notes
- Main branch: `feature/frame-based-byol` - Uses different frames from same audio file
- Worktree branch: `feature/angle-based-byol` - Uses different audio files from same angle
- Located in: `worktrees/angle-based-byol/`

## Data Organization
```
root/
├─ angle_00/
│    ├─ clip_000.npy
│    ├─ clip_001.npy
│    └─ ...
├─ angle_01/
│    └─ ...
```

## Last Updated
2025-10-06 (Results‑only commits; removed planning‑only commit allowance)
