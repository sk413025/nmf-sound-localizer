# Cross-Material Geometry Audit

Use this reference when the user asks:

- which factors matter most
- which quantities can be computed now from current artifacts
- whether there is a candidate "universal equation"
- whether low rank, separability, coherence, or another metric best explains performance
- for a scorecard, factor audit, or metric comparison across materials or runs

Workflow:

1. Lock the representation.
   - State the exact `H` representation before computing anything.
   - Typical choices: raw `H`, `|H|`, centered magnitude `|H| - row_mean(|H|)`, low-rank projection.
2. Separate metric families.
   - compression or low-rank metrics
   - separability or overlap metrics
   - energy or amplitude summaries
   - downstream task metrics such as `Top-1`, `MAE`, or tolerance accuracy
3. Keep evidence executed.
   - Prefer committed bundles under `results/<run_name>/`.
   - If a new audit is needed, create a new run bundle rather than leaving terminal-only numbers.
4. Report results in four buckets.
   - stable findings
   - rejected or weak candidates
   - current best candidate mechanism
   - missing evidence
5. Keep manuscript claims conservative.
   - A candidate explanatory form is not yet a law unless validation goes beyond descriptive fit on the current small set.

Preferred outputs:

- `SUMMARY.md`
- machine-readable `csv` or `json`
- reproduction script and command
- optional figures that show ranking, overlap, or separability

Preferred answer structure:

1. What is stable already
2. What does not explain the result well
3. What currently looks most promising
4. What can be computed immediately versus what needs new experiments

If the user wants manuscript-ready language after the audit, route the interpretation step to `paper-submission`.
