# Fig. 6 Factor Importance Report

Date: 2026-03-27  
Audience: internal decision note  
Posture: strict evidence only

## Reader Guide

This note is written for readers both inside and outside the immediate field.
The technical question is:

**Why are some everyday objects better than others at preserving where a sound came from?**

The five key terms in this report mean:

- **Low rank**: many direction measurements can be summarized by a small number
  of recurring vibration patterns
- **Geometry separability**: different directions leave clearly different
  fingerprints instead of overlapping with each other
- **Cross-material subspace overlap**: how similar the dominant pattern family
  of one object is to that of another object
- **Conditioning**: how fragile decoding becomes when the fingerprints are too
  similar or too compressed in an unfavorable way
- **Coherence**: a broad average similarity measure that is useful for
  inspection, but too coarse to explain task quality on its own

In plain language, the current story is:

- low rank tells us that the object compresses the sound field into a manageable
  set of patterns
- separability tells us whether that compression still keeps different
  directions distinguishable

## Executive Answer

The current evidence does **not** support a single universal-equation variable.
It supports a factor hierarchy.

The working answer is:

1. **Geometry separability** is the current leading candidate driver of how
   well an object preserves sound direction.
2. **Cross-material subspace overlap** and **dictionary conditioning** are
   secondary structural factors.
3. **Low-rankness itself** is a shared physical substrate, but weak as a
   standalone predictor of task quality.
4. **Raw coherence** should not be used as the main summary variable for Fig. 6.

The shortest interdisciplinary summary is:

> A good object-level directional encoder is not just compact. It must also keep
> different directions visibly distinct after that compression.

| Factor family | Current tier | Representative metrics | Current read |
| --- | --- | --- | --- |
| Geometry separability | Tier 1 | `full_far_mean`, `band_local_far_gap`, `rank95_far_collision_gt_0_8` | Best current explanation for whether different directions stay distinguishable |
| Cross-material subspace overlap | Tier 2 | `mean_top3_subspace_overlap_to_others` | Stable candidate, but not yet a law |
| Dictionary conditioning | Tier 2 | `raw_condition_number` | Secondary constraint on how usable the fingerprints remain |
| Low-rankness itself | Tier 3 | `effective_rank_centered_mag` | Necessary background, weak standalone predictor |
| Raw coherence | Tier 4 | `raw_mean_coherence` | Too coarse to serve as the main Fig. 6 explanation |

![](assets/factor_tier_matrix.png)

This tiered matrix is an evidence view, not a causal percentage decomposition.
The current dataset is too small for a defensible variance-share claim across
families.

## Evidence Basis and Ranking Rule

This note uses only executed artifacts from:

- `results/fig06_universal_equation_factor_audit_20260327_173647`
- `results/fig06_cross_material_geometry_20260327_171243`
- `figures/review_artifacts/fig06`

The factor audit uses the commit-aligned centered-magnitude representation:

`H_c = |H| - row_mean(|H|)`

In plain language, this means:

- start from the magnitude-only direction-frequency fingerprint
- remove the average level at each frequency
- keep only the direction-dependent contrast

The ranking rule in this note is intentionally conservative:

- primary outcome: held-out `MAE`
- supporting evidence: descriptive material-level correlations and family-level
  ablation/permutation checks
- caution only: `Top-1`, because the pooled `Top-1` model does not beat the
  intercept-only baseline

![](assets/baseline_vs_model_summary.png)

This is the key gating decision for the whole report:

- `Top-1` pooled log-loss: `0.5461`
- `Top-1` intercept-only pooled log-loss: `0.5458`
- `Top-1` pooled accuracy: `0.7676`
- `Top-1` intercept-only pooled accuracy: `0.7676`
- `MAE` pooled error: `4.81 deg`
- `MAE` intercept-only pooled error: `7.00 deg`

So the current question is not "which factors explain Top-1?"  
It is "which factors reduce held-out angular error?"

## Low Rank Matters, But It Is Not Enough

The cross-material story still begins with low rank.

The current Fig. 6 low-rank panel shows that low-dimensional structure persists
across the five materials rather than collapsing into material-specific noise.
That part of the physical story remains intact.

![](assets/fig06_panel_c_low_rank_context.png)

For a non-specialist reader, low rank is best understood as:

> the object does not respond with an unlimited number of unrelated patterns;
> instead, a small set of dominant structural modes explains most of what is
> measured.

That matters because it makes single-point directional encoding physically
plausible in the first place.

But low-rankness alone is weak as a quality predictor:

- `effective_rank_centered_mag` stays in the final feature set, but its `MAE`
  sign consistency is only `0.5956`
- its descriptive material-level correlation with `MAE` is only `r = 0.0376`
- the broader `global centered low-rank` family has weak family-level support
  in the current held-out checks

The current read is therefore:

- low rank is the **shared physical substrate**
- low rank is **not** the most useful standalone explanation for why one object
  encodes direction better than another

## Tier 1: Geometry Separability Is the Leading Candidate Driver

The strongest current evidence points to geometry, not raw rank.

The candidate driver is the separability of the centered-magnitude manifold:

- how different far-apart directions remain from each other
- how large the local-vs-far gap stays
- how often distant directions collapse into the same fingerprint inside the
  informative band

This is the most important conceptual distinction in the report:

- **low rank** asks whether the object compresses the sound field
- **geometry separability** asks whether that compressed representation still
  preserves directional identity

An object can therefore be low rank and still be a poor directional encoder if
its compressed fingerprints overlap too much.

The best current MAE-side representatives are:

- `full_far_mean`: `MAE` sign consistency `0.8644`
- `band_local_far_gap`: `MAE` sign consistency `0.6880`
- `rank95_far_collision_gt_0_8`: `MAE` sign consistency `0.6732`

The strongest descriptive material-level signals also live in the same
geometry family:

- `band_far_collision_gt_0_8`: `r(MAE) = -0.6324`
- `band_max_nonlocal_mean`: `r(MAE) = -0.6087`
- `band_local_far_gap`: `r(MAE) = -0.4297`

![](assets/fig06_panel_e_frequency_structure_context.png)

![](assets/mae_factor_importance.png)

This is the main reason geometry separability is the current Tier 1 family:

- it is where the strongest descriptive signals appear
- it contributes several of the top-ranked `MAE` features
- it matches the intuitive physical picture that downstream failure occurs when
  different directions start looking too similar

The family-level support is still mixed, so this is a **leading candidate
driver**, not a settled law.

## Tier 2: Subspace Overlap and Conditioning Are Secondary Structural Factors

### Cross-material subspace overlap

`mean_top3_subspace_overlap_to_others` is one of the most stable single
features in the current factor audit:

- `MAE` selection frequency: `1.000`
- `MAE` sign consistency: `0.7772`
- `Top-1` descriptive correlation: `r = -0.5599`

![](assets/top3_subspace_overlap_heatmap.png)

The interpretation is straightforward:

- different objects share a low-rank physical skeleton
- but they do not share the **same** dominant pattern space
- objects with more unfavorable overlap structure appear harder to use as
  directional encoders

For a broad scientific audience, this means:

> there is a shared physical principle across objects, but each object realizes
> that principle with its own geometry.

This is scientifically useful, but still better treated as a secondary factor
family until stronger held-out support accumulates.

### Dictionary conditioning

`raw_condition_number` is weaker than geometry but stronger than low-rankness
alone:

- `MAE` selection frequency: `1.000`
- `MAE` sign consistency: `0.7428`
- descriptive correlation with `MAE`: `r = -0.3005`

It also has the clearest positive family-level effect in the current MAE-side
checks. That makes conditioning a plausible Tier 2 structural constraint.

The current read is:

- conditioning still matters
- but it behaves more like a secondary limit on usable geometry than the main
  explanatory variable

## Tier 3 and Tier 4: What Should Not Lead the Story

### Low-rankness itself

Low-rankness should stay in the story, but in the correct role:

- it explains why single-point direction encoding is physically plausible
- it does not, by itself, explain which object becomes a better encoder

So low-rankness belongs in the **substrate** layer of the narrative, not the
top explanatory layer.

### Raw coherence

Raw coherence is the clearest variable to deprioritize:

- it was dropped by collinearity filtering
- it does not remain competitive in the final feature set
- it has already produced a misleading screening story historically

So the current report treats raw coherence as **Tier 4 / unsupported** for the
main Fig. 6 summary.

For non-specialists, the key point is:

> "cleaner overall signal" is not the same as "better directional code."

## Manuscript-Safe Reading Today

The following statements are safe or nearly safe under the current evidence:

- **Safe**: bounded objects share a low-rank direction-frequency encoding
  structure
- **Safe**: low rank alone does not explain cross-material outcome differences
- **Near-safe**: geometry separability appears to govern downstream quality more
  directly than rank alone
- **Exploratory**: cross-material subspace overlap and conditioning are likely
  secondary constraints on usable encoding capacity
- **Not safe yet**: a single universal equation, a coherence-led explanation,
  or a Top-1-based importance hierarchy

One manuscript-adjacent wording candidate would be:

> Cross-material transfer preserves a low-rank physical encoding substrate, but
> downstream sensing quality is more directly aligned with the separability of
> the resulting manifold than with low-rankness alone.

An even more accessible version for interdisciplinary discussion would be:

> Everyday objects can compress directional sound into a small set of structural
> patterns, but good directional sensing requires those compressed patterns to
> stay distinct rather than collapse onto one another.

The first sentence is closer to manuscript language. The second sentence is
closer to seminar language for mixed audiences.

## Appendix: Why Top-1 Is Caution Only

The `Top-1` factor ranking is included for inspection, but it should not drive
the current scientific summary.

![](assets/fig06_panel_d_screening_context.png)

![](assets/top1_factor_importance_caution.png)

The important point is not that `Top-1` is useless.
It is that, under the current five-material setting, `Top-1` is too close to
the intercept-only baseline to serve as the primary ranking outcome for factor
importance.
