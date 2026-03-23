# Claim→citation matrix (figure-ordered spine)

This document turns the **nine final figures** in `paper/figures/` into a figure-ordered manuscript spine (Fig. 1 → Fig. 9). For each figure panel, it lists manuscript-ready claims and maps them to:

- **Internal support**: relevant working notes in `paper/references/derived/` and `paper/references/exp/`.
- **External citations**: BibTeX keys in `paper/references/references.bib`.
- **Evidence needed**: which result/analysis should substantiate the claim (can be `TBD` if not yet formalized).

## Legend

- `panel_anchor`: the figure panel (e.g., `Fig1a`, `Fig2c`).
- `paper_location`: where this claim would live in a figure-ordered manuscript flow.
- `internal_support`: internal note files that justify the logic or define the terms.
- `citation_keys`: BibTeX keys that should exist in `paper/references/references.bib`.
- `hedging/limits`: suggested conservative language to avoid overclaiming.

---

## Fig. 1 (asset: `paper/figures/fig01_paradigm-shift.jpg`)

**Takeaway:** A complex structure converts incident direction into a reproducible, direction-dependent single-point spectral signature (physical encoding), motivating single-sensor DOA inference.

**Bridge to Fig. 2:** The “chaos” is structured: a small number of dominant modes/channels explain most variance, enabling a structured dictionary view.

| claim_id | panel_anchor | claim_text | paper_location | internal_support | needs_evidence | citation_keys | hedging/limits |
|---|---|---|---|---|---|---|---|
| C1.1 | Fig1a | The experimental system measures single-point structural vibration induced by an incident acoustic field using a laser Doppler vibrometer (LDV) on a target structure. | Spine §1 (opening) | `paper/figures/Figure-Legends.md`, `paper/references/derived/svd03.md` | Fig. 1a photo + setup description |  | Descriptive; no external citation required. |
| C1.2 | Fig1a | The measured time-domain vibration can appear irregular due to multimodal structural dynamics and complex scattering, motivating frequency-domain analysis. | Spine §1 | `paper/references/derived/svd03.md` | Example waveform + spectrogram (if shown elsewhere) | `timoshenko1959plates` | Keep qualitative unless quantitative stationarity/chaos metrics are reported. |
| C1.3 | Fig1b | Under small-amplitude linear dynamics, the structure’s response to acoustic forcing is well approximated by linear superposition of a limited number of modes/channels. | Spine §1 → §2 transition | `paper/references/derived/svd.md`, `paper/references/derived/svd02.md`, `paper/references/derived/svd03.md` | Explicit “linearity” validation protocol (TBD) | `timoshenko1959plates` | State assumptions (linearity, time-invariance during measurement). |
| C1.4 | Fig1b | Different incidence directions change modal coupling weights, yielding direction-dependent spectral fingerprints measurable at a single point. | Spine §1 (core claim) | `paper/references/derived/svd03.md`, `paper/references/exp/exp03.md` | Per-angle spectral separability analysis (TBD) | `dipassio2023doa_single_sensor`, `rutowski2024reverb_single_sensor` | “Direction-dependent” rather than “unique” unless bijectivity is proven. |
| C1.5 | Fig1b | The structure can be viewed as a physical encoder that maps DOA to a transfer function \(H(f,\theta)\) observed at the sensor location. | Spine §1 (model statement) | `paper/references/exp/exp04.md`, `paper/references/exp/exp06.md` | Definition of \(H\) and measurement protocol (TBD) |  | Avoid claiming full system identification; describe \(H\) as empirical frequency response for discrete angles. |
| C1.6 | Fig1b | This physical encoding enables single-sensor DOA inference without a conventional microphone array by exploiting structure-induced dispersion signatures. | Spine §1 (positioning) | `paper/references/exp/exp08.md` | Comparison statement against array-centric DOA (TBD) | `krim1996array`, `dipassio2023doa_single_sensor` | Phrase as “without a microphone array at the sensing front-end” (we still use a structure). |

---

## Fig. 2 (asset: `paper/figures/fig02_svd-physical-dictionary.jpg`)

**Takeaway:** The physical encoding is **low-dimensional** and **structured**: a few dominant spectral–spatial modes are sufficient, and their factorization yields a physics-structured dictionary \(D\).

**Bridge to Fig. 3:** Once cast as a structured sparse inverse problem over \(D\), inference can be performed with an unrolled, physics-guided sparse selection network.

| claim_id | panel_anchor | claim_text | paper_location | internal_support | needs_evidence | citation_keys | hedging/limits |
|---|---|---|---|---|---|---|---|
| C2.1 | Fig2a | The transfer operator/matrix exhibits rapidly decaying singular values, indicating that a small number of dominant modes/channels capture most response energy (effective low rank). | Spine §2 (mechanism) | `paper/references/derived/svd.md` | SVD plot generation protocol + threshold definition (TBD) | `davy2015eigenchannels` | Specify rank criterion (energy fraction or noise-floor threshold). |
| C2.2 | Fig2a | Low effective rank provides a physics-motivated sparsity/low-dimensionality prior for subsequent inverse inference from single-point spectra. | Spine §2 | `paper/references/derived/svd.md` | Link to reconstruction conditioning (TBD) | `davy2015eigenchannels` | Avoid “breaks physical limits”; use “reduces effective degrees of freedom”. |
| C2.3 | Fig2b | Each dominant response component can be interpreted as a frequency-selective spectrum coupled with a direction-selective directivity pattern, functioning as a virtual directional sensing channel. | Spine §2 | `paper/references/derived/svd02.md`, `paper/references/derived/svd03.md` | Define decomposition used (SVD vs modal model) (TBD) | `timoshenko1959plates`, `davy2015eigenchannels` | Clarify whether this is an interpretive model or an explicitly estimated factorization. |
| C2.4 | Fig2c | A structured physical dictionary \(D\) can be constructed by combining spectral and directional components, producing atoms with distinct dispersion signatures for mode–angle combinations. | Spine §2 (dictionary definition) | `paper/references/exp/exp01.md`, `paper/references/exp/exp06.md` | Dictionary construction spec (inputs/normalization) (TBD) |  | Keep “constructed” vs “learned” terminology consistent with the actual pipeline. |
| C2.5 | Fig2c | Given \(D\), single-point observations can be represented with a sparse set of atoms, motivating greedy/sparse selection methods as baselines and building blocks. | Spine §2 → §3 transition | `paper/references/exp/exp01.md` | Sparsity statistics (selected atom counts, stability) (TBD) | `tropp2007omp` | Do not claim exact sparsity without reporting distributions. |
| C2.6 | Fig2c | This framing turns DOA estimation into a structured sparse inverse problem: infer a sparse code whose atom indices map to discrete angles. | Spine §2 (problem statement) | `paper/references/exp/exp04.md`, `paper/references/exp/exp06.md` | Formal definition of index→angle mapping (TBD) | `tropp2007omp` | Make discrete-angle assumption explicit. |

---

## Fig. 3 (asset: `paper/figures/fig03_unrolled-attention-omp.jpg`)

**Takeaway:** A physics-guided deep unrolling architecture performs iterative sparse selection with residual updates, using attention-based gating to improve routing decisions over the physical dictionary.

**Bridge to Fig. 4:** The learned routing improves robustness under noise and reveals which architectural components matter.

| claim_id | panel_anchor | claim_text | paper_location | internal_support | needs_evidence | citation_keys | hedging/limits |
|---|---|---|---|---|---|---|---|
| C3.1 | Fig3 (stage view) | Inference is implemented as a deep unrolled optimization: repeated stages update a sparse representation and a residual consistent with the fixed dictionary \(D\). | Spine §3 (method) | `paper/references/exp/exp06.md` | Precise stage equations and loss terms (TBD) | `monga2021unrolling` | State which parts are fixed physics (D) vs learned (routing/updates). |
| C3.2 | Fig3 (physics match) | Each stage computes a physics-based matching score \(g = D^\top r_t\), analogous to correlation-based atom selection in sparse pursuit methods. | Spine §3 | `paper/references/exp/exp01.md` | Explicit definition of \(r_t\) and normalization (TBD) | `tropp2007omp` | Clarify whether complex-valued handling uses magnitude/real/whitening. |
| C3.3 | Fig3 (attention gating) | A transformer module produces attention weights that gate the sparse update, enabling data-driven, context-dependent selection beyond fixed heuristics. | Spine §3 | `paper/references/exp/exp05.md`, `paper/references/exp/exp06.md` | Router logits/probability logging (TBD) | `vaswani2017attention` | Avoid claiming “causality”; describe as learned routing conditioned on the observation embedding. |
| C3.4 | Fig3 (discrete selection) | Discrete/sparse selection can be approximated during training using differentiable relaxations for categorical choices, enabling end-to-end learning of selection behaviour. | Spine §3 | `paper/references/exp/exp05.md` | Confirm actual implementation (hard/soft, top-K) (TBD) | `jang2016gumbel` | Only include if the final model truly uses Gumbel/relaxation. |
| C3.5 | Fig3 (mapping) | The accumulated sparse vector \(x_T\) is mapped from selected atom indices to a DOA estimate \(\hat{\theta}\), yielding a physically grounded decision rule. | Spine §3 (output definition) | `paper/references/exp/exp04.md` | Mapping definition + discrete angle set (TBD) |  | Avoid “ground truth” language; this is a deterministic mapping given the dictionary design. |
| C3.6 | Fig3 (consistency) | Residual updates enforce reconstruction consistency (\(r_{t+1}=r_t-D\Delta x\)), which regularizes selection and supports interpretability. | Spine §3 | `paper/references/exp/exp06.md` | Reconstruction error curves across stages (TBD) | `monga2021unrolling` | Avoid overclaiming interpretability; tie to explicit diagnostics (Fig. 5). |

---

## Fig. 4 (asset: `paper/figures/fig04_noise-robustness-ablation.jpg`)

**Takeaway:** The physics-aware routing improves accuracy under additive noise, and ablations isolate which components drive the gains.

**Bridge to Fig. 5:** Beyond accuracy, the learned routing exhibits physically consistent structure and reduces systematic selection errors.

| claim_id | panel_anchor | claim_text | paper_location | internal_support | needs_evidence | citation_keys | hedging/limits |
|---|---|---|---|---|---|---|---|
| C4.1 | Fig4b | The physics-aware model achieves higher validation accuracy than baselines across multiple SNR levels, with the largest gains under lower SNR. | Spine §4 (results) |  | Underlying numeric table for Fig. 4b (TBD) |  | Keep strictly tied to reported SNR levels and splits. |
| C4.2 | Fig4a | The no-transformer ablation degrades clean-SNR performance, indicating that learned routing contributes materially even before additive noise is introduced. | Spine §4 |  | Ablation configuration + training protocol (TBD) |  | Ensure fair comparison (same training budget, seeds, and preprocessing). |
| C4.3 | Fig4a | A fixed heuristic sparse selection baseline underperforms learned routing, suggesting that data-driven gating better handles model mismatch than a hand-designed selector. | Spine §4 | `paper/references/exp/exp01.md` | Define the heuristic score and selection budget (TBD) | `tropp2007omp` | Specify the heuristic (e.g., magnitude correlation) precisely. |
| C4.4 | Fig4a | Dense routing underperforms sparse physics-aware routing, supporting the role of structured sparsity aligned with the physical dictionary. | Spine §4 |  | Define “dense routing” ablation (TBD) |  | Avoid implying a universal sparse>dense statement; limit to this setting. |
| C4.5 | Fig4a | The clean-SNR ablation ordering is consistent across multiple independent runs, with trial-level scatter exposing the stability of the ranking. | Spine §4 (reporting) |  | Provide trial-level results + test specification (TBD) |  | Use statistical language carefully; include effect sizes and n. |

---

## Fig. 5 (asset: `paper/figures/fig05_structure-macro-selection.png`)

**Takeaway:** The learned router aligns with the physical angle manifold at both the structure level and the all-angle selection-statistics level.

**Bridge to Fig. 6:** Once the global structure is established, angle-specific distributions reveal the same mechanism at representative directions.

| claim_id | panel_anchor | claim_text | paper_location | internal_support | needs_evidence | citation_keys | hedging/limits |
|---|---|---|---|---|---|---|---|
| C5.1 | Fig5a | The physical `H` manifold exhibits diagonal locality, showing that nearby angles remain more correlated than distant angles. | Spine §5 (global structure) | `paper/references/exp/exp05.md` | Definition of the `H` correlation map and normalization | `timoshenko1959plates` | Treat as an empirical manifold diagnostic, not a proof of a specific plate model. |
| C5.2 | Fig5a | The learned `QK` routing structure is also strongly diagonal, indicating that the model has internalized the locality of the physical angle manifold. | Spine §5 | `paper/references/exp/exp05.md` | Attention/routing extraction protocol | `vaswani2017attention` | Phrase as “indicating” or “consistent with” rather than causal proof. |
| C5.3 | Fig5b | Across all angles, analytical OMP spreads selection mass over a diffuse set of off-diagonal experts. | Spine §5 (macro robustness) | `paper/references/exp/exp05.md` | Selection-probability calculation from `g_energy_expert` and labels | `tropp2007omp` | Keep the statement descriptive unless an off-diagonal mass summary is reported. |
| C5.4 | Fig5b | The physics-aware model remains sharply diagonal across the full angle range, showing globally stable expert selection. | Spine §5 | `paper/references/exp/exp05.md` | Selection-probability calculation from `scores_expert` and labels |  | Avoid claiming perfect determinism unless quantified. |
| C5.5 | Fig5a-b | The agreement between manifold structure and all-angle selection statistics supports the interpretation that routing exploits physical manifold smoothness rather than memorizing isolated cases. | Spine §5 (interpretation) | `paper/references/exp/exp05.md` | Joint bridge between structure and macro panels |  | Use “supports the interpretation” rather than stronger causal language. |

---

## Fig. 6 (asset: `paper/figures/fig06_angle-specific-mechanism.png`)

**Takeaway:** At representative directions, learned routing sharpens the correct-atom peak and suppresses structured off-axis alternatives.

**Bridge to Figs. 7–8:** The same mechanism can be decomposed into full-band and narrower band-specific diagnostics.

| claim_id | panel_anchor | claim_text | paper_location | internal_support | needs_evidence | citation_keys | hedging/limits |
|---|---|---|---|---|---|---|---|
| C6.1 | Fig6a | At `55°`, the transformer baseline assigns more probability to the correct atom than the no-transformer ablation. | Spine §6 (angle-specific diagnostics) | `paper/references/exp/exp04.md` | Correct-probability extraction from angle-55 distributions |  | Report exact percentages only if frozen in the final figure/caption. |
| C6.2 | Fig6a | The no-transformer variant redistributes mass onto secondary off-axis atoms at `55°`, indicating weaker routing concentration. | Spine §6 | `paper/references/exp/exp04.md` | Distribution tails or ranked off-axis peaks |  | Phrase as a comparative pattern rather than a mechanistic proof. |
| C6.3 | Fig6b | The same baseline-versus-ablation separation persists at `100°`, showing that the effect is not limited to a single representative direction. | Spine §6 | `paper/references/exp/exp04.md` | Correct-probability extraction from angle-100 distributions |  | Avoid overgeneralizing beyond the reported representative angles. |
| C6.4 | Fig6a-b | Representative-angle distributions make the routing gain quantitative: learned routing increases the dominant correct-atom peak while preserving a sparse probability profile. | Spine §6 (interpretation) | `paper/references/exp/exp04.md` | Joint summary statistic across selected angles |  | Keep tied to the selected angles and current ablation family. |

---

## Fig. 7 (asset: `paper/figures/fig07_bandwise-routing-analysis-part1.png`)

**Takeaway:** The case-study routing mechanism persists in full-band and low-band views, and the corrected no-smoothing rerun preserves the same target-angle localization.

**Bridge to Fig. 8:** The remaining bands determine how far this localization persists as the frequency window moves upward.

| claim_id | panel_anchor | claim_text | paper_location | internal_support | needs_evidence | citation_keys | hedging/limits |
|---|---|---|---|---|---|---|---|
| C7.1 | Fig7a | The original smoothed full-band diagnostic concentrates the case-study signal near the true angle while keeping the two scoring families visually separable. | Spine §7 (band decomposition) | `results/fig5_b3_line_traceback_20260312/README.md` | Smoothed full-band PDF lineage and rendering parameters |  | Treat this as a diagnostic view, not a new quantitative benchmark. |
| C7.2 | Fig7b | Removing smoothing preserves the same target-angle localization in the corrected rerun, showing that the effect is not an artefact of smoothing alone. | Spine §7 | `results/fig5_b3_line_traceback_20260312/README.md` | No-smoothing full-band PDF lineage and corrected peak-marker semantics |  | Phrase as “preserves” rather than “proves robustness”. |
| C7.3 | Fig7c | The `300–500 Hz` window retains clear angle-dependent structure, indicating that low frequencies carry a substantial portion of the routing evidence. | Spine §7 | `results/fig5_b3_line_traceback_20260312/README.md` | Low-band PDF plus case-study angle reference |  | Keep the statement qualitative unless a bandwise concentration score is computed. |

---

## Fig. 8 (asset: `paper/figures/fig08_bandwise-routing-analysis-part2.png`)

**Takeaway:** Mid- and high-band diagnostics remain structured but become weaker and more variable than the low-band views.

**Bridge to Fig. 9:** After resolving the mechanism across scales and bands, the paper turns to whether the same physical logic extends across materials.

| claim_id | panel_anchor | claim_text | paper_location | internal_support | needs_evidence | citation_keys | hedging/limits |
|---|---|---|---|---|---|---|---|
| C8.1 | Fig8a | The `500–1000 Hz` band still retains angle-dependent contrast aligned with the case-study target. | Spine §8 (band continuation) | `results/fig5_b3_line_traceback_20260312/README.md` | `500–1000 Hz` PDF and case-study angle reference |  | Keep the statement descriptive rather than ranking the band globally. |
| C8.2 | Fig8b | The `1000–2000 Hz` band remains structured but shows visibly weaker and noisier contrast than the lower bands. | Spine §8 | `results/fig5_b3_line_traceback_20260312/README.md` | `1000–2000 Hz` PDF |  | Use cautious comparative language unless a quantitative band ordering is added. |
| C8.3 | Fig8c | The `2000–3000 Hz` band retains residual angle dependence, indicating that the mechanism is distributed across frequency rather than confined to a single narrow resonance. | Spine §8 | `results/fig5_b3_line_traceback_20260312/README.md` | `2000–3000 Hz` PDF |  | “Residual angle dependence” is safer than a stronger reliability claim. |

---

## Fig. 9 (asset: `paper/figures/fig09_cross-material-universality.jpg`)

**Takeaway:** The physical encoding mechanism persists across diverse materials, and the physics-aware inference maintains robust DOA accuracy where analytical baselines degrade with complexity.

| claim_id | panel_anchor | claim_text | paper_location | internal_support | needs_evidence | citation_keys | hedging/limits |
|---|---|---|---|---|---|---|---|
| C9.1 | Fig9a | The evaluation spans targets with increasing physical complexity (geometry, damping, anisotropy, and internal structure). | Spine §9 (generalization) |  | Provide material/geometry description table (TBD) |  | Keep descriptive; avoid implying a formal complexity metric unless defined. |
| C9.2 | Fig9b | Learned dictionaries/response heatmaps show shared dispersion-signature structure across materials, suggesting a common physical encoding mechanism. | Spine §9 | `paper/references/derived/svd02.md` | Define how heatmaps are computed and normalized (TBD) | `timoshenko1959plates` | “Suggesting” rather than “demonstrating universality” unless formal invariants are shown. |
| C9.3 | Fig9c | Physics-aware AI maintains low DOA estimation error across materials, whereas analytical OMP degrades substantially as structural complexity increases. | Spine §9 (core result) |  | Underlying numeric RMSE table + per-material n (TBD) | `tropp2007omp` | Keep comparison bounded to evaluated materials and settings. |
| C9.4 | Fig9c | Robust cross-material performance indicates that physics-consistent sparse routing reduces sensitivity to target-specific modal details. | Spine §9 (interpretation) | `paper/references/exp/exp09.md` | Sensitivity analysis or failure cases (TBD) |  | Use “indicates” or “is consistent with”; list known limitations explicitly. |
