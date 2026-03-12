# Figure legends (Nature Communications style)

This document provides concise manuscript-ready legends for the nine final main-paper figure assets in `paper/figures/`:
`fig01_paradigm-shift.jpg`, `fig02_svd-physical-dictionary.jpg`, `fig03_unrolled-attention-omp.jpg`, `fig04_noise-robustness-ablation.jpg`, `fig05_structure-macro-selection.png`, `fig06_angle-specific-mechanism.png`, `fig07_bandwise-routing-analysis-part1.png`, `fig08_bandwise-routing-analysis-part2.png`, and `fig09_cross-material-universality.jpg`.

## Fig. 1 (asset: `fig01_paradigm-shift.jpg`)

![](fig01_paradigm-shift.jpg)

**Fig. 1 | From chaotic acoustic scattering to sparse physical order in complex-media sensing.**
a, Photograph of the experimental setup (loudspeaker excitation, acrylic sensor plate and laser Doppler vibrometer (LDV)); inset shows a representative single-point vibration waveform exhibiting complex, seemingly chaotic fluctuations.
b, Conceptual schematic illustrating that different incidence directions excite distinct combinations of a small number of structural modes, whose spectral superposition yields direction-specific single-point “spectral fingerprints”.

## Fig. 2 (asset: `fig02_svd-physical-dictionary.jpg`)

![](fig02_svd-physical-dictionary.jpg)

**Fig. 2 | Physical encoding via spectral–spatial modes and construction of a structured dictionary.**
a, Singular-value spectrum showing rapid decay, indicating that the measured structural response is dominated by a small set of modes.
b, Modal decomposition into frequency-selective spectra \(u_r(f)\) and direction-selective polar patterns \(v_r(\theta)\), forming virtual directional sensing channels.
c, Structured physical dictionary \(D\) assembled by combining modal spectra and directivity patterns to produce distinct mode–angle atoms with characteristic dispersion signatures.

## Fig. 3 (asset: `fig03_unrolled-attention-omp.jpg`)

![](fig03_unrolled-attention-omp.jpg)

**Fig. 3 | Physics-guided deep unrolled network with attention-based gating for sparse DOA inference.**
At stage \(t\), the residual \(r_t\) is correlated with the static physics dictionary \(D\) to form a physical match \(g=D^\top r_t\); a transformer encoder outputs attention weights that gate sparse updates \(\Delta x\), followed by residual update \(r_{t+1}=r_t-D\Delta x\).
Unrolling across stages accumulates a sparse vector \(x_T\), which is mapped to the final DOA estimate \(\hat{\theta}\).

## Fig. 4 (asset: `fig04_noise-robustness-ablation.jpg`)

![](fig04_noise-robustness-ablation.jpg)

**Fig. 4 | Performance under additive noise and architectural ablations.**
a, Validation accuracy under additive white noise (SNR = 10, 5 and 0 dB) comparing the full physics-aware model, a no-transformer variant, and a fixed heuristic baseline; points denote independent trials and horizontal bars indicate means.
b, Ablation of core components comparing the full model with no-transformer, dense routing, and fixed heuristic baselines.

## Fig. 5 (asset: `fig05_structure-macro-selection.png`)

![](fig05_structure-macro-selection.png)

**Fig. 5 | Global structure alignment and macro selection robustness.**
a, The physical `H`-matrix manifold and the learned `QK` routing structure both exhibit strong diagonal locality, indicating that the learned router aligns with the geometry of the physical angle manifold.
b, All-angle selection-probability heatmaps comparing traditional OMP with physics-aware AI; OMP spreads mass over persistent off-diagonal experts, whereas the learned router remains sharply diagonal across the full angle range.

## Fig. 6 (asset: `fig06_angle-specific-mechanism.png`)

![](fig06_angle-specific-mechanism.png)

**Fig. 6 | Angle-specific routing distributions sharpen the correct atom.**
a, Probability distributions at `55°` comparing the transformer baseline with the no-transformer ablation; the baseline concentrates more mass on the correct atom and suppresses secondary off-axis peaks.
b, The same comparison at `100°`, again showing a sharper correct-atom peak when learned routing is present.

## Fig. 7 (asset: `fig07_bandwise-routing-analysis-part1.png`)

![](fig07_bandwise-routing-analysis-part1.png)

**Fig. 7 | Band-wise routing diagnostics I: full-band and low-band views.**
a, Full-band (`300–3000 Hz`) routing diagnostic with the original smoothed rendering.
b, Full-band (`300–3000 Hz`) routing diagnostic with smoothing removed in the corrected rerun.
c, `300–500 Hz` band diagnostic, isolating the low-frequency contribution to the same case-study mechanism.

## Fig. 8 (asset: `fig08_bandwise-routing-analysis-part2.png`)

![](fig08_bandwise-routing-analysis-part2.png)

**Fig. 8 | Band-wise routing diagnostics II: mid- and high-band views.**
a, `500–1000 Hz` routing diagnostic.
b, `1000–2000 Hz` routing diagnostic.
c, `2000–3000 Hz` routing diagnostic.

## Fig. 9 (asset: `fig09_cross-material-universality.jpg`)

![](fig09_cross-material-universality.jpg)

**Fig. 9 | Universal physical encoding across diverse materials and robust cross-material performance.**
a, Targets spanning a broad spectrum of material and geometric complexity (acrylic plate, paper cup, wooden board, cardboard box, and a laptop shell).
b, Representative dictionary/response heatmaps for each material, highlighting shared dispersion-signature structure despite differing physical properties.
c, DOA estimation error (RMSE) across materials comparing analytical OMP and physics-aware AI, showing degradation of OMP under increasing complexity and stable low error for the physics-aware model.
