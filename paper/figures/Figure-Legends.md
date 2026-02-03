# Figure legends (Nature Communications style)

This document provides concise manuscript-ready legends for the six final figure assets in `paper/figures/`:
`fig01_paradigm-shift.jpg`, `fig02_svd-physical-dictionary.jpg`, `fig03_unrolled-attention-omp.jpg`, `fig04_noise-robustness-ablation.jpg`, `fig05_routing-mechanism-analysis.png`, and `fig06_cross-material-universality.jpg`.

## Fig. 1 (asset: `fig01_paradigm-shift.jpg`)

![](fig01_paradigm-shift.jpg)

**Fig. 1 | From chaotic acoustic scattering to sparse physical order in complex-media sensing.**
a, Photograph of the experimental setup (loudspeaker excitation, acrylic sensor plate and laser Doppler vibrometer (LDV)); inset shows a representative single-point vibration waveform exhibiting complex, seemingly chaotic fluctuations.
b, Conceptual schematic illustrating that different incidence directions excite distinct combinations of a small number of structural modes, whose spectral superposition yields direction-specific single-point “spectral fingerprints”.

## Fig. 2 (asset: `fig02_svd-physical-dictionary.jpg`)

![](fig02_svd-physical-dictionary.jpg)

**Fig. 2 | Physical encoding via spectral–spatial modes and construction of a structured dictionary.**
a, Singular-value spectrum showing rapid decay, indicating that the measured structural response is dominated by a small set of modes (sparsity/low-rank structure).
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
a, Validation accuracy under additive white noise (SNR = 10, 5 and 0 dB) comparing the full physics-aware model, a no-transformer variant, and a fixed heuristic baseline; points denote independent trials and horizontal bars indicate means (two-sided t-test, ***P < 0.001).
b, Ablation of core components comparing the full model with no-transformer, dense routing, and fixed heuristic baselines.

## Fig. 5 (asset: `fig05_routing-mechanism-analysis.png`)

![](fig05_routing-mechanism-analysis.png)

**Fig. 5 | Deciphering model behaviour across scales: attention structure, micro-mechanism and macro-robustness.**
a, Global self-attention map exhibiting a physics-consistent near-diagonal correlation structure across the physical manifold.
b, Micro-level case study (\(\theta_{\mathrm{true}}=60^\circ\)) comparing analytical OMP and physics-aware selection against ground truth, and the resulting angular estimate.
c, Selection-probability statistics across all angles showing off-diagonal errors for traditional OMP and a sharp diagonal alignment for physics-aware AI, indicating globally consistent physical selection.

## Fig. 6 (asset: `fig06_cross-material-universality.jpg`)

![](fig06_cross-material-universality.jpg)

**Fig. 6 | Universal physical encoding across diverse materials and robust cross-material performance.**
a, Targets spanning a broad spectrum of material and geometric complexity (acrylic plate, paper cup, wooden board, cardboard box and a laptop shell).
b, Representative dictionary/response heatmaps for each material, highlighting shared dispersion-signature structure despite differing physical properties.
c, DOA estimation error (RMSE) across materials comparing analytical OMP and physics-aware AI, showing degradation of OMP under increasing complexity and stable low error for the physics-aware model.
