# Research Context Summary and Related Work (NC Strategy)

Purpose: summarize the content of `docs/research_context/comprehensive_research_narrative.md`, align it with the Nature Communications (NC) submission strategy, and list high-quality related journals and conferences.

---

## Summary of `comprehensive_research_narrative.md`

- **Core thesis**: Acoustic scattering is a high-dimensional mapping that can be decoded rather than treated as noise. Physics-learning synergy is necessary; pure physics achieves 1.7% accuracy and pure deep learning achieves 2.7%, while the integrated physics-aware routing achieves 93.5% on 37-class DOA.
- **Central innovation**: The network architecture instantiates the physical formula via deep unrolling. The inner loop uses IS-divergence multiplicative updates; the outer loop uses KL-regularized policy optimization (PPO/GRPO). QK attention selects physical atoms from a structured dictionary.
- **Key results**: High accuracy on DOA, robustness down to 0 dB, cross-material generality, and real-time continuous tracking.
- **Ablation evidence**: Rigorous ablations quantify the necessity of the Transformer, sparsity routing, and physics dictionary. Statistical significance is reported (p < 0.001) with large effect sizes.
- **Physical foundation**: Plate vibration is modeled via modal decomposition and acoustic-structural coupling; LDV provides non-intrusive measurement of the modal superposition. A physical dictionary D is constructed from directional transfer functions H_d, and speech content is modeled via IS-NMF.
- **Method evolution**: The project progresses from pure physics to pure DL to physics-informed sparse coding and finally to physics-aware neural routing, with explicit failures motivating each step.

---

## Alignment with NC Submission Strategy

- **Conceptual shift**: Reframes scattering from noise to information-rich mapping, consistent with NC’s preference for paradigm shifts.
- **Mechanistic depth**: Formal physical derivations (plate equation, modal coupling, IS/KL geometry, advantage derivation) provide mechanism-level rigor.
- **Causal evidence**: Strong ablations with statistical testing demonstrate necessity of hybrid design, supporting causal claims rather than correlation.
- **Generality and impact**: Cross-material universality and 0 dB robustness indicate learning of invariant physical laws, with broad sensing implications.
- **Interdisciplinary scope**: Combines acoustics, optics (LDV), signal processing, and ML (unrolling + attention + RL), matching NC’s interdisciplinary emphasis.

---

## High-Quality Related Journals (Selected)

**Acoustics / LDV / Sensing**
- Journal of the Acoustical Society of America (JASA)
- Journal of Sound and Vibration (JSV)
- Mechanical Systems and Signal Processing (MSSP)
- Review of Scientific Instruments

**Physics / Wave Phenomena / Inverse Problems**
- Physical Review X
- Physical Review Letters
- Inverse Problems
- Journal of Computational Physics
- Communications Physics
- Nature Physics

**Optics / Scattering / Imaging**
- Optica
- Optics Express
- Optics Letters
- Light: Science & Applications

**ML / Signal Processing / Imaging**
- IEEE Signal Processing Magazine
- IEEE Transactions on Signal Processing
- IEEE Transactions on Image Processing
- International Journal of Computer Vision

**General High-Impact**
- Nature Communications
- Science Advances
- PNAS

---

## High-Quality Related Conferences (Selected)

**Signal Processing / Acoustics**
- ICASSP
- International Congress on Acoustics (ICA)
- IEEE Ultrasonics Symposium (IUS)

**ML / Vision / Imaging**
- NeurIPS
- ICML
- ICLR
- CVPR
- ICCV/ECCV
- SIGGRAPH

**Optics / Photonics**
- CLEO
- SPIE Photonics West (Computational Imaging tracks)

---

## High-Quality Related Papers (With DOI)

**LDV and Acoustic Field Measurement**
- Laser Doppler vibrometry and near-field acoustic holography. MSSP (2006). DOI: 10.1016/j.ymssp.2005.11.011
- Visualising scattering underwater acoustic fields using LDV. JSV (2007). DOI: 10.1016/j.jsv.2007.04.026
- Transducer characterization by LDV. JASA (2009). DOI: 10.1121/1.4783677
- Laser Doppler multi-beam differential vibrometry. JASA (2020). DOI: 10.1121/1.5147034

**Wave Physics / Scattering / Time Reversal**
- Time reversal in acoustics. Contemporary Physics (1996). DOI: 10.1080/00107519608230338
- An overview of time-reversal acoustics. JASA (2008). DOI: 10.1121/1.2933288
- On optimisation for inverse acoustic scattering (full/limited aperture). Inverse Problems (1989). DOI: 10.1088/0266-5611/5/2/009
- Inverse acoustic scattering by small-obstacle expansion. Inverse Problems (2008). DOI: 10.1088/0266-5611/24/3/035022

**Physics-Guided ML / Inverse Problems**
- CNNs for inverse problems in imaging (review). IEEE SPM (2017). DOI: 10.1109/MSP.2017.2739299
- Physics-informed neural networks. JCP (2019). DOI: 10.1016/j.jcp.2018.10.045
- Physics-constrained DL for acoustic inverse scattering. MSSP (2022). DOI: 10.1016/j.ymssp.2021.108190
- NN warm-start for inverse acoustic obstacle scattering. JCP (2023). DOI: 10.1016/j.jcp.2023.112341
- Deep Unfolding for Snapshot Compressive Imaging. IJCV (2023). DOI: 10.1007/s11263-023-01844-4

**Non-contact Acoustic Decoding / Scattering Operators**
- Far-Field Subwavelength Acoustic Imaging by DL. PRX (2020). DOI: 10.1103/PhysRevX.10.031029
- The visual microphone. ACM TOG (2014). DOI: 10.1145/2601097.2601119
- Event-Based Visual Microphone. ICASSP (2023). DOI: 10.1109/ICASSP49357.2023.10094677
- Transmission matrix inversion through scattering media. Optics Express (2017). DOI: 10.1364/OE.25.027234
- Online learning of transmission matrix in dynamic media. Optica (2023). DOI: 10.1364/OPTICA.479962
