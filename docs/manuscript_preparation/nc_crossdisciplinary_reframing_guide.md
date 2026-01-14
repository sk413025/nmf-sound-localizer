# Nature Communications Cross-Disciplinary Reframing Guide

**Related Documents**:
- For editor psychology and decision tree: see `editor_perspective_analysis.md`
- For full literature timeline and DOIs: see `../related_work_development_map.md`
- For auditory physiology details: see `auditory_physiology_literature.md`

---

Purpose: Provide a detailed, editor-facing writing scaffold that keeps the main narrative accessible to multi-disciplinary readers while preserving technical depth in Methods and Supplementary.

Scope: Use this guide to restructure Introduction and Related Work so the editor can quickly identify (1) cross-disciplinary positioning, (2) the gap, and (3) a clear reviewer pool.

---

## 1) Editor-first goals (fast decision criteria)

- Clarity: A one-sentence positioning statement visible within the first paragraph.
- Gap: A single explicit sentence that states the missing intersection.
- Boundary: A short line that limits claims and lowers review risk.
- Reviewer cues: A few domain keywords that make reviewer selection obvious.

---

## 2) Narrative layers (what goes where)

**Layer 0 (Hook, 1-2 sentences)**  
Use a concept-level idea that any field can understand:
Scattering is not noise; it is a direction-dependent code.

**Layer 1 (Mechanistic intuition, 1 short paragraph)**  
Explain why a single point can carry direction: spectral cues can encode space.

**Layer 2 (Method concept, 1 paragraph)**  
State the approach without deep technical terms: a physics dictionary plus a learned decoder.

**Layer 3 (Evidence, 1 paragraph)**  
State the key results and why they matter (robustness, generalization).

**Layer 4 (Details)**  
Move algorithmic details, hyperparameters, and derivations to Methods/Supplementary.

---

## 3) Main text vs Methods vs Supplementary (detail allocation)

Main text should contain:
- Cross-disciplinary positioning and the gap.
- Mechanism-level intuition.
- Result-level statements with plain language context.
- Boundary statement (what we do not claim).

Methods should contain:
- Model architecture specifics.
- Exact optimization steps (NMF/IS, unrolling details).
- Implementation and measurement details.

Supplementary should contain:
- Full derivations and parameter sweeps.
- Additional ablations, diagnostics, and extended baselines.

---

## 4) Positioning + gap templates (editor ready)

**Positioning templates**  
1) Single-point acoustic localization at the intersection of monaural spectral cues, non-contact vibrometry, and physics-informed inverse problems.  
2) A physics-informed, single-sensor DOA method that treats natural structural modes as a direction-dependent spectral encoder.  
3) Non-contact, single-point DOA inference using an explicit physical dictionary and a learned decoder.

**Gap template**  
Prior work covers each axis separately (monaural cues, optical/vibrometry sensing, physics-informed inversion), but no method jointly infers DOA from a single remote vibration measurement with an explicit physical dictionary.

**Boundary statement (required)**  
This is a physics-inspired analogy, not a physiological model or an HRTF simulation.

---

## 5) Cross-disciplinary translation table (plain language)

| Specialized term | Plain-language equivalent | Where to keep detail |
| --- | --- | --- |
| IS divergence in NMF | A scale-invariant way to match spectral shapes | Methods/Supplementary |
| Unrolling | Turning an iterative solver into a trainable network | Methods |
| Physics dictionary | Precomputed spectral templates from structural modes | Main text |
| LDV | Non-contact vibration measurement using a laser | Main text |
| DOA | Direction of arrival (source direction) | Main text |

---

## 6) Related Work structure (NC-friendly)

Use three short sections, each with 2-3 anchors:

1) Biological inspiration: monaural spectral cues  
   - Demonstrate that spectral cues can encode direction [A3, A4, A8, A10, A12].

2) Non-contact vibrometry for acoustic sensing  
   - Demonstrate feasibility of remote acoustic recovery [L6, L7, L8].

3) Physics-informed decoding for localization  
   - Provide the inverse-problem and dictionary-learning foundations [R17, R27, R28, R31].  
   - Contrast with single-sensor DOA using engineered scatterers [R35, R36, R37, R38, R39].

---

## 7) Claim-to-evidence mapping (minimal anchors)

| Claim | Evidence anchors |
| --- | --- |
| Spectral cues can carry direction at a single point | [A3, A4, A8, A10, A12] |
| Non-contact vibration sensing can recover acoustic content | [L6, L7, L8] |
| A physical dictionary yields interpretable inversion | [R27, R28, R31] |
| Single-sensor DOA exists but relies on engineered structures | [R35, R36, R37, R38, R39] |
| Physics-informed inverse scattering provides prior structure | [R17] |

---

## 8) Example NC-style paragraph (ready to paste)

We position this work at the intersection of monaural spectral cues, non-contact vibrometry, and physics-informed inverse problems. Auditory studies show that spectral cues alone can encode direction and can be learned in monaural listening [A3, A4, A8, A10, A12]. Optical and vibrometry systems demonstrate that non-contact vibration measurements can recover acoustic content [L6, L7, L8]. However, prior physics-informed localization typically assumes arrays or engineered scatterers and does not jointly infer DOA from a single remote vibration measurement with an explicit physical dictionary [R17, R35, R36, R37, R38, R39]. We fill this gap by treating natural structural modes as a direction-dependent spectral encoder and learning the inverse mapping, while explicitly framing this as a physics-inspired analogy rather than a physiological model.

---

## 9) Reviewer pool cues (for editor confidence)

Use these keywords in the cover letter or a short note to the editor:
- monaural spectral cues localization
- laser Doppler vibrometry acoustic sensing
- physics-informed inverse problems acoustic localization
- single-sensor DOA sparse coding

---

## 10) Integration plan (how to use this file)

1) Copy the positioning sentence and boundary statement into `manuscript/main_text/introduction.md`.
2) Convert Section 6 into the Related Work outline inside `docs/related_work_development_map.md`.
3) Keep the reference registry below as the minimal anchor set.

---

## 11) Reference registry (explicit and real)

**Auditory physiology and monaural localization**  
- [A3] D. W. Batteau, "The role of the pinna in human localization," Proceedings of the Royal Society B, 168(1011):158-180, 1967. DOI: 10.1098/rspb.1967.0058  
- [A4] E. A. G. Shaw, "Transformation of sound-pressure level from the free field to the eardrum in the horizontal plane," Journal of the Acoustical Society of America, 56(6):1848-1861, 1974. DOI: 10.1121/1.1903522  
- [A8] M. M. Van Wanrooij and A. J. Van Opstal, "Contribution of head shadow and pinna cues to chronic monaural sound localization," Journal of Neuroscience, 24(17):4163-4171, 2004. DOI: 10.1523/JNEUROSCI.4163-03.2004  
- [A10] F. L. Wightman and D. J. Kistler, "Monaural sound localization revisited," Journal of the Acoustical Society of America, 101(2):1050-1063, 1997. DOI: 10.1121/1.418029  
- [A12] P. M. Hofman, J. G. A. Van Riswick, and A. J. Van Opstal, "Relearning sound localization with new ears," Nature Neuroscience, 1(5):417-421, 1998. DOI: 10.1038/2226

**Optical and vibrometry-based acoustic sensing**  
- [L6] Y.-H. Lai et al., "Optical Microphone-Based Speech Reconstruction System With Deep Learning for Individuals With Hearing Loss," IEEE Transactions on Biomedical Engineering, 70(11):3195-3206, 2023. DOI: 10.1109/TBME.2023.3285437  
- [L7] Y.-H. Lai et al., "Study of optical-based speech acquisition system using vibration signals from speakers' medical masks," JASA Express Letters, 2(5), 2022. DOI: 10.1121/10.0010491  
- [L8] Y.-H. Lai et al., "Miniaturized Fabry-Perot fiber-optic microphone based on capillary tube and hydrogel diaphragm," Optics and Laser Technology, 184, 2025. DOI: 10.1016/j.optlastec.2025.112582

**Physics-informed inverse problems and dictionary learning**  
- [R17] "Physics-constrained deep learning for acoustic inverse scattering," Mechanical Systems and Signal Processing, 2022. DOI: 10.1016/j.ymssp.2021.108190  
- [R27] D. D. Lee and H. S. Seung, "Learning the parts of objects by non-negative matrix factorization," Nature, 401:788-791, 1999. DOI: 10.1038/44565  
- [R28] C. Fevotte, N. Bertin, and J.-L. Durrieu, "Nonnegative Matrix Factorization with the Itakura-Saito Divergence: With Application to Music Analysis," Neural Computation, 21(3):793-830, 2009. DOI: 10.1162/neco.2008.04-08-771  
- [R31] V. Monga, Y. Li, and Y. C. Eldar, "Algorithm Unrolling: Interpretable, Efficient Deep Learning for Signal and Image Processing," IEEE Signal Processing Magazine, 38(2):18-44, 2021. DOI: 10.1109/MSP.2020.3016905

**Single-sensor DOA with engineered scattering**  
- [R35] "Direction of arrival estimation of an acoustic wave using a single structural vibration sensor," Journal of Sound and Vibration, 2023. DOI: 10.1016/j.jsv.2023.117671  
- [R36] S. El Badawy and I. Dokmanic, "Direction of Arrival With One Microphone, a Few LEGOs, and Non-Negative Matrix Factorization," IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2018. DOI: 10.1109/TASLP.2018.2867081  
- [R37] Y. Jiang and C. He, "Spatial information coding with artificially engineered structures for acoustic and elastic wave sensing," Frontiers in Physics, 2022. DOI: 10.3389/fphy.2022.1024964  
- [R38] "EarCase: Sound Source Localization Leveraging Mini Acoustic Structure Equipped Phone Cases for Hearing-challenged People," MobiHoc, 2023. DOI: 10.1145/3565287.3610270  
- [R39] "Owlet: Enabling Spatial Information in Ubiquitous Acoustic Devices," MobiSys, 2021. DOI: 10.1145/3458864.3467880

