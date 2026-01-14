# Related Work Development Map (NC Draft Support)

Purpose: provide a detailed development map across themes, papers, and manuscript claims, and a lineage-driven Related Work draft with citation positions.

Note on citations: this file uses placeholder labels [R#]. Replace with BibTeX keys in the manuscript and the actual references list.

---

## Development Timeline (Paper-Level Lineage)

- 1989-2008: Foundational inverse acoustic scattering formalizes ill-posed recovery and sensitivity to aperture and misfit design, establishing the need for strong priors and careful measurement [R5, R6].
- 1996-2008: Time-reversal acoustics demonstrates that multipath and scattering can be harnessed to refocus energy, reframing disorder as a reversible mapping rather than noise [R1, R4].
- 2006-2009: LDV paired with near-field acoustic holography and scattering visualization shows non-contact optical readout for high-fidelity surface velocity and field structure [R7, R8, R9].
- 2014-2017: Visual microphones show micro-motion readout yields audio signals; transmission-matrix inversion formalizes scattering as an operator [R11, R20].
- 2017-2023: CNN-based inverse solvers and physics-informed frameworks introduce data-driven and physics-guided priors for inverse problems [R13, R14, R17].
- 2020-2025: Deep learning enables subwavelength acoustic imaging and scattering surrogates; unrolling yields interpretable inverse architectures; dynamic transmission matrices and multi-beam LDV enable continuous tracking [R16, R15, R18, R21, R10].

---

## Conceptual Lineage (Theme-to-Theme Development)

- Wave reciprocity -> time reversal: scattering becomes invertible in principle, indicating that multipath carries recoverable information [R1, R4].
- Time reversal -> inverse scattering: inversion is possible but ill-conditioned, motivating structured priors and robust measurement [R5, R6].
- Inverse scattering -> LDV measurement: high-fidelity non-contact sensing is needed to avoid perturbing the system and to reduce noise amplification [R7, R8].
- LDV -> optical acoustic decoding: optical sensing captures acoustic field signatures, aligning with non-contact sound recovery [R7, R9, R11].
- Scattering media -> transmission matrix: scattering modeled as a linear operator suggests a learnable mapping perspective applicable to acoustic scattering [R20].
- Inverse problems -> physics-guided learning: data-driven inference stabilizes inversion when guided by physics constraints [R13, R14, R17].

---

## Methodological Lineage (Learning/Architecture Development)

- CNN priors -> structured inverse solvers: learned priors replace hand-crafted regularization in inverse imaging [R13].
- Physics-informed learning -> physical constraints: governing equations constrain the hypothesis space to improve generalization [R14].
- Physics constraints -> unrolled solvers: algorithm unrolling encodes iterative physics as trainable layers [R15].
- Unrolling -> physics-aware decoders: unrolled structures support architectures that instantiate the physical formula rather than only regularize it [R15].
- Optical sensing -> real-time decoding: event-based visual microphones show low-latency optical audio decoding [R12].
- Dynamic scattering -> continuous tracking: online transmission matrices show scattering operators can be updated in real time [R21].

---

## Claim-to-Literature Map (Manuscript Positioning)

- Scattering as high-dimensional mapping: time-reversal and transmission-matrix work show scattering is structured and invertible in principle [R1, R4, R20]; the gap is acoustic scattering decoded via LDV without contact or arrays.
- Modal sparsity and physical dictionary: inverse-problem priors and unrolling imply low-dimensional structure is exploitable [R13, R15]; the contribution is extracting a physics dictionary directly from modal evidence in LDV signals.
- Physics-aware decoder (unrolled physics): PINNs and physics-constrained learning embed equations as constraints [R14, R17], but the architecture instantiates the physical formula itself rather than only a loss term.
- Hybrid necessity (physics + learning): prior work shows benefits of physics constraints, but rarely demonstrates that pure physics or pure DL fails so dramatically; the ablation makes the necessity explicit [R17, R18].
- 0 dB robustness: subwavelength acoustic imaging demonstrates learnability but not extreme low-SNR stability [R16]; results extend robustness into the 0 dB regime.
- Mechanistic alignment (diagonal agreement): unrolled models are interpretable, yet explicit formula-level alignment is rarely shown [R15, R14]; diagonal evidence provides mechanism-level validation.
- Universality across materials: transmission-matrix approaches often require medium-specific calibration [R20, R21]; cross-material generality argues for a shared physical signature.
- Continuous tracking beyond grids: dynamic scattering and event-based sensing imply continuous updates are feasible [R21, R12]; continuous physics provides a principled route beyond grid discretization.

---

## Related Work (NC-Style Draft, Lineage-Driven)

### Wave physics and scattering as structure
Classical inverse acoustic scattering frames recovery from scattered fields as ill-posed and highly sensitive to aperture and modeling choices, which makes strong priors essential [R5, R6]. In parallel, time-reversal acoustics demonstrates that multipath and scattering are not merely noise but a reversible mapping that can refocus energy, indicating that complex media encode recoverable structure [R1, R4]. These foundations motivate a shift from suppressing scattering to decoding it.

### Non-contact measurement of acoustic fields
Laser Doppler vibrometry (LDV) provides non-contact surface-velocity measurements and has been paired with near-field acoustic holography to reconstruct spatial distributions, while also visualizing scattering fields and enabling transducer characterization [R7, R8, R9]. LDV thus offers a high-fidelity optical readout of acoustic phenomena without disturbing the system. Prior LDV work predominantly targets measurement and visualization rather than decoding the latent physical structure embedded in scattering.

### Physics-guided inference and unrolled inverse solvers
CNN-based approaches to inverse imaging show that learned priors can replace explicit regularization and significantly stabilize reconstruction [R13]. Physics-informed neural networks and physics-constrained learning incorporate governing equations or constraints to improve generalization beyond training data [R14, R17]. Algorithm unrolling bridges physics and learning by turning iterative solvers into trainable architectures with interpretable internal states [R15]. These advances suggest that embedding physics in model structure yields robust inversion, yet few methods instantiate the physical formula as the network itself.

### Learning-based acoustic decoding and scattering operators
Deep learning has enabled subwavelength acoustic imaging and neural surrogates for inverse scattering, indicating that acoustic mappings are learnable under constrained measurements [R16, R18]. Non-contact sound recovery from visual measurements shows that optical sensing can recover acoustic information from surface motion [R11, R12]. In optics, transmission-matrix formulations model scattering media as linear operators and enable inversion and online adaptation [R20, R21], providing a conceptual analog to acoustic scattering as a high-dimensional mapping. These developments set the stage for a physics-aware decoder that uses LDV measurements to explicitly invert scattering structure and generalize across materials.

---

## Reference Map (Placeholders)

[R1] Time reversal in acoustics. Contemporary Physics (1996). DOI: 10.1080/00107519608230338
[R4] An overview of time-reversal acoustics. JASA (2008). DOI: 10.1121/1.2933288
[R5] On an optimisation method for the full- and the limited-aperture problem in inverse acoustic scattering. Inverse Problems (1989). DOI: 10.1088/0266-5611/5/2/009
[R6] Inverse acoustic scattering by small-obstacle expansion of a misfit function. Inverse Problems (2008). DOI: 10.1088/0266-5611/24/3/035022
[R7] Laser Doppler vibrometry and near-field acoustic holography. Mechanical Systems and Signal Processing (2006). DOI: 10.1016/j.ymssp.2005.11.011
[R8] Visualising scattering underwater acoustic fields using LDV. Journal of Sound and Vibration (2007). DOI: 10.1016/j.jsv.2007.04.026
[R9] Transducer characterization by LDV. JASA (2009). DOI: 10.1121/1.4783677
[R10] Laser Doppler multi-beam differential vibrometry. JASA (2020). DOI: 10.1121/1.5147034
[R11] The visual microphone. ACM Transactions on Graphics (2014). DOI: 10.1145/2601097.2601119
[R12] Event-Based Visual Microphone. ICASSP (2023). DOI: 10.1109/ICASSP49357.2023.10094677
[R13] CNNs for inverse problems in imaging: A review. IEEE Signal Processing Magazine (2017). DOI: 10.1109/MSP.2017.2739299
[R14] Physics-informed neural networks. Journal of Computational Physics (2019). DOI: 10.1016/j.jcp.2018.10.045
[R15] Deep Unfolding for Snapshot Compressive Imaging. International Journal of Computer Vision (2023). DOI: 10.1007/s11263-023-01844-4
[R16] Far-Field Subwavelength Acoustic Imaging by Deep Learning. Physical Review X (2020). DOI: 10.1103/PhysRevX.10.031029
[R17] Physics-constrained deep learning for acoustic inverse scattering. Mechanical Systems and Signal Processing (2022). DOI: 10.1016/j.ymssp.2021.108190
[R18] Neural network warm-start for inverse acoustic obstacle scattering. Journal of Computational Physics (2023). DOI: 10.1016/j.jcp.2023.112341
[R20] Transmission matrix inversion in scattering media. Optics Express (2017). DOI: 10.1364/OE.25.027234
[R21] Online learning of transmission matrix in dynamic media. Optica (2023). DOI: 10.1364/OPTICA.479962
