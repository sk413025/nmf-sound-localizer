# Related Work Development Map (NC Draft Support)

Purpose: provide a detailed development map across themes, papers, and manuscript claims, and a lineage-driven Related Work draft with citation positions.

Note on citations: this file uses placeholder labels [R#]. Replace with BibTeX keys in the manuscript and the actual references list.

---

## Development Timeline (Paper-Level Lineage)

- 1989-2008: Foundational inverse acoustic scattering formalizes ill-posed recovery and sensitivity to aperture and misfit design, establishing the need for strong priors and careful measurement [R5, R6].
- 1996-2008: Time-reversal acoustics demonstrates that multipath and scattering can be harnessed to refocus energy, reframing disorder as a reversible mapping rather than noise [R1, R4].
- 2006-2009: LDV paired with near-field acoustic holography and scattering visualization shows non-contact optical readout for high-fidelity surface velocity and field structure [R7, R8, R9].
- 2007-2011: Transmission-matrix optics establishes scattering as a controllable linear operator and enables focusing through disordered media [R32, R33, R20].
- 2014-2017: Visual microphones show micro-motion readout yields audio signals; transmission-matrix inversion formalizes scattering as an operator [R11, R20].
- 2017-2023: CNN-based inverse solvers and physics-informed frameworks introduce data-driven and physics-guided priors for inverse problems [R13, R14, R17].
- 2009-2021: Sparse inference and algorithm unrolling mature from convex optimization (FISTA) to interpretable deep architectures [R30, R31, R15].
- 1999-2011: NMF foundations and IS-divergence optimization formalize parts-based representations and scale-invariant spectral modeling [R27, R28, R29].
- 1986-1996: Classical array/DOA foundations (MUSIC, ESPRIT, beamforming, parametric reviews) define the baseline context that single-point LDV departs from [R23, R24, R25, R26].
- 2020-2025: Deep learning enables subwavelength acoustic imaging and scattering surrogates; dynamic transmission matrices and multi-beam LDV enable continuous tracking [R16, R18, R21, R10].
- 2018-2023: Single-sensor DOA via engineered scattering structures demonstrates that spatial information can be encoded without microphone arrays; NMF and deep learning decode direction from frequency-dependent signatures [R35, R36, R37, R38, R39].
- 1967-2004: Auditory physiology establishes that the pinna acts as a direction-dependent spectral filter; monaural localization studies demonstrate that single-ear spectral cues suffice for spatial perception when learned mappings are available [A1, A3, A8].
- 2017-2020: Deep learning for auditory signal processing shows effectiveness in speech enhancement and cochlear implant applications, establishing learned representations for auditory-relevant features [L2, L4].

---

## Conceptual Lineage (Theme-to-Theme Development)

- Wave reciprocity -> time reversal: scattering becomes invertible in principle, indicating that multipath carries recoverable information [R1, R4].
- Time reversal -> inverse scattering: inversion is possible but ill-conditioned, motivating structured priors and robust measurement [R5, R6].
- Inverse scattering -> LDV measurement: high-fidelity non-contact sensing is needed to avoid perturbing the system and to reduce noise amplification [R7, R8].
- LDV -> optical acoustic decoding: optical sensing captures acoustic field signatures, aligning with non-contact sound recovery [R7, R9, R11].
- Scattering media -> transmission matrix: scattering modeled as a linear operator suggests a learnable mapping perspective applicable to acoustic scattering [R32, R33, R20].
- Inverse problems -> physics-guided learning: data-driven inference stabilizes inversion when guided by physics constraints [R13, R14, R17].
- Scattering structures -> single-sensor DOA: engineered acoustic structures encode spatial information in frequency signatures, enabling direction estimation without arrays [R35, R36, R37, R38, R39].
- Array baselines -> single-point DOA: classical array methods define the limitations of multi-sensor assumptions, motivating single-point LDV approaches [R22, R23, R24, R25, R26].
- Pinna acoustics -> spectral spatial encoding: human auditory system demonstrates that scattering structures encode direction into frequency signatures, inspiring analogous engineered and natural approaches [A1, A3, A8].
- Monaural localization -> learning-based decoding: biological evidence that single-ear spectral cues require learned mappings supports the necessity of data-driven decoders [A8, A12].

---

## Methodological Lineage (Learning/Architecture Development)

- CNN priors -> structured inverse solvers: learned priors replace hand-crafted regularization in inverse imaging [R13].
- Physics-informed learning -> physical constraints: governing equations constrain the hypothesis space to improve generalization [R14].
- Physics constraints -> unrolled solvers: algorithm unrolling encodes iterative physics as trainable layers [R15].
- Convex optimization -> unrolling: FISTA provides the iterative template later unrolled into trainable architectures [R30, R31].
- Unrolling -> physics-aware decoders: unrolled structures support architectures that instantiate the physical formula rather than only regularize it [R15].
- Optical sensing -> real-time decoding: event-based visual microphones show low-latency optical audio decoding [R12].
- Dynamic scattering -> continuous tracking: online transmission matrices show scattering operators can be updated in real time [R21].
- NMF foundations -> IS-divergence modeling: NMF and IS-divergence formalize parts-based spectral modeling for source content estimation [R27, R28, R29].

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
- Single-sensor DOA without arrays: engineered scattering structures enable single-microphone DOA [R35, R36, R38, R39]; the gap is using natural plate vibration modes rather than artificial structures, and non-contact LDV rather than embedded microphones.
- Classical DOA baselines: array-based methods such as MUSIC and ESPRIT set the dominant paradigm that single-point LDV-based decoding challenges [R23, R24, R25, R26].
- NMF/IS-divergence foundations: IS-geometry and NMF establish the mathematical basis for the speech dictionary and multiplicative updates used in the inner loop [R27, R28, R29].
- Unrolling foundations: iterative optimization (FISTA) and unrolling reviews motivate the physics-aware unrolled architecture [R30, R31].
- Transmission-matrix analogy: optical transmission matrices formalize scattering as a linear operator, supporting the high-dimensional mapping viewpoint [R32, R33, R20].
- Biological inspiration (pinna analogy): human auditory system uses pinna as direction-dependent spectral filter [A1, A3]; the contribution is using natural plate modes rather than evolved (pinna) or engineered (metamaterial) structures, with non-contact LDV measurement.
- Learning from spectral cues: monaural localization studies show spectral-to-spatial mappings require learning [A8, A12]; the physics-aware decoder explicitly instantiates this mapping via a physical dictionary rather than implicit neural adaptation.

---

## Related Work (NC-Style Draft, Lineage-Driven)

### Wave physics and scattering as structure
Classical inverse acoustic scattering frames recovery from scattered fields as ill-posed and highly sensitive to aperture and modeling choices, which makes strong priors essential [R5, R6]. In parallel, time-reversal acoustics demonstrates that multipath and scattering are not merely noise but a reversible mapping that can refocus energy, indicating that complex media encode recoverable structure [R1, R4]. These foundations motivate a shift from suppressing scattering to decoding it.

### Non-contact measurement of acoustic fields
Laser Doppler vibrometry (LDV) provides non-contact surface-velocity measurements and has been paired with near-field acoustic holography to reconstruct spatial distributions, while also visualizing scattering fields and enabling transducer characterization [R7, R8, R9]. LDV thus offers a high-fidelity optical readout of acoustic phenomena without disturbing the system. Prior LDV work predominantly targets measurement and visualization rather than decoding the latent physical structure embedded in scattering.

### Physics-guided inference and unrolled inverse solvers
CNN-based approaches to inverse imaging show that learned priors can replace explicit regularization and significantly stabilize reconstruction [R13]. Physics-informed neural networks and physics-constrained learning incorporate governing equations or constraints to improve generalization beyond training data [R14, R17]. Algorithm unrolling bridges physics and learning by turning iterative solvers into trainable architectures with interpretable internal states [R15]. These advances suggest that embedding physics in model structure yields robust inversion, yet few methods instantiate the physical formula as the network itself.

### Learning-based acoustic decoding and scattering operators
Deep learning has enabled subwavelength acoustic imaging and neural surrogates for inverse scattering, indicating that acoustic mappings are learnable under constrained measurements [R16, R18]. Non-contact sound recovery from visual measurements shows that optical sensing can recover acoustic information from surface motion [R11, R12]. In optics, transmission-matrix formulations model scattering media as linear operators and enable inversion and online adaptation [R32, R33, R20, R21], providing a conceptual analog to acoustic scattering as a high-dimensional mapping. These developments set the stage for a physics-aware decoder that uses LDV measurements to explicitly invert scattering structure and generalize across materials.

### Single-sensor DOA via engineered scattering structures
Recent work demonstrates that engineered acoustic structures can encode spatial information for single-sensor direction-of-arrival estimation. El Badawy and Dokmanić show that embedding a microphone in a scattering structure (LEGO bricks) creates frequency-dependent directional signatures decodable via NMF [R36]. Garg et al. extend this with 3D-printed metamaterial enclosures and deep learning for low-power IoT deployment [R39]. A comprehensive review by Jiang and He surveys spatial information coding with artificially engineered structures across acoustic and elastic wave sensing [R37]. Mobile implementations include EarCase, which achieves 3.7° mean error using phone-case scattering structures [R38], and single-vibration-sensor DOA estimation on structural plates [R35]. These approaches establish the principle that scattering encodes direction, but rely on engineered structures and embedded sensors. The present work differs by using natural plate vibration modes as the scattering medium and non-contact LDV measurement, eliminating the need for custom structures or sensor embedding.

### Classical DOA baselines and NMF foundations
Array-based DOA methods such as MUSIC and ESPRIT, along with beamforming reviews, define the classical multi-sensor paradigm and its assumptions [R23, R24, R25, R26]. For single-point settings, the reliance on spatial sampling motivates alternative encodings such as engineered scattering structures. On the content modeling side, NMF provides the parts-based spectral decomposition used to estimate source content, and IS-divergence offers scale-invariant fitting appropriate for power spectra [R27, R28, R29]. These foundations support the inner-loop content estimation and clarify the departure from array-based baselines.

### Unrolling and optimization foundations
Iterative optimization methods such as FISTA provide the canonical template for sparse inference, later generalized via algorithm unrolling into interpretable, trainable networks for inverse problems [R30, R31]. This lineage justifies the physics-aware unrolled architecture and its link to classical optimization geometry.

### Biological inspiration: spatial hearing and monaural localization
The ability to localize sounds using spectral cues from a single ear has been well established in human auditory physiology [A1, A3]. The pinna acts as a direction-dependent spectral filter, encoding elevation and azimuth information into frequency-domain signatures that the auditory system learns to decode [A4, A8]. Studies of monaural localization in unilaterally deaf individuals demonstrate that spectral cues alone can provide sufficient spatial information when combined with learned mappings [A8, A12]. This biological mechanism—a scattering structure encoding spatial information into spectral features—directly inspires the present approach, where natural plate vibration modes serve an analogous role to the pinna, and the LDV measurement replaces cochlear transduction. Unlike engineered scattering structures [R36, R39], plate modes arise naturally from structural dynamics, and unlike human learning that requires weeks of adaptation [A12], the physics-aware decoder explicitly instantiates the spectral-to-direction mapping through a pre-computed physical dictionary. Deep learning approaches have demonstrated effectiveness in related auditory signal processing tasks including speech enhancement for hearing-impaired listeners [L2], suggesting that learned representations can successfully capture auditory-relevant features under challenging conditions.

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
[R22] High-resolution frequency-wavenumber spectrum analysis. Proceedings of the IEEE (1969). DOI: 10.1109/PROC.1969.7278
[R23] Multiple emitter location and signal parameter estimation. IEEE Transactions on Antennas and Propagation (1986). DOI: 10.1109/TAP.1986.1143830
[R24] ESPRIT - Estimation of signal parameters via rotational invariance techniques. IEEE Transactions on Acoustics, Speech, and Signal Processing (1989). DOI: 10.1109/29.32276
[R25] Beamforming: a versatile approach to spatial filtering. IEEE ASSP Magazine (1988). DOI: 10.1109/53.665
[R26] Two decades of array signal processing research: the parametric approach. IEEE Signal Processing Magazine (1996). DOI: 10.1109/79.526899
[R27] Learning the parts of objects by non-negative matrix factorization. Nature (1999). DOI: 10.1038/44565
[R28] Nonnegative Matrix Factorization with the Itakura-Saito Divergence: With Application to Music Analysis. Neural Computation (2009). DOI: 10.1162/neco.2008.04-08-771
[R29] Algorithms for Nonnegative Matrix Factorization with the beta-divergence. Neural Computation (2011). DOI: 10.1162/neco_a_00168
[R30] A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems. SIAM Journal on Imaging Sciences (2009). DOI: 10.1137/080716542
[R31] Algorithm Unrolling: Interpretable, Efficient Deep Learning for Signal and Image Processing. IEEE Signal Processing Magazine (2021). DOI: 10.1109/MSP.2020.3016905
[R32] Measuring the Transmission Matrix in Optics: An Approach to the Study and Control of Light Propagation in Disordered Media. Physical Review Letters (2010). DOI: 10.1103/PhysRevLett.104.100601
[R33] Focusing coherent light through opaque strongly scattering media. Optics Letters (2007). DOI: 10.1364/OL.32.002309
[R34] Time-Reversal Acoustics in Biomedical Engineering. Annual Review of Biomedical Engineering (2003). DOI: 10.1146/annurev.bioeng.5.040202.121630
[R35] Direction of arrival estimation of an acoustic wave using a single structural vibration sensor. Journal of Sound and Vibration (2023). DOI: 10.1016/j.jsv.2023.117671
[R36] Direction of Arrival With One Microphone, a Few LEGOs, and Non-Negative Matrix Factorization. IEEE/ACM Transactions on Audio, Speech, and Language Processing (2018). DOI: 10.1109/TASLP.2018.2867081
[R37] Spatial information coding with artificially engineered structures for acoustic and elastic wave sensing. Frontiers in Physics (2022). DOI: 10.3389/fphy.2022.1024964
[R38] EarCase: Sound Source Localization Leveraging Mini Acoustic Structure Equipped Phone Cases for Hearing-challenged People. MobiHoc (2023). DOI: 10.1145/3565287.3610270
[R39] Owlet: Enabling Spatial Information in Ubiquitous Acoustic Devices. MobiSys (2021). DOI: 10.1145/3458864.3467880

### Auditory Physiology and Spatial Hearing [A1-A12]

[A1] Spatial Hearing: The Psychophysics of Human Sound Localization. Blauert, J. MIT Press (1997). DOI: 10.1121/1.392109
[A3] The role of the pinna in human localization. Batteau, D.W. Proceedings of the Royal Society B (1967). DOI: 10.1098/rspb.1967.0058
[A4] Transformation of sound-pressure level from the free field to the eardrum. Shaw, E.A.G. JASA (1974). DOI: 10.1121/1.1903522
[A5] Spectral cues used in the localization of sound sources on the median plane. Hebrank, J. & Wright, D. JASA (1974). DOI: 10.1121/1.1903520
[A6] The influence of pinnae-based spectral cues on sound localization. Musicant, A.D. & Butler, R.A. JASA (1984). DOI: 10.1121/1.390773
[A8] Contribution of Head Shadow and Pinna Cues to Chronic Monaural Sound Localization. Van Wanrooij, M.M. & Van Opstal, A.J. Journal of Neuroscience (2004). DOI: 10.1523/JNEUROSCI.4163-03.2004
[A9] Monaural sound localization: Acute versus chronic unilateral impairment. Slattery, W.H. & Middlebrooks, J.C. Hearing Research (1994). DOI: 10.1016/0378-5955(94)90053-1
[A10] Monaural sound localization revisited. Wightman, F.L. & Kistler, D.J. JASA (1997). DOI: 10.1121/1.418029
[A11] Adapting to supernormal auditory localization cues. Shinn-Cunningham, B.G. et al. JASA (1998). DOI: 10.1121/1.423088
[A12] Relearning sound localization with new ears. Hofman, P.M. et al. Nature Neuroscience (1998). DOI: 10.1038/2226

### Advisor Publications (Prof. Ying-Hui Lai) [L1-L5]

[L1] A Deep Denoising Autoencoder Approach to Improving the Intelligibility of Vocoded Speech in Cochlear Implant Simulation. Lai, Y.-H. et al. IEEE TBME (2017). DOI: 10.1109/TBME.2016.2613960
[L2] Deep Learning-Based Noise Reduction Approach to Improve Speech Intelligibility for Cochlear Implant Recipients. Lai, Y.-H. et al. Ear and Hearing (2018). DOI: 10.1097/AUD.0000000000000537
[L3] An Audio-Visual Speech Enhancement Model Using Multimodal Deep Learning. Lai, Y.-H., Tsao, Y.-C., & Chen, F. IEEE TETCI (2018). DOI: 10.1109/TETCI.2017.2784878
[L4] Speech enhancement for hearing-impaired listeners using deep neural networks with auditory-mask motivated loss function. Lai, Y.-H. et al. JASA (2019). DOI: 10.1121/1.5094063
[L5] Time-frequency attention for monaural speech enhancement. Lai, Y.-H. et al. IEEE ICASSP (2020). DOI: 10.1109/ICASSP40776.2020.9054182
