# Methods

## Experimental Apparatus

Laser Doppler vibrometry (LDV) measurements were acquired using a Polytec vibrometer system mounted on a vibration-isolated optical table (Newport). The sensor plate consisted of a polymethyl methacrylate (acrylic) sheet measuring approximately 30 × 30 × 0.3 cm, positioned on foam pads to approximate free-edge boundary conditions. This mounting configuration was chosen to allow natural modal vibrations to develop without the damping or frequency shifts that would be introduced by rigid clamping or adhesive attachment. The foam supports were positioned at the corners to minimize their influence on the plate's fundamental vibration modes while providing adequate mechanical stability.

An omnidirectional loudspeaker (effective frequency response 100 Hz–10 kHz) was positioned 1.5 m from the plate center at the same height as the measurement point. This distance was selected to ensure far-field acoustic conditions while maintaining adequate signal-to-noise ratio at the measurement location. The angular position of the loudspeaker relative to the plate normal was varied systematically to cover 37 directions spanning 0° to 180° in 5° increments, providing uniform angular sampling of the frontal hemisphere.

The measurement laser was focused near the plate center, approximately 5 cm from the geometric centroid. This location was chosen to avoid nodal lines of low-order structural modes, where vibration amplitude approaches zero and directional sensitivity would be compromised. Single-point measurement at a fixed location ensures that all directional information is encoded through spectral variations rather than spatial sampling.

The acoustic environment was semi-reverberant, with foam padding on adjacent walls to reduce flutter echoes while preserving realistic scattering conditions representative of practical deployment scenarios. Background acoustic noise levels were approximately 40 dB SPL. The semi-reverberant conditions were deliberately chosen rather than anechoic treatment to evaluate robustness under conditions closer to real-world applications, where perfect acoustic isolation is rarely achievable.

## Speech260 Dataset

The Speech260 dataset comprises single-point LDV measurements of plate vibrations induced by speech from 260 speakers with diverse linguistic content. For each of the 37 angular positions, 260 speech clips were recorded sequentially, yielding 9,620 total samples. Each clip corresponds to a 2-second speech utterance, providing sufficient temporal extent to capture multiple phonetic units while remaining computationally tractable.

Raw audio waveforms were sampled at 16 kHz, chosen to adequately capture the speech frequency range (300–3,000 Hz) while reducing computational overhead compared to higher sampling rates. Preprocessing included voice activity detection to remove leading and trailing silence, temporal synchronization to a common reference, and amplitude normalization to unit root-mean-square value. These preprocessing steps ensure that variations in recording levels and alignment do not confound the directional information encoded in spectral structure.

The dataset was partitioned into training (80%) and validation (20%) subsets using a deterministic rule based on clip index: every fifth clip (indices 0, 5, 10, ...) was assigned to the validation set. This stratified split ensures that each of the 37 angular classes contains proportionally equal representation in both subsets (208 training samples and 52 validation samples per angle), preventing class imbalance from affecting model evaluation. The deterministic split rule enables exact reproduction of training and evaluation conditions.

## Signal Processing

Raw waveforms were transformed to time-frequency representations using the short-time Fourier transform (STFT). We applied a 2,048-point FFT with Hann windowing and 75% overlap (hop length of 512 samples), computing magnitude spectra with DC offset removed. The FFT size was selected to provide adequate frequency resolution (~7.8 Hz per bin) for resolving structural resonances while maintaining temporal resolution sufficient for tracking speech dynamics.

From the full spectrum, frequency bins corresponding to the 300–3,000 Hz band were retained, yielding 346 frequency bins per frame. This band encompasses the primary spectral content of speech, including the fundamental frequency range (typically 80–400 Hz for adult speakers, but with harmonics extending into the kilohertz range) and first several formants critical for phonetic identity. The lower bound of 300 Hz excludes low-frequency mechanical noise and structural modes below the acoustic coupling threshold, while the upper bound of 3,000 Hz captures the most information-dense speech frequencies while excluding high-frequency content above the plate's modal density.

The same STFT configuration was applied consistently to the transfer function matrix, speech spectral basis, and LDV measurements to ensure frequency alignment across all processing stages. Frequency grid mismatch between these components would introduce systematic errors in dictionary construction and direction estimation.

## Transfer Function Measurement

The direction-dependent transfer function matrix **H** ∈ ℝ^{346×37} was measured using broadband calibration signals played sequentially from each of the 37 angular positions. For each direction, broadband noise (300–3,000 Hz, flat spectrum) was presented for 10 seconds, the LDV response was recorded, STFT-transformed using the parameters above, and averaged across approximately 300 frames to obtain a stable spectral estimate with reduced variance.

Each column of **H** represents the spectral signature of one incident direction, encoding how the plate's modal response varies with source angle. The physical interpretation is that different incident angles excite different combinations of plate eigenmodes with different amplitudes: the spatial pattern of acoustic pressure across the plate surface varies with incident direction, and this pattern couples differently to each structural mode depending on the mode shape. The result is that each direction produces a unique spectral fingerprint through direction-selective modal filtering.

Transfer function columns were normalized to unit L2 norm to ensure that direction estimates are not biased by overall amplitude differences between angular positions. This normalization preserves the relative spectral shape while removing scale variations that could arise from measurement inconsistencies.

## Physical Dictionary Construction

We constructed a physics-derived dictionary by combining the transfer function matrix with a speech spectral basis, encoding the assumption that the observed spectrum arises from speech content filtered by an angular transfer function.

First, an unsupervised spectral model was trained on a corpus of speech data to learn a set of 50 spectral atoms representing canonical speech patterns. This basis was learned using non-negative matrix factorization with Itakura-Saito divergence [@fevotte2009_is; @lee1999_nmf], which is appropriate for audio spectra because it preserves the scale-invariant properties of perceptual loudness and naturally enforces non-negativity constraints on spectral magnitudes. The basis atoms capture characteristic spectral shapes including vowel formant structures, consonant noise patterns, and transitional phonetic features.

The 50-atom basis was compressed via k-means clustering to 8 representative atoms, reducing the dictionary size and computational burden while preserving spectral diversity. The physics dictionary **D** ∈ ℝ^{346×296} was then formed by combining each of the 37 directional transfer functions with each of the 8 speech atoms (Hadamard product), yielding 296 direction-conditioned spectral templates. Each atom encodes a specific combination: one speech spectral pattern filtered through one angular transfer function.

This dictionary structure provides several advantages: explicit physical interpretability (we can identify which spectral patterns and directions contribute to any prediction), systematic ablation capability (we can test whether the directional or spectral structure is necessary), and reduced sample complexity (the structured dictionary constrains the hypothesis space, enabling learning with fewer examples than would be required for unconstrained function approximation).

## Physics-Aware Decoder Architecture

The decoder architecture combines sparse dictionary selection with transformer-based feature extraction, designed to exploit physical structure while retaining the flexibility to adapt to data variations not captured by the idealized dictionary.

The input spectrum (346-dimensional magnitude vector) is first embedded into a 128-dimensional feature space via a learned linear projection. This dimensionality was chosen to provide sufficient representational capacity while limiting model complexity. The embedded features are processed by a transformer encoder [@vaswani2017_attention] comprising a single layer with two attention heads. The transformer encoder learns to extract features relevant for direction discrimination through self-attention over the frequency dimension, enabling the model to learn which frequency bands and frequency relationships are most informative for direction estimation.

The transformer output generates attention weights over dictionary atoms via scaled dot-product attention. Query vectors are computed from the transformer output, while key vectors are derived from the dictionary atoms through a learned projection. The attention mechanism implements soft selection over the 296 dictionary atoms, with the attention weights indicating which direction-conditioned spectral templates are relevant for the current input.

Simultaneously, we compute the physical correlation **g** = **D**^T **y** between the input spectrum **y** and each dictionary atom, quantifying how well each direction-conditioned template explains the observation based on inner-product similarity. This physical correlation provides a complementary signal grounded in the forward model, encoding which atoms are geometrically aligned with the observation regardless of learned features.

The final routing decision combines both signals through element-wise multiplication of attention weights and normalized physical correlation. This gating mechanism ensures that only atoms receiving both high learned attention and high physical correlation contribute to the direction estimate, preventing the model from hallucinating directions unsupported by physical evidence while allowing the learned component to adapt to variations not captured by the idealized dictionary.

The gated representation is aggregated across atoms within each direction (L2 pooling) to produce 37 direction logits, which are trained with cross-entropy loss to maximize classification accuracy.

### Model Parameters

The complete model contains approximately 320,000 parameters, substantially smaller than typical vision or language transformers. This compact size reflects the structured nature of the problem: the physics dictionary provides strong prior constraints, reducing the representational burden on learned components.

## Training Procedure

Models were trained using the Adam optimizer with learning rate 10^{−3}, weight decay 10^{−4}, β_1 = 0.9, and β_2 = 0.999. The learning rate was selected based on preliminary experiments to provide stable convergence without oscillation. Weight decay provides L2 regularization to prevent overfitting on the training set.

Training proceeded for 20 epochs with batch size 32, with the best checkpoint selected based on validation accuracy. This training duration was sufficient for convergence across all experimental conditions, as verified by monitoring training and validation loss curves. Early stopping was not employed; instead, the fixed 20-epoch schedule was used consistently across all conditions to ensure fair comparison.

A fixed random seed (42) was applied throughout to ensure reproducibility of weight initialization, data shuffling order, and any stochastic operations (e.g., dropout, though not used in the final architecture). This deterministic configuration enables exact reproduction of all reported results.

## Ablation Study Design

To isolate the contributions of the physical dictionary and learned components, we conducted systematic ablation experiments under four conditions:

The **full model** (Physics-Aware AI) includes the complete architecture: transformer encoder for feature extraction, physical dictionary for structured representation, and gated routing combining learned attention with physical correlation. This serves as the reference condition against which all ablations are compared.

The **no-transformer ablation** removes the transformer encoder, replacing it with direct linear projection from input features to routing weights. The physical dictionary and correlation-based routing are retained. This tests whether adaptive feature extraction via self-attention is necessary beyond linear transformations, isolating the contribution of the learned component.

The **dense-routing ablation** removes the physical dictionary entirely, replacing the physics-aware routing with a standard dense classification head that maps transformer features directly to 37-class logits. This tests whether the physical structure provides essential inductive bias beyond what the transformer can learn from data alone, isolating the contribution of the physics component.

The **fixed-heuristic baseline** removes all learned parameters, predicting direction as the argmax of physical correlation θ̂ = argmax(**D**^T **y**). This establishes the accuracy achievable with physics alone, providing a lower bound on performance without any learned adaptation.

All ablation conditions used identical preprocessing, data splits, and training procedures where applicable, ensuring that performance differences reflect architectural contributions rather than experimental confounds.

## Evaluation Metrics

The primary metric was top-1 classification accuracy: the percentage of validation samples for which the predicted direction (argmax of the output distribution) matched the ground-truth angle. With 37 direction classes uniformly distributed, chance-level performance is 2.7% (1/37). We additionally computed per-angle accuracy to diagnose directional biases or systematic confusions between particular angle pairs.

Robustness to measurement noise was evaluated by adding white Gaussian noise to validation samples at controlled signal-to-noise ratios (10, 5, and 0 dB SNR). SNR was computed as 10 log₁₀(P_signal / P_noise) relative to original signal power. Accuracy degradation curves across SNR levels quantified the model's ability to extract directional information under increasing noise corruption, providing insight into operational limits.

## Statistical Analysis

Statistical significance was assessed across n = 5 independent training runs initialized with different random seeds (42, 1, 2, 3, 4). For each experimental condition, we report mean accuracy ± standard deviation. This repeated measurement protocol accounts for variability due to weight initialization and minibatch sampling order.

Pairwise comparisons between the full model and each ablation condition used two-sided independent-samples t-tests with significance threshold α = 0.001. The stringent significance threshold accounts for multiple comparisons while maintaining adequate power given the large effect sizes observed.

Effect sizes were quantified using Cohen's d, computed as the difference in means divided by the pooled standard deviation. Effect sizes exceeding 0.8 are conventionally considered large; all reported comparisons exhibited effect sizes substantially exceeding this threshold, indicating practically significant performance differences.

## Data and Code Availability

The Speech260 dataset and pre-trained model weights are available upon reasonable request to the corresponding author. Source code for data preprocessing, model training, evaluation, and figure generation will be made available at a public repository upon publication. All experiments can be reproduced using the provided configuration files and random seed settings.
