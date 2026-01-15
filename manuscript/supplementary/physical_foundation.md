# Supplementary Note: Physical Foundation of the Decoder Architecture

This supplementary note provides the complete mathematical derivation from the wave equation governing plate vibration to the neural network architecture used for direction estimation. The derivation establishes that the decoder architecture is not an arbitrary neural network design but emerges as a principled solution to the physics-constrained inverse problem.

## 1. Problem Definition

The experimental configuration involves:
- **Input**: Incident acoustic wave from direction (θ, φ), creating a spatial pressure distribution P(x, y; θ, φ, ω) across the plate surface
- **System**: A thin vibrating plate governed by linear elasticity (small-amplitude vibration assumption)
- **Output**: Point measurement of plate velocity W(x₀, y₀, ω) at the LDV sensor location

The goal is to derive the mapping from incident direction to measured spectrum, then invert this mapping using a physics-aware architecture.

## 2. Governing Equation: Kirchhoff-Love Plate Theory

For a thin plate under the Kirchhoff-Love assumptions (plane sections remain plane, shear deformation negligible), the frequency-domain governing equation is:

$$
\mathcal{L}_\omega W(x, y, \omega) = P(x, y; \theta, \phi, \omega)
\tag{1}
$$

where the linear operator is:

$$
\mathcal{L}_\omega = D\nabla^4 - \rho h \omega^2 + i\omega c_d
$$

with:
- D = Eh³/[12(1-ν²)]: flexural rigidity
- E: Young's modulus
- h: plate thickness
- ν: Poisson's ratio
- ρ: density
- c_d: damping coefficient

**Key insight**: Because the system is linear (small-amplitude vibration), the mapping from input P to output W is necessarily linear. This linearity is the foundation for the transfer matrix representation.

## 3. Green's Function Formulation

For any linear system, we can define a Green's function G satisfying:

$$
\mathcal{L}_\omega G((x, y), (x', y'), \omega) = \delta(x - x')\delta(y - y')
\tag{2}
$$

The solution at any point is then the convolution:

$$
W(x_0, y_0, \omega) = \iint_\Omega G((x_0, y_0), (x', y'), \omega) \cdot P(x', y'; \theta, \phi, \omega) \, dA'
\tag{3}
$$

This integral representation is the continuous analog of the matrix equation **y** = **H** **x** that will emerge after discretization.

## 4. Modal Decomposition

The plate's response can be expanded in terms of its natural vibration modes. The eigenvalue problem for free vibration is:

$$
D\nabla^4 \Phi_r(x, y) = \rho h \omega_r^2 \Phi_r(x, y)
\tag{4}
$$

with orthonormality:

$$
\iint_\Omega \rho h \, \Phi_r \Phi_s \, dA = m_r \delta_{rs}
\tag{5}
$$

The displacement field is expanded as:

$$
W(x, y, \omega) = \sum_r \Phi_r(x, y) \cdot q_r(\omega)
\tag{6}
$$

**Physical interpretation**: Any vibration pattern is a superposition of the structure's natural mode shapes Φ_r, weighted by modal coordinates q_r that depend on frequency and excitation.

## 5. Modal Participation Factor (Spatial Matching)

Substituting the modal expansion into the governing equation and using orthogonality, each modal coordinate satisfies:

$$
q_r(\omega) = \frac{F_r(\omega)}{m_r(\omega_r^2 - \omega^2 + i 2\zeta_r \omega_r \omega)}
\tag{7}
$$

where the **modal force** (generalized force) is:

$$
F_r(\omega; \theta, \phi) = \iint_\Omega \Phi_r(x, y) \cdot P(x, y; \theta, \phi, \omega) \, dA
\tag{8}
$$

**Key insight**: The modal force F_r is an inner product between the pressure field P and the mode shape Φ_r. This spatial matching determines how efficiently each mode is excited by sound from a given direction. The direction-dependence of P(x, y; θ, φ) creates direction-dependent modal excitation, which is the physical basis for direction encoding.

## 6. Point Measurement

The measured response at the sensor location (x₀, y₀) is:

$$
Y(\omega; \theta, \phi) = W(x_0, y_0, \omega) = \sum_r \Phi_r(x_0, y_0) \cdot q_r(\omega)
\tag{9}
$$

Substituting Equation (7):

$$
Y(\omega; \theta, \phi) = \sum_r \underbrace{\frac{\Phi_r(x_0, y_0)}{m_r(\omega_r^2 - \omega^2 + i 2\zeta_r \omega_r \omega)}}_{\text{modal frequency response}} \cdot \underbrace{F_r(\omega; \theta, \phi)}_{\text{spatial matching}}
\tag{10}
$$

This equation shows that the measured spectrum is determined by:
1. **Modal frequency response**: Peaks at resonant frequencies ω ≈ ω_r
2. **Spatial matching**: Direction-dependent excitation via F_r

## 7. Transfer Matrix Construction

Discretizing frequency (M bins) and direction (N angles), we form vectors:

$$
\mathbf{y}_j = \begin{bmatrix} Y(\omega_1; \theta_j, \phi_j) \\ \vdots \\ Y(\omega_M; \theta_j, \phi_j) \end{bmatrix} \in \mathbb{C}^M
\tag{11}
$$

The transfer matrix collects all directions:

$$
\mathbf{H} = [\mathbf{y}_1, \mathbf{y}_2, \ldots, \mathbf{y}_N] \in \mathbb{C}^{M \times N}
\tag{12}
$$

For a single active direction with unit energy:

$$
\mathbf{y} = \mathbf{H} \mathbf{x} + \mathbf{n}
\tag{13}
$$

where **x** ∈ ℝ^N is a one-hot vector indicating the source direction.

## 8. Low-Rank Structure from Modal Decomposition

Defining the modal frequency fingerprint vector:

$$
\mathbf{u}_r = \begin{bmatrix} \frac{\Phi_r(x_0, y_0)}{m_r(\omega_r^2 - \omega_1^2 + i 2\zeta_r \omega_r \omega_1)} \\ \vdots \\ \frac{\Phi_r(x_0, y_0)}{m_r(\omega_r^2 - \omega_M^2 + i 2\zeta_r \omega_r \omega_M)} \end{bmatrix} \in \mathbb{C}^M
\tag{14}
$$

and the spatial matching coefficient:

$$
b_{rj} = F_r(\theta_j, \phi_j)
\tag{15}
$$

The transfer matrix has the low-rank factorization:

$$
\mathbf{H} = \sum_{r=1}^R \mathbf{u}_r \mathbf{b}_r^T = \mathbf{U}_{\text{phys}} \mathbf{B}
\tag{16}
$$

where **U**_phys ∈ ℂ^{M×R} contains R modal fingerprints and **B** ∈ ℂ^{R×N} contains spatial matching coefficients.

**Key insight**: If only R << min(M, N) modes are significant, **H** is approximately rank-R. This low-rank structure is why SVD reveals rapid singular value decay.

## 9. SVD Analysis

The singular value decomposition:

$$
\mathbf{H} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^H = \sum_{i=1}^{\min(M,N)} \sigma_i \mathbf{u}_i \mathbf{v}_i^H
\tag{17}
$$

relates to the physical decomposition:
- **Physical basis** (Eq. 16): **H** = **U**_phys **B** (modes may not be orthogonal)
- **SVD basis** (Eq. 17): **H** = **U** **Σ** **V**^H (optimal orthogonal basis, energy-ordered)

The rapid decay of singular values σᵢ confirms the low-rank modal structure: only 10-20 effective modes capture most of the variance.

## 10. From Sparse Coding to Neural Network Architecture

### 10.1 The Inverse Problem

Given observation **y**, estimate direction by solving:

$$
\min_{\mathbf{x}} D_{\text{IS}}(\mathbf{y} \| \mathbf{D}\mathbf{x}) + \lambda \|\mathbf{x}\|_1
\tag{18}
$$

where D_IS is the Itakura-Saito divergence, appropriate for power spectra.

### 10.2 Multiplicative Update Rules

The IS-divergence admits multiplicative updates:

$$
\mathbf{z} \leftarrow \mathbf{z} \odot \frac{\mathbf{W}^T (\mathbf{y} / \hat{\mathbf{y}}^2)}{\mathbf{W}^T (1 / \hat{\mathbf{y}}) + \lambda}
\tag{19}
$$

where ⊙ denotes element-wise product and ŷ = **W** **z** is the current reconstruction.

### 10.3 Deep Unrolling

Each iteration of Equation (19) becomes one network layer:

$$
\mathbf{x}^{(t+1)} = \text{Layer}_t(\mathbf{x}^{(t)}, \mathbf{y}, \mathbf{D}; \theta_t)
$$

with learnable parameters θ_t replacing fixed algorithm constants.

### 10.4 Attention-Gated Routing

The multiplicative structure of Equation (19) motivates the gating mechanism:

$$
\Delta \mathbf{x} = \mathbf{w}_{\text{att}} \odot \mathbf{g}
\tag{20}
$$

where:
- **w**_att = softmax(**q** · **K**^T / √d_k): learned attention weights
- **g** = **D**^T **y**: physical correlation

This element-wise product ensures that updates require both:
1. High learned attention (data-driven relevance)
2. High physical correlation (physics-grounded support)

## Summary

The derivation chain is:

$$
\text{Wave Equation} \xrightarrow{\text{linearity}} \text{Green's function} \xrightarrow{\text{modal expansion}} \text{Low-rank } \mathbf{H} \xrightarrow{\text{sparse coding}} \text{NMF updates} \xrightarrow{\text{unrolling}} \text{Network layers}
$$

The neural network architecture is therefore not an arbitrary design but the mathematical consequence of:
1. Linear plate dynamics (Kirchhoff-Love equation)
2. Modal superposition principle
3. Low-rank structure from finite number of significant modes
4. IS-divergence optimization for sparse coding
5. Deep unrolling of iterative algorithms

This physics-grounded derivation ensures that the learned decoder respects the same structural constraints as the underlying acoustic-structural system.

## Correspondence Table

| Main Text Concept | Physical Symbol | Equation |
|-------------------|-----------------|----------|
| Transfer function matrix | **H** | (12) |
| Modal participation factor | F_r | (8) |
| Spectral fingerprint | **u**_r | (14) |
| Low-rank structure | **H** = **U**_phys **B** | (16) |
| Physical correlation | **g** = **D**^T **y** | (20) |
| Multiplicative gating | **w**_att ⊙ **g** | (20) |

## References

The derivation follows standard structural dynamics [@graff1975_wave_motion; @leissa1993_vibration_plates] combined with sparse coding theory [@fevotte2009_is] and algorithm unrolling [@monga2021_unrolling].
