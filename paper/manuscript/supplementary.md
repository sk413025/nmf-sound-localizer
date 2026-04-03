# Supplementary Information

**Object-specific directional readout in calibrated passive objects**

Sound direction enters the measured LDV spectrum through the object's continuous structural response, and the following derivations trace how that response becomes a standardized directional fingerprint and a local-overlap decoding problem in the acrylic reference-object study and its object-specific extensions. The reduced-order surrogate isolates the local angle geometry, whereas the guided solver writes that local continuity into staged residual updates in the full standardized feature space and the OMP baseline provides the hard-selection comparison.

## Supplementary Methods 1. Single-spectrum response from the continuous plate model

Incidence direction enters the single-LDV spectrum in the idealized plate model through the continuous structural response.

Under small-amplitude linear structural dynamics, the out-of-plane displacement field \(W(x,y,\omega;\theta)\) at angular frequency \(\omega\) and incidence direction \(\theta\) satisfies a linear frequency-domain operator equation

$$
\mathcal L_\omega W(\cdot,\cdot,\omega;\theta) = P(\cdot,\cdot,\omega;\theta),
\tag{S1}
$$

where \(P\) is the effective distributed loading induced by the incident sound field and \(\mathcal L_\omega\) collects geometry, material parameters, boundary conditions, and damping.

For a thin plate under Kirchhoff-Love assumptions, one representative operator is

$$
\mathcal L_\omega
=
D_p\nabla^4 - \rho t\,\omega^2 + i\omega c_d,
\tag{S2}
$$

where \(D_p\) is the bending stiffness, \(\rho t\) is the areal mass density, and \(c_d\) is an effective damping term. This operator provides one concrete starting point for the derivation and motivates the approximate linear time-invariant response used over the analyzed window.

For a fixed LDV measurement point \((x_L,y_L)\), the single-point velocity response is

$$
Y(\omega;\theta) = V(x_L,y_L,\omega;\theta) = i\omega\,W(x_L,y_L,\omega;\theta).
\tag{S3}
$$

With Green's function \(G_\omega\), the same response can be written as

$$
Y(\omega;\theta)
=
i\omega
\iint_\Omega
G_\omega\!\big((x_L,y_L),(x',y')\big)\,
P(x',y',\omega;\theta)\,dA'.
\tag{S4}
$$

Equation (S4) makes the forward map explicit: direction changes the effective forcing pattern \(P\), and the structure converts that forcing into a single measured spectrum through the Green's kernel seen by the LDV point.

When \(R\) structural modes contribute appreciably in the analysis band, the response admits the modal expansion

$$
Y(\omega;\theta)
\approx
\sum_{m=1}^{R}
\underbrace{
\frac{i\omega\,\phi_m(x_L,y_L)}
{m_m\left(\omega_m^2-\omega^2+i2\zeta_m\omega_m\omega\right)}
}_{s_m(\omega)}
\underbrace{
\iint_\Omega \phi_m(x,y)\,P(x,y,\omega;\theta)\,dA
}_{\alpha_m(\theta,\omega)}.
\tag{S5}
$$

Suppressing the explicit \(\omega\)-dependence in the coupling term gives the compact form used in Eq. (1) of the main text,

$$
Y(\omega;\theta) \approx \sum_{m=1}^{R} s_m(\omega)\,\alpha_m(\theta),
\tag{S6}
$$

In words, the measured single-point spectrum is a superposition of modal spectral fingerprints \(s_m(\omega)\), each weighted by a direction-dependent participation factor \(\alpha_m(\theta,\omega)\).

Sampling frequency at \(\omega_1,\dots,\omega_F\) and direction at \(\theta_1,\dots,\theta_E\) gives the ideal complex transfer matrix

$$
\mathcal H_{k,e} = Y(\omega_k;\theta_e).
\tag{S7}
$$

Using (S5), this sampled matrix obeys

$$
\mathcal H_{k,e} = \sum_{m=1}^{R} s_m(\omega_k)\,\alpha_m(\theta_e,\omega_k).
\tag{S8}
$$

If the modal couplings are approximately separable or vary only slowly with frequency over the retained band, so that \(\alpha_m(\theta_e,\omega_k)\approx a_m(\theta_e)\), then

$$
\mathcal H \approx S A^\top,
\qquad
S_{k,m}=s_m(\omega_k),
\qquad
A_{e,m}=a_m(\theta_e),
\qquad
\mathrm{rank}(\mathcal H)\lesssim R.
\tag{S9}
$$

The singular value decomposition of \(\mathcal H\) is therefore an orthogonal re-expression of the same modal structure at this approximate level. This low-rank physical picture motivates the later reduced-order view, and sparsity enters after the discrete angle-grid surrogate introduced in Supplementary Methods 2.

## Supplementary Methods 2. Standardized fingerprints, \(H\), and the Fig. 2 reduced view

The experiments use a standardized fingerprint dictionary \(H\) built from measured single-point spectra, and Fig. 2 summarizes the centered-magnitude matrix \(H_{\mathrm{fig}}\) derived from that dictionary.

The measured waveform at angle index \(e\) and trial index \(n\) is the single-point LDV velocity signal \(v_{e,n}(t)\). After short-time Fourier transformation, the complex coefficient at frequency bin \(k\) and frame \(t\) is

$$
V_{e,n}[k,t] = \mathrm{STFT}\{v_{e,n}\}[k,t].
\tag{S10}
$$

The analysis uses the time-averaged power statistic

$$
\widehat{S}_{e,n}[k] = \frac{1}{T_{e,n}} \sum_{t=1}^{T_{e,n}} \left|V_{e,n}[k,t]\right|^2,
\tag{S11}
$$

followed by a log-power transform

$$
y_{e,n}[k] = \log_{10}\!\left(\widehat{S}_{e,n}[k] + \epsilon\right).
\tag{S12}
$$

The analysis band is restricted to \(300\text{--}3{,}000\) Hz, giving \(F=346\) retained frequency bins in this study.

Using the white-noise calibration set \(\mathcal C\), we compute a per-frequency calibration mean and scale,

$$
\mu[k] = \frac{1}{|\mathcal C|} \sum_{(e,n)\in\mathcal C} y_{e,n}[k],
\qquad
\sigma[k] = \sqrt{\frac{1}{|\mathcal C|-1}\sum_{(e,n)\in\mathcal C}\!\left(y_{e,n}[k]-\mu[k]\right)^2 + \epsilon_\sigma},
\tag{S13}
$$

and define the standardized fingerprint

$$
\tilde y_{e,n}[k] = \frac{y_{e,n}[k]-\mu[k]}{\sigma[k]}.
\tag{S14}
$$

The angle-indexed calibration dictionary is the mean standardized fingerprint at each angle,

$$
h_e = \frac{1}{N_e}\sum_{n\in\mathcal C_e}\tilde y_{e,n} \in \mathbb{R}^F,
\qquad
H=[h_1,\dots,h_E]\in\mathbb{R}^{F\times E},
\tag{S15}
$$

where \(\mathcal C_e\) is the set of calibration trials at angle \(e\). This \(H\) is the experimental dictionary used for downstream analysis and inference. It is built from standardized measured fingerprints, whereas \(\mathcal H\) from Supplementary Methods 1 is the ideal complex transfer matrix.

To isolate angle-dependent structure from shared magnitude offsets, we analyze the centered-magnitude matrix

$$
H_{\mathrm{fig}}[k,e]
=
|H[k,e]|
-
\frac{1}{E}\sum_{e'=1}^{E}|H[k,e']|,
\tag{S16}
$$

whose singular value decomposition is

$$
H_{\mathrm{fig}} = U\Sigma V^\top.
\tag{S17}
$$

The early saturation in Fig. 2 is therefore an empirical property of \(H_{\mathrm{fig}}\), after the nonlinear steps \(\mathcal H \mapsto |\,\mathcal H\,|\), trial averaging, log compression, and row-wise centering. This compact measured structure motivates a local-overlap decoding picture rather than a search over unrelated angle templates.

Calibration turns the smooth physical response in Eq. (S6) into an angle-indexed dictionary of measured fingerprints and carries the local angle ordering into measured space. In the full standardized feature space, a held-out fingerprint can be written as

$$
\tilde y \approx Hx,
\qquad
\|x\|_0 \le K,
\tag{S18}
$$

where \(x\) is the surrogate coefficient vector, a dominant coefficient marks the source angle, and any additional support captures overlap among nearby calibrated directions or residual noise. Operationally, \(K\) is the residual-correction budget or pursuit depth. Equation (S18) therefore expresses the compact modal picture as a discrete local-overlap model on the calibrated angle grid.

Projecting the same local-overlap surrogate into a retained singular subspace gives

$$
z = U_r^\top \tilde y,
\qquad
A = U_r^\top H,
\tag{S19}
$$

giving the reduced-order surrogate

$$
z \approx A x,
\qquad
\|x\|_0 \le K.
\tag{S20}
$$

This projected form makes the reduced-order geometry behind Eq. 2 explicit. It is a bridge for describing why nearby calibrated directions compete locally, not the implemented inference surface itself. The decoder used in Figs. 4 and 5 operates in the full \(F=346\) standardized feature space on \(\tilde y\), \(H\), and the grouped dictionary \(D\) introduced below, rather than on the projected pair \((z, A)\).

## Supplementary Methods 3. Hard OMP on the reduced-order surrogate

On the reduced-order surrogate, hard OMP iteratively selects the strongest direction on the calibrated angle grid and then refits on the active support. The exact greedy recursion is

$$
r_0 = z,
\qquad
\mathcal S_0 = \varnothing.
\tag{S21}
$$

For general non-unit-norm columns, the correlation score can be written as

$$
c_e(r) = \frac{|\langle a_e,r\rangle|}{\|a_e\|_2+\varepsilon_c}.
\tag{S22}
$$

If the columns are unit-normalized, (S22) reduces to the raw inner-product magnitude. At stage \(t\), hard-OMP selects

$$
j_t = \arg\max_{e\in\{1,\dots,E\}} c_e(r_{t-1}),
\qquad
\mathcal S_t = \mathcal S_{t-1}\cup\{j_t\}.
\tag{S23}
$$

The coefficient estimate on the active support is the least-squares solution

$$
x_{\mathcal S_t}^{(t)}
=
\arg\min_{u\in\mathbb{R}^{|\mathcal S_t|}} \|z-A_{\mathcal S_t}u\|_2^2
=
\left(A_{\mathcal S_t}^\top A_{\mathcal S_t}\right)^{-1}A_{\mathcal S_t}^\top z,
\tag{S24}
$$

with all inactive entries set to zero. The updated residual is

$$
r_t = z - A_{\mathcal S_t}x_{\mathcal S_t}^{(t)}.
\tag{S25}
$$

By the normal equations for (S24),

$$
A_{\mathcal S_t}^\top r_t = 0,
\tag{S26}
$$

so the residual is orthogonal to the currently selected span. After \(K\) stages, the angle prediction is

$$
\hat\theta = \theta_{\arg\max_e |x_e^{(K)}|}.
\tag{S27}
$$

Equations (S21)-(S27) define the exact hard-OMP recursion for the reduced-order surrogate \((z,A,x)\). Figure 3 does not plot that full recursion. Instead, it isolates the first discrete commitment in the full standardized space: starting from \(r_0=\tilde y\), the plotted stage-0 score aggregates the grouped inner-product magnitudes \(|g_0[e,m]| = |\langle \tilde y,d_{e,m}\rangle|\) within each direction group. When that stage-0 score concentrates near the matched direction, the current fingerprint remains locally separable; when it spreads across neighboring groups, the fingerprint becomes locally ambiguous and an immediate first choice becomes unstable on held-out speech. The noise-response curves in Fig. 3f use that same stage-0 diagnostic but remain a separate surface from Fig. 3d: the white-noise branch is recomputed on synthetic noisy white-noise datasets, whereas the speech branch comes from a separate five-seed speech-plus-babble sweep.

## Supplementary Methods 4. Routed updates in the full standardized fingerprint space

The guided solver is a physics-guided residual-correction readout with learned local gating. It operates directly in the full standardized feature space and replaces immediate one-angle commitment with learned local gating. Its role is narrow: preserve the contiguous angle neighborhood implied by the calibrated dictionary long enough for subtraction to act on a physically plausible local support. At each stage, the solver scores the current residual against the physical dictionary, routes that evidence across nearby direction groups, applies a gated update, and then recomputes the residual. Relative to Supplementary Methods 3, the residual-correction scaffold stays the same; what changes is the selection rule.

The guided solver uses a grouped dictionary

$$
D=[d_{e,m}] \in \mathbb{R}^{F\times (EM)},
\tag{S28}
$$

whose columns are partitioned by direction \(e\in\{1,\dots,E\}\) and source-atom index \(m\in\{1,\dots,M\}\). The coefficient vector is partitioned accordingly as \(x_t=\{x_t^{(e,m)}\}\), and the initialization is

$$
r_0=\tilde y,
\qquad
x_0=0.
\tag{S29}
$$

At stage \(t\), the physics-consistent match score is the correlation between the current residual and the grouped dictionary:

$$
g_t = D^\top r_t.
\tag{S30}
$$

Unlike the reduced-order hard-OMP recursion in Supplementary Methods 3, the routed solver does not commit to one support element and refit. Instead it distributes weight across a local neighborhood of angle groups before subtraction.

The learned routing branch preserves the physical match score in (S30) and redistributes that evidence across nearby directions. Its scientific job is to keep the measured local continuity active before subtraction. The model first computes atom-level compatibility scores from the current residual and the dictionary tokens. It then pools those atom-level scores within each direction to produce one routing score \(s_t^{(\mathrm{exp})}[e]\) per direction. In the reported implementation, that pooling step is an L2 norm over the per-direction atom scores. The implementation then converts the direction-level scores into routing weights with a Gumbel-family gating rule,

$$
w_t = \mathrm{GumbelSoftmax}\!\left(s_t^{(\mathrm{exp})};\tau\right),
\qquad
w_t \in \Delta^{E-1}.
\tag{S31}
$$

Within each selected direction group, the model forms an atom-level gate \(u_t^{(e)}\in\Delta^{M-1}\) in the same way. The combined gate is therefore

$$
W_t[e,m] = w_t[e]\,u_t^{(e)}[m].
\tag{S32}
$$

The routed sparse update is

$$
\Delta x_t[e,m] = W_t[e,m]\; g_t[e,m],
\tag{S33}
$$

and the coefficient and residual recursions are

$$
x_{t+1} = x_t + \eta\,\Delta x_t,
\qquad
r_{t+1} = r_t - D(\eta\,\Delta x_t) = \tilde y - D x_{t+1},
\tag{S34}
$$

where \(\eta\) is a learned step size. These equations define the staged residual-correction scaffold: each stage computes a physical correlation, routes that evidence across the local angle ordering, applies a gated coefficient update, and then redefines the residual for the next stage.

Training and evaluation do not decode direction from \(\|x_K^{(e)}\|_2\). Instead they read out expert-level routing scores. Let

$$
\bar s[e] = \frac{1}{K_{\mathrm{sup}}}\sum_{t=1}^{K_{\mathrm{sup}}} s_t^{(\mathrm{exp})}[e],
\qquad
\hat\theta = \theta_{\arg\max_e \bar s[e]},
\tag{S35}
$$

where \(K_{\mathrm{sup}}=1\) in the configuration studied here, so the readout is the first-stage expert score. The composite training objective is

$$
\mathcal L = \alpha\,\mathcal L_{\mathrm{rec}} + \beta\,\mathcal L_{\mathrm{mono}} + \gamma\,\mathcal L_{\mathrm{cls}},
\qquad
(\alpha,\beta,\gamma)=(1.0,\,0.2,\,0.5),
\tag{S36}
$$

with

$$
\mathcal L_{\mathrm{rec}} = \|r_K\|_2^2,
\qquad
\mathcal L_{\mathrm{cls}} = \mathrm{CE}(\bar s,e^\star),
\tag{S37}
$$

and stagewise monotonicity regularizer

$$
\mathcal L_{\mathrm{mono}}
=
\sum_{t=0}^{K-2}
\max\!\left(0,\|r_{t+1}\|_2-\|r_t\|_2\right).
\tag{S38}
$$

Equation (S36) gives the main loss. During early training, an auxiliary cross-entropy term is added for the first 10 epochs to stabilize expert assignment before the full routed objective dominates. The routed solver keeps the OMP residual-correction scaffold but replaces exact least-squares refitting with learned local routing. In the reported implementation, the local gate uses a transformer parameterization together with a straight-through Gumbel approximation; their manuscript-facing role is to preserve local physical continuity before subtraction rather than to define a separate black-box inference problem.

## Supplementary Methods 5. Fig. 2 and Fig. 3 summary statistics

Figs. 2 and 3 summarize angle separability and similarity with the following descriptive statistics.

Let \(\tilde y_{e,n}\in\mathbb{R}^F\) denote the standardized fingerprint of trial \(n\) at angle \(e\). For any two vectors \(u,v\in\mathbb{R}^F\), the Pearson correlation coefficient is

$$
\rho(u,v)
=
\frac{\sum_{k=1}^{F}(u_k-\bar u)(v_k-\bar v)}
{\sqrt{\sum_{k=1}^{F}(u_k-\bar u)^2}\sqrt{\sum_{k=1}^{F}(v_k-\bar v)^2}},
\tag{S39}
$$

where \(\bar u\) and \(\bar v\) are sample means across frequency bins.

For angle \(e\), the within-angle correlation set is

$$
\mathcal W_e
=
\left\{
\rho(\tilde y_{e,n},\tilde y_{e,n'})
\;:\;
n<n'
\right\},
\tag{S40}
$$

and the corresponding mean is

$$
\bar r_{\mathrm{within}}(e) = \frac{1}{|\mathcal W_e|}\sum_{\rho\in\mathcal W_e}\rho.
\tag{S41}
$$

The between-angle correlation set anchored at angle \(e\) is

$$
\mathcal B_e
=
\left\{
\rho(\tilde y_{e,n},\tilde y_{e',n'})
\;:\;
e'\neq e
\right\},
\tag{S42}
$$

with mean

$$
\bar r_{\mathrm{between}}(e) = \frac{1}{|\mathcal B_e|}\sum_{\rho\in\mathcal B_e}\rho.
\tag{S43}
$$

The per-angle discriminability margin plotted in Fig. 3c is

$$
\Delta r(e) = \bar r_{\mathrm{within}}(e) - \bar r_{\mathrm{between}}(e).
\tag{S44}
$$

The pooled violin-plot summaries in Fig. 3a,b are obtained by aggregating all elements of \(\mathcal W_e\) and \(\mathcal B_e\) over angles. For two pooled correlation samples \(X\) and \(Y\), Cohen's \(d\) is

$$
d = \frac{\bar X-\bar Y}{s_p},
\qquad
s_p^2 =
\frac{(n_X-1)s_X^2 + (n_Y-1)s_Y^2}{n_X+n_Y-2}.
\tag{S45}
$$

The reported \(P\)-values come from the two-sided Mann-Whitney \(U\) test applied to the same pooled samples. The inter-angle similarity statistic used for Fig. 2f is the correlation matrix of the angle-indexed prototypes,

$$
S_{e,e'} = \rho(h_e,h_{e'}).
\tag{S46}
$$

These are descriptive statistics on the executed standardized fingerprints. They summarize directional separability in the experimentally constructed feature space and do not by themselves constitute a first-principles identifiability theorem.
