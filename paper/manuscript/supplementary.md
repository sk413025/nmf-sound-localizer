# Supplementary Information

**Non-contact acoustic direction sensing via physical encoding in everyday objects**

This Supplementary Information gives the derivation chain behind the manuscript's decoding model in the same order a reader would encounter it physically: from the continuous plate equation to Green's function, modal superposition, ideal transfer matrix, low-rank interpretation, sparse direction-grid surrogate, and finally the deep-unrolled routed solver. Three objects are kept distinct throughout: the ideal complex response \(\mathcal H\) from the continuous physical model, the executed standardized fingerprint dictionary \(H=[h_1,\dots,h_E]\), and the centered-magnitude analysis matrix \(H_{\mathrm{fig}}\) used for Fig. 2.

## Supplementary Methods 1. Continuous physical model: Kirchhoff-Love operator, Green's function, modal decomposition, and ideal transfer matrix

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

where \(D_p\) is the bending stiffness, \(\rho t\) is the areal mass density, and \(c_d\) is an effective damping term. This operator gives a concrete starting point for the derivation chain, although the manuscript requires only that the response be approximately linear and time-invariant over the analysis window.

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

If only \(R\) structural modes contribute appreciably in the analysis band, the response admits the modal expansion

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

The manuscript's Eq. 1 is the compact notation

$$
Y(\omega;\theta) \approx \sum_{m=1}^{R} s_m(\omega)\,\alpha_m(\theta),
\tag{S6}
$$

where the explicit \(\omega\)-dependence in the coupling term has been suppressed into a compact notation. In words, the measured single-point spectrum is a superposition of modal spectral fingerprints \(s_m(\omega)\), each weighted by a direction-dependent participation factor \(\alpha_m(\theta,\omega)\).

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

The singular value decomposition of \(\mathcal H\) is therefore an orthogonal re-expression of the same modal structure only at this approximate level. This low-rank physical picture motivates the later reduced-order view, but sparsity enters only after the discrete angle-grid surrogate is introduced in Supplementary Methods 2.

## Supplementary Methods 2. Executed observable construction, standardized dictionary \(H\), centered-magnitude SVD, and reduced-order sparse surrogate

The measured waveform at angle index \(e\) and trial index \(n\) is the single-point LDV velocity signal \(v_{e,n}(t)\). After short-time Fourier transformation, the complex coefficient at frequency bin \(k\) and frame \(t\) is

$$
V_{e,n}[k,t] = \mathrm{STFT}\{v_{e,n}\}[k,t].
\tag{S10}
$$

The time-averaged power statistic used in the paper is

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

The executed angle-indexed calibration dictionary is the mean standardized fingerprint at each angle,

$$
h_e = \frac{1}{N_e}\sum_{n\in\mathcal C_e}\tilde y_{e,n} \in \mathbb{R}^F,
\qquad
H=[h_1,\dots,h_E]\in\mathbb{R}^{F\times E},
\tag{S15}
$$

where \(\mathcal C_e\) is the set of calibration trials at angle \(e\). This \(H\) is the experimental dictionary used for downstream analysis and inference. It is not the ideal complex matrix \(\mathcal H\) from Supplementary Methods 1.

Figure 2 does not decompose \(\mathcal H\) directly. Matching the committed Fig. 2 generator, it analyzes the centered-magnitude matrix

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

The early saturation reported in Fig. 2 is therefore an empirical property of \(H_{\mathrm{fig}}\), after the nonlinear steps \(\mathcal H \mapsto |\,\mathcal H\,|\), trial averaging, log compression, and row-wise centering.

The low-rank picture motivates a reduced-order surrogate, but sparsity enters only after introducing a discrete single-source angle-grid approximation. In full standardized feature space, that surrogate can be written as

$$
\tilde y \approx Hx,
\qquad
\|x\|_0 \le K,
\tag{S18}
$$

where a dominant coefficient identifies the source angle and any additional support captures local directional overlap or noise. Operationally, \(K\) is the residual-correction budget or pursuit depth, not a claim that the physical transfer matrix itself is sparse.

For exposition around the manuscript's Eq. 2, one may further project the standardized fingerprint and calibration dictionary into a retained singular subspace,

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

The reported runtime path does not apply PCA/SVD before decoding and instead operates directly in the full \(F=346\) standardized feature space.

## Supplementary Methods 3. Analytical hard-OMP recursion for the reduced-order surrogate

The exact analytical hard-OMP baseline is defined for the reduced-order surrogate (S20). Let the initial residual and support be

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

Equations (S21)-(S27) define the exact hard-OMP recursion for the reduced-order surrogate. The displayed Fig. 3 panels are slightly different: they use a correlation-based greedy / OMP-family diagnostic on the same calibration dictionary rather than a literal panel-by-panel visualization of every least-squares refit step. Concretely, the plotted score aggregates \(|\langle y,d_{e,m}\rangle|\) within each direction group.

## Supplementary Methods 4. Deep unrolling and attention-gated routed pursuit in full standardized feature space

The guided solver studied in the manuscript operates in the full standardized feature space with a grouped dictionary

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

The learned routing branch does not replace (S30); it gates it. Let \(q_{t,e,m}^{(\mathrm{QK})}\) denote the atom-level query-key score produced from the current residual and dictionary tokens. Aggregating those atom-level scores within each direction yields expert-level routing scores \(s_t^{(\mathrm{exp})}[e]\). In the implementation used here, the default expert aggregation is an L2 norm over the per-direction atom scores. Direction-level routing weights are then obtained from a Gumbel-family gating rule,

$$
w_t = \mathrm{GumbelSoftmax}\!\left(s_t^{(\mathrm{exp})};\tau\right),
\qquad
w_t \in \Delta^{E-1}.
\tag{S31}
$$

Within each selected direction group, an atom-level gate \(u_t^{(e)}\in\Delta^{M-1}\) is formed analogously. The combined gate is therefore

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

where \(\eta\) is a learned step size. These equations define the deep-unrolled residual-correction scaffold: each stage computes a physical correlation, routes that evidence across the angle manifold, applies a gated coefficient update, and then redefines the residual for the next stage.

Training and evaluation do not decode direction from \(\|x_K^{(e)}\|_2\). Instead they read out expert-level routing scores. Let

$$
\bar s[e] = \frac{1}{K_{\mathrm{sup}}}\sum_{t=1}^{K_{\mathrm{sup}}} s_t^{(\mathrm{exp})}[e],
\qquad
\hat\theta = \theta_{\arg\max_e \bar s[e]},
\tag{S35}
$$

where the reported configuration uses \(K_{\mathrm{sup}}=1\), so the readout is the first-stage expert score. The composite training objective is

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

Equation S36 gives the main loss. In the cited primary guided-solver run, the implementation also includes an auxiliary early-epoch teacher-warmup cross-entropy addend for the first 10 epochs. The routed solver is OMP-inspired rather than a literal least-squares unrolling of Supplementary Methods 3. The `use_hard_gumbel` switch is optional in the codebase; in the cited primary guided-solver run it is enabled, so the forward routing step uses the straight-through hard-Gumbel variant within the broader Gumbel-family formulation above.

## Supplementary Methods 5. Discriminability and similarity-statistic definitions

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
