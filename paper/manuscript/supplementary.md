# Supplementary Information

**A recurring directional code across passive objects**

The same directional code traced in the main text is developed here from a continuous single-point response into a measured fingerprint dictionary and then into two inference forms: a local-overlap surrogate whose classical hard-commitment limit fractures the local neighborhood and a grouped full-space surface on which that neighborhood can be preserved before subtraction. The final sections give the descriptive statistics behind Figs. 2 and 3 and the bounded cross-object interpretation behind Fig. 6.

## Supplementary Methods 1. Single-spectrum response from the continuous plate model

Passive structural vibration can carry a directional code that is compact and locally ordered at one fixed measurement point before any decoder enters the story. This section derives that single-point physical picture.

Under small-amplitude linear structural dynamics, the out-of-plane displacement field \(W(x,y,\omega;\theta)\) satisfies a linear frequency-domain operator equation for each angular frequency \(\omega\) and incidence direction \(\theta\),

$$
\mathcal L_\omega W(\cdot,\cdot,\omega;\theta) = P(\cdot,\cdot,\omega;\theta),
\tag{S1}
$$

where \(P\) is the effective distributed loading induced by the incident sound field. The operator \(\mathcal L_\omega\) collects geometry, material parameters, boundary conditions, and damping.

For a thin plate under Kirchhoff-Love assumptions, one representative operator is

$$
\mathcal L_\omega
=
D_p\nabla^4 - \rho t\,\omega^2 + i\omega c_d,
\tag{S2}
$$

where \(D_p\) is the bending stiffness, \(\rho t\) is the areal mass density, and \(c_d\) is an effective damping term. This operator gives one concrete starting point for the derivation. It also motivates the approximate linear time-invariant response used over the analyzed window.

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

In words, the single-point response spectrum is a superposition of modal spectral fingerprints \(s_m(\omega)\). Each fingerprint is weighted by a direction-dependent participation factor \(\alpha_m(\theta,\omega)\).

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

If the modal couplings are approximately separable or vary only slowly with frequency over the retained band, so that \(\alpha_m(\theta_e,\omega_k)\approx b_m(\theta_e)\), then

$$
\mathcal H \approx S B^\top,
\qquad
S_{k,m}=s_m(\omega_k),
\qquad
B_{e,m}=b_m(\theta_e),
\qquad
\mathrm{rank}(\mathcal H)\lesssim R.
\tag{S9}
$$

The singular value decomposition of \(\mathcal H\) is therefore an orthogonal re-expression of the same shared modal structure at this approximate level. Directional variation enters mainly through the weights on a limited response basis, so nearby angles remain related before any discrete decoder is imposed. Figure 1b gives the corresponding physical-principle schematic, and Supplementary Methods 2 follows the same structure through measurement, averaging, logarithmic compression, and standardization to obtain the empirical dictionary \(H\) analyzed in Figs. 1 and 2.

## Supplementary Methods 2. Measured fingerprints, local-overlap surrogates, and the Fig. 2 reduced view

Matched calibration carries the sampled response of Supplementary Methods 1 into an empirical dictionary of measured fingerprints on the same angle-frequency grid. The carried-over symbols are therefore the sampled direction index \(e\leftrightarrow\theta_e\), the sampled frequency index \(k\leftrightarrow\omega_k\), and the ideal sampled surface \(\mathcal H_{k,e}=Y(\omega_k;\theta_e)\).

The main text suppresses clip indices whenever one measured fingerprint is discussed in isolation. Here those indices are restored explicitly: \(n\) denotes repeated calibration trials at one sampled direction, and \(t\) denotes short-time Fourier frames within a trial. With that notation in place, this section writes the measured local-overlap geometry first in the full standardized feature space, where the fingerprints are naturally observed, and then in the reduced singular subspace used for the Fig. 2 view and the hard-commitment surrogate.

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

The calibrated dictionary \(H\) is the mean standardized fingerprint at each angle,

$$
h_e = \frac{1}{N_e}\sum_{n\in\mathcal C_e}\tilde y_{e,n} \in \mathbb{R}^F,
\qquad
H=[h_1,\dots,h_E]\in\mathbb{R}^{F\times E},
\tag{S15}
$$

where \(\mathcal C_e\) is the set of calibration trials at angle \(e\). This calibrated dictionary \(H\) is the empirical dictionary of standardized measured fingerprints that carries local angle ordering on the calibration grid. By contrast, \(\mathcal H\) from Supplementary Methods 1 is the ideal complex transfer matrix before measurement and nonlinear preprocessing.

In that sense, \(\mathcal H_{k,e}\) and \(H[k,e]\) live on the same sampled angle-frequency grid but not at the same descriptive level. The former is the ideal complex response at sampled \((\omega_k,\theta_e)\). The latter is the empirical mean fingerprint that remains after repeated trials, power averaging, logarithmic compression, and standardization have converted those sampled responses into measured fingerprints.

Written as a processing chain rather than an equality, the ideal sampled surface \(\mathcal H_{k,e}\) is carried into the calibrated dictionary \(H\) through repeated trials, time-frequency power statistics, logarithmic compression, and standardization:

$$
\mathcal H_{k,e}
\;\leadsto\;
V_{e,n}[k,t]
\;\leadsto\;
\widehat{S}_{e,n}[k]
\;\leadsto\;
y_{e,n}[k]
\;\leadsto\;
\tilde y_{e,n}[k]
\;\leadsto\;
h_e.
$$

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

The early saturation in Fig. 2 is therefore an empirical property of \(H_{\mathrm{fig}}\), after the nonlinear steps \(\mathcal H \mapsto |\,\mathcal H\,|\), trial averaging, log compression, and row-wise centering. The compactness seen in Fig. 2 is not a property of the ideal transfer matrix alone. It is the measured geometry that remains after those processing steps, and that measured geometry is what the local-overlap model must explain.

Calibration carries the smooth physical response in Eq. (S6) into a discrete dictionary of measured fingerprints while preserving the local angle ordering on the calibration grid. In the full standardized feature space, a held-out fingerprint can then be written as

$$
\tilde y \approx Hx,
\qquad
\|x\|_0 \le K,
\tag{S18}
$$

where \(x\) is the surrogate coefficient vector. A dominant coefficient marks the source angle, and any additional support captures overlap among nearby calibrated directions or residual noise. Operationally, \(K\) is the residual-correction budget or pursuit depth. Equation (S18) is therefore the full-space local-overlap model in the standardized fingerprint coordinates, not a choice among unrelated angle templates.

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

Equations (S18) and (S20) are not two different inference models. They are the same local-overlap surrogate written first in the full standardized coordinates \((\tilde y,H)\) and then in the retained singular coordinates \((z,A)\). The reduced pair \((z,A)\) makes the competition among nearby calibrated directions explicit and supplies the natural surface for the classical hard-commitment baseline in Supplementary Methods 3.

The full standardized surface remains the natural measurement space. For the readout analysis that follows, we enrich that same calibrated surface into a grouped dictionary by assigning each direction group \(e\) a small within-direction atom set indexed by \(m\),

$$
D=[d_{e,m}] \in \mathbb{R}^{F\times (EM)}.
\tag{S21}
$$

The grouped dictionary \(D\) lives on the same full standardized surface as \(\tilde y\), but it refines each direction group into \(M\) within-direction atoms instead of collapsing that group to one mean fingerprint. Supplementary Methods 3 uses this finer grouped surface only to expose the pre-refit failure at stage 0. Supplementary Methods 4 uses the same grouped surface for neighborhood-preserving updates.

The roles of \(H\) and \(D\) are therefore different. The matrix \(H=[h_1,\dots,h_E]\) averages the calibration fingerprints within each sampled angle and provides one empirical mean fingerprint \(h_e\) per direction, so it defines the coarse local-overlap geometry across the \(E\) sampled angles. The grouped dictionary \(D=[d_{e,m}]\) stays on that same \(F\)-dimensional standardized calibration surface but keeps a finer within-direction expansion by replacing each single mean fingerprint \(h_e\) with an \(M\)-atom set. In that sense, \(D\) is not a second measurement surface or a different physical model. It is a more resolved readout surface built from the same calibrated fingerprint space on which \(H\) was defined.

## Supplementary Methods 3. Hard OMP and stage-0 failure on the local-overlap surrogate

Supplementary Methods 3 begins from the reduced surrogate introduced in Supplementary Methods 2. Equations (S19)-(S20) rewrote the full standardized local-overlap model on the retained singular subspace as the pair \((z,A)\), with \(z=U_r^\top\tilde y\) and \(A=U_r^\top H\). This reduced view does not define a second inference model. It is the same calibrated local-overlap geometry written in the coordinates that make competition among nearby directions easiest to see. Hard OMP is written on that reduced surrogate because premature single-angle commitment is most transparent there. The grouped full-space dictionary \(D=[d_{e,m}]\) is carried forward only afterward, when we return to the full standardized surface to inspect the same failure before any refit has altered the residual.

$$
\rho_0 = z,
\qquad
\mathcal S_0 = \varnothing.
\tag{S22}
$$

Let \(a_e\) denote the \(e\)-th column of \(A\). For general non-unit-norm columns, the correlation score can then be written as

$$
c_e(\rho) = \frac{|\langle a_e,\rho\rangle|}{\|a_e\|_2+\varepsilon_c}.
\tag{S23}
$$

If the columns are unit-normalized, (S23) reduces to the raw inner-product magnitude. At stage \(t\), hard OMP selects

$$
j_t = \arg\max_{e\in\{1,\dots,E\}} c_e(\rho_{t-1}),
\qquad
\mathcal S_t = \mathcal S_{t-1}\cup\{j_t\}.
\tag{S24}
$$

The coefficient estimate on the active support is the least-squares solution

$$
x_{\mathcal S_t}^{(t)}
=
\arg\min_{u\in\mathbb{R}^{|\mathcal S_t|}} \|z-A_{\mathcal S_t}u\|_2^2
=
\left(A_{\mathcal S_t}^\top A_{\mathcal S_t}\right)^{-1}A_{\mathcal S_t}^\top z,
\tag{S25}
$$

with all inactive entries set to zero. The updated residual is

$$
\rho_t = z - A_{\mathcal S_t}x_{\mathcal S_t}^{(t)}.
\tag{S26}
$$

By the normal equations for (S25),

$$
A_{\mathcal S_t}^\top \rho_t = 0,
\tag{S27}
$$

so the residual is orthogonal to the currently selected span. After \(K\) stages, the angle prediction is

$$
\hat\theta = \theta_{\arg\max_e |x_e^{(K)}|}.
\tag{S28}
$$

Equations (S22)-(S28) therefore isolate the classical hard-commitment limit of the same reduced surrogate introduced in Supplementary Methods 2. The reduced coordinates make the local competition explicit, but they do not remove the shared neighborhood structure inherited from the calibrated dictionary. The failure comes from the commitment rule: selection and orthogonalization occur too early, before locally shared evidence has been consolidated across neighboring directions. Once one direction is chosen, evidence that is physically shared across that neighborhood is forced into the reduced residual \(\rho_t\) rather than retained as a coherent local band.

That same failure is already visible before the first refit. We therefore return from the reduced surrogate \((z,A)\) to the grouped full standardized surface built in Supplementary Methods 2 and inspect the grouped evidence before any support update has acted. We denote this pre-update grouped match by

$$
g_0 = D^\top \tilde y,
\tag{S29}
$$

which is the \(t=0\) case of the same grouped correlation surface that drives the routed updates below. The direction-level form of this pre-update grouped match, used in Fig. 3, is

$$
g_0^{(\mathrm{grp})}[e]
=
\sum_{m=1}^{M}
\left|g_0[e,m]\right|.
$$

When that diagnostic concentrates near one direction, local separability remains high. When it spreads across neighboring groups, the fingerprint remains locally ordered but local separability weakens. The directional code is still present. What fails is the rule that one direction should be chosen before that shared neighborhood evidence has been pooled.

This pre-update grouped match is the quantity behind the speech-side local separability results in Fig. 3d-f. Figure 3d accumulates the normalized pre-update grouped match across increasing angular radii to show how much local support remains confined to the local neighborhood before any residual correction. Figures 3e and 3f then compare exact support with local support on that same grouped evidence. The scientific question is therefore the same as in the main text: not whether the code disappears under speech, but how much of it survives as broadened local support before any routed update can sharpen it.

## Supplementary Methods 4. Routed updates on the grouped full standardized surface

Supplementary Methods 4 stays on the same calibrated local-overlap geometry introduced in Supplementary Methods 2. There that geometry was written in reduced form as \((z,A)\) to expose competition among nearby directions, and Supplementary Methods 3 used that reduced view to isolate how hard commitment fails. Here we return to the grouped full standardized surface \(D=[d_{e,m}]\) and the full fingerprint \(\tilde y\), because subtraction should act where the local neighborhood itself is still visible rather than after that neighborhood has already been collapsed into a one-angle choice.

The mathematical change from Supplementary Methods 3 is only the commitment rule. Hard OMP chose one direction and refit immediately on the reduced surrogate. The routed update first consolidates evidence across the local neighborhood and subtracts only afterward. The direction-level score tracked in the main text is denoted \(s_t[e]\), and here it is developed from the grouped atom-level routing quantities that produce it.

Its columns are partitioned by direction \(e\in\{1,\dots,E\}\) and source-atom index \(m\in\{1,\dots,M\}\). The grouped coefficient state \(x_t=\{x_t^{(e,m)}\}\) now lives on \(D\) rather than on the direction-level matrices \(H\) or \(A\). The initialization is

$$
r_0=\tilde y,
\qquad
x_0=0.
\tag{S30}
$$

Each stage of the routed update has four parts: compute the physical match score, pool that score across neighboring directions, convert the pooled evidence into a gated local update, and then redefine the residual. The formulas below follow that order. At stage \(t\), the correlation between the current residual and the grouped dictionary is

$$
g_t = D^\top r_t.
\tag{S31}
$$

This is the same grouped correlation surface whose ungated \(t=0\) case appeared in (S29). It is the physical evidence term in the routed recursion: \(g_t[e,m]\) measures how strongly the current residual aligns with grouped atom \((e,m)\). The difference from hard commitment is what happens next. Instead of collapsing that evidence to one support element before subtraction, the routed update preserves a broad local band long enough for the residual update to act on the measured geometry rather than on an artificially sharpened support.

Direction-level pooling and local gating then consolidate that match score across nearby directions. The routing branch first forms atom-level compatibility scores

$$
s_t^{(\mathrm{atom})}[e,m]
=
\frac{\langle q_t,k_{e,m}\rangle}{\sqrt{d_k}},
\tag{S32}
$$

where \(q_t\) is a learned query derived from the current residual, \(k_{e,m}\) is the learned key associated with atom \((e,m)\), and \(d_k\) is the key dimension. The quantities \(g_t[e,m]\) and \(s_t^{(\mathrm{atom})}[e,m]\) therefore play different roles: \(g_t[e,m]\) is the physical correlation between the current residual and grouped atom \((e,m)\), whereas \(s_t^{(\mathrm{atom})}[e,m]\) is the learned routing score that controls how that physical evidence is pooled and gated before subtraction. Only \(g_t\) enters the coefficient update in (S37). The routing scores do not replace the physical match; they regulate how strongly each part of that match is allowed to contribute.

Pooling the atom-level routing scores within each direction group produces the direction-level routing score \(s_t[e]\) that is tracked across stages in the main text. At this grouped resolution, the atom index \(m\) remains explicit and the direction score simply records the total routed evidence accumulated within each local neighborhood.

$$
s_t[e]
=
\left(
\sum_{m=1}^{M}
\left|s_t^{(\mathrm{atom})}[e,m]\right|^2
\right)^{1/2}.
\tag{S33}
$$

The pooled direction scores are then converted into routing weights with a Gumbel-family gating rule,

$$
w_t = \mathrm{GumbelSoftmax}\!\left(s_t;\tau\right),
\qquad
w_t \in \Delta^{E-1}.
\tag{S34}
$$

Operationally, this gate concentrates a broad local match onto one physically plausible neighborhood before subtraction. It is not the same object as the stage-0 physical summary \(g_0^{(\mathrm{grp})}\) above. The latter reports how the raw physical match is distributed across direction groups before any routing. The former determines how that physical match is pooled and gated once the routed update is allowed to act.

Within each retained direction group, the same atom-level compatibility scores define an atom gate

$$
u_t^{(e)}
=
\mathrm{GumbelSoftmax}\!\left(\left|s_t^{(\mathrm{atom})}[e,:]\right|;\tau_a\right),
\qquad
u_t^{(e)} \in \Delta^{M-1}.
\tag{S35}
$$

The combined gate is therefore

$$
W_t[e,m] = w_t[e]\,u_t^{(e)}[m].
\tag{S36}
$$

The routed sparse update is

$$
\Delta x_t[e,m] = W_t[e,m]\; g_t[e,m],
\tag{S37}
$$

The gated local update then enters the coefficient and residual recursions,

$$
x_{t+1} = x_t + \eta\,\Delta x_t,
\qquad
r_{t+1} = r_t - D(\eta\,\Delta x_t) = \tilde y - D x_{t+1},
\tag{S38}
$$

where \(\eta\) is a learned step size. These equations close the staged residual-correction scaffold: each stage computes a physical correlation, routes that evidence across the local angle ordering, applies a gated coefficient update, and then redefines the residual for the next stage. The routing step matters here only because it lets subtraction respect the measured directional code rather than destroy it.

Training and evaluation do not decode direction from \(\|x_K^{(e)}\|_2\). Instead they read out expert-level routing scores. Let

$$
\bar s[e] = \frac{1}{K_{\mathrm{sup}}}\sum_{t=1}^{K_{\mathrm{sup}}} s_t[e],
\qquad
\hat\theta = \theta_{\arg\max_e \bar s[e]},
\tag{S39}
$$

where \(K_{\mathrm{sup}}=1\) in the configuration studied here, so the readout is the first-stage direction-level routing score. The composite training objective is

$$
\mathcal L = \alpha\,\mathcal L_{\mathrm{rec}} + \beta\,\mathcal L_{\mathrm{mono}} + \gamma\,\mathcal L_{\mathrm{cls}},
\qquad
(\alpha,\beta,\gamma)=(1.0,\,0.2,\,0.5),
\tag{S40}
$$

with

$$
\mathcal L_{\mathrm{rec}} = \|r_K\|_2^2,
\qquad
\mathcal L_{\mathrm{cls}} = \mathrm{CE}(\bar s,e^\star),
\tag{S41}
$$

and stagewise monotonicity regularizer

$$
\mathcal L_{\mathrm{mono}}
=
\sum_{t=0}^{K-2}
\max\!\left(0,\|r_{t+1}\|_2-\|r_t\|_2\right).
\tag{S42}
$$

Equations (S39)-(S42) close the same recursion with a readout, a reconstruction loss, and a monotonic residual regularizer defined on the neighborhood-preserving updates. During early training, an auxiliary cross-entropy term is added for the first 10 epochs to stabilize expert assignment before the full routed objective dominates. The guided solver still follows the OMP-style residual-correction scaffold, but it replaces exact least-squares refitting with learned local routing. In the reported implementation, a compact transformer parameterization provides the atom-level compatibility scores in (S32), and straight-through Gumbel approximations convert those scores into the stagewise gates in (S34) and (S35), so that broad local matches can be focused before subtraction within the staged residual updates. That implementation detail matters here only because it preserves the local neighborhood that calibration had already exposed.

Figure 5 follows the same local-support story from admissibility to final prediction. For any row-normalized support distribution \(p_i[e]\) on the angle grid \(\{\theta_e\}_{e=1}^E\), local support is summarized by the cumulative neighborhood mass

$$
m_i(r)=\sum_{e:\,|\theta_e-\theta_i|\le r} p_i[e].
$$

Figure 5a applies this statistic to three aligned stages: the speech pre-update grouped match \(g_0^{(\mathrm{grp})}\), the first guided-step validation replay from Supplementary Methods 4, and the final guided clean confusion matrix after row normalization. Figure 5b then applies it to the final clean confusion matrices of the guided solver, router-bypass ablation, OMP baseline, and dense routing, so the family comparison stays on the same local-support measure throughout. Across all three stages, the question is the same one posed in the Results: how much nearby-angle support remains local as the readout sharpens?

To quantify how strongly final local support follows the local neighborhood in Fig. 5f, let

$$
w_{15}^{(m)} = \frac{1}{E}\sum_{i=1}^{E} m_i^{(m)}(15^\circ)
$$

denote the mean local support inside \(15^\circ\) for decoder family \(m\), computed from the corresponding row-normalized clean confusion matrix. Let \(q_i^{(m)}\) denote the decoder's anglewise local-support profile, defined as the row-normalized mass that family \(m\) keeps inside the \(\pm 15^\circ\) neighborhood around target angle \(i\), and let \(q_i^{(H)}\) denote the corresponding profile of the local neighborhood derived from the calibrated neighborhood geometry. The family-to-neighborhood profile agreement is then

$$
\rho_{\mathrm{prof}}^{(m)} = \rho\!\left(q^{(m)}, q^{(H)}\right),
$$

where \(\rho(\cdot,\cdot)\) is the Pearson correlation from Eq. (S43). The primary bar score reported in Fig. 5f is

$$
A^{(m)}
=
w_{15}^{(m)}
\left[
0.75 + 0.25\left(\frac{\rho_{\mathrm{prof}}^{(m)}+1}{2}\right)
\right],
$$

that is, local support inside \(15^\circ\) multiplied by a bounded profile-agreement factor in \([0.75, 1]\). This construction keeps local support as the dominant term while rewarding families whose anglewise local-support ordering also matches the local neighborhood. The open-circle annotations on the same panel report only the corresponding whole-map correlation between the row-normalized clean confusion map and the calibrated angle-angle matrix that quantifies that same local neighborhood, used as a secondary reference rather than as the primary family-ranking quantity.

For the Fig. 5f null check, we compare the measured and learned structures against a permutation null that shuffles the angle ordering of the learned map while preserving its marginal values. The observed profile and whole-map correlations exceed the corresponding 95th-percentile permutation nulls, so the reported agreement is stronger than shuffled angle structure would permit.

## Supplementary Methods 5. Fig. 2 and Fig. 3 compactness, neighborhood, and local separability summaries

The Fig. 2 and Fig. 3 panels summarize measured geometry on two aligned but distinct surfaces. Figure 2 stays on the calibration-side centered-magnitude matrix \(H_{\mathrm{fig}}\). Figure 3 first builds angle-conditioned centered summaries for calibration and speech to ask whether the same directional code survives realistic source variation, and then returns to the pre-update grouped match \(g_0^{(\mathrm{grp})}\) to ask how much support remains exact and how much has already broadened into the local neighborhood before any residual correction has acted.

For any two vectors \(u,v\in\mathbb{R}^F\), the Pearson correlation coefficient is

$$
\rho(u,v)
=
\frac{\sum_{k=1}^{F}(u_k-\bar u)(v_k-\bar v)}
{\sqrt{\sum_{k=1}^{F}(u_k-\bar u)^2}\sqrt{\sum_{k=1}^{F}(v_k-\bar v)^2}},
\tag{S43}
$$

where \(\bar u\) and \(\bar v\) are sample means across frequency bins.

The inter-angle similarity statistic used for Fig. 1e is the correlation matrix of the angle-indexed mean fingerprints,

$$
S_{e,e'} = \rho(h_e,h_{e'}).
\tag{S44}
$$

For Fig. 2d and Fig. 2f, we quantify the same local neighborhood on the analysis surface \(H_{\mathrm{fig}}\) with the matrix

$$
S^{(\mathrm{ctr})}_{e,e'} = \rho\!\left(H_{\mathrm{fig}}[:,e], H_{\mathrm{fig}}[:,e']\right).
\tag{S45}
$$

Figure 2d plots the mean of Eq. (S45) at fixed angular separation, so the panel reports how quickly positive local ordering decays on the centered calibration surface.

Figure 2f gives a complementary graph view of that same centered local neighborhood. Its affinity matrix is the positive part of Eq. (S45),

$$
A_{e,e'} =
\begin{cases}
\max(S^{(\mathrm{ctr})}_{e,e'}, 0), & e \neq e',\\
0, & e = e',
\end{cases}
$$

and the displayed coordinates are the first two nontrivial eigenvectors of the symmetric normalized graph Laplacian

$$
L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2},
\qquad
D_{e,e} = \sum_{e'} A_{e,e'}.
$$

This graph view is therefore constructed as a graph embedding of the centered structure of that same local neighborhood implied by Eq. (S45). It is not the singular-coordinate surrogate of Eqs. (S19)-(S20), which remains the reduced-order model used for the hard-commitment analysis.

To compare calibration and speech on a matched summary representation in Fig. 3a-c, let \(\tilde y^{(c)}_{e,n}\in\mathbb{R}^F\) denote the standardized fingerprint of clip \(n\) at angle \(e\) under condition \(c\in\{\mathrm{cal},\mathrm{speech}\}\). We form the angle-conditioned mean magnitude profile

$$
\mu^{(c)}_e[k]
=
\frac{1}{N^{(c)}_e}\sum_{n=1}^{N^{(c)}_e}\left|\tilde y^{(c)}_{e,n}[k]\right|,
\tag{S46}
$$

and then center across angle,

$$
M^{(c)}[k,e]
=
\mu^{(c)}_e[k]
- \frac{1}{E}\sum_{e'=1}^{E}\mu^{(c)}_{e'}[k].
\tag{S47}
$$

The mirrored compactness curve in Fig. 3a is the cumulative singular-value energy of \(M^{(c)}\),

$$
\mathcal E^{(c)}(r)
=
\frac{\sum_{j=1}^{r}\sigma_j(M^{(c)})^2}
{\sum_{j}\sigma_j(M^{(c)})^2},
\tag{S48}
$$

where \(\sigma_j(M^{(c)})\) are the singular values of the centered summary matrix. The corresponding speech-side neighborhood similarity matrix is

$$
S^{(c)}_{e,e'} = \rho\!\left(M^{(c)}[:,e], M^{(c)}[:,e']\right),
\tag{S49}
$$

and Fig. 3b plots the mean of Eq. (S49) at fixed angular separation. Figure 3c displays the same similarity family in split-triangle form, with calibration in the lower-left triangle and speech in the upper-right triangle.

Figure 3d-f then return to the ungated grouped-match surface from Supplementary Methods 3. Let

$$
\pi_i[e]
=
\frac{g^{(\mathrm{grp})}_{0,i}[e]}
{\sum_{e'=1}^{E} g^{(\mathrm{grp})}_{0,i}[e']},
\tag{S50}
$$

denote the row-normalized stage-0 support across direction groups for clip \(i\) with ground-truth angle index \(e_i\). The cumulative mass within angular radius \(r\) is

$$
m_i(r)
=
\sum_{|\theta_e-\theta_{e_i}|\le r}\pi_i[e],
\tag{S51}
$$

and Fig. 3d plots the condition-wise mean of Eq. (S51) over the radius sequence \(r=0^\circ,5^\circ,\dots,30^\circ\). Exact support is the radius-zero case

$$
a_i^{(0)} = \mathbf{1}\!\left[\arg\max_e g^{(\mathrm{grp})}_{0,i}[e] = e_i\right],
\tag{S52}
$$

whereas local support at tolerance \(\tau\) is

$$
a_i^{(\tau)}
=
\mathbf{1}\!\left[|\theta_{\arg\max_e g^{(\mathrm{grp})}_{0,i}[e]}-\theta_{e_i}| \le \tau\right].
\tag{S53}
$$

Figure 3e reports the angle-wise mean of Eq. (S52) for calibration and speech, and Fig. 3f compares the speech-side angle-wise means of exact support and local support from Eqs. (S52) and (S53) with \(\tau=10^\circ\).

These are descriptive statistics on the executed standardized fingerprints and grouped stage-0 support surfaces. They summarize compactness, neighborhood width, and local separability in the experimentally constructed feature space; they do not by themselves constitute a first-principles identifiability theorem.

## Supplementary Methods 6. Cross-object descriptor interpretation for Fig. 6

Figure 6 examines recurrence across a tested set of structurally distinct passive-object archetypes rather than a constitutive catalog of five nominal materials.

The five objects in Fig. 6 span visibly different structural layouts: flat plate, curved shell, corrugated hollow shell, orthotropic board, and thin consumer-device shell with cavity. The manuscript therefore interprets them at the level of coupled material-structure archetypes rather than nominal material name alone. The data show that the directional code recurs across this tested archetype set.

The local neighborhood, compactness, and shared-response overlap across objects used in Fig. 6 are response-level summaries derived from the executed measurements. For each object, the centered response matrix is the row-wise mean-centered magnitude matrix \(|H|-\mathrm{mean}_\theta(|H|)\). The width of the local neighborhood is the first angular separation at which the mean centered-\(|H|\) inter-angle correlation becomes non-positive. The effective rank is the entropy-equivalent rank of the same centered-\(|H|\) singular-value spectrum,

$$
r_{\mathrm{eff}}
=
\exp\!\left(-\sum_j p_j \log p_j\right),
\qquad
p_j = \frac{\sigma_j^2}{\sum_\ell \sigma_\ell^2},
$$

where \(\sigma_j\) are the singular values of the centered-\(|H|\) matrix. The descriptor used in Fig. 6d is the shared-response overlap across objects. Formally, it is the mean top-3 overlap burden: the material-wise average of pairwise mean squared canonical correlations between each object's top-3 centered-\(|H|\) subspace and those of the other objects. In other words, the formal descriptor asks how strongly each object's leading directional-response subspace overlaps with those of the other objects. The local neighborhood, compactness, and shared-response overlap across objects in the bridge analysis therefore remain response-level descriptors of the measured geometry.

The five objects differ in stiffness, damping, anisotropy, shell-versus-plate geometry, layering, and cavity structure, but Fig. 6 does not isolate any one of those factors as a controlled causal variable. Instead, it asks whether a common directional code survives across structurally different passive substrates. The answer supported by the executed measurements is yes: each object retains structured angle-frequency fingerprints, a finite neighborhood of positive local ordering, and above-chance single-point readout under matched calibration.

The same cross-object view also clarifies what orders performance. The paper-cup branch makes the contrast explicit: it carries the strongest overall \(|H|\) energy in this set, yet its neighboring directions overlap more broadly and its readout is worse than cardboard. Across this tested archetype set, local separability tracks directional usability more closely than energy alone.

Finally, Fig. 6e shows that recurrence does not require one shared object-specific informative band. The selected contrast window shifts across objects, and the recovered directional code takes different spectral forms across those object-specific informative bands. What persists across the set is not one universal spectrum. It is one directional code expressed through different structural bands.

## Supplementary Table 1. Cross-object archetype descriptors and executed response summaries

| Object | Structural archetype | Stiffness class | Damping class | Anisotropy class | CompressionIndex | SeparabilityIndex | Top-1 (%) | Within 10° (%) | MAE (deg) |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cardboard box | Corrugated hollow packaging shell | Low | High | High | 0.0 | 45.8 | 81.1 | 95.5 | 2.61 |
| Wooden board | Orthotropic solid board | Mid-high | Moderate | High | 75.0 | 22.2 | 79.3 | 97.3 | 5.09 |
| Acrylic plate | Homogeneous flat polymer plate | Mid | Moderate | Low | 100.0 | 44.4 | 78.4 | 93.7 | 3.83 |
| Paper cup | Curved coated paperboard shell | Low | Moderate-high | Medium | 37.5 | 69.4 | 77.5 | 91.9 | 5.99 |
| Laptop shell | Thin consumer-device shell with cavity | High | Low | Low-mid | 37.5 | 68.1 | 67.6 | 92.8 | 5.23 |
