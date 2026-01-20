# Physics-Derived RTG-OMP (From First Principles)

This note derives an **RTG-influential** variant of OMP for the Mic→LDV lag-selection problem **from scratch**, starting from a minimal physical model. The goal is to make **RTG (Return-To-Go / goal-conditioning)** enter the *teacher/oracle* decision process in a **physically interpretable** way, rather than being an extra input that the optimal solver can ignore.

This document is intentionally written **without relying on prior experiment constraints** (e.g., “always run to `K_max`”, fixed epsilon floors, or a deterministic teacher). It then maps the derived algorithm back to practical implementation in this repo.

---

## 1) Problem Setup (Minimal Physical Assumptions)

### 1.1 LTI relationship between Mic and LDV

Assume a linear time-invariant approximation in a short window:

\[
y(t) = (h * x)(t) + e(t)
\]

- \(x(t)\): microphone signal (air pressure proxy)
- \(y(t)\): LDV signal (surface velocity/displacement proxy)
- \(h(\tau)\): effective impulse response capturing propagation + structural dynamics + multipath
- \(e(t)\): unexplained component (sensor noise + model mismatch + other forces)

This is the only “physics” we strictly need: linear superposition in a short time window.

### 1.2 Sparse delay-path approximation (physically interpretable model class)

In many practical short-window regimes, an interpretable approximation is:

\[
h(\tau) \approx \sum_{k=1}^{K} w_k \, \delta(\tau-\tau_k)
\quad\Rightarrow\quad
y(t) \approx \sum_{k=1}^{K} w_k\, x(t-\tau_k) + e(t)
\]

Interpretation:
- each selected \(\tau_k\) corresponds to an “effective path / delay component”
- \(K\) corresponds to “how many effective paths we allow”, i.e., **model complexity**

This is the origin of “select a small set of lags”.

---

## 2) Frequency-Domain / Windowed Linear Algebra Formulation

For a short window of length `Tw` (e.g., `Tw` STFT frames) and a fixed frequency bin \(f\), define:

- \(y_f \in \mathbb{C}^{Tw}\): LDV target vector in that window at frequency \(f\)
- build a lag dictionary from the Mic history:
  - \(D_f \in \mathbb{C}^{Tw \times M}\), where each column is the Mic vector at some lag \(\tau\)
  - \(M = 2\cdot\text{max_lag} + 1\)

Select a set \(S\subseteq \{1,\dots,M\}\) and solve complex least squares:

\[
\hat{w} = \arg\min_{w}\ \|y_f - D_{f,S} w\|_2^2,
\qquad
\hat{y}_f = D_{f,S}\hat{w},
\qquad
r_f = y_f - \hat{y}_f
\]

This is a standard sparse approximation / pursuit problem per frequency bin.

---

## 3) Baseline OMP (Why RTG Can Be Ignored)

### 3.1 Fixed objective with a fixed budget

Classical “fixed K” objective:

\[
\min_{S,|S|\le K}\ \|y_f - D_{f,S} w\|_2^2
\]

OMP is a greedy approximation: iteratively pick one atom (one lag) and refit by LS.

### 3.2 The key reason RTG is not required here

If the teacher is deterministic and the objective is fixed (same \(K\), same error metric), then the next action is (approximately) a function of the current residual / correlation state:

\[
a_k \approx \arg\max_j |\langle d_j, r_k\rangle|
\quad\Rightarrow\quad
a_k = f(s_k)
\]

If the teacher’s action distribution does not change with RTG, then in a supervised imitation objective the optimal classifier can ignore RTG:

\[
P(a\mid s,\text{RTG}) = P(a\mid s)
\]

This does **not** invalidate Decision Transformer in general; it only says RTG must correspond to a factor that **actually changes the optimal/teacher behavior** under the data distribution.

---

## 4) What “RTG-Influential” Means (Necessary Condition)

To make RTG matter, RTG must parameterize a **family of physically meaningful objectives** (or constraints) such that:

\[
\exists s:\ \arg\max_a P_{\text{teacher}}(a\mid s,\text{RTG}=r_1) \neq
\arg\max_a P_{\text{teacher}}(a\mid s,\text{RTG}=r_2)
\]

In plain words:
- for *similar states*, different RTG values lead to different “correct” actions

This requires RTG to encode a real **goal / cost / constraint / uncertainty**.

---

## 5) Physics-Derived Ways to Introduce RTG into OMP (Three Interpretable Knobs)

Below are three RTG semantics that are physically defensible in this domain.

### 5.1 RTG as a reconstruction goal (specification)

RTG0 encodes a required “explained energy” level, e.g., 0.6 vs 0.9. This is physically meaningful because:
- beyond a point, reducing residual energy further often fits measurement noise or non-Mic-driven components

This introduces a **STOP** decision (variable horizon).

### 5.2 RTG as a complexity cost (model selection / path-count penalty)

More selected lags correspond to a more complex physical explanation (more effective paths). A natural MAP derivation:

- assume \(e \sim \mathcal{CN}(0,\sigma^2 I)\)
- add a prior penalizing large sets: \(p(S) \propto \exp(-\lambda |S|)\)

Then MAP becomes:

\[
\min_{S,w}\ \|y - D_S w\|_2^2 + \lambda |S|
\]

Here \(\lambda\) is a physically meaningful “complexity penalty”. RTG can parameterize \(\lambda\).

### 5.3 RTG as uncertainty / noise scale (soft/stochastic pursuit)

If measurement uncertainty is high, “argmax corr” is overconfident. A Bayesian/approximate sampling interpretation leads to:

\[
P(a=j\mid s) \propto \exp\left(\frac{|\langle d_j,r\rangle|}{\tau}\right)
\]

Temperature \(\tau\) can be interpreted as an uncertainty scale. RTG can parameterize \(1/\tau\) (confidence).

This produces teacher trajectory diversity in a physically interpretable way (uncertainty-aware pursuit).

---

## 6) A Concrete “RTG-OMP” (Goal + Complexity + Noise Floor)

This section defines a single solver that:
- keeps the same physical forward model (lag dictionary + LS projection)
- introduces RTG via **goal** and **complexity/uncertainty**
- yields RTG-dependent actions and/or RTG-dependent stopping in a controlled way

### 6.1 Definitions

Let:
- energy \(E(r) = \|r\|_2^2\) (per-frequency-bin window energy)
- initial energy \(E_0 = E(y)\)
- explained energy ratio: \(\text{cap} = 1 - E(r)/\max(E_0,\epsilon)\)

Introduce a **noise floor** \(E_{\text{noise}}\) (per window, per frequency, or per band):
- physically: energy level below which additional fit is not meaningful
- statistically: an estimate of \(\sigma^2 \cdot Tw\) in that bin/window

RTG (2D example):
- `RTG0 = goal_capture` in \([0,1]\)
- `RTG1 = budget_fraction` in \([0,1]\) (or remaining steps; any monotone mapping is fine)

### 6.2 Objective family

Define a family of objectives parameterized by \(\lambda\):

\[
\min_{S,w}\ E(y - D_S w) + \lambda |S|
\]

But we solve it greedily with a **marginal gain** rule.

### 6.3 Marginal gain (physically interpretable decision statistic)

At step \(k\), with current set \(S_k\) and residual \(r_k\), consider adding atom \(j\):

\[
\Delta E(j) = E(r_k) - E(r_{k+1}^{(j)})
\]

where \(r_{k+1}^{(j)}\) is the residual after adding \(j\) and refitting by LS.

### 6.4 RTG enters via a decision threshold (STOP rule)

Define a threshold \(\lambda(\text{RTG}, E_{\text{noise}}, \text{urgency})\).

Greedy decision:

- compute \(\Delta E(j)\) for candidate atoms \(j\)
- let \(j^\* = \arg\max_j \Delta E(j)\)
- if \(\Delta E(j^\*) \le \lambda(\cdot)\): **STOP**
- else: add \(j^\*\) and continue

This is still “OMP-like” (greedy pursuit), but now it is **goal-conditioned** and **complexity-aware**.

### 6.5 Mapping RTG to urgency and lambda (one reasonable design)

Define urgency as: “how much remaining progress is required per remaining step”.

- remaining required capture: \(g = \max(0,\ \text{RTG0} - \text{cap}_k)\)
- remaining steps \(L\) derived from RTG1 and max budget:
  - e.g., \(L = \lceil \text{RTG1}\cdot K_{\max}\rceil\) or directly use remaining steps if RTG1 encodes it
- urgency: \(u = g / \max(L,1)\)

Then a simple monotone threshold design:

\[
\lambda = \alpha \cdot E_{\text{noise}} \cdot (1 - u)
\]

Interpretation:
- high urgency (need lots of progress quickly) ⇒ \(u \uparrow\) ⇒ \(\lambda \downarrow\): accept smaller marginal gains (keep adding paths)
- low urgency ⇒ \(\lambda \uparrow\): stop earlier to avoid fitting noise / extra paths

This is only one mapping; the key is monotonicity and physical interpretability.

---

## 7) When RTG Should Change Actions (Not Only Stopping)

If RTG only affects stopping, early actions may remain identical (always pick the largest correlation first), especially in easy regimes.

To make RTG change **which lag** is selected at early steps, you need at least one of:

### 7.1 Lookahead pursuit (planning under limited remaining steps)

If remaining steps are small but the goal is high (high urgency), the best next atom may differ from the myopic best-correlation atom, particularly when dictionary atoms are coherent/redundant.

One-step lookahead:
- pick top-N candidate atoms under current score
- for each candidate, simulate greedy completion for `L-1` steps
- choose the candidate yielding the best final objective value

This remains physically consistent (same forward model, same LS updates), but becomes a “planner”.

### 7.2 Risk-aware objective (stability vs immediate gain)

Myopic selection can pick an atom that is locally best but makes the remaining residual hard to explain with the remaining budget (ill-conditioning, coherence).

A physically defensible risk-aware score may include a conditioning penalty:

\[
\text{score}(j) = \Delta E(j) - \beta \cdot \text{cond}(D_{S_k \cup \{j\}})
\]

This encodes: “prefer atoms that improve fit without making the model unstable”.

RTG can modulate \(\beta\) (higher RTG0/urgency ⇒ lower \(\beta\), tolerate more conditioning risk).

---

## 8) Relationship to Soft-OMP (Physical Interpretation)

Soft-OMP can be interpreted as uncertainty-aware pursuit:

\[
P(a=j\mid s) \propto \exp(\Delta E(j)/\tau)
\]

- \(\tau\) is a noise/uncertainty scale
- RTG can encode \(1/\tau\) (confidence)

This makes teacher trajectories diverse in a way that is physically meaningful: higher uncertainty produces more exploration in lag selection.

Important distinction:
- this RTG is a “policy knob” (uncertainty/exploration), not necessarily a “quality target”

If you want RTG to encode *quality targets*, use goal/cost semantics (Sections 6–7).

---

## 9) Practical Integration Guidance (How This Maps to the Repo)

This repo currently has a deterministic argmax OMP for lag trajectories and a separate RTG definition for DT training/eval. To make RTG physically meaningful and influential, the teacher must be changed to produce RTG-dependent behavior.

### 9.1 Teacher data generation

Current teacher:
- `scripts/h_exploration/generate_lag_omp.py` runs deterministic argmax OMP to `K_max`
- then slices prefixes to create “variable-K” episodes

To implement RTG-OMP:
- add **STOP**: trajectories have variable length based on RTG0/RTG1 and a noise-aware rule
- add **RTG-conditioned objective parameters**:
  - goal (RTG0) and budget/urgency (RTG1)
  - or cost λ (derived from RTG)
- optionally add lookahead for urgency-sensitive behavior

### 9.2 Model training

If the teacher now genuinely depends on RTG, then supervised training on (state, RTG)→action can learn RTG usage.

Critical training requirement:
- the dataset must include multiple RTG settings **for the same (or similar) states**, otherwise the model cannot learn the dependence.

### 9.3 Evaluation

If you keep evaluating always to `K_max`, RTG effects that primarily control stopping will be invisible. Evaluate with:
- variable horizon (stop allowed)
- metrics that reflect both performance and cost:
  - `(final_capture, steps_used)` or Pareto curves

---

## 10) Diagnostics (To Verify RTG Enters the Physics, Not Just the Network)

RTG must change teacher behavior in measurable ways. Recommended per-window/per-bin diagnostics:

**Teacher-side (RTG-OMP)**
- `cap_k` trajectory: capture vs step
- `steps_used` (stop step)
- `max_deltaE_k`, `deltaE_selected_k`, and their ratio
- `lambda_k` (threshold used)
- `E_noise` estimate and `E_residual_k`

**RTG sensitivity**
- action-change rate across RTG sweeps for the same input window/bin
- KL divergence between action distributions (if stochastic) across RTG values

**Sanity checks**
- if RTG0 is higher (harder goal), teacher should (on average) use more steps or accept smaller gains
- if complexity penalty is higher, teacher should stop earlier and/or prefer atoms with larger marginal gains

---

## 11) Summary (Design Principles)

To derive an RTG-influential OMP from physical principles:

1) Keep the forward model: lag dictionary + LS projection (physics-consistent).
2) Introduce RTG as a physically meaningful knob:
   - **goal** (how much explainable energy is required),
   - **complexity cost** (how many effective paths we are willing to use),
   - **uncertainty/noise scale** (how confident we are in greedy correlation).
3) Implement RTG in the teacher by changing the **objective family** and/or **stopping rule**.
4) Evaluate with variable horizon and multi-objective metrics so RTG effects are observable.

