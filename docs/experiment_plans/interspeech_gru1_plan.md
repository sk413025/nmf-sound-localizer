# Experiment Plan: Interspeech GRU1

## 1. terminology Alignment (Based on commit c183fd4)

To ensure consistency with the manuscript/interspeech-2026 branch, we align our terminology as follows:

- **Physical Framework**: Adopt the **LTI (Linear Time-Invariant)** framework for time-domain observation models.
- **Transfer Function**: Use **CTF (Convolutive Transfer Function)** with explicit residual terms $\eta(f,n)$ and **Mic-LDV impedance coupling** ($Z_s$ physics).
- **Algorithm**:
  - **OMP**: Refer to as **Physics-motivated $\ell_0$ objective** or **Greedy Energy Capture principle**.
  - **Agent/Model**: Frame the architecture within the **MDP (Markov Decision Process)** formalization.
  - **Targets**: Refer to targets (like RTG) as **physics-meaningful energy targets**.
- **Embedding**: Emphasize **frequency embedding necessity**.

## 2. Dataset Structure

We explicitly record the data structure for the `SpeechData` to be used in this experiment (`boy1` subject).

**Root Path**: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData`

**Subjects**:
- `boy1`, `boy2`, `boy3`, `boy4`, `boy5`, `boy6`
- `girl1`, `girl2`, `girl3`, `girl4`, `girl6`, `girl7`

**Target Subject (`boy1`) Structure**:
Location: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1`

Subdirectories:
- `LDV/`: Contains LDV recordings (e.g., `boy1_papercup_LDV_001.wav`...)
- `MIC/`: Contains Microphone recordings.
- `target/`: Target signals.

## 3. Experiment Objective

Reproduction and extension of the **Full Spectrum Greedy Energy Capture (OMP)** and **MDP Agent (DTmin)** experiments (originally defined in commit `b482ddf`) using the **boy1** MIC and LDV data.

**Key Parameters (inherited from b482ddf):**
- Spectrum: Full Spectrum (5-1024 bin)
- Budget (K): 8 (Optimized for "Reverb Tail" capture as per $\ell_0$ objective)
- Lag: 16
- Epochs: 100
- objective: Validate performance of Frequent-Aware MDP Agent on the new subject data.

