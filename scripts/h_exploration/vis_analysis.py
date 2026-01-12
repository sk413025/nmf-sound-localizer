import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.h_exploration.dataset_lag import DoALagDataset
from scripts.h_exploration.train_dt_lag_seq_rtg import SeqDT_FreqAware
import argparse
import logging
from pathlib import Path

# Configure Plotting style
plt.style.use('seaborn-v0_8-darkgrid')

def run_omp_step_by_step(X_chunk, Y_chunk, freq_idx, K_max=3):
    """
    Run OMP for a specific bin and return the trajectory of choices.
    """
    device = X_chunk.device
    Tw = 16
    MaxLag = 16
    
    # 1. Dictionary Construction
    # X_chunk: (MaxLag + Tw, F)
    # Target Bin y: (Tw,)
    y = Y_chunk[:, freq_idx] # (Tw)
    
    # Construct Dict for this freq
    D = torch.zeros(Tw, MaxLag, dtype=X_chunk.dtype, device=device)
    for k in range(MaxLag):
        # Lag k: X[16-k : 16+Tw-k]
        x_lag = X_chunk[16-k : 16+Tw-k, freq_idx]
        D[:, k] = x_lag
        
    # Normalize Dict
    D_norms = torch.linalg.vector_norm(D.abs(), dim=0, keepdim=True) + 1e-8
    D = D / D_norms
    
    residual = y.clone()
    trajectory = []
    
    initial_energy = torch.linalg.vector_norm(y.abs())**2
    
    for k in range(K_max):
        # Calc Correlations (Projections)
        corrs = D.conj().T @ residual
        
        # Selection
        best_lag = torch.argmax(corrs.abs()).item()
        val = corrs[best_lag]
        
        # Update
        contribution = D[:, best_lag] * val
        residual = residual - contribution
        
        current_energy = torch.linalg.vector_norm(residual.abs())**2
        reduction = (initial_energy - current_energy) / initial_energy
        
        trajectory.append({
            "step": k,
            "lag": best_lag,
            "reduction": reduction.item(),
            "corr_profile": corrs.abs().cpu().numpy()
        })
        
    return trajectory

def get_saliency_map(model, x, rtg, freq_idx_val):
    """
    Compute Input Gradient Saliency for the predicted action.
    """
    model.zero_grad()
    
    # Prepare Inputs with Gradient
    x_tensor = x.clone().detach().unsqueeze(0).requires_grad_(True) # (1, K, M)
    rtg_tensor = rtg.clone().detach().unsqueeze(0) # (1, K)
    freq_tensor = torch.tensor([freq_idx_val], device=x.device) # (1,)
    
    # Forward
    logits, _ = model.rnn(model.layer_norm(
        model.state_embed(model.corr_norm(x_tensor)) + 
        model.rtg_embed(rtg_tensor.unsqueeze(-1)) + 
        model.freq_embed(freq_tensor).unsqueeze(1).expand(-1, x_tensor.size(1), -1)
    ))
    # Logits: (1, K, hidden) -> Head -> (1, K, M_lags)
    # Wait, model.forward returns (out, hidden). I need to pass through Head.
    
    # Re-implement forward part to go all the way to head
    # Or just use the model as is. 
    # Calling the model directly handles everything? 
    # No, the provided class forward returns RNN output. Head is separate?
    # Let's check the code: 
    # The class `SeqDT_FreqAware` forward returns `out, _`.
    # And there is `self.head`.
    # But `forward` DOES NOT CALL HEAD.
    # The training loop calls `logits = model.head(out)`.
    
    # So we do:
    out, _ = model.rnn(model.layer_norm(
        model.state_embed(model.corr_norm(x_tensor)) + 
        model.rtg_embed(rtg_tensor.unsqueeze(-1)) + 
        model.freq_embed(freq_tensor).unsqueeze(1).expand(-1, x_tensor.size(1), -1)
    ))
    pred_logits = model.head(out) # (1, K, M)
    
    # We want to explain the decision at the LAST step (or specific step)
    # Let's average gradients over all steps 0..K-1
    target_score = pred_logits.max()
    target_score.backward()
    
    # Get Gradient at input x
    saliency = x_tensor.grad.abs().squeeze(0).cpu().numpy() # (K, M)
    return saliency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/vis_analysis")
    args = parser.parse_args()
    
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load Model
    model = SeqDT_FreqAware(M_lags=16, d_model=128, hidden_dim=256, max_freq=1025)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # 2. Get Data
    dataset = DoALagDataset("/Users/sbplab/LDV-data-processed/speech260_original_16k_no_edge_sync_vad_normalized", 
                            "/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized", 
                            angle=90.0)
    item = dataset[0]
    mic_stft = item["mic_stft"].to(device)
    ldv_stft = item["ldv_stft"].to(device)
    
    # Chunk
    Tw = 16
    MaxLag = 16
    start_t = 32
    X_chunk = mic_stft[start_t-MaxLag : start_t+Tw]
    Y_chunk = ldv_stft[start_t : start_t+Tw]
    
    bins_to_compare = [50, 150, 250]
    
    # ==========================================
    # PART 1: OMP Trajectory Visualization
    # ==========================================
    print("Generating OMP Trajectories...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    saliency_maps = {}
    
    for i, b in enumerate(bins_to_compare):
        traj = run_omp_step_by_step(X_chunk, Y_chunk, b)
        
        lags = [t['lag'] for t in traj]
        steps = [t['step'] for t in traj]
        reductions = [t['reduction'] for t in traj]
        
        # Plot Trajectory
        ax = axes[i]
        ax.plot(steps, lags, marker='o', linestyle='-', linewidth=2, label=f'Bin {b}')
        ax.set_title(f"Frequency Bin {b}\nFinal Red: {reductions[-1]:.1%}")
        ax.set_xlabel("Optimization Step")
        ax.set_xticks([0, 1, 2])
        if i == 0: ax.set_ylabel("Selected Lag Index")
        ax.set_ylim(-1, 16)
        
        # Prepare Data for Saliency
        # specific_corrs from trajectory
        corrs_seq = np.stack([t['corr_profile'] for t in traj]) # (3, 16)
        actions_seq = np.array([t['lag'] for t in traj])
        
        # Run Saliency
        x_in = torch.tensor(corrs_seq, device=device).float()
        # Fake RTG for visualization (assuming perfect RTG)
        rtg_in = torch.tensor([1.0, 0.5, 0.25], device=device) # Decay assumption
        
        sal = get_saliency_map(model, x_in, rtg_in, b)
        
        saliency_maps[b] = sal
        
    plt.suptitle("OMP Trajectories across Frequencies")
    plt.tight_layout()
    plt.savefig(out_path / "omp_trajectories.png")
    print(f"Saved OMP Trajectory plot to {out_path}/omp_trajectories.png")
    
    # ==========================================
    # PART 2: Saliency Map (Attention Proxy)
    # ==========================================
    print("Generating Saliency Maps...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    for i, b in enumerate(bins_to_compare):
        sal = saliency_maps[b] # (3, 16)
        
        sns.heatmap(sal.T, ax=axes[i], cmap="viridis", cbar=True, yticklabels=True)
        axes[i].set_title(f"Model Saliency: Bin {b}")
        axes[i].set_xlabel("Step (0, 1, 2)")
        if i == 0: axes[i].set_ylabel("Lag Index Input (0-15)")
        axes[i].invert_yaxis()
        
    plt.suptitle("Model Input Importance (Gradient Saliency)")
    plt.tight_layout()
    plt.savefig(out_path / "model_saliency.png")
    print(f"Saved Saliency plot to {out_path}/model_saliency.png")
    
    # ==========================================
    # PART 3: Embedding Similarity
    # ==========================================
    print("Generating Embedding Similarity Matrix...")
    embeds = model.freq_embed.weight.detach().cpu().numpy() # (1025, d)
    # Take range 0-300
    embeds = embeds[0:300]
    
    # Compute Cosine Sim
    norm = np.linalg.norm(embeds, axis=1, keepdims=True)
    embeds_norm = embeds / (norm + 1e-8)
    sim_matrix = embeds_norm @ embeds_norm.T
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, cmap="RdBu_r", center=0)
    plt.title("Frequency Embedding Similarity (0-300Hz)")
    plt.xlabel("Freq Bin ID")
    plt.ylabel("Freq Bin ID")
    plt.savefig(out_path / "freq_embedding_sim.png")
    print(f"Saved Embedding Similarity to {out_path}/freq_embedding_sim.png")

if __name__ == "__main__":
    main()
