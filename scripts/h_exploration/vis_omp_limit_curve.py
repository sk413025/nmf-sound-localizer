
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from scripts.h_exploration.dataset_lag import DoALagDataset

# Set style to match academic/report quality
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)

def run_omp_k_scan(Dict_tensor, Targets, k_list):
    """
    Run OMP for various K values to find the limit.
    """
    F, Tw, MaxLag = Dict_tensor.shape
    device = Dict_tensor.device
    
    # Store results for each K in k_list
    results = {}
    
    Residuals = Targets.clone()
    Active_Sets = torch.zeros(F, max(k_list), dtype=torch.long, device=device) - 1
    
    D_norms = torch.linalg.vector_norm(Dict_tensor.abs(), dim=1, keepdim=True) + 1e-8
    Dict_Norm = Dict_tensor / D_norms
    
    initial_energy = torch.linalg.vector_norm(Targets, dim=1).squeeze()**2
    
    max_k = max(k_list)
    
    for k in range(max_k):
        # 1. Correlations
        Corrs = torch.bmm(Dict_Norm.conj().transpose(1, 2), Residuals)
        Abs_Corrs = torch.abs(Corrs).squeeze(2)
        
        if k > 0:
            for b in range(F):
                prev = Active_Sets[b, :k]
                valid_prev = prev[prev >= 0]
                Abs_Corrs[b, valid_prev] = -1.0
                
        # 2. Select
        Best_Lags = torch.argmax(Abs_Corrs, dim=1)
        Active_Sets[:, k] = Best_Lags
        
        # 3. Update
        for b in range(F):
            indices = Active_Sets[b, :k+1]
            A_active = Dict_tensor[b, :, indices]
            y_b = Targets[b, :, 0]
            
            h = torch.linalg.lstsq(A_active, y_b).solution.flatten()
            recon = A_active @ h
            Residuals[b, :, 0] = y_b - recon
            
        # Check if current k is a checkpoint
        current_k_step = k + 1
        if current_k_step in k_list:
            current_energy = torch.linalg.vector_norm(Residuals, dim=1).squeeze()**2
            mask = initial_energy > 1e-9
            reductions = np.zeros(F)
            if mask.sum() > 0:
                 r = (initial_energy[mask] - current_energy[mask]) / initial_energy[mask]
                 reductions[mask.cpu().numpy()] = r.cpu().numpy()
            
            results[current_k_step] = reductions
            
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_clips", type=int, default=30)
    args = parser.parse_args()
    
    device = torch.device("cpu")
    
    dataset = DoALagDataset("/Users/sbplab/LDV-data-processed/speech260_original_16k_no_edge_sync_vad_normalized", 
                            "/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized", 
                            angle=90.0)
    
    # Dense scan for K=1..20 to show the curve, then sparse
    k_dense = list(range(1, 21)) # 1 to 20
    k_sparse = [25, 30, 40, 50]
    k_test = sorted(k_dense + k_sparse)
    
    start_bin = 5
    end_bin = 1024
    Tw = 16
    MaxLag = 50 # Set high enough to allow K=50
    start_t = MaxLag + 16
    
    all_reductions = {k: [] for k in k_test}
    
    print(f"Generating OMP Curve (K=1..50) using {args.max_clips} clips...")
    
    for i in tqdm(range(min(len(dataset), args.max_clips))):
        item = dataset[i]
        mic = item["mic_stft"].to(device)
        ldv = item["ldv_stft"].to(device)
        
        if mic.shape[0] < start_t + Tw:
            continue

        X_chunk = mic[start_t-MaxLag : start_t+Tw, start_bin:end_bin]
        Y_chunk = ldv[start_t : start_t+Tw, start_bin:end_bin]
        
        n_bins = X_chunk.shape[1]
        
        Dict_tensor = torch.zeros(n_bins, Tw, MaxLag, dtype=X_chunk.dtype, device=device)
        for k in range(MaxLag):
            start_row = MaxLag - k
            end_row = MaxLag + Tw - k
            Dict_tensor[:, :, k] = X_chunk[start_row : end_row, :].T
            
        Targets = Y_chunk.T.unsqueeze(2)
        
        # Only run up to min(MaxLag, 50) - but MaxLag is 50 here.
        batch_results = run_omp_k_scan(Dict_tensor, Targets, k_list=k_test)
        
        for k in k_test:
            all_reductions[k].append(batch_results[k])
            
    # Compute stats
    k_values = []
    means = []
    
    for k in k_test:
        mean_red = np.mean(np.concatenate(all_reductions[k]))
        k_values.append(k)
        means.append(mean_red)
        
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Setup Dual Axis
    ax2 = ax1.twinx()

    # 1. Cumulative Energy Reduction (Like SVD "Cumulative fraction")
    color1 = 'tab:red'
    ax1.set_xlabel('Number of OMP Steps (K)')
    ax1.set_ylabel('Total Energy Reduction (%)', color=color1)
    ax1.plot(k_values, means, color=color1, marker='s', label='Cumulative Energy Explained')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)

    # 2. Marginal Gain (Like SVD "Singular value")
    # Calculate delta
    deltas = [means[0]] # first point gain
    for i in range(1, len(means)):
        deltas.append(means[i] - means[i-1])
            
    color2 = 'tab:blue'
    ax2.set_ylabel('Marginal Gain (Delta Reduction)', color=color2)
    ax2.plot(k_values, deltas, color=color2, marker='o', linestyle='--', label='Marginal Gain (Information per Step)')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_yscale('log') # SVD plots often use log scale for singular values
    
    # Titles and Annotations
    plt.title('OMP Information Saturation Analysis\n(K Steps vs Reduction, Full Spectrum 5-1024 bin)')
    
    # Mark K=16 Saturation
    if 16 in k_values:
        idx = k_values.index(16)
        val = means[idx]
        ax1.annotate(f'Saturation\nK=16 ({val:.1%})', 
                     xy=(16, val), xytext=(20, val-0.2),
                     arrowprops=dict(facecolor='black', shrink=0.05))
        ax1.axvline(x=16, color='gray', linestyle=':', alpha=0.5)

    # Note
    plt.tight_layout()
    out_file = '/Users/jnrle/Documents/LDVReorientation/worktrees/exp-freq-aware-policy/results/omp_limit_curve.png'
    plt.savefig(out_file, dpi=300)
    
    print(f"Plot saved to {out_file}")
    
    # Print data for user
    print("\nPlot Data:")
    for k, m, d in zip(k_values, means, deltas):
        print(f"K={k:<2}: Red={m:.2%} (+{d:.2%})")

if __name__ == "__main__":
    main()
