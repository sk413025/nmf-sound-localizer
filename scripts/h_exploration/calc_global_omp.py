import torch
import numpy as np
import argparse
from scripts.h_exploration.dataset_lag import DoALagDataset
import logging

def run_omp_per_bin(X_col, y_col, K_max=3):
    """
    X_col: (MaxLag+Tw)
    y_col: (Tw)
    Returns: initial_energy, final_residuals
    """
    Tw = y_col.shape[0]
    MaxLag = 16
    
    # Dict construction
    D = torch.zeros(Tw, MaxLag, dtype=X_col.dtype, device=X_col.device)
    for k in range(MaxLag):
        D[:, k] = X_col[16-k : 16+Tw-k]
        
    D_norms = torch.linalg.vector_norm(D.abs(), dim=0, keepdim=True) + 1e-8
    D = D / D_norms
    
    residual = y_col.clone()
    
    for k in range(K_max):
        # Corr
        corrs = D.conj().T @ residual
        best_lag = torch.argmax(corrs.abs())
        val = corrs[best_lag]
        
        # Update
        contribution = D[:, best_lag] * val
        residual = residual - contribution
        
    return residual

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_idx", type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load Dataset
    dataset = DoALagDataset("/Users/sbplab/LDV-data-processed/speech260_original_16k_no_edge_sync_vad_normalized", 
                            "/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized", 
                            angle=90.0)
    
    item = dataset[args.clip_idx]
    mic_stft = item["mic_stft"].to(device)
    ldv_stft = item["ldv_stft"].to(device)
    
    # Define Range (Bins 5 to 300)
    start_bin = 5
    end_bin = 300
    
    # Define Time Chunk (Same as training/inspection)
    Tw = 16
    MaxLag = 16
    start_t = 32 # Arbitrary stable chunk
    
    X_chunk = mic_stft[start_t-MaxLag : start_t+Tw] # (32, 1025)
    Y_chunk = ldv_stft[start_t : start_t+Tw]        # (16, 1025)
    
    total_initial_energy = 0.0
    total_residual_energy = 0.0
    
    print(f"Calculating Global OMP Reduction (Independent per Bin) for Bins {start_bin}-{end_bin}...")
    print(f"Clip {args.clip_idx}, Time {start_t}")
    
    # Iterate Bins
    for b in range(start_bin, end_bin):
        X_b = X_chunk[:, b]
        Y_b = Y_chunk[:, b]
        
        initial = torch.linalg.vector_norm(Y_b.abs())**2
        
        # Run OMP (K=3)
        final_res = run_omp_per_bin(X_b, Y_b, K_max=3)
        final = torch.linalg.vector_norm(final_res.abs())**2
        
        total_initial_energy += initial.item()
        total_residual_energy += final.item()
        
    global_reduction = (total_initial_energy - total_residual_energy) / total_initial_energy
    
    print("-" * 30)
    print(f"Total Initial Energy: {total_initial_energy:.4f}")
    print(f"Total Residual Energy: {total_residual_energy:.4f}")
    print(f"Global Reduction: {global_reduction:.2%}")
    print("-" * 30)
    
    if global_reduction > 0.75:
        print("RESULT: YES, Global Reduction > 75%")
    else:
        print("RESULT: NO, Global Reduction <= 75%")

if __name__ == "__main__":
    main()
