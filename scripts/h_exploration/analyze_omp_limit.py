import torch
import numpy as np
import argparse
import logging
from tqdm import tqdm
from scripts.h_exploration.dataset_lag import DoALagDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_omp_k_scan(Dict_tensor, Targets, k_list=[3, 10, 16]):
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
            # Handle potential zero initial energy
            mask = initial_energy > 1e-9
            reductions = np.zeros(F)
            if mask.sum() > 0:
                 r = (initial_energy[mask] - current_energy[mask]) / initial_energy[mask]
                 reductions[mask.cpu().numpy()] = r.cpu().numpy()
            
            results[current_k_step] = reductions
            
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_clips", type=int, default=10)
    parser.add_argument("--max_lag", type=int, default=50, help="Maximum history lag size (M)")
    args = parser.parse_args()
    
    device = torch.device("cpu") # Force CPU
    
    dataset = DoALagDataset("/Users/sbplab/LDV-data-processed/speech260_original_16k_no_edge_sync_vad_normalized", 
                            "/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized", 
                            angle=90.0)
    
    # K steps to test
    MaxLag = args.max_lag
    # Scan K from 3 up to MaxLag
    k_targets = [3, 8, 16, 24, 32, 40, 48, 50]
    k_test = [k for k in k_targets if k <= MaxLag]
    if MaxLag not in k_test:
        k_test.append(MaxLag)
    
    start_bin = 5
    end_bin = 1024
    Tw = 16
    start_t = MaxLag + 16 # Ensure we have enough history
    
    all_reductions = {k: [] for k in k_test}
    
    print(f"Scanning OMP Limits (K={k_test}) with MaxLag={MaxLag} on full spectrum...")
    
    for i in tqdm(range(min(len(dataset), args.max_clips))):
        item = dataset[i]
        mic = item["mic_stft"].to(device)
        ldv = item["ldv_stft"].to(device)
        
        # Ensure file is long enough
        if mic.shape[0] < start_t + Tw:
            continue

        X_chunk = mic[start_t-MaxLag : start_t+Tw, start_bin:end_bin]
        Y_chunk = ldv[start_t : start_t+Tw, start_bin:end_bin]
        
        n_bins = X_chunk.shape[1]
        
        Dict_tensor = torch.zeros(n_bins, Tw, MaxLag, dtype=X_chunk.dtype, device=device)
        for k in range(MaxLag):
            # X_chunk start index 0 corresponds to t - MaxLag
            # We want lag k: starts at t - k  => index: (t - k) - (t - MaxLag) = MaxLag - k
            start_row = MaxLag - k
            end_row = MaxLag + Tw - k
            Dict_tensor[:, :, k] = X_chunk[start_row : end_row, :].T
            
        Targets = Y_chunk.T.unsqueeze(2)
        
        batch_results = run_omp_k_scan(Dict_tensor, Targets, k_list=k_test)
        
        for k in k_test:
            all_reductions[k].append(batch_results[k])
            
    # Aggregate
    print("\n" + "="*40)
    print(" OMP LIMIT ANALYSIS (Energy Reduction)")
    print("="*40)
    
    base_red = 0
    
    for k in k_test:
        data = np.concatenate(all_reductions[k])
        mean_red = np.mean(data)
        
        if k == 3:
            base_red = mean_red
            print(f"K = {k:<2} (Current) : {mean_red:.2%}")
        else:
            gain = mean_red - base_red
            print(f"K = {k:<2}           : {mean_red:.2%} (+{gain:.2%} gain)")
            
    print("-" * 40)
    print("Interpretation:")
    print(f"If K={max(k_test)} is much higher than K=3, then K=3 is the bottleneck.")
    print(f"If K={max(k_test)} is close to K=3, then the limit is physics/noise (OMP cannot solve it).")

if __name__ == "__main__":
    main()
