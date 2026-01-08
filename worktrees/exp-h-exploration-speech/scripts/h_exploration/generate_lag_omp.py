import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scripts.h_exploration.dataset_lag import DoALagDataset, create_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def solve_ls_complex(X, y):
    # X: (K, M), y: (K,)
    if X.shape[1] == 0:
        return torch.zeros(0, dtype=X.dtype, device=X.device)
    # Use lstsq
    # Expected: X (Batch, M), y(Batch, 1) if batched. Here single problem.
    sol = torch.linalg.lstsq(X, y).solution
    return sol

def run_omp_lag(X_history, y_target, K_max=4):
    """
    X_history: (M, F) - M past frames for F frequencies
    y_target: (F,) - Current target frame for F frequencies
    
    We solve independent OMP for each Frequency Bin f?
    Or Jointly?
    Sparse Filter usually implies localized in time.
    Let's do per-frequency OMP, BUT this is slow for 1024 bins.
    
    Let's process a Batch of frequencies at once using tensor ops.
    Dictionary per freq f: D_f = X_history[:, f]. 
    Wait, Dict columns should be lags.
    So D_f has shape (1, M) if we only have 1 sample (current time t).
    We need more samples to solve LS.
    
    We need a Time Window for LS stability.
    Let's say we solve for window W=10 frames.
    y_window: (W, F)
    Dictionary D_frame(t): [x(t), x(t-1), ... x(t-M)]
    
    Standard Convolutive OMP:
    Construct Toeplitz matrix?
    
    Simplification:
    We process 1 "Snapshot" of time T.
    We just want to find which lag $\tau$ explains $y(t)$ best.
    But solving LS on 1 point is trivial (w = y/x).
    We need statistical significance over time.
    
    So we solve for a Window of frames T_w.
    Target: y[t:t+Tw, f] (Vector of length Tw)
    Dictionary Atoms: shifted x vectors.
    Atom k: x[t-k : t-k+Tw, f] (Vector of length Tw)
    
    So for frequency f:
    Target Y (Tw,)
    Dictionary A (Tw, M) where col k is lag k.
    
    We can batch this over frequencies F.
    Input Batches: (Tw, F), Dict (Tw, M, F)
    """
    
    # Inputs:
    # X_window: (Tw + MaxLag, F)
    # Y_window: (Tw, F)
    # We want to use X to predict Y.
    
    Tw, F = y_target.shape
    M = X_history.shape[0] - Tw # Max Lag
    
    # Construct Dictionary Tensor: (F, Tw, M)
    # For each freq, we have matrix Tw x M
    
    # Pre-unfold X into dictionary
    # X_history is (Tw+M, F)
    # We want A[f, :, k] = X_history[M-k : M-k+Tw, f] ?
    # Let lag 0 be x[t]. lag k be x[t-k].
    # If Y starts at T0 (relative to X start), then Y[0] is aligned with X[T0].
    # Then Lag k is X[T0-k].
    # Ideally X_history is aligned such that X_history[-Tw:] is Lag 0.
    
    # Let's assume input X_history contains [x(t-M) ... x(t+Tw-1)].
    # Y contains [y(t) ... y(t+Tw-1)].
    # Then Dictionary col k corresponds to starting at index M-k in X_history.
    
    Dict_tensor = torch.zeros(F, Tw, M, dtype=X_history.dtype, device=X_history.device)
    for k in range(M):
        # Lag k
        # X range: [M-1-k : M-1-k+Tw] ?
        # Let's align simple: Last Tw samples of X are Lag 0.
        # X_history size: M + Tw
        # Lag 0: X[M : M+Tw]
        # Lag k: X[M-k : M+Tw-k]
        
        start = M - k
        end = M + Tw - k
        # Careful with indices.
        # If X has length M+Tw.
        # Lag 0 starts at M.
        # Lag M starts at 0.
        slice_k = X_history[M-k : M+Tw-k, :] # (Tw, F)
        Dict_tensor[:, :, k] = slice_k.T # (F, Tw)
        
    Targets = y_target.T.unsqueeze(2) # (F, Tw, 1)
    
    # Now we have Batch OMP problem:
    # Batch B=F.
    # A: (B, Tw, M)
    # y: (B, Tw, 1)
    
    # Vectorized OMP
    # Residual
    Residuals = Targets.clone()
    Active_Sets = torch.zeros(F, K_max, dtype=torch.long, device=X_history.device) - 1
    
    # Store Trajectories
    # We can't easily store trajectories for all 1024 freqs per step.
    # Just return final Active Sets?
    # Or simplified trajectory: (step, chosen_lag, residual_norm)
    
    Trajectories = []
    
    # Normalized Dictionary for correlation?
    # OMP usually requires normalized atoms.
    # Calculate norms
    Norms = torch.norm(Dict_tensor, dim=1, keepdim=True) + 1e-8 # (F, 1, M)
    Dict_Norm = Dict_tensor / Norms
    
    for k in range(K_max):
        # 1. Correlations: A^H * r
        # (B, M, Tw) @ (B, Tw, 1) -> (B, M, 1)
        Corrs = torch.bmm(Dict_Norm.conj().transpose(1, 2), Residuals)
        Abs_Corrs = torch.abs(Corrs).squeeze(2) # (B, M)
        
        # Mask selected
        # How to mask in batch? 
        # Scatter -inf
        if k > 0:
            # Gather previously selected
            # Simple loop masking for now
            for b in range(F):
                prev = Active_Sets[b, :k]
                valid_prev = prev[prev >= 0]
                Abs_Corrs[b, valid_prev] = -1.0
                
        # 2. Select
        Best_Lags = torch.argmax(Abs_Corrs, dim=1) # (F,)
        
        # Update Sets
        Active_Sets[:, k] = Best_Lags
        
        # 3. Projection (LS)
        # This is hard to fully vectorize efficiently in PyTorch without loop over batch 
        # because each batch has different active columns.
        # But F=1025.
        # We can implement a loop over F (slow) or use Cholesky update (complex).
        # For "Generation", loop over F is acceptable if Tw is small.
        
        # To speed up, maybe just do 1 step OMP (Matching Pursuit)?
        # Or just simulate K=1?
        # Let's do K=3, loop over F is okay-ish.
        
        # Update Residuals loop
        for b in range(F):
            indices = Active_Sets[b, :k+1]
            # Form A_active (Tw, k+1)
            # Unnormalized Dict
            A_b = Dict_tensor[b, :, indices].T # (Tw, k+1) NO wait
            # Dict_tensor [b, :, indices] is (Tw, k+1)
            A_active = Dict_tensor[b, :, indices] # (Tw, k+1)
            y_b = Targets[b, :, 0] # (Tw,)
            
            # LS
            h = torch.linalg.lstsq(A_active, y_b).solution
            recon = A_active @ h
            Residuals[b, :, 0] = y_b - recon
            
    return Active_Sets.cpu().numpy() # (F, K)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mic_root", type=str, default="/Users/sbplab/LDV-data-processed/speech260_original_16k_no_edge_sync_vad_normalized")
    parser.add_argument("--ldv_root", type=str, default="/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--angle", type=float, default=90.0)
    parser.add_argument("--max_items", type=int, default=100)
    args = parser.parse_args()
    
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Params
    Tw = 16 # Window size for stability
    MaxLag = 16
    
    dataset = DoALagDataset(args.mic_root, args.ldv_root, angle=args.angle)
    # Subset
    if args.max_items:
        indices = list(range(min(len(dataset), args.max_items)))
        dataset = torch.utils.data.Subset(dataset, indices)
        
    loader = create_dataloader(dataset, batch_size=1)
    
    all_data = []
    
    logger.info("Processing...")
    for i, batch in tqdm(enumerate(loader)):
        mic_stft = batch["mic_stft"][0] # (T_full, F)
        ldv_stft = batch["ldv_stft"][0]
        
        T_full, F_bins = mic_stft.shape
        
        # Stride through time
        stride = 32
        for t in range(MaxLag, T_full - Tw, stride):
            # Window Analysis
            # X Context: Needs [t-MaxLag : t+Tw]
            # Wait, definition:
            # Lag 0: X[t : t+Tw]
            # Lag Max: X[t-MaxLag : t+Tw-MaxLag]
            # So we need X from t-MaxLag to t+Tw.
            # Dict expects [MaxLag+Tw] rows.
            
            X_chunk = mic_stft[t-MaxLag : t+Tw] # (MaxLag+Tw, F)
            Y_chunk = ldv_stft[t : t+Tw]       # (Tw, F)
            
            if X_chunk.shape[0] < MaxLag+Tw or Y_chunk.shape[0] < Tw:
                continue
                
            active_lags = run_omp_lag(X_chunk, Y_chunk, K_max=3) # (F, K)
            
            # Store statistic
            # Just store counts of lag 0, 1, 2...
            # Space saving: store histogram of selected lags per chunk
            
            # Flatten active lags
            lags_flat = active_lags.flatten()
            
            blob = {
                "clip": i,
                "time": t,
                "lags_hist": np.bincount(lags_flat, minlength=MaxLag)
            }
            all_data.append(blob)
            
    torch.save(all_data, out_path / "lag_stats.pt")
    logger.info(f"Saved stats to {out_path}")

if __name__ == "__main__":
    main()
