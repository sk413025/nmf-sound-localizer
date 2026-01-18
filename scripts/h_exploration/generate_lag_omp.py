import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scripts.h_exploration.dataset_lag import DoALagDataset, create_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_omp_lag_capture(X_history, y_target, K_max=16, Lag_Min=-32, Lag_Max=32):
    """
    X_history: (L, F) where L is large enough to cover range.
               With Lag_Min=-32 (Future), Lag_Max=32 (Past).
               Convention: y[t] = sum x[t-k] * w_k
    y_target: (Tw, F)
    
    Returns:
        Active_Sets: (F, K)
        Weights: (F, K) complex
        Trajectory_Data: List of dicts per step
    """
    
    Tw, F = y_target.shape
    device = X_history.device
    
    # Lag Range
    # Lags k from Lag_Min to Lag_Max
    # e.g. -32 to 32
    # Total Atoms M = Lag_Max - Lag_Min + 1 (e.g. 65)
    Lags = torch.arange(Lag_Min, Lag_Max + 1, device=device)
    M = len(Lags)

    # Dictionary Construction
    # D[:, :, m] corresponds to Lag Lags[m]
    # x[t - Lags[m]]
    # X_history passed in should center around t=0 at some specific index.
    # To simplify, we assume X_history is passed as [t - Lag_Max : t + Tw - Lag_Min]
    # Let's define the center.
    # Actually, simpler: Main loop ensures X_history is big enough.
    # We define X_history as starting from (t - Lag_Max).
    # So index 0 corresponds to t - Lag_Max.
    # Index at t is Lag_Max.
    # For Lag k: need x[t-k ... t+Tw-k]
    # Start idx in X_history: (t-k) - (t-Lag_Max) = Lag_Max - k
    
    Dict_tensor = torch.zeros(F, Tw, M, dtype=X_history.dtype, device=device)
    
    for m, k in enumerate(Lags):
        # k is the actual lag value (e.g. -32)
        # Start index in X_history relative to t=0
        # If X_history starts at t - Lag_Max
        start_idx = int(Lag_Max - k)
        end_idx = int(start_idx + Tw)
        
        # Valid check?
        if end_idx > X_history.shape[0]:
             # Pad or clamp? Should be handled by data fetcher
             raise ValueError(f"X_history too short for Lag {k}. Need {end_idx}, got {X_history.shape[0]}")

        slice_k = X_history[start_idx : end_idx, :] 
        Dict_tensor[:, :, m] = slice_k.T # (F, Tw)
        
    Targets = y_target.T.unsqueeze(2) # (F, Tw, 1)
    
    Residuals = Targets.clone()
    Active_Sets = torch.zeros(F, K_max, dtype=torch.long, device=X_history.device) - 1
    Weights = torch.zeros(F, K_max, dtype=X_history.dtype, device=X_history.device)
    
    Norms = torch.norm(Dict_tensor, dim=1, keepdim=True) + 1e-8
    Dict_Norm = Dict_tensor / Norms
    
    # Store trajectory data
    # (F, K_max, M)
    All_Corrs = []
    All_Actions = []
    All_Reductions = [] # Store cumulative reduction at each step
    
    initial_norms = torch.norm(Targets, dim=1).squeeze() # (F)
    initial_energy = initial_norms ** 2
    
    current_norms = initial_norms.clone()
    
    # Run OMP to absolute max first
    for k in range(K_max):
        # 1. Correlations
        Corrs = torch.bmm(Dict_Norm.conj().transpose(1, 2), Residuals)
        Abs_Corrs = torch.abs(Corrs).squeeze(2) # (B, M)
        
        # Store raw correlations 
        All_Corrs.append(Abs_Corrs.cpu())
        
        # Mask
        if k > 0:
            for b in range(F):
                prev = Active_Sets[b, :k]
                valid_prev = prev[prev >= 0]
                Abs_Corrs[b, valid_prev] = -1.0
                
        # 2. Select
        Best_Lags = torch.argmax(Abs_Corrs, dim=1) # (F,)
        Active_Sets[:, k] = Best_Lags
        
        All_Actions.append(Best_Lags.cpu())
        
        # 3. Projection & Weights
        for b in range(F):
            indices = Active_Sets[b, :k+1]
            A_active = Dict_Norm[b, :, indices] # (Tw, k+1)
            y_b = Targets[b, :, 0] # (Tw,)
            
            # LS
            h = torch.linalg.lstsq(A_active, y_b).solution.flatten()
            
            # Store current weights
            Weights[b, :k+1] = h
            
            recon = A_active @ h
            Residuals[b, :, 0] = y_b - recon
            
        # Calc cumulative reduction so far (Energy Based)
        current_energy_vec = torch.norm(Residuals, dim=1).squeeze() ** 2
        reductions = (initial_energy - current_energy_vec) / (initial_energy + 1e-6)
        All_Reductions.append(reductions.cpu())

    # Stack full execution
    Full_Corrs = torch.stack(All_Corrs, dim=1)         # (F, K_max, M)
    Full_Actions = torch.stack(All_Actions, dim=1)     # (F, K_max)
    Full_Reductions = torch.stack(All_Reductions, dim=1) # (F, K_max)
    
    return Full_Corrs, Full_Actions, Full_Reductions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mic_root", type=str, default="/Users/sbplab/LDV-data-processed/speech260_original_16k_no_edge_sync_vad_normalized")
    parser.add_argument("--ldv_root", type=str, default="/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--angle", type=float, default=90.0)
    parser.add_argument("--max_items", type=int, default=100)
    parser.add_argument("--hop_length", type=int, default=None, help="Override STFT hop length (e.g. 160 for 10ms)")
    parser.add_argument("--variants_per_clip", type=int, default=5, help="Number of random K variants to extract per clip")
    parser.add_argument("--gain", type=float, default=1.0, help="Scale mic/ldv STFT by this factor to avoid epsilon-floor artifacts")
    
    # New Params for MaxLag
    parser.add_argument("--max_lag", type=int, default=32, help="Max lag (both future and past). i.e. [-max_lag, max_lag]")
    parser.add_argument("--max_k", type=int, default=32, help="Max OMP steps (budget K)")
    parser.add_argument("--tw", type=int, default=16, help="Time window length for prediction")
    
    args = parser.parse_args()
    
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Params
    Tw = args.tw
    Lag_Max = args.max_lag
    Lag_Min = -args.max_lag # Symmetric
    Absolute_K = args.max_k 

    logger.info(f"Config: Lag [{Lag_Min}, {Lag_Max}], Tw={Tw}, K={Absolute_K}, Gain={args.gain}")

    dataset = DoALagDataset(args.mic_root, args.ldv_root, angle=args.angle, hop_length=args.hop_length)
    if args.max_items:
        indices = list(range(min(len(dataset), args.max_items)))
        dataset = torch.utils.data.Subset(dataset, indices)
    
    loader = create_dataloader(dataset, batch_size=1)
    
    all_trajectories = []
    
    logger.info(f"Processing Trajectories... Hop: {args.hop_length if args.hop_length else 'Default'}")
    
    stride = 32
    
    for i, batch in tqdm(enumerate(loader)):
        mic_stft = batch["mic_stft"][0] 
        ldv_stft = batch["ldv_stft"][0]
        
        T_full, F_bins = mic_stft.shape
        
        # Determine strict bounds for t
        # Valid t must allow [t - Lag_Max] to be >= 0
        # And [t + Tw - 1 - Lag_Min] to be < T_full
        # Note: end index in slice is exclusive, so [t + Tw - Lag_Min] <= T_full
        
        start_t = Lag_Max
        end_t_limit = T_full - Tw + Lag_Min # Lag_Min is negative, so this shrinks the range
        
        if end_t_limit <= start_t:
            continue
            
        for t in range(start_t, end_t_limit, stride):
            # Context window: 
            # Needs to cover from [t - Lag_Max] to [t + Tw - Lag_Min]
            
            chunk_start = t - Lag_Max
            chunk_end = t + Tw - Lag_Min
            
            X_chunk = mic_stft[chunk_start : chunk_end]
            Y_chunk = ldv_stft[t : t+Tw]

            if args.gain != 1.0:
                X_chunk = X_chunk * float(args.gain)
                Y_chunk = Y_chunk * float(args.gain)
            
            if X_chunk.shape[0] < (chunk_end - chunk_start) or Y_chunk.shape[0] < Tw:
                continue
                
            # Run OMP to absolute max K
            # Lag_Min, Lag_Max must be passed
            f_corrs, f_actions, f_reds = run_omp_lag_capture(
                X_chunk, Y_chunk, 
                K_max=Absolute_K,
                Lag_Min=Lag_Min,
                Lag_Max=Lag_Max
            )
            
            # Generate Random Variants
            # We want to create N variants with random K length \in [1, MaxK]
            
            possible_ks = list(range(1, Absolute_K + 1))
            
            if args.variants_per_clip >= Absolute_K:
                selected_ks = possible_ks
            else:
                # Random selection without replacement
                selected_ks = np.random.choice(possible_ks, size=args.variants_per_clip, replace=False)
            
            for k_len in selected_ks:
                # Slice the trajectory
                # k_len is 1-based index, so slice [:k_len]
                
                sub_corrs = f_corrs[:, :k_len, :]
                sub_actions = f_actions[:, :k_len]
                sub_reductions = f_reds[:, :k_len]
                
                # The "Target" for this episode is the reduction achieved at the FINAL step of this sub-trajectory
                # i.e. index [k_len-1]
                
                final_scores_k = sub_reductions[:, -1] # (F,)
                
                all_trajectories.append({
                    "corrs": sub_corrs.half(),
                    "actions": sub_actions.to(torch.int8),
                    "reductions": sub_reductions.half(),
                    "scores": final_scores_k.half() # This becomes the Initial RTG base
                })

            # if len(all_trajectories) > 500: break 
    
    torch.save(all_trajectories, out_path / "lag_trajectories.pt")
    logger.info(f"Saved {len(all_trajectories)} blocks to {out_path} (Variants: {args.variants_per_clip})")


if __name__ == "__main__":
    main()
