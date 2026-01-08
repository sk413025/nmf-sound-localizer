import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scripts.h_exploration.dataset_lag import DoALagDataset, create_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_omp_lag_capture(X_history, y_target, K_max=4):
    """
    Returns:
        Active_Sets: (F, K)
        Weights: (F, K) complex
        Trajectory_Data: List of dicts per step
    """
    
    Tw, F = y_target.shape
    M = X_history.shape[0] - Tw # Max Lag
    M = min(M, 16) # Clamp M just in case logic drifts

    Dict_tensor = torch.zeros(F, Tw, M, dtype=X_history.dtype, device=X_history.device)
    for k in range(M):
        slice_k = X_history[16-k : 16+Tw-k, :] # Fixed indexing relative to MaxLag=16 assumed
        # Actually passed X_history is sized: MaxLag + Tw
        # If MaxLag=16, then indices 0..15 are history, 16 is t0.
        # k=0 (Lag 0) -> X at t, t+1... -> indices [16 : 16+Tw]
        # k=1 (Lag 1) -> X at t-1... -> indices [15 : 15+Tw]
        start = 16 - k
        end = 16 + Tw - k
        slice_k = X_history[start : end, :]
        Dict_tensor[:, :, k] = slice_k.T # (F, Tw)
        
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
    
    initial_norms = torch.norm(Targets, dim=1).squeeze() # (F)
    
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
            A_active = Dict_tensor[b, :, indices] # (Tw, k+1)
            y_b = Targets[b, :, 0] # (Tw,)
            
            # LS
            h = torch.linalg.lstsq(A_active, y_b).solution.flatten() # (k+1,)
            
            # Store current weights
            Weights[b, :k+1] = h
            
            recon = A_active @ h
            Residuals[b, :, 0] = y_b - recon
            
    final_norms = torch.norm(Residuals, dim=1).squeeze() # (F)
    
    score_improvement = (initial_norms - final_norms) / (initial_norms + 1e-6)
    
    traj = {
        "correlations": torch.stack(All_Corrs, dim=1), # (F, K, M)
        "actions": torch.stack(All_Actions, dim=1),    # (F, K)
        "scores": score_improvement.cpu() # (F)
    }
            
    return Active_Sets.cpu().numpy(), Weights.cpu().numpy(), traj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mic_root", type=str, default="/Users/sbplab/LDV-data-processed/speech260_original_16k_no_edge_sync_vad_normalized")
    parser.add_argument("--ldv_root", type=str, default="/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--angle", type=float, default=90.0)
    parser.add_argument("--max_items", type=int, default=100)
    parser.add_argument("--hop_length", type=int, default=None, help="Override STFT hop length (e.g. 160 for 10ms)")
    args = parser.parse_args()
    
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Params
    Tw = 16 
    MaxLag = 16

    dataset = DoALagDataset(args.mic_root, args.ldv_root, angle=args.angle, hop_length=args.hop_length)
    if args.max_items:
        indices = list(range(min(len(dataset), args.max_items)))
        dataset = torch.utils.data.Subset(dataset, indices)
    
    loader = create_dataloader(dataset, batch_size=1)
    
    all_trajectories = []
    
    logger.info(f"Processing Trajectories... Hop: {args.hop_length if args.hop_length else 'Default'}")
    
    stride = 32
    # Ensure dataset_lag returns context of length MaxLag + Tw
    # Actually my previous implementation of dataset_lag might need checking on chunk sizes.
    # dataset_lag returns FULL CLIP stft.
    
    for i, batch in tqdm(enumerate(loader)):
        mic_stft = batch["mic_stft"][0] 
        ldv_stft = batch["ldv_stft"][0]
        
        T_full, F_bins = mic_stft.shape
        
        for t in range(MaxLag, T_full - Tw, stride):
            # Context window: [t - MaxLag : t + Tw]
            # MaxLag=16, Tw=16 -> Length 32
            # Indices: 0..15 (History), 16..31 (Target Win)
            # t needs to be at least MaxLag
            
            X_chunk = mic_stft[t-MaxLag : t+Tw]
            Y_chunk = ldv_stft[t : t+Tw]
            
            if X_chunk.shape[0] < MaxLag+Tw or Y_chunk.shape[0] < Tw:
                continue
                
            _, _, traj = run_omp_lag_capture(X_chunk, Y_chunk, K_max=3)
            
            corrs = traj["correlations"] # (F, K, M)
            actions = traj["actions"]
            scores = traj["scores"]
            
            # Save block
            all_trajectories.append({
                "corrs": corrs.half(),
                "actions": actions.to(torch.int8),
                "scores": scores.half()
            })

            # if len(all_trajectories) > 500: break # Safety limit for quick test? 
            # No, user said "start measuring". Let's run full set of 100 items.
    
    torch.save(all_trajectories, out_path / "lag_trajectories.pt")
    logger.info(f"Saved {len(all_trajectories)} blocks to {out_path}")

if __name__ == "__main__":
    main()
