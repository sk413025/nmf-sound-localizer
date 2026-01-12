import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scripts.h_exploration.dataset import DoAPairDataset, create_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def solve_ls(X, y):
    # X: (T, K), y: (T,)
    # Solve h = (X^H X)^-1 X^H y
    # Uses torch.linalg.lstsq
    # Expected shapes: X (Batch, T, K), y (Batch, T, 1) or similar.
    # Here we process single frequency, so (T, K).
    
    if X.shape[1] == 0:
        return torch.zeros(0, dtype=X.dtype, device=X.device)
        
    solution = torch.linalg.lstsq(X, y).solution # (K,)
    return solution

def run_omp(X, y, K_max=6):
    """
    Run OMP for a single frequency bin.
    X: (T, C) Complex
    y: (T,) Complex
    """
    T, C = X.shape
    
    residual = y.clone()
    active_indices = []
    
    trajectory = []
    
    current_h = torch.zeros(C, dtype=X.dtype, device=X.device)
    
    for k in range(K_max):
        # 1. Correlations
        # alpha = X^H * r
        correlations = torch.matmul(X.conj().T, residual) # (C,)
        
        # State: Magnitude of correlations? or Complex?
        # DTmin usually takes Real inputs. Let's take Magnitude for now, or concat Real/Imag.
        # Let's use Magnitude of correlations as the "Observation" of alignment.
        state_obs = torch.abs(correlations) 
        
        # 2. Select Atom
        # Mask already selected
        correlations_masked = state_obs.clone()
        if active_indices:
            correlations_masked[active_indices] = -1.0
            
        best_idx = torch.argmax(correlations_masked).item()
        
        # Check if already selected (should not happen with masking)
        if best_idx in active_indices:
            break
            
        # 3. Update Set
        active_indices.append(best_idx)
        
        # 4. Solve LS
        X_active = X[:, active_indices] # (T, k+1)
        h_active = solve_ls(X_active, y) # (k+1,)
        
        # 5. Update Residual
        # Recon = X_active * h_active
        recon = torch.matmul(X_active, h_active)
        residual = y - recon
        
        # Calc Reconstruction Error/Reward
        mse = torch.mean(torch.abs(residual)**2).item()
        # Reward = -MSE or Improvement?
        # Let's store pure MSE for now.
        
        step_data = {
            "step": k,
            "observation": state_obs.cpu().numpy(), # (C,)
            "action": best_idx,
            "mse": mse,
            "active_set": list(active_indices)
        }
        trajectory.append(step_data)
        
        if mse < 1e-6:
            break
            
    return trajectory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mic_root", type=str, default="/Users/sbplab/LDV-data-processed/speech260_original_16k_no_edge_sync_vad_normalized")
    parser.add_argument("--ldv_root", type=str, default="/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_items", type=int, default=100, help="For testing, limit items")
    args = parser.parse_args()
    
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    dataset = DoAPairDataset(args.mic_root, args.ldv_root)
    # Subset for testing
    if args.max_items:
        indices = list(range(min(len(dataset), args.max_items)))
        dataset = torch.utils.data.Subset(dataset, indices)
        
    loader = create_dataloader(dataset, batch_size=1, shuffle=False)
    
    logger.info(f"Processing {len(loader)} clips...")
    
    all_trajectories = []
    
    for i, batch in tqdm(enumerate(loader)):
        X_full = batch["X"][0] # (F, T, C)
        Y_full = batch["Y"][0] # (F, T)
        angle = batch["angle"][0]
        
        # Process subset of frequencies to save space? Or all?
        # Let's process a stride of frequencies.
        F_bins = X_full.shape[0]
        
        # Stride 10 for diversity
        for f in range(0, F_bins, 10):
            X_f = X_full[f] # (T, C)
            y_f = Y_full[f] # (T,)
            
            traj = run_omp(X_f, y_f, K_max=X_f.shape[1])
            
            # Pack metadata
            traj_blob = {
                "clip_idx": i,
                "freq_bin": f,
                "angle": angle,
                "steps": traj
            }
            all_trajectories.append(traj_blob)
            
    # Save
    save_path = out_path / "omp_trajectories.pt"
    torch.save(all_trajectories, save_path)
    logger.info(f"Saved {len(all_trajectories)} trajectories to {save_path}")

if __name__ == "__main__":
    main()
