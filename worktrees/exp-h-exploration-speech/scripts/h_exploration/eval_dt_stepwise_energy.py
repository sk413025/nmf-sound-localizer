import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scripts.h_exploration.dataset_lag import DoALagDataset, create_dataloader
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StepwiseDT(nn.Module):
    def __init__(self, M_lags=16, d_model=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(M_lags, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, M_lags)
        )
        
    def forward(self, x):
        return self.net(x)

def run_omp_with_agent(X_history, y_target, agent_model, K_max=3, method="model"):
    """
    Run OMP but ask agent_model for indices instead of argmax(Corr).
    method: "model" or "lag0"
    """
    Tw, F = y_target.shape
    M = 16 # Fixed
    
    # Build Dictionary
    Dict_tensor = torch.zeros(F, Tw, M, dtype=X_history.dtype, device=X_history.device)
    for k in range(M):
        start = 16 - k
        end = 16 + Tw - k
        slice_k = X_history[start : end, :]
        Dict_tensor[:, :, k] = slice_k.T 
        
    Targets = y_target.T.unsqueeze(2) # (F, Tw, 1)
    
    Residuals = Targets.clone()
    Active_Sets = torch.zeros(F, K_max, dtype=torch.long, device=X_history.device) - 1
    
    # Complex Norm fix
    Norms = torch.sqrt(torch.sum(Dict_tensor.real**2 + Dict_tensor.imag**2, dim=1, keepdim=True)) + 1e-8
    Dict_Norm = Dict_tensor / Norms
    
    T_sq = Targets.squeeze(2)
    initial_energy = torch.sum(T_sq.real**2 + T_sq.imag**2, dim=1) # (F,)
    
    for k in range(K_max):
        # 1. State: Correlations
        Corrs = torch.bmm(Dict_Norm.conj().transpose(1, 2), Residuals)
        Abs_Corrs = torch.abs(Corrs).squeeze(2) # (F, M)
        
        # 2. Agent Action
        if method == "model":
            # Input to agent: Abs_Corrs (F, M)
            with torch.no_grad():
                logits = agent_model(Abs_Corrs.float()) # (F, M)
                if k > 0:
                    for b in range(F):
                        # Mask globally for simplicity or just sequential?
                        # Mask strictly chosen ones
                        prev = Active_Sets[b, :k]
                        valid_prev = prev[prev >= 0]
                        logits[b, valid_prev] = -float('inf')

                actions = torch.argmax(logits, dim=1) # (F,)
        elif method == "lag0":
            # Always pick Lag 0, then Lag 1, then Lag 2
            # Or just Lag 0? No, we need K actions.
            # Let's pick k (0, 1, 2)
            actions = torch.full((F,), k, dtype=torch.long, device=X_history.device)
            
        Active_Sets[:, k] = actions
        
        # 3. Projection (True Physics)
        for b in range(F):
            indices = Active_Sets[b, :k+1]
            A_active = Dict_tensor[b, :, indices]
            y_b = Targets[b, :, 0]
            
            h = torch.linalg.lstsq(A_active, y_b).solution.flatten()
            recon = A_active @ h
            Residuals[b, :, 0] = y_b - recon
            
    R_sq = Residuals.squeeze(2)
    final_energy = torch.sum(R_sq.real**2 + R_sq.imag**2, dim=1)
    reduction = (initial_energy - final_energy) / (initial_energy + 1e-8)
    
    return reduction # (F,)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mic_root", type=str, default="/Users/sbplab/LDV-data-processed/speech260_original_16k_no_edge_sync_vad_normalized")
    parser.add_argument("--ldv_root", type=str, default="/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized")
    parser.add_argument("--model_path", type=str, default="results/dtmin_lag/dt_lag_stepwise.pth")
    parser.add_argument("--hop_length", type=int, default=160)
    parser.add_argument("--method", type=str, default="model", choices=["model", "lag0"])
    args = parser.parse_args()
    
    # Load Model if needed
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = None
    if args.method == "model":
        model = StepwiseDT(M_lags=16, d_model=256)
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
    
    # Load Data
    dataset = DoALagDataset(args.mic_root, args.ldv_root, angle=90.0, hop_length=args.hop_length)
    indices = list(range(20)) # Small subset for eval
    dataset = torch.utils.data.Subset(dataset, indices)
    loader = create_dataloader(dataset, batch_size=1)
    
    all_reductions = []
    
    logger.info(f"Evaluating Agent ({args.method})...")
    
    stride = 32
    MaxLag = 16
    Tw = 16
    
    for i, batch in tqdm(enumerate(loader)):
        mic_stft = batch["mic_stft"][0].to(device)
        ldv_stft = batch["ldv_stft"][0].to(device)
        
        T_full, F_bins = mic_stft.shape
        
        for t in range(MaxLag, T_full - Tw, stride):
            X_chunk = mic_stft[t-MaxLag : t+Tw]
            Y_chunk = ldv_stft[t : t+Tw]
            
            if X_chunk.shape[0] < MaxLag+Tw or Y_chunk.shape[0] < Tw:
                continue
                
            reductions = run_omp_with_agent(X_chunk, Y_chunk, model, K_max=3, method=args.method)
            all_reductions.append(reductions.cpu().numpy())
            
    all_reductions = np.concatenate(all_reductions)
    
    mean_red = np.mean(all_reductions)
    median_red = np.median(all_reductions)
    
    logger.info(f"Energy Reduction ({args.method}):")
    logger.info(f"Mean: {mean_red:.4f} ({mean_red*100:.2f}%)")
    logger.info(f"Median: {median_red:.4f}")

if __name__ == "__main__":
    main()
