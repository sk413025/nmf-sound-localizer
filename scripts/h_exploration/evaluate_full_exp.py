import torch
import numpy as np
import argparse
import logging
from tqdm import tqdm
from pathlib import Path
from scripts.h_exploration.dataset_lag import DoALagDataset
from scripts.h_exploration.train_dt_lag_seq_rtg import SeqDT_FreqAware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_omp_oracle(Dict_tensor, Targets, K_max=3):
    """
    Run Standard OMP (Oracle) for all bins in parallel.
    Dict_tensor: (F, Tw, MaxLag)
    Targets: (F, Tw, 1)
    """
    F, Tw, MaxLag = Dict_tensor.shape
    device = Dict_tensor.device
    
    Residuals = Targets.clone()
    Active_Sets = torch.zeros(F, K_max, dtype=torch.long, device=device) - 1
    
    # Normalize Dict
    D_norms = torch.linalg.vector_norm(Dict_tensor.abs(), dim=1, keepdim=True) + 1e-8
    Dict_Norm = Dict_tensor / D_norms
    
    for k in range(K_max):
        # 1. Correlations
        Corrs = torch.bmm(Dict_Norm.conj().transpose(1, 2), Residuals)
        Abs_Corrs = torch.abs(Corrs).squeeze(2) # (F, M)
        
        # Mask
        if k > 0:
            for b in range(F):
                prev = Active_Sets[b, :k]
                valid_prev = prev[prev >= 0]
                Abs_Corrs[b, valid_prev] = -1.0
                
        # 2. Select
        Best_Lags = torch.argmax(Abs_Corrs, dim=1) # (F,)
        Active_Sets[:, k] = Best_Lags
        
        # 3. Update Residuals (Physics)
        for b in range(F):
            indices = Active_Sets[b, :k+1]
            A_active = Dict_tensor[b, :, indices] # (Tw, k+1)
            y_b = Targets[b, :, 0]
            
            h = torch.linalg.lstsq(A_active, y_b).solution.flatten()
            recon = A_active @ h
            Residuals[b, :, 0] = y_b - recon
            
    return Residuals

def run_dtmin_policy(model, Dict_tensor, Targets, freq_indices, K_max=3):
    """
    Run DTmin Policy for all bins in parallel.
    """
    F, Tw, MaxLag = Dict_tensor.shape
    device = Dict_tensor.device
    
    Residuals = Targets.clone()
    Active_Sets = torch.zeros(F, K_max, dtype=torch.long, device=device) - 1
    
    D_norms = torch.linalg.vector_norm(Dict_tensor.abs(), dim=1, keepdim=True) + 1e-8
    Dict_Norm = Dict_tensor / D_norms
    
    # Initial Calculation for Reduction Tracking
    init_energy = torch.linalg.vector_norm(Targets, dim=1).squeeze()**2 # (F,)
    current_energy = init_energy.clone()
    
    # State Tracking
    # Past Correlations needs to be grown
    past_corrs = None
    past_rtg = None
    
    # Desired RTG: We always want 100% reduction eventually
    # RTG = 1.0 - achieved_reduction
    # At step 0: achieved=0.0 -> RTG=1.0
    
    for k in range(K_max):
        # 1. Observation
        Corrs = torch.bmm(Dict_Norm.conj().transpose(1, 2), Residuals)
        Obs_Corrs = Corrs.abs().squeeze(2) # (F, M)
        
        # Calculate RTG
        achieved_red = (init_energy - current_energy) / (init_energy + 1e-9)
        rtg_val = torch.clamp(1.0 - achieved_red, 0.0, 1.0) # (F,)
        
        # Update Sequence Inputs
        if k == 0:
            past_corrs = Obs_Corrs.unsqueeze(1) # (F, 1, M)
            past_rtg = rtg_val.unsqueeze(1)     # (F, 1)
        else:
            past_corrs = torch.cat([past_corrs, Obs_Corrs.unsqueeze(1)], dim=1)
            past_rtg = torch.cat([past_rtg, rtg_val.unsqueeze(1)], dim=1)
            
        # 2. Model Forward
        # freq_indices: (F,)
        logits = model(past_corrs, past_rtg, freq_indices) # (F, k+1, M)
        
        # Take action for current step
        step_logits = logits[:, -1, :] # (F, M)
        
        # Mask previous actions
        if k > 0:
            for b in range(F):
                prev = Active_Sets[b, :k]
                valid_prev = prev[prev >= 0]
                step_logits[b, valid_prev] = -1e9
                
        actions = torch.argmax(step_logits, dim=1)
        Active_Sets[:, k] = actions
        
        # 3. Physics Update
        for b in range(F):
            indices = Active_Sets[b, :k+1]
            A_active = Dict_tensor[b, :, indices]
            y_b = Targets[b, :, 0]
            
            h = torch.linalg.lstsq(A_active, y_b).solution.flatten()
            recon = A_active @ h
            Residuals[b, :, 0] = y_b - recon
            
        # Update Energy
        current_energy = torch.linalg.vector_norm(Residuals, dim=1).squeeze()**2
        
    return Residuals

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_clips", type=int, default=10)
    args = parser.parse_args()
    
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu") # Force CPU to avoid MPS lstsq issues
    
    # Load Model
    model = SeqDT_FreqAware(M_lags=16, d_model=128, hidden_dim=256, max_freq=1025)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # Dataset
    dataset = DoALagDataset("/Users/sbplab/LDV-data-processed/speech260_original_16k_no_edge_sync_vad_normalized", 
                            "/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized", 
                            angle=90.0)
    
    # Params
    Tw = 16
    MaxLag = 16
    start_t = 32
    start_bin = 5
    end_bin = 1024 # Full range now
    
    omp_reductions = []
    dt_reductions = []
    
    logger.info(f"Evaluating on {args.max_clips} clips... Range: {start_bin}-{end_bin}")
    
    for i in tqdm(range(min(len(dataset), args.max_clips))):
        item = dataset[i]
        mic_stft = item["mic_stft"].to(device)
        ldv_stft = item["ldv_stft"].to(device)
        
        # Chunk
        X_chunk = mic_stft[start_t-MaxLag : start_t+Tw]
        Y_chunk = ldv_stft[start_t : start_t+Tw]
        
        # Freq Slice
        X_chunk = X_chunk[:, start_bin:end_bin]
        Y_chunk = Y_chunk[:, start_bin:end_bin]
        
        n_bins = X_chunk.shape[1]
        
        # Build Batch Dicts
        Dict_tensor = torch.zeros(n_bins, Tw, MaxLag, dtype=X_chunk.dtype, device=device)
        for k in range(MaxLag):
            x_lag = X_chunk[16-k : 16+Tw-k, :]
            Dict_tensor[:, :, k] = x_lag.T
            
        Targets = Y_chunk.T.unsqueeze(2)
        
        # 1. Oracle (OMP)
        res_omp = run_omp_oracle(Dict_tensor, Targets)
        
        # 2. DTmin
        freq_indices = torch.arange(start_bin, end_bin, device=device)
        res_dt = run_dtmin_policy(model, Dict_tensor, Targets, freq_indices)
        
        # Calculate Energies
        init_E = torch.linalg.vector_norm(Targets, dim=1).squeeze()**2
        omp_E = torch.linalg.vector_norm(res_omp, dim=1).squeeze()**2
        dt_E = torch.linalg.vector_norm(res_dt, dim=1).squeeze()**2
        
        # Valid Mask (Avoid divide by zero)
        mask = init_E > 1e-9
        
        red_omp = (init_E[mask] - omp_E[mask]) / init_E[mask]
        red_dt = (init_E[mask] - dt_E[mask]) / init_E[mask]
        
        omp_reductions.append(red_omp.cpu().numpy())
        dt_reductions.append(red_dt.cpu().numpy())
        
    all_omp = np.concatenate(omp_reductions)
    all_dt = np.concatenate(dt_reductions)
    
    # Analyze by Bands
    # Total bins per clip = 1019 (5 to 1024)
    # We have flat array (N_clips * N_bins)
    # Let's reshape to (N_clips, N_bins) for easier indexing if clips are uniform.
    # dataset[i] might have different lengths? No, STFT F dim is constant.
    n_clips = len(omp_reductions)
    n_bins_per_clip = omp_reductions[0].shape[0]
    
    omp_matrix = all_omp.reshape(n_clips, n_bins_per_clip) # (10, 1019)
    
    # Define Bands (Indices relative to start_bin=5)
    # Low: Bin 5-100 (Indices 0-95)
    # Mid: Bin 100-500 (Indices 95-495)
    # High: Bin 500-1024 (Indices 495-1019)
    
    low_band = omp_matrix[:, 0:95]
    mid_band = omp_matrix[:, 95:495]
    high_band = omp_matrix[:, 495:]
    
    print("\n" + "="*40)
    print(" EVALUATION RESULTS (Energy Reduction)")
    print("="*40)
    print(f"Total Bins Evaluated: {len(all_omp)}")
    print(f"OMP (Oracle) Mean Reduction: {np.mean(all_omp):.2%}")
    print(f"DTmin (Model) Mean Reduction: {np.mean(all_dt):.2%}")
    print(f"Ratio (DT/OMP): {np.mean(all_dt)/np.mean(all_omp):.2%}")
    
    print("\n--- OMP Performance by Frequency Band ---")
    print(f"Low Freq (Bin 5-100)   : {np.mean(low_band):.2%}")
    print(f"Mid Freq (Bin 100-500) : {np.mean(mid_band):.2%}")
    print(f"High Freq (Bin 500+)   : {np.mean(high_band):.2%}")
    print("-" * 40)

if __name__ == "__main__":
    evaluate()
