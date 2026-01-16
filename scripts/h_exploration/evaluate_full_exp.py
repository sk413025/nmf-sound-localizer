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
        res_omp = run_omp_oracle(Dict_tensor, Targets, K_max=8)
        
        # 2. DTmin
        freq_indices = torch.arange(start_bin, end_bin, device=device)
        res_dt = run_dtmin_policy(model, Dict_tensor, Targets, freq_indices, K_max=8)
        
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
    # Total bins per clip might vary slightly if some are masked, but typically it is fixed range.
    # We essentially flattened everything.
    # But we know the order: It iterates clips, and for each clip it iterates bins start_bin to end_bin.
    # So the stream is: [Clip0_Bin5, Clip0_Bin6... Clip0_Bin1024, Clip1_Bin5....]
    
    n_bins_range = end_bin - start_bin # 1024 - 5 = 1019
    total_samples = len(all_omp)
    
    # Verify shape
    if total_samples % n_bins_range != 0:
        logger.warning(f"Shape Mismatch! Total {total_samples} not divisible by range {n_bins_range}")
        # Fallback to global mean
    else:
        n_clips_actual = total_samples // n_bins_range
        print(f"Reshaping data for {n_clips_actual} clips x {n_bins_range} bins...")
        
        omp_matrix = all_omp.reshape(n_clips_actual, n_bins_range)
        dt_matrix = all_dt.reshape(n_clips_actual, n_bins_range)
        
        # Define Band Indices (Relative to start_bin 5)
        # Low: 5 - 300  (Indices 0 - 295)
        # Mid: 300 - 800 (Indices 295 - 795)
        # High: 800 - 1024 (Indices 795 - 1019)
        
        idx_300 = 300 - start_bin
        idx_800 = 800 - start_bin
        
        print("\n--- Detailed Band Analysis ---")
        
        bands = {
            "Low (5-300Hz Bins)": (0, idx_300),
            "Mid (300-800Hz Bins)": (idx_300, idx_800),
            "High (800-1024Hz Bins)": (idx_800, n_bins_range)
        }
        
        print(f"{'Band':<25} | {'OMP (Oracle)':<12} | {'DTmin':<12} | {'Recovery':<10}")
        print("-" * 65)
        
        for name, (s, e) in bands.items():
            omp_band = omp_matrix[:, s:e]
            dt_band = dt_matrix[:, s:e]
            
            m_omp = np.mean(omp_band)
            m_dt = np.mean(dt_band)
            rec = m_dt / m_omp if m_omp > 1e-6 else 0
            
            print(f"{name:<25} | {m_omp:.2%}     | {m_dt:.2%}     | {rec:.2%}")
            
    print("\n" + "="*40)
    print(" EVALUATION RESULTS (Energy Reduction)")
    print("="*40)
    print(f"Total Bins Evaluated: {len(all_omp)}")
    print(f"OMP (Oracle) Mean Reduction: {np.mean(all_omp):.2%}")
    print(f"DTmin (Model) Mean Reduction: {np.mean(all_dt):.2%}")
    print(f"Ratio (DT/OMP): {np.mean(all_dt)/np.mean(all_omp):.2%}")
    print("-" * 40)

if __name__ == "__main__":
    evaluate()
