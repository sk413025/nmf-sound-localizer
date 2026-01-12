import torch
import numpy as np
import argparse
from scripts.h_exploration.dataset_lag import DoALagDataset
from scripts.h_exploration.train_dt_lag_seq_rtg import SeqDT_FreqAware
import logging

logging.basicConfig(level=logging.ERROR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--clip_idx", type=int, default=0)
    args = parser.parse_args()
    
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu") # Force CPU to debug math
    
    # 1. Load Model
    model = SeqDT_FreqAware(M_lags=16, d_model=128, hidden_dim=256, max_freq=1025)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # 2. Load Real Physics Data
    dataset = DoALagDataset("/Users/sbplab/LDV-data-processed/speech260_original_16k_no_edge_sync_vad_normalized", 
                            "/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized", 
                            angle=90.0)
    
    item = dataset[args.clip_idx]
    mic_stft = item["mic_stft"].to(device)
    ldv_stft = item["ldv_stft"].to(device)
    
    # Chunk Definitions
    Tw = 16
    MaxLag = 16
    start_t = 32
    K_steps = 3
    
    start_bin = 5
    end_bin = 300
    
    X_chunk = mic_stft[start_t-MaxLag : start_t+Tw] # (32, 1025)
    Y_chunk = ldv_stft[start_t : start_t+Tw]        # (16, 1025)
    
    # =========================================================================
    # Run Model Inference Across All Bins in Parallel
    # =========================================================================
    print(f"Validating Global DTmin Performance (Bins {start_bin}-{end_bin})...")
    
    # We will compute physics updates PER BIN to get true reduction
    # But we want to batch inference if possible.
    # The model takes (B, K, M). We can treat frequencies as Batch dim B=F.
    
    # Prepare Dicts for all bins (needed for physics update)
    # Shape: (F_sub, Tw, MaxLag)
    n_bins = end_bin - start_bin
    
    # 1. Build Dictionary (Physics Engine)
    # This is efficient vectorized dict construction
    Dict_tensor = torch.zeros(n_bins, Tw, MaxLag, dtype=X_chunk.dtype, device=device)
    for k in range(MaxLag):
        # Slice X for Lag k corresponding to all bins
        # X_chunk shape (32, 1025)
        # We need X[16-k : 16+Tw-k, start_bin:end_bin]
        x_lag = X_chunk[16-k : 16+Tw-k, start_bin:end_bin] # (Tw, n_bins)
        Dict_tensor[:, :, k] = x_lag.T # (n_bins, Tw)
        
    # Normalize Dicts
    D_norms = torch.linalg.vector_norm(Dict_tensor.abs(), dim=1, keepdim=True) + 1e-8
    Dict_Norm = Dict_tensor / D_norms
    
    # Prepare Targets / Residuals
    Targets = Y_chunk[:, start_bin:end_bin].T.unsqueeze(2) # (n_bins, Tw, 1)
    Residuals = Targets.clone()
    
    total_initial_energy = torch.sum(torch.linalg.vector_norm(Targets.abs(), dim=1)**2).item()
    
    # Freq Indices for Model Context
    freq_indices = torch.arange(start_bin, end_bin, device=device) # (n_bins,)
    
    # Initial RTG (Goal): "I want 100% reduction" -> RTG=1.0
    # Shape: (n_bins, 1) 
    # Wait, model needs (B, K). We build it iteratively?
    # Actually, model is sequence model.
    # At step 0: Input Corr(0), RTG(0)=1.0, FreqID
    # At step 1: Input Corr(1), RTG(1)=Actual_RTG, FreqID
    
    # State tracking
    current_rtg = torch.ones(n_bins, 1, device=device) # (n_bins, 1)
    
    for k in range(K_steps):
        # A. Observe Environment (Physics) -> Get Correlations
        # (n_bins, Tw, M) @ (n_bins, Tw, 1) -> (n_bins, M, 1)
        # Note: Dict_Norm is (B, Tw, M). Conj Transpose -> (B, M, Tw)
        Corrs = torch.bmm(Dict_Norm.conj().transpose(1, 2), Residuals) # (n_bins, M, 1)
        Obs_Corrs = Corrs.abs().squeeze(2) # (n_bins, M)
        
        # B. Model Decision
        # Input to model: 
        #   x: (B, step+1, M) sequence of observations? 
        #   The model is SeqDT. It expects sequence of length K.
        #   In inference, we feed [o_0] -> a_0. Then [o_0, o_1] -> a_1.
        #   We need to verify if model can handle generating step-by-step or needs padding.
        #   Usually HuggingFace DT uses padding. Our implementation uses simple GRU.
        #   GRU needs visible data.
        
        if k == 0:
            past_corrs = Obs_Corrs.unsqueeze(1) # (B, 1, M)
            past_rtg = current_rtg # (B, 1)
        else:
            # Append new observation
            new_corr = Obs_Corrs.unsqueeze(1)
            past_corrs = torch.cat([past_corrs, new_corr], dim=1) # (B, k+1, M)
            
            # RTG Updating
            # RTG_next = RTG_now - reward. Reward = reduction gain.
            # But here let's simplify: Input "Desired Return" which decays?
            # Or just feed the "Goal Remaining".
            # Initial Goal = 1.0 (reduce 100%).
            # After step 0, if we reduced 60%, remaining goal is 0.4?
            # Let's check training logic in `dataset_lag.py` read previously?
            # Ideally: RTG = Final_Target - Cumulative_Reduction_So_Far
            # If we want to reach 100% reduction (Target=1.0), RTG = 1.0 - Current_Red.
            
            # Calculate current reduction state
            current_energy_vec = torch.linalg.vector_norm(Residuals.abs(), dim=1).squeeze()**2
            initial_energy_vec = torch.linalg.vector_norm(Targets.abs(), dim=1).squeeze()**2 + 1e-9
            
            current_cum_red = (initial_energy_vec - current_energy_vec) / initial_energy_vec
            
            # Target = 1.0 (We want perfection)
            # rtg_val = 1.0 - current_cum_red
            rtg_val = torch.clamp(1.0 - current_cum_red, 0.0, 1.0)
            if rtg_val.ndim == 0:
                rtg_val = rtg_val.unsqueeze(0).unsqueeze(1)
            elif rtg_val.ndim == 1:
                rtg_val = rtg_val.unsqueeze(1)
            
            past_rtg = torch.cat([past_rtg, rtg_val], dim=1)
            
        # Forward Pass
        # x: (B, L, M)
        # rtg: (B, L)
        # freq: (B,)
        # Return: logits (B, L, M)
        
        logits, _ = model.rnn(
             model.layer_norm(
                model.state_embed(model.corr_norm(past_corrs)) + 
                model.rtg_embed(past_rtg.unsqueeze(-1)) + 
                model.freq_embed(freq_indices).unsqueeze(1).expand(-1, past_corrs.size(1), -1)
            )
        )
        logits = model.head(logits) # (B, L, M)
        
        # Take last step logic
        step_logits = logits[:, -1, :] # (B, M)
        
        # Greedy Action
        actions = torch.argmax(step_logits, dim=1) # (B,)
        
        # C. Physics Update (Execute Action)
        # For each bin, apply action
        # Vectorized Update
        # Dict_Norm: (B, Tw, M)
        # actions: (B,) -> need to gather
        # actions_view: (B, 1, 1) -> expand to (B, Tw, 1)
        
        batch_indices = torch.arange(n_bins, device=device)
        # Selected atoms: (B, Tw)
        selected_atoms = Dict_Norm[batch_indices, :, actions] # (B, Tw)
        
        # Projection amount
        # residual: (B, Tw, 1)
        # atom: (B, Tw) -> (B, Tw, 1)
        atoms_exp = selected_atoms.unsqueeze(2)
        
        # Coeff: atom.H @ residual
        # (B, 1, Tw) @ (B, Tw, 1) -> (B, 1, 1)
        coeffs = torch.bmm(atoms_exp.conj().transpose(1, 2), Residuals)
        
        # Warning: OMP orthogonalizes. 
        # "Greedy" step-by-step update is Matching Pursuit (MP), not OMP.
        # But for K=3, the difference is small if dictionary atoms are somewhat orthogonal.
        # OMP requires re-projecting on ALL selected atoms so far.
        # Implementing full vectorized OMP update is complex.
        # Let's stick to MP (Residual Update) for speed, which is a lower bound on performance.
        # If MP gets 90%, OMP gets >90%.
        
        contribution = atoms_exp * coeffs
        
        # DEBUG BLOCK
        if n_bins == 1:
            start_e = torch.linalg.vector_norm(Residuals.abs()).item()**2
            atom_norm = torch.linalg.vector_norm(selected_atoms.abs()).item()
            coeff_mag = coeffs.abs().item()
            proj_norm = torch.linalg.vector_norm(contribution.abs()).item()**2
            
            # Prediction of next energy: E_new = E_old - |coeff|^2 (if atom norm=1)
            pred_red = coeff_mag**2
            
            print(f"Step {k}: AtomNorm={atom_norm:.4f}, CoeffMag={coeff_mag:.4f}, ProjEnergy={proj_norm:.6f}")
            print(f"  Pre-Update Energy: {start_e:.6f}")
        
        Residuals = Residuals - contribution
        
        if n_bins == 1:
            end_e = torch.linalg.vector_norm(Residuals.abs()).item()**2
            print(f"  Post-Update Energy: {end_e:.6f} (Delta={start_e - end_e:.6f})")
            
    # Final Calculation
    final_energies = torch.linalg.vector_norm(Residuals.abs(), dim=1).squeeze()**2
    total_final_energy = torch.sum(final_energies).item()
    
    dt_global_red = (total_initial_energy - total_final_energy) / total_initial_energy
    
    print("-" * 40)
    print(f"Total Initial Energy : {total_initial_energy:.4f}")
    print(f"Total Final Energy   : {total_final_energy:.4f}")
    print(f"DTmin Global Reduction: {dt_global_red:.2%}")
    print("-" * 40)

if __name__ == "__main__":
    main()
