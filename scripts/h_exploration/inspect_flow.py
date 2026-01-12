import torch
import numpy as np
from scripts.h_exploration.dataset_lag import DoALagDataset, create_dataloader
from scripts.h_exploration.train_dt_lag_seq_rtg import SeqDT_FreqAware # Modified Import
from scripts.h_exploration.generate_lag_omp import run_omp_lag_capture
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_data_flow():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True) 
    # data_path is used to infer dataset location or load subset. 
    # But for this inspection script we might load raw wavs directly via Dataset class.
    # Let's keep it simple and assume standard paths or args.
    args = parser.parse_args()

    # 1. Setup Data & Model
    model_path = args.model_path
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Init Model (Must match training config)
    # FreqAware Model has freq_embed
    model = SeqDT_FreqAware(M_lags=16, d_model=128, hidden_dim=256, max_freq=1025)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logger.info(f"Loaded trained model from {model_path}")
    except Exception as e:
        logger.error(f"Could not load model: {e}")
        return

    model.to(device)
    model.eval()
    
    # 2. Get One Real Clip Chunk
    # Using defaults from generate_lag_omp.py
    dataset = DoALagDataset("/Users/sbplab/LDV-data-processed/speech260_original_16k_no_edge_sync_vad_normalized", 
                            "/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized", 
                            angle=90.0)
    # Just take 1st item
    item = dataset[0] 
    mic_stft = item["mic_stft"].to(device) # (T, F) complex
    ldv_stft = item["ldv_stft"].to(device) # (T, F) complex
    
    # Slice a chunk
    Tw = 16
    MaxLag = 16
    start_t = MaxLag
    
    X_chunk = mic_stft[start_t-MaxLag : start_t+Tw] # (32, 257)
    Y_chunk = ldv_stft[start_t : start_t+Tw]        # (16, 257)
    
    print("\n=== SYSTEM INPUTS (Physics Level) ===")
    print(f"Mic Chunk (X): Shape={X_chunk.shape}, Dtype={X_chunk.dtype}")
    print(f"LDV Chunk (Y): Shape={Y_chunk.shape}, Dtype={Y_chunk.dtype}")
    print("Interpretation: X contains history + current window. Y contains current window target.")

    # 3. Run OMP (Teacher) Step-by-Step Simulation
    print("\n=== OMP (Teacher) PROCESS ===")
    # Re-implementing simplified parts of run_omp_lag_capture to print internals
    
    F = Y_chunk.shape[1]
    K_max = 3
    
    # Dictionary Construction
    # We construct 16 Lags.
    # Lag 0: X[16:32] (Current Tw)
    # Lag 1: X[15:31] (Shifted back by 1)
    Dict_tensor = torch.zeros(F, Tw, MaxLag, dtype=X_chunk.dtype, device=device)
    for k in range(MaxLag):
        Dict_tensor[:, :, k] = X_chunk[16-k : 16+Tw-k, :].T
        
    Targets = Y_chunk.T.unsqueeze(2) # (F, Tw, 1)
    Residuals = Targets.clone()
    
    # We will track one specific Frequency bin for clarity: Bin 50
    b_idx = 50
    print(f"Inspecting Frequency Bin {b_idx}...")
    
    # Store history for DTmin validation
    history_corrs = []
    history_rtg = []
    history_freqs = [] # New: Track Freq
    
    # Safe norm for MPS/Complex
    # Fix B: Use torch.linalg.vector_norm(x.abs()) for clarity and MPS support
    initial_norm = torch.linalg.vector_norm(Targets[b_idx].abs()).item()
    print(f"Initial Energy (Norm): {initial_norm:.4f}")
    
    omp_reduction_history = []
    
    for k in range(K_max):
        print(f"\n--- Step {k} ---")
        
        # OMP Step 1: Correlation (Inner Product)
        D_b = Dict_tensor[b_idx]
        R_b = Residuals[b_idx]
        
        # Normalize Dict atoms - Fix B applied
        D_norms = torch.linalg.vector_norm(D_b.abs(), dim=0, keepdim=True) + 1e-8
        D_normed = D_b / D_norms
        
        # Correlation Calculation (Complex)
        Corr_Complex = D_normed.conj().T @ R_b 
        
        # --- Diagnostic 1: Phase Information ---
        Corr_Phase = torch.angle(Corr_Complex).flatten()
        
        # DTmin Input: Absolute Value
        Corr_Abs = torch.abs(Corr_Complex).flatten() 
        
        # Normalize Corr for display
        best_lag = torch.argmax(Corr_Abs).item()
        print(f"OMP Choice: Lag {best_lag} (Val={Corr_Abs[best_lag]:.4f})")
        
        # Print Phase Info for Top 3
        top3_indices = torch.topk(Corr_Abs, 3).indices
        print(f"  > Top 3 Candidates Phase Check (Hidden Variable):")
        for idx in top3_indices:
            idx = idx.item()
            mag = Corr_Abs[idx].item()
            pha = Corr_Phase[idx].item()
            print(f"    Lag {idx:2d}: Mag={mag:.4f}, Phase={pha:6.3f} rad ({np.degrees(pha):6.1f} deg)")
            
        # --- Diagnostic 2: Ablation (Magnitude vs Phase-Diff Proxy) ---
        Corr_Real = Corr_Complex.real.flatten()
        rank_mag = torch.argsort(Corr_Abs, descending=True)
        rank_real = torch.argsort(Corr_Real.abs(), descending=True)
        
        print(f"  > Ablation Proxy Check (Sensitivity to Phase):")
        print(f"    Top 1 (Mag): Lag {rank_mag[0]} (Val={Corr_Abs[rank_mag[0]]:.4f})")
        print(f"    Top 1 (Real): Lag {rank_real[0]} (Val={Corr_Real[rank_real[0]].abs():.4f})")
        
        if rank_mag[0] != rank_real[0]:
            print(f"    [!] CRITICAL: Phase Rotation changes selection! (Real-only proxy fails).")
            print(f"        Interpretation: Phase is an essential 'Hidden Variable' for deciding NEXT step dynamics.")
            print(f"        (Without phase, a world-model cannot predict correct residue subtraction).")
        else:
            print(f"    [~] Passive: Real-only proxy matches Magnitude selection.")
            
        # Physics Update (OMP/MP) - CLEAN FIRST PRINCIPLES
        # We use the Normalized Atom from the selection step to ensure consistency.
        # Coeff alpha = d_norm^H * r (since ||d_norm||=1)
        # Update: r <- r - alpha * d_norm
        
        atom_normed = D_normed[:, best_lag:best_lag+1] # (Tw, 1) Normalized
        alpha = Corr_Complex[best_lag] # The complex correlation IS the projection coefficient for unit vectors
        
        # Validation: Re-calculate alpha explicitly to be sure
        # alpha_check = atom_normed.conj().T @ R_b
        # print(f"Alpha Check: {alpha.item():.4f} vs {alpha_check.item():.4f}")
        
        projection = atom_normed * alpha
        R_b = R_b - projection
        Residuals[b_idx] = R_b 
        
        curr_norm = torch.linalg.vector_norm(R_b.abs()).item()
        reduction = (initial_norm - curr_norm) / (initial_norm + 1e-6)
        print(f"Energy Remaining: {curr_norm:.4f} (Reduction: {reduction:.2%})")
        omp_reduction_history.append(reduction)
        
        # Collect Data for DTmin offline check
        history_corrs.append(Corr_Abs)
        history_rtg.append(0.5) # Dummy Value for the passive check
        history_freqs.append(b_idx) # Track correct bin

    print("\n=== DTmin (Student) I/O Check ===")
    
    # Fix A: Ensure tensor device matches model correctly
    device_model = next(model.parameters()).device
    seq_corrs = torch.stack(history_corrs).unsqueeze(0).float().to(device_model)
    seq_rtg = torch.tensor(history_rtg).unsqueeze(0).float().to(device_model)
    seq_freqs = torch.tensor(history_freqs).unsqueeze(0).long().to(device_model) # (1, K) ? No, Model expects (B, 1) or (B)? 
    
    # Model forward expects freq_idx: (B,)
    # But here we have a sequence. Wait, how was model trained?
    # In train script: self.freq_embed(freq_idx).unsqueeze(1).expand(-1, K, -1)
    # So it expects a single freq per batch item. Correct.
    seq_freq_scalar = torch.tensor([b_idx]).long().to(device_model) # (1,)
    
    print(f"DTmin Input State (Corrs): Shape={seq_corrs.shape}")
    
    with torch.no_grad():
        logits = model(seq_corrs, seq_rtg, seq_freq_scalar)
    
    last_step_logits = logits[0, -1, :]
    print(f"final passive logits: {last_step_logits[:5].detach().cpu().numpy()}")

    
    # --- EXPERIMENT 4B: INTERACTIVE ROLLOUT (Policy Mode) ---
    print("\n\n=== EXPERIMENT: Interactive Rollout (Policy Mode) ===")
    print("Goal: Test if DTmin selection + TRUE PHYSICS UPDATE works (Environment-in-the-loop).")
    
    # Reset Environment
    R_interactive = Targets.clone()
    interactive_corrs = []
    interactive_rtg_hist = []
    
    # Target RTG Strategy: Demand 100% reduction (Target=1.0) 
    # Or demand the OMP achieved reduction? Let's demand 1.0 (Greedy)
    target_return = 1.0 
    current_reduction = 0.0
    
    dt_reduction_history = []
    
    for k in range(K_max):
        print(f"\n[Interactive Step {k}]")
        
        # 1. Physics: Calculate State (Complex Correlations)
        D_b = Dict_tensor[b_idx]
        R_b = R_interactive[b_idx]
        D_norms = torch.linalg.vector_norm(D_b.abs(), dim=0, keepdim=True) + 1e-8
        D_normed = D_b / D_norms
        Corr_Complex = D_normed.conj().T @ R_b 
        Corr_Abs = torch.abs(Corr_Complex).flatten()
        
        # 2. Prepare Input for Student
        interactive_corrs.append(Corr_Abs)
        
        # RTG = Target_Return - Current_Reduction
        # (This is how we tell the model 'how much is left to do')
        step_rtg = target_return - current_reduction
        interactive_rtg_hist.append(step_rtg)
        
        # Stack Sequence
        inp_seq_corrs = torch.stack(interactive_corrs).unsqueeze(0).float().to(device_model)
        inp_seq_rtg = torch.tensor(interactive_rtg_hist).unsqueeze(0).float().to(device_model)
        # Use simple scalar freq
        inp_freq = torch.tensor([b_idx]).long().to(device_model)
        
        # 3. Model Decision
        with torch.no_grad():
            logits = model(inp_seq_corrs, inp_seq_rtg, inp_freq)
            
        # Get Action for current step (last in sequence)
        step_logits = logits[0, -1, :]
        action_idx = torch.argmax(step_logits).item()
        
        print(f"  > DTmin sees Corrs Max={Corr_Abs.max().item():.4f}, RTG={step_rtg:.4f}")
        print(f"  > DTmin Selects: Lag {action_idx}")
        
        # 4. Physics Update (True Environment Feedback) - CLEAN FIRST PRINCIPLES
        # Consistent with the "Teacher" loop above
        D_norms = torch.linalg.vector_norm(D_b.abs(), dim=0, keepdim=True) + 1e-8
        D_normed = D_b / D_norms
        
        atom_normed = D_normed[:, action_idx:action_idx+1]
        
        # Calculate projection coefficient on the unit atom
        # (Since we haven't computed the correlation for THIS specific lag in the diagnostics strictly,
        # we re-compute it or grab it from Corr_Complex if we trust indices match)
        # Safest to recompute simple dot product
        alpha = atom_normed.conj().T @ R_b
        
        projection = atom_normed * alpha
        R_b = R_b - projection
        R_interactive[b_idx] = R_b # Commit update
        
        # 5. Measure Result
        curr_norm = torch.linalg.vector_norm(R_b.abs()).item()
        current_reduction = (initial_norm - curr_norm) / (initial_norm + 1e-6)
        dt_reduction_history.append(current_reduction)
        print(f"  > Result: Energy={curr_norm:.4f}, Reduction={current_reduction:.2%}")

    print("\n=== FINAL COMPARISON (3 Steps) ===")
    print(f"{'Step':<5} | {'OMP Red.':<12} | {'DTmin (Pol) Red.':<18}")
    print("-" * 40)
    for k in range(K_max):
        omp_r = omp_reduction_history[k] if k < len(omp_reduction_history) else 0
        dt_r = dt_reduction_history[k] if k < len(dt_reduction_history) else 0
        print(f"{k:<5} | {omp_r:.2%}       | {dt_r:.2%}")
        
    print("-" * 40)
    final_delta = omp_reduction_history[-1] - dt_reduction_history[-1]
    if final_delta > 0.05:
         print(f"CONCLUSION: OMP beats Policy-Mode DTmin by {final_delta:.2%}.")
         print("Hypothesis: Even with correct correlations, the model fails to pick the interaction-optimal atom.")
    else:
         print(f"CONCLUSION: DTmin matches/beats OMP (Diff {final_delta:.2%}).")
         print("Hypothesis: Phase loss is NOT the bottleneck if Environment Feedback is provided.")

if __name__ == "__main__":
    inspect_data_flow()
