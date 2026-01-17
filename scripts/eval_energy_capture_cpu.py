
import os
import torch
import torch.nn as nn
import numpy as np
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# Hack to allow imports if needed, but we will copy classes for robustness
sys.path.append(os.getcwd())
# Try importing dataset
from scripts.h_exploration.dataset_lag import DoALagDataset

# --- MODEL DEFINITION (Copied from train_dt_lag_seq_rtg.py) ---
class SeqDT_FreqAware(nn.Module):
    def __init__(self, M_lags=16, d_model=128, hidden_dim=256, n_layers=2, max_freq=1025):
        super().__init__()
        self.rtg_embed = nn.Linear(1, d_model)
        self.state_embed = nn.Linear(M_lags, d_model)
        self.freq_embed = nn.Embedding(max_freq, d_model)
        self.corr_norm = nn.LayerNorm(M_lags)
        self.layer_norm = nn.LayerNorm(d_model)
        self.rnn = nn.GRU(
            input_size=d_model, 
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, M_lags)
        )
        
    def forward(self, x, rtg, freq_idx, mask=None):
        # x: (B, K, M)
        B, K, M = x.shape
        x_emb = self.state_embed(x)
        r_emb = self.rtg_embed(rtg.unsqueeze(-1))
        f_emb = self.freq_embed(freq_idx).unsqueeze(1).expand(-1, K, -1) # (B, K, D)
        
        h = x_emb + r_emb + f_emb
        h = self.layer_norm(h)
        
        # RNN
        out, _ = self.rnn(h)
        logits = self.head(out)
        return logits

# --- HELPER FUNCTIONS ---
def manual_complex_norm(tensor, dim=None, keepdim=False):
    if dim is None:
        return torch.sqrt(tensor.real.pow(2) + tensor.imag.pow(2) + 1e-10)
    else:
        return torch.sqrt(tensor.real.pow(2).sum(dim=dim, keepdim=keepdim) + tensor.imag.pow(2).sum(dim=dim, keepdim=keepdim) + 1e-10)

def safe_lstsq_solve(H_S, y):
    """
    Solve min ||y - H_S * x|| using CPU to avoid MPS complex limitation.
    H_S: (F, Tw, k) - actually we process per frequency bin usually in the loop
    But here we might process single bin.
    """
    # Inputs assumed to be (Tw, K) and (Tw) complex
    device = H_S.device
    H_cpu = H_S.detach().cpu().to(torch.complex64)
    y_cpu = y.detach().cpu().to(torch.complex64)
    
    try:
        sol = torch.linalg.lstsq(H_cpu, y_cpu).solution
    except:
        sol = torch.linalg.pinv(H_cpu) @ y_cpu
        
    return sol.to(device)

def run_physics_step(Dict_Norm, Residuals, Active_Sets, k, device):
    """
    Performs the projection and residual update for step k.
    Dict_Norm: (F, Tw, M)
    Residuals: (F, Tw, 1)
    Active_Sets: (F, K_max) - we use up to k
    """
    F = Dict_Norm.shape[0]
    # We Iterate over F to solve LS per frequency (batched LS on CPU might be complex)
    # Pytorch's linalg.lstsq supports batching
    
    # Gather Active Atoms: (F, Tw, k+1)
    # This is tricky to batch if k varies, but here k is fixed for all F
    indices = Active_Sets[:, :k+1] # (F, k+1)
    
    # Gather: need to select appropriate M per F
    # Dict_Norm[i, :, indices[i]]
    # Expand indices to (F, Tw, k+1) for gathering? No, simpler to loop or use refined gather
    # For Eval Speed, looping F is acceptable if Batch=1 (1 clip)
    
    current_norms_sq = torch.zeros(F, device=device)
    
    # CPU Batch Accumulation
    # We can move whole Arrays to CPU once and do batch op
    
    D_cpu = Dict_Norm.cpu()
    R_cpu = Residuals.cpu()
    Idx_cpu = indices.cpu()
    Y_cpu = (Residuals + 0).cpu() # Wait, Residuals changes. We need original Target? 
    # Actually OMP projects onto original target Y using selected set.
    # But usually we keep updating residual. 
    # Standard OMP: x = argmin ||y - Dx||. r = y - Dx.
    # So we need Y (Target).
    
    pass 

def eval_clip(model, dataset, idx, device, max_k=8, rtg_scale=1.0):
    """
    Evaluates one clip.
    Returns: 
       dt_energy (K_max,)
       omp_energy (K_max,)
    """
    # 1. Load Data
    item = dataset[idx]
    mic_stft = item['mic_stft'].to(device) # (T, F)
    ldv_stft = item['ldv_stft'].to(device) # (T, F)
    
    # 2. Construct History Matrix (Dict)
    # Identical to generate_lag_omp.py
    T_total, F = ldv_stft.shape
    Tw = 16 # Window Size
    M_lags = 16
    
    # Limit T to avoid edge effects
    # Start at 16 (MaxLag)
    # End at T_total - Tw
    valid_starts = range(M_lags, T_total - Tw, Tw) # Non-overlapping windows for eval? 
    # Or just one window? 
    # For robustness, let's take a set of windows
    
    dt_energies = []
    omp_energies = []
    
    model.eval()
    
    # Pre-move model to device
    
    for start_t in valid_starts: # Iterate windows
        # Extract Block
        end_t = start_t + Tw
        y_block = ldv_stft[start_t:end_t, :].T # (F, Tw)
        
        # Build Dictionary for this block
        # X_history involves previous M_lags
        # Indices in mic_stft: 
        # For Lag 0: [start_t : end_t]
        # For Lag k: [start_t-k : end_t-k]
        
        D = torch.zeros(F, Tw, M_lags, dtype=mic_stft.dtype, device=device)
        for k in range(M_lags):
            # lag k
            s = start_t - k
            e = s + Tw
            D[:, :, k] = mic_stft[s:e, :].T
            
        # Normalize Dictionary
        Norms = manual_complex_norm(D, dim=1).unsqueeze(1) + 1e-8
        D_norm = D / Norms
        
        # Targets
        Y = y_block.unsqueeze(2) # (F, Tw, 1)
        initial_energy = manual_complex_norm(Y.squeeze(), dim=1)**2
        
        # --- OMP Run ---
        # --- DTmin Run ---
        
        # Prepare Batch for Model
        # Model expects (B, K, M) - but we run step-by-step
        # Actually Model is GRU. We can pass state.
        
        # Init
        res_dt = Y.clone()
        res_omp = Y.clone()
        
        indices_dt = torch.zeros(F, max_k, dtype=torch.long, device=device)
        indices_omp = torch.zeros(F, max_k, dtype=torch.long, device=device)
        
        mask_dt = torch.zeros(F, M_lags, dtype=torch.bool, device=device)
        mask_omp = torch.zeros(F, M_lags, dtype=torch.bool, device=device)
        
        # Hidden State for DT
        hidden = None
        
        accum_dt = []
        accum_omp = []
        
        # Freq Idx
        freq_ids = torch.arange(F, device=device)
        
        for k in range(max_k):
            # 1. Calc Correlations
            # DT
            corrs_dt = torch.bmm(D_norm.conj().transpose(1, 2), res_dt).squeeze(2) # (F, M)
            abs_corrs_dt = torch.abs(corrs_dt)
            
            # OMP
            corrs_omp = torch.bmm(D_norm.conj().transpose(1, 2), res_omp).squeeze(2)
            abs_corrs_omp = torch.abs(corrs_omp)
            
            # Masking
            abs_corrs_dt[mask_dt] = -1.0
            abs_corrs_omp[mask_omp] = -1.0
            
            # 2. Selection
            # OMP
            act_omp = torch.argmax(abs_corrs_omp, dim=1)
            
            # DT
            # Prepare input: (F, 1, M)
            # RTG: We want full reduction. 
            # Current Red = (Init - Cur) / Init. 
            # Target Return = 1.0 - Current_Red = Cur / Init.
            # So RTG input should be Current_Residual_Ratio?
            # In training: rtg = (Final - Prev)
            # If we want MAX possible red, we ask for (1.0 - achieved_so_far)
            
            cur_E_dt = manual_complex_norm(res_dt.squeeze(), dim=1)**2
            vals = cur_E_dt / (initial_energy + 1e-6) # Remaining Energy Ratio
            # We want to reduce all of valid
            # So rtg = vals.
            
            rtg_in = vals # (F,)
            
            # Forward
            # x: (F, 1, M), rtg: (F, 1), freq: (F,)
            logits = model(
                abs_corrs_dt.unsqueeze(1), 
                rtg_in.unsqueeze(1),
                freq_ids
            ) # (F, 1, M)
            
            logits = logits.squeeze(1)
            # Mask logits
            logits[mask_dt] = -float('inf')
            
            act_dt = torch.argmax(logits, dim=1)
            
            # Update Masks
            # One hot
            mask_dt.scatter_(1, act_dt.unsqueeze(1), True)
            mask_omp.scatter_(1, act_omp.unsqueeze(1), True)
            
            indices_dt[:, k] = act_dt
            indices_omp[:, k] = act_omp
            
            # 3. Physics Update (Projection)
            # We use the CPU Safe Solve
            # Need to solve for specific F
            # We can parallelize via CPU batching
            
            # Collect selected atoms so far
            # indices_dt: (F, K)
            
            # To avoid slow mechanics, we just update residual using orthogonal projection
            # New atom k. 
            # Actually standard OMP updates residual by projecting onto span of ALL selected.
            
            # --- DT Update ---
            D_cpu = D_norm.cpu()
            Y_cpu = Y.cpu()
            
            # For each F, solve
            # This is the bottleneck. We do it for all F.
            
            # Vectorized CPU approach?
            # Construct Active Dictionary Batch
            # (F, Tw, k+1) via gather
            
            # Gather Function
            def gather_atoms(Dict, Ids_till_k):
                # Dict: (F, Tw, M)
                # Ids: (F, k+1)
                # Output: (F, Tw, k+1)
                F, Tw, M = Dict.shape
                K_curr = Ids_till_k.shape[1]
                
                # Expand Ids to (F, Tw, K_curr)
                Ids_exp = Ids_till_k.unsqueeze(1).expand(-1, Tw, -1)
                return torch.gather(Dict, 2, Ids_exp)

            active_dt = gather_atoms(D_cpu, indices_dt[:, :k+1].cpu())
            active_omp = gather_atoms(D_cpu, indices_omp[:, :k+1].cpu())
            
            # Solve
            # lstsq(A, B). A: (F, Tw, K), B: (F, Tw, 1)
            # torch.linalg.lstsq supports batch dims
            # Note: A must be full rank usually.
            
            try:
                sol_dt = torch.linalg.lstsq(active_dt, Y_cpu).solution
            except:
                # Fallback per sample if batch fails
                # print("Batch lstsq failed")
                sol_dt = torch.zeros(F, active_dt.shape[2], 1, dtype=torch.complex64)
                for f in range(F):
                    sol_dt[f] = torch.linalg.pinv(active_dt[f]) @ Y_cpu[f]
            
            try:
                sol_omp = torch.linalg.lstsq(active_omp, Y_cpu).solution
            except:
                 sol_omp = torch.zeros(F, active_omp.shape[2], 1, dtype=torch.complex64)
                 for f in range(F):
                     sol_omp[f] = torch.linalg.pinv(active_omp[f]) @ Y_cpu[f]
            
            # Recon
            recon_dt = active_dt @ sol_dt
            recon_omp = active_omp @ sol_omp
            
            # Update Residuals (back to GPU for Model Next Step)
            res_dt = (Y_cpu - recon_dt).to(device)
            res_omp = (Y_cpu - recon_omp).to(device)
            
            # Calc Energy Capture
            # ||Recon||^2 / ||Y||^2
            E_recon_dt = manual_complex_norm(recon_dt.squeeze(), dim=1)**2
            E_recon_omp = manual_complex_norm(recon_omp.squeeze(), dim=1)**2
            
            ratio_dt = E_recon_dt / (initial_energy.cpu() + 1e-6)
            ratio_omp = E_recon_omp / (initial_energy.cpu() + 1e-6)
            
            accum_dt.append(ratio_dt) # (F,)
            accum_omp.append(ratio_omp)
            
        # Stack results for this window
        dt_energies.append(torch.stack(accum_dt, dim=1)) # (F, K)
        omp_energies.append(torch.stack(accum_omp, dim=1))
        
    return torch.cat(dt_energies, dim=0), torch.cat(omp_energies, dim=0)

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Setup
    mic_root = "data/derived/boy1_processed/MIC"
    ldv_root = "data/derived/boy1_processed/LDV"
    angle = 0.0 # Test Angle 0
    
    dataset = DoALagDataset(
        mic_root, ldv_root, angle=angle, 
        freq_min=0, freq_max=8000
    )
    print(f"Dataset Size: {len(dataset)}")
    
    # 2. Load Model
    model = SeqDT_FreqAware(M_lags=16, d_model=128, hidden_dim=256).to(device)
    ckpt_path = "results/interspeech_gru1/model_varK/dt_freq_aware_best.pth"
    
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Model Loaded.")
    except Exception as e:
        print(f"Model Load Failed: {e}")
        # Try finding it
        import glob
        cands = glob.glob("results/**/*.pth", recursive=True)
        print(f"Candidates: {cands}")
        return

    # 3. Eval Loop
    means_dt = []
    means_omp = []
    
    # Evaluate first 10 clips
    # for i in range(min(10, len(dataset))):
    for i in tqdm(range(min(10, len(dataset)))):
        dt_E, omp_E = eval_clip(model, dataset, i, device)
        # dt_E: (N_windows*F, K)
        
        # Average over all windows/freqs
        m_dt = dt_E.mean(dim=0).cpu().numpy()
        m_omp = omp_E.mean(dim=0).cpu().numpy()
        
        means_dt.append(m_dt)
        means_omp.append(m_omp)
        
    # Final Stats
    res_dt = np.mean(np.array(means_dt), axis=0) # (K,)
    res_omp = np.mean(np.array(means_omp), axis=0)
    
    print("\nXXX RESULTS XXX")
    print(f"K\tDTmin\tOMP\t% of OMP")
    for k in range(len(res_dt)):
        print(f"{k+1}\t{res_dt[k]:.4f}\t{res_omp[k]:.4f}\t{res_dt[k]/res_omp[k]*100:.1f}%")

if __name__ == "__main__":
    main()
