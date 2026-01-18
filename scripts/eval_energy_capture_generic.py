import os
import torch
import torch.nn as nn
import numpy as np
import sys
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from scripts.h_exploration.dataset_lag import DoALagDataset

# --- MODEL DEFINITION ---
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

def gather_atoms(Dict, Ids_till_k):
    # Dict: (F, Tw, M)
    # Ids: (F, k+1)
    # Output: (F, Tw, k+1)
    F, Tw, M = Dict.shape
    K_curr = Ids_till_k.shape[1]
    
    # Expand Ids to (F, Tw, K_curr)
    Ids_exp = Ids_till_k.unsqueeze(1).expand(-1, Tw, -1)
    return torch.gather(Dict, 2, Ids_exp)

def eval_clip(model, dataset, idx, device, max_lag=16, max_k=16, Tw=16):
    """
    Evaluates one clip.
    Returns: 
       dt_energy (N_windows*F_bins, K_max)
       omp_energy (N_windows*F_bins, K_max)
    """
    
    # Lag Range
    Lag_Min = -max_lag
    Lag_Max = max_lag
    Lags = torch.arange(Lag_Min, Lag_Max + 1, device=device)
    M_lags = len(Lags)

    item = dataset[idx]
    mic_stft = item['mic_stft'].to(device) # (T, F)
    ldv_stft = item['ldv_stft'].to(device) # (T, F)
    
    T_total, F = ldv_stft.shape
   
    # Valid Start calculation
    start_limit = Lag_Max
    end_limit = T_total - Tw + Lag_Min # Lag_Min is negative (-32). So T - 16 - 32.
    
    # Ensure end_limit > start_limit
    if end_limit <= start_limit:
         return torch.zeros(0, max_k), torch.zeros(0, max_k)
    
    valid_starts = range(start_limit, end_limit, Tw) 
    
    dt_energies = []
    omp_energies = []
    
    model.eval()
    
    if len(valid_starts) == 0:
        return torch.zeros(0, max_k), torch.zeros(0, max_k)

    for start_t in valid_starts:
        end_t = start_t + Tw
        y_block = ldv_stft[start_t:end_t, :].T # (F, Tw)
        
        # Build Dictionary for this block
        D = torch.zeros(F, Tw, M_lags, dtype=mic_stft.dtype, device=device)
        
        for m, k_lag in enumerate(Lags):
            # lag k_lag
            # Input slice: [start_t - k_lag : end_t - k_lag]
            s = int(start_t - k_lag)
            e = int(s + Tw)
            D[:, :, m] = mic_stft[s:e, :].T
            
        # Normalize Dictionary
        Norms = manual_complex_norm(D, dim=1).unsqueeze(1) + 1e-8
        D_norm = D / Norms
        
        # Targets
        Y = y_block.unsqueeze(2) # (F, Tw, 1)
        initial_energy = manual_complex_norm(Y.squeeze(), dim=1)**2
        
        # Init Tracking
        res_dt = Y.clone()
        res_omp = Y.clone()
        
        indices_dt = torch.zeros(F, max_k, dtype=torch.long, device=device)
        indices_omp = torch.zeros(F, max_k, dtype=torch.long, device=device)
        
        mask_dt = torch.zeros(F, M_lags, dtype=torch.bool, device=device)
        mask_omp = torch.zeros(F, M_lags, dtype=torch.bool, device=device)
        
        accum_dt = []
        accum_omp = []
        
        freq_ids = torch.arange(F, device=device)
        
        for k in range(max_k):
            # 1. Correlations
            corrs_dt = torch.bmm(D_norm.conj().transpose(1, 2), res_dt).squeeze(2) # (F, M)
            abs_corrs_dt = torch.abs(corrs_dt)
            
            corrs_omp = torch.bmm(D_norm.conj().transpose(1, 2), res_omp).squeeze(2)
            abs_corrs_omp = torch.abs(corrs_omp)
            
            # Masking
            abs_corrs_dt[mask_dt] = -1.0
            abs_corrs_omp[mask_omp] = -1.0
            
            # 2. Selection
            # OMP Oracle
            act_omp = torch.argmax(abs_corrs_omp, dim=1)
            
            # DT Agent
            cur_E_dt = manual_complex_norm(res_dt.squeeze(), dim=1)**2
            vals = cur_E_dt / (initial_energy + 1e-6) 
            rtg_in = vals # We want to eliminate remaining energy
            
            # Forward
            # x input: correlations (F, M). Reshape to (F, 1, M) as seq len 1
            logits = model(
                abs_corrs_dt.unsqueeze(1), 
                rtg_in.unsqueeze(1),
                freq_ids
            ) 
            logits = logits.squeeze(1)
            logits[mask_dt] = -float('inf')
            
            act_dt = torch.argmax(logits, dim=1)
            
            # Update Masks
            mask_dt.scatter_(1, act_dt.unsqueeze(1), True)
            mask_omp.scatter_(1, act_omp.unsqueeze(1), True)
            
            indices_dt[:, k] = act_dt
            indices_omp[:, k] = act_omp
            
            # 3. Physics Update (Projection) - CPU Batch
            # Move to CPU for solve to avoid MPS complex issues
            D_cpu = D_norm.cpu()
            Y_cpu = Y.cpu() # Should be original Y for projection
            
            # Note: We should project onto ORIGINAL Y for correct OMP behavior.
            # But the code inside eval_energy_capture_cpu.py was using Residuals?
            # No, standard OMP projects Y onto Active Set.
            # We use Y_orig_cpu
            Y_orig_cpu = Y.cpu()
            
            active_dt = gather_atoms(D_cpu, indices_dt[:, :k+1].cpu())
            active_omp = gather_atoms(D_cpu, indices_omp[:, :k+1].cpu())
            
            # Solve DT
            try:
                # Batched lstsq
                sol_dt = torch.linalg.lstsq(active_dt, Y_orig_cpu).solution
            except:
                sol_dt = torch.zeros(F, active_dt.shape[2], 1, dtype=torch.complex64)
                for f in range(F):
                    sol_dt[f] = torch.linalg.pinv(active_dt[f]) @ Y_orig_cpu[f]
            
            # Solve OMP
            try:
                sol_omp = torch.linalg.lstsq(active_omp, Y_orig_cpu).solution
            except:
                 sol_omp = torch.zeros(F, active_omp.shape[2], 1, dtype=torch.complex64)
                 for f in range(F):
                     sol_omp[f] = torch.linalg.pinv(active_omp[f]) @ Y_orig_cpu[f]
            
            # Recon
            recon_dt = active_dt @ sol_dt
            recon_omp = active_omp @ sol_omp
            
            # Update Residuals
            res_dt = (Y_orig_cpu - recon_dt).to(device)
            res_omp = (Y_orig_cpu - recon_omp).to(device)
            
            # Metrics
            E_recon_dt = manual_complex_norm(recon_dt.squeeze(), dim=1)**2
            E_recon_omp = manual_complex_norm(recon_omp.squeeze(), dim=1)**2
            
            ratio_dt = E_recon_dt / (initial_energy.cpu() + 1e-6)
            ratio_omp = E_recon_omp / (initial_energy.cpu() + 1e-6)
            
            accum_dt.append(ratio_dt) # (F,)
            accum_omp.append(ratio_omp)
        
        if len(accum_dt) > 0:
            dt_energies.append(torch.stack(accum_dt, dim=1))
            omp_energies.append(torch.stack(accum_omp, dim=1))
        
    if len(dt_energies) == 0:
        return torch.zeros(0, max_k), torch.zeros(0, max_k)

    return torch.cat(dt_energies, dim=0), torch.cat(omp_energies, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mic_root", type=str, required=True)
    parser.add_argument("--ldv_root", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/eval_capture")
    parser.add_argument("--max_lag", type=int, default=32)
    parser.add_argument("--max_k", type=int, default=32)
    parser.add_argument("--tw", type=int, default=16)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Dataset
    # We use broadband to scan all freqs
    dataset = DoALagDataset(
        args.mic_root, args.ldv_root, angle=90.0, # Default angle 90
        freq_min=0, freq_max=8000
    )
    print(f"Dataset Size: {len(dataset)}")
    
    # Model
    # Calculate M_lags from max_lag
    # e.g. 32 -> -32..32 -> 65
    M_lags = args.max_lag * 2 + 1
    print(f"Initializing Model with M_lags={M_lags}")
    
    model = SeqDT_FreqAware(M_lags=M_lags, d_model=128, hidden_dim=256).to(device)
    
    try:
        # Load weights only option if strict check fails?
        # weights_only=False/True depending on pytorch version.
        state = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(state)
        print("Model Loaded.")
    except Exception as e:
        print(f"Model Load Failed: {e}")
        return

    # Evaluation
    print("Collecting raw data...")
    
    Total_DT = [] 
    Total_OMP = [] 
    
    # Eval 10 clips
    N_clips = min(10, len(dataset))
    
    for i in tqdm(range(N_clips)):
        # Gets (N_windows * F, K)
        dt_flat, omp_flat = eval_clip(
            model, dataset, i, device, 
            max_lag=args.max_lag, max_k=args.max_k, Tw=args.tw
        )
        
        if dt_flat.shape[0] == 0: continue
            
        # Reshape to (Windows, F, K)
        item = dataset[i]
        F_dim = item['mic_stft'].shape[1]
        
        num_windows = dt_flat.shape[0] // F_dim
        if num_windows == 0: continue
            
        dt_3d = dt_flat.reshape(num_windows, F_dim, -1)
        omp_3d = omp_flat.reshape(num_windows, F_dim, -1)
        
        # Mean over windows
        dt_mean = dt_3d.mean(dim=0).cpu() # (F, K)
        omp_mean = omp_3d.mean(dim=0).cpu()
        
        Total_DT.append(dt_mean)
        Total_OMP.append(omp_mean)

    if len(Total_DT) == 0:
        print("No valid data collected.")
        return

    Grand_DT = torch.stack(Total_DT, dim=0).mean(dim=0) # (F, K)
    Grand_OMP = torch.stack(Total_OMP, dim=0).mean(dim=0)
    
    # Save Raw
    torch.save({"DT": Grand_DT, "OMP": Grand_OMP}, os.path.join(args.out_dir, "eval_stats.pt"))
    
    # Analyze Bands
    fs = 16000
    n_fft = 2048
    freq_bins = torch.linspace(0, fs/2, n_fft//2 + 1)
    
    BANDS = {
        "Sub-Bass (0-60Hz)": (0, 60),
        "Bass (60-250Hz)": (60, 250),
        "Low Mids (250-500Hz)": (250, 500),
        "Mids (500-2k Hz)": (500, 2000),
        "Upper Mids (2k-4k Hz)": (2000, 4000),
        "Presence (4k-6k Hz)": (4000, 6000),
        "Highs (6k-8k Hz)": (6000, 8000)
    }
    
    plot_data = {}
    
    for band_name, (f_start, f_end) in BANDS.items():
        mask = (freq_bins >= f_start) & (freq_bins < f_end)
        indices = torch.where(mask)[0]
        indices = indices[indices < Grand_DT.shape[0]]
        
        if len(indices) == 0: continue
            
        band_dt = Grand_DT[indices, :].mean(dim=0).numpy()
        band_omp = Grand_OMP[indices, :].mean(dim=0).numpy()
        band_eff = band_dt / (band_omp + 1e-6) * 100
        
        plot_data[band_name] = {"DT": band_dt, "OMP": band_omp, "Eff": band_eff}
        
    # Table
    print("\nXXX DETAILED BAND ANALYSIS XXX")
    print(f"{'Band':<25} | {'K':<3} | {'DT':<6} | {'OMP':<6} | {'Eff %':<6}")
    print("-" * 60)
    
    # Dynamic K points
    k_vals = np.array([1, 4, 8, 16, 32, args.max_k])
    k_vals = k_vals[k_vals <= args.max_k]
    k_points = sorted(list(set(k_vals.astype(int) - 1)))
    
    for name, data in plot_data.items():
        for k_idx in k_points:
            if k_idx < len(data['DT']):
                k = k_idx + 1
                curr_dt = data['DT'][k_idx]
                curr_omp = data['OMP'][k_idx]
                curr_eff = data['Eff'][k_idx]
                print(f"{name:<25} | {k:<3} | {curr_dt:.4f} | {curr_omp:.4f} | {curr_eff:.1f}%")
        print("-" * 60)
        
    # Plotting
    if args.visualize:
        K_steps = np.arange(1, Grand_DT.shape[1] + 1)
        
        plt.figure(figsize=(10, 6))
        for name, data in plot_data.items():
            plt.plot(K_steps, data["Eff"], marker='o', label=name)
        plt.title("Efficiency by Band")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.out_dir, "efficiency.png"))
        
        rows = (len(plot_data) + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(15, 4*rows))
        axes = axes.flatten()
        for i, (name, data) in enumerate(plot_data.items()):
            ax = axes[i]
            ax.plot(K_steps, data["OMP"], 'k--', label="OMP")
            ax.plot(K_steps, data["DT"], 'b-', label="DT")
            ax.set_title(name)
            if i==0: ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "capture.png"))

if __name__ == "__main__":
    main()
