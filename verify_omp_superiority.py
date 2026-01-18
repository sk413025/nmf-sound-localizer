import argparse
import logging
import torch
import numpy as np
from tqdm import tqdm
from scripts.h_exploration.dataset_lag import DoALagDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def manual_complex_norm(tensor, dim=None):
    if dim is None:
        return torch.sqrt(tensor.real.pow(2) + tensor.imag.pow(2) + 1e-10)
    return torch.sqrt(
        tensor.real.pow(2).sum(dim=dim) + tensor.imag.pow(2).sum(dim=dim) + 1e-10
    )

def gather_atoms(Dict, Ids_till_k):
    # Dict: (F, Tw, M)
    # Ids: (F, k)
    Tw = Dict.shape[1]
    ids_exp = Ids_till_k.unsqueeze(1).expand(-1, Tw, -1)
    return torch.gather(Dict, 2, ids_exp)

def run_verify(mic_root, ldv_root, angle=90.0, hop_length=160, device='cpu', all_angles=False):
    tw = 32
    max_lag = 50
    M = 2 * max_lag + 1
    
    # We will test specific Ks
    test_ks = [1, 2, 4, 8, 16] 
    
    dataset = DoALagDataset(
        mic_root,
        ldv_root,
        angle=None if all_angles else angle,
        hop_length=hop_length,
        normalize_audio=True,
    )
    
    # Analyze first 5 clips to be safe and robust
    n_clips = min(len(dataset), 3)
    
    # Accumulators for Weighted Average
    # [K][Method] -> Energy
    total_reduction = {k: {'omp': 0.0, 'rnd': 0.0} for k in test_ks}
    total_initial = 0.0
    
    logger.info(f"Verifying OMP vs Random on {n_clips} clips. Tw={tw}, MaxLag={max_lag}")
    
    for idx in range(n_clips):
        item = dataset[idx]
        
        # Robust load
        if isinstance(item["mic_stft"], torch.Tensor):
            mic_stft = item["mic_stft"].to(device) * 100.0
        else:
            mic_stft = torch.from_numpy(item["mic_stft"]).to(device) * 100.0

        if isinstance(item["ldv_stft"], torch.Tensor):
            ldv_stft = item["ldv_stft"].to(device) * 100.0
        else:
            ldv_stft = torch.from_numpy(item["ldv_stft"]).to(device) * 100.0
        
        T, F = mic_stft.shape
        start_t = max_lag
        end_t_limit = T - tw - max_lag
        
        # Stride to save time
        for t in range(start_t, end_t_limit, 64):
            y_block = ldv_stft[t:t+tw, :].T # (F, Tw)
            
            # Init Energy
            initial_energy = manual_complex_norm(y_block, dim=1)**2
            total_initial += initial_energy.sum().item()
            
            # Dictionary
            D = torch.zeros(F, tw, M, dtype=mic_stft.dtype, device=device)
            lags = torch.arange(-max_lag, max_lag + 1)
            for m, lag in enumerate(lags):
                s = t - lag
                e = s + tw
                D[:, :, m] = mic_stft[s:e, :].T
            
            # Norms for matching
            d_norms = manual_complex_norm(D, dim=1).unsqueeze(1) + 1e-9
            D_norm = D / d_norms
            
            # --- OMP Execution (Track path) ---
            res = y_block.unsqueeze(2).clone()
            active_ids = torch.zeros(F, 0, dtype=torch.long, device=device)
            
            # Optimization: We can reuse the path. 
            # Run up to max(test_ks)
            max_test_k = max(test_ks)
            
            # Store reductions for current window
            # Map k -> reduction_vector (F,)
            omp_reductions = {} 
            rnd_reductions = {}
            
            # OMP Loop
            for k_step in range(1, max_test_k + 1):
                # 1. Corr
                corrs = torch.bmm(D_norm.conj().transpose(1, 2), res).squeeze(2)
                abs_corrs = torch.abs(corrs)
                
                # Mask prev
                if active_ids.shape[1] > 0:
                    # scatter mask
                    # brute force for safety on cpu
                    for b in range(F):
                        abs_corrs[b, active_ids[b]] = -1.0
                
                # Select
                best = torch.argmax(abs_corrs, dim=1).unsqueeze(1)
                active_ids = torch.cat([active_ids, best], dim=1)
                
                if k_step in test_ks:
                    # Project
                    atoms = gather_atoms(D, active_ids)
                    sol = torch.linalg.lstsq(atoms, y_block.unsqueeze(2)).solution
                    recon = atoms @ sol
                    res_k = y_block.unsqueeze(2) - recon
                    
                    # Calc Energy Reduction
                    current_res_E = manual_complex_norm(res_k.squeeze(2), dim=1)**2
                    red = (initial_energy - current_res_E)
                    # Sum over F
                    total_reduction[k_step]['omp'] += red.sum().item()
                    
                    # Update residual for NEXT step? 
                    # Ideally OMP updates residual using orthogonal projection.
                    # The `res` var in loop should be updated.
                    res = res_k

            # --- Random Loop ---
            # Random is independent for each K? Or sequential?
            # Standard "Random Baseline" usually means picking K random atoms at once.
            for k_step in test_ks:
                # Average over 5 trials
                trial_reds = 0.0
                for _ in range(5):
                    rand_ids = torch.randint(0, M, (F, k_step), device=device) # with replacement is fine for random baseline, or existing code used permutation
                    atoms_r = gather_atoms(D, rand_ids)
                    sol_r = torch.linalg.lstsq(atoms_r, y_block.unsqueeze(2)).solution
                    recon_r = atoms_r @ sol_r
                    res_r = y_block.unsqueeze(2) - recon_r
                    current_res_E_r = manual_complex_norm(res_r.squeeze(2), dim=1)**2
                    red_r = (initial_energy - current_res_E_r)
                    trial_reds += red_r.sum().item()
                
                total_reduction[k_step]['rnd'] += (trial_reds / 5.0)

    # Report
    logger.info(f"{'K':<5} | {'OMP (W%)':<12} | {'Rnd (W%)':<12} | {'Gap':<12} | {'Ratio(O/R)':<12}")
    logger.info("-" * 65)
    
    for k in sorted(test_ks):
        omp_score = total_reduction[k]['omp'] / (total_initial + 1e-6)
        rnd_score = total_reduction[k]['rnd'] / (total_initial + 1e-6)
        gap = omp_score - rnd_score
        ratio = omp_score / (rnd_score + 1e-6)
        logger.info(f"{k:<5} | {omp_score:.4f}       | {rnd_score:.4f}       | {gap:+.4f}       | {ratio:.2f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mic_root", type=str, required=True)
    parser.add_argument("--ldv_root", type=str, required=True)
    parser.add_argument("--angle", type=float, default=90.0)
    parser.add_argument("--all_angles", action="store_true")
    args = parser.parse_args()
    
    run_verify(args.mic_root, args.ldv_root, angle=args.angle, all_angles=args.all_angles)
