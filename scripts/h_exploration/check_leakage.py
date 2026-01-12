import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scripts.h_exploration.dataset_lag import DoALagDataset
import scipy.signal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_spectral_flatness(mag_spec):
    """
    Geometric mean / Arithmetic mean. 
    Close to 1 -> White Noise.
    Close to 0 -> Tonal/Speech.
    """
    # mag_spec: (F,)
    gmean = torch.exp(torch.mean(torch.log(mag_spec + 1e-10)))
    amean = torch.mean(mag_spec)
    return (gmean / (amean + 1e-10)).item()

def run_integrity_check():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mic_root", type=str, default="/Users/sbplab/LDV-data-processed/speech260_original_16k_no_edge_sync_vad_normalized")
    parser.add_argument("--ldv_root", type=str, default="/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized")
    args = parser.parse_args()
    
    # Force CPU for analysis to avoid MPS missing ops like vdot
    device = torch.device("cpu")
    
    dataset = DoALagDataset(args.mic_root, args.ldv_root, angle=90.0)
    print(f"Dataset Size: {len(dataset)}")
    
    # Check 3 random clips
    indices = [0, 50, 100]
    
    for idx in indices:
        if idx >= len(dataset): continue
        print(f"\n[Checking Clip {idx}]")
        
        item = dataset[idx]
        mic = item["mic_stft"].to(device) # (T, F)
        ldv = item["ldv_stft"].to(device) # (T, F)
        
        # 1. Identity / Duplication Leakage
        # ------------------------------------------------
        # Normalize both to verify shape matching
        mic_flat = torch.abs(mic).flatten()
        ldv_flat = torch.abs(ldv).flatten()
        
        mic_norm = mic_flat / mic_flat.max()
        ldv_norm = ldv_flat / ldv_flat.max()
        
        mse = torch.mean((mic_norm - ldv_norm)**2).item()
        corr_matrix = torch.corrcoef(torch.stack([mic_flat, ldv_flat]))
        pearson_rho = corr_matrix[0, 1].item()
        
        print(f"  > Identity Check:")
        print(f"    Pearson Corr (Global): {pearson_rho:.4f}")
        print(f"    Normalized MSE: {mse:.6f}")
        
        if pearson_rho > 0.99:
            print("    [!] WARNING: Extremely high raw correlation. Data might be duplicated or trivial.")
        else:
            print("    [OK] Data is distinct.")

        # 2. Lag 0 Baseline vs Optimal (Task Difficulty)
        # ------------------------------------------------
        # Take a chunk
        start_t = 32
        Tw = 16
        MaxLag = 16
        
        X_chunk = mic[start_t-MaxLag : start_t+Tw] # (32, F)
        Y_chunk = ldv[start_t : start_t+Tw]        # (16, F)
        
        # Check specific bin (e.g. 50)
        b = 50
        x_col = X_chunk[:, b]
        y_col = Y_chunk[:, b]
        
        # Lag 0 Prediction (Just X[16:] scaled)
        lag0_pred = x_col[16:32]
        # Optimal Scaling for Lag 0
        # alpha = (a . y) / (a . a)
        alpha = torch.vdot(lag0_pred, y_col) / (torch.vdot(lag0_pred, lag0_pred) + 1e-10)
        res_lag0 = y_col - alpha * lag0_pred
        
        init_E = torch.linalg.vector_norm(y_col)**2
        lag0_E = torch.linalg.vector_norm(res_lag0)**2
        red_lag0 = 1.0 - (lag0_E / init_E)
        
        print(f"  > Task Difficulty (Bin {b}):")
        print(f"    Lag 0 Reduction: {red_lag0:.2%}")
        
        if red_lag0 > 0.95:
             print("    [!] WARNING: Lag 0 solves the task. Optimization is unnecessary.")
        else:
             print(f"    [OK] Lag 0 is not enough ({red_lag0:.2%}). Search is required.")

        # 3. Speech vs Noise (Spectral Analysis)
        # ------------------------------------------------
        # Check Spectral Flatness of input MIC
        # Average mag spectrum over time
        mag_avg = torch.mean(torch.abs(mic), dim=0).cpu() # (F,)
        sfm = compute_spectral_flatness(mag_avg)
        
        print(f"  > Signal Content Check:")
        print(f"    Spectral Flatness Measure (SFM): {sfm:.4f}")
        if sfm > 0.6:
            print("    [?] Signal looks like White Noise (High SFM).")
        else:
            print("    [OK] Signal looks like Speech/Tonal (Low SFM).")
            
        # 4. Silence Detection
        energy_profile = torch.norm(mic, dim=1) # (T,)
        silent_frames = torch.sum(energy_profile < 1e-4)
        print(f"    Silent Frames: {silent_frames}/{len(energy_profile)}")

        
        # 5. Coherence Check (Linearity Leakage)
        # ------------------------------------------------
        # Use scipy for coherence
        f, Cxy = scipy.signal.coherence(mic[:, b].cpu().numpy(), ldv[:, b].cpu().numpy(), fs=16000, nperseg=32)
        avg_coherence = np.mean(Cxy)
        print(f"  > Linearity (Coherence):")
        print(f"    Avg Coherence (Bin {b}): {avg_coherence:.4f}")
        
        if avg_coherence > 0.99:
            print("    [!] WARNING: Perfect Linear Coherence. Likely simulated or simple gain transfer.")
            
        if idx == 0:
            # Plot Spectrogram
            plt.figure(figsize=(10, 6))
            plt.subplot(2, 1, 1)
            plt.imshow(torch.log10(torch.abs(mic).T + 1e-6).numpy(), aspect='auto', origin='lower')
            plt.title("Mic Spectrogram (Clip 0)")
            plt.subplot(2, 1, 2)
            plt.imshow(torch.log10(torch.abs(ldv).T + 1e-6).numpy(), aspect='auto', origin='lower')
            plt.title("LDV Spectrogram (Clip 0)")
            plt.tight_layout()
            plt.savefig("spectrum_check.png")
            print("\n    [Saved] spectrum_check.png")
        
if __name__ == "__main__":
    run_integrity_check()
