import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--hop_length", type=int, default=160)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data = torch.load(args.weights_path, weights_only=False)
    weights_sum = data["weights_sum"] # (F, MaxLag) complex
    counts = data["counts"]           # (F, MaxLag) int
    
    # Compute Average Effective Weights (Impulse Response)
    # Total frames per frequency can be estimated since OMP selects fixed K usually.
    # But let's check counts distribution.
    # Assuming K=3 fixed roughly.
    total_active = np.sum(counts, axis=1, keepdims=True) # (F, 1)
    K_omp = 3.0 # Defined in generate_lag_omp
    total_frames = total_active / K_omp
    
    # H_eff = Sum / Total_Frames
    #       = (Sum / Counts) * (Counts / Total_Frames)
    #       = h_avg_conditional * prob_selection
    
    # Use h_eff for transfer function
    h_eff = np.zeros_like(weights_sum)
    non_zero_frames = total_frames > 0
    # Broadcast
    h_eff = weights_sum / total_frames
    
    # Use h_eff for plotting
    h_avg = h_eff
    
    F_bins, MaxLag = h_avg.shape
    
    # Visualization 1: Magnitude of weights per Lag per Frequency
    # Heatmap: X=Lag, Y=Freq
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(h_avg), aspect='auto', origin='lower', extent=[0, MaxLag, 0, F_bins], cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Lag (Frames)')
    plt.ylabel('Frequency Bin')
    plt.title(f'Average Lag Weights Magnitude (Hop={args.hop_length})')
    plt.savefig(out_dir / "h_avg_magnitude.png")
    plt.close()
    
    # Visualization 2: H(omega) - Frequency Response of the Lag Filter
    # Interpret lags as time samples with dt = hop_length / fs
    # FFT over the Lag dimension
    
    H_omega = np.fft.fft(h_avg, axis=1) # (F, MaxLag)
    # The frequency axis here corresponds to "Modulation Frequency"
    # Sampling rate = fs / hop_length = 16000 / 160 = 100 Hz
    # Freqs = [0, 100/16, ... 100/2] = [0, 6.25, ... 50] Hz
    
    mod_freqs = np.fft.fftfreq(MaxLag, d=args.hop_length/16000.0)
    
    # We only care about positive freqs usually, but let's just plot magnitude heatmap
    # Shift so 0 is center
    H_omega_shifted = np.fft.fftshift(H_omega, axes=1)
    freqs_shifted = np.fft.fftshift(mod_freqs)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(H_omega_shifted), aspect='auto', origin='lower', 
               extent=[freqs_shifted[0], freqs_shifted[-1], 0, F_bins], cmap='inferno')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Modulation Frequency (Hz)')
    plt.ylabel('Carrier Frequency Bin')
    plt.title(f'Lag Filter Frequency Response H(omega) (Hop={args.hop_length})')
    plt.savefig(out_dir / "h_omega_magnitude.png")
    plt.close()
    
    # Visualization 3: Phase of Lag 0 vs Frequency
    # Does Lag 0 have a consistent phase across frequencies? (Linear phase = delay)
    lag0_phase = np.angle(h_avg[:, 0])
    plt.figure(figsize=(12, 4))
    plt.plot(lag0_phase)
    plt.xlabel('Frequency Bin')
    plt.ylabel('Phase (rad)')
    plt.title('Phase of Lag 0 Weight (Instantaneous)')
    plt.savefig(out_dir / "lag0_phase.png")
    plt.close()
    
    # Dump simple stats
    logger.info(f"Mean Magnitude at Lag 0: {np.mean(np.abs(h_avg[:, 0])):.4f}")
    logger.info(f"Mean Magnitude at Lag 3: {np.mean(np.abs(h_avg[:, 3])):.4f} (expected peak if 30ms delay)")
    
    # Calculate Center of Gravity of Delay
    # Sum over F
    h_mean_lags = np.mean(np.abs(h_avg), axis=0) # (MaxLag,)
    lags = np.arange(MaxLag)
    cog = np.sum(lags * h_mean_lags) / np.sum(h_mean_lags)
    logger.info(f"Center of Gravity (Lags): {cog:.2f} (frames)")
    logger.info(f"Center of Gravity (Time): {cog * args.hop_length / 16000 * 1000:.2f} ms")
    
    logger.info(f"Saved plots to {out_dir}")

if __name__ == "__main__":
    main()
