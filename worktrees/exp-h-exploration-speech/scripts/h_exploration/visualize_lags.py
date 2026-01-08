import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    data = torch.load(args.stats_path, weights_only=False)
    
    # Aggregate Histograms
    total_hist = None
    
    for item in data:
        hist = item["lags_hist"]
        if total_hist is None:
            total_hist = np.zeros_like(hist)
        total_hist += hist
        
    # Plot
    plt.figure(figsize=(8, 5))
    lags = range(len(total_hist))
    plt.bar(lags, total_hist)
    plt.title("Lag Selection Distribution (Impulse Response Structure)")
    plt.xlabel("Lag (Frames) [16kHz, hop=512 -> 32ms/lag]")
    plt.ylabel("Selection Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path / "lag_distribution.png")
    plt.close()
    
    logger.info(f"Saved plot to {out_path}")
    
    # Analyze
    logger.info(f"Total Histogram: {total_hist}")
    peak_lag = np.argmax(total_hist)
    logger.info(f"Most significant lag: {peak_lag}")

if __name__ == "__main__":
    main()
