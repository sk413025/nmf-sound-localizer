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
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/omp_performance")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading {args.data_path}...")
    raw_data = torch.load(args.data_path, weights_only=False)
    
    # Collect scores
    all_scores = []
    
    for block in raw_data:
        # block["scores"] is (F,)
        s = block["scores"].float().numpy()
        all_scores.append(s)
        
    scores_flat = np.concatenate(all_scores)
    
    # Statistics
    mean_score = np.mean(scores_flat)
    median_score = np.median(scores_flat)
    std_score = np.std(scores_flat)
    
    logger.info(f"Total Samples (F x Time): {len(scores_flat)}")
    logger.info(f"Mean Residual Reduction: {mean_score:.4f} ({mean_score*100:.2f}%)")
    logger.info(f"Median Residual Reduction: {median_score:.4f}")
    logger.info(f"Std Dev: {std_score:.4f}")
    
    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(scores_flat, bins=100, range=(0, 1), alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(mean_score, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_score:.2f}')
    plt.axvline(median_score, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_score:.2f}')
    
    plt.xlabel('Residual Reduction (Score)')
    plt.ylabel('Count')
    plt.title('OMP Performance: Residual Energy Reduction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "omp_residual_reduction_hist.png")
    plt.close()
    
    # Cumulative Distribution
    sorted_scores = np.sort(scores_flat)
    cdf = np.arange(len(sorted_scores)) / float(len(sorted_scores))
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_scores, cdf)
    plt.xlabel('Residual Reduction')
    plt.ylabel('CDF')
    plt.title('Cumulative Distribution of Performance')
    plt.grid(True)
    plt.savefig(out_dir / "omp_performance_cdf.png")
    plt.close()
    
    # Deciles
    deciles = np.percentile(scores_flat, np.arange(0, 101, 10))
    logger.info(f"Deciles: {deciles}")

if __name__ == "__main__":
    main()
