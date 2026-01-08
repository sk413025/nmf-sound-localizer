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
    parser.add_argument("--traj_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading {args.traj_path}...")
    data = torch.load(args.traj_path, weights_only=False)
    
    # 1. Action Distribution
    actions = []
    mses = []
    
    # Sample a subset for plotting if too large
    subset = data[:1000] if len(data) > 1000 else data
    
    for item in subset:
        for step in item["steps"]:
            actions.append(step["action"])
            mses.append(step["mse"])
            
    # Plot Action Hist
    plt.figure(figsize=(6, 4))
    plt.hist(actions, bins=range(max(actions)+2), align='left', rwidth=0.8)
    plt.title("Action Distribution (Channel Selection)")
    plt.xlabel("Channel Index")
    plt.ylabel("Count")
    plt.xticks(range(max(actions)+1))
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path / "action_distribution.png")
    plt.close()
    
    # Plot MSE Distribution
    plt.figure(figsize=(6, 4))
    plt.hist(mses, bins=50)
    plt.title("MSE Distribution (Reconstruction Error)")
    plt.xlabel("MSE")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path / "mse_distribution.png")
    plt.close()
    
    logger.info(f"Saved plots to {out_path}")
    logger.info(f"Unique Actions: {set(actions)}")
    logger.info(f"Average MSE: {np.mean(mses)}")

if __name__ == "__main__":
    main()
