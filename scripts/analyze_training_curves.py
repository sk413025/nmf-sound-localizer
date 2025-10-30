#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze and visualize train/test loss curves from DT training history.
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Apply canonical plotting style from AGENTS.md
NATURE_ACCENT_PALETTE = [
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#56B4E9",  # Sky blue
    "#CC79A7",  # Reddish purple
    "#009E73",  # Bluish green
    "#F0E442",  # Yellow
    "#D55E00",  # Vermillion (warnings)
    "#999999",  # Neutral grey
]

mpl.rcParams.update({
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "figure.dpi": 150,
    "text.color": "#000000",
    "axes.edgecolor": "#4D4D4D",
    "axes.labelcolor": "#000000",
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.grid": False,
    "axes.linewidth": 1.0,
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.facecolor": "#FFFFFF",
})


def load_checkpoint_history(ckpt_path: Path):
    """Load training history from checkpoint."""
    import torch
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    return ckpt.get('history', [])


def plot_loss_curves(history, out_path: Path):
    """Plot train vs test loss curves."""
    if not history:
        print("No history data found!")
        return
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h.get('train_loss_eval', h.get('train_loss', np.nan)) for h in history]
    test_loss = [h.get('test_loss', np.nan) for h in history]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot lines
    ax.plot(epochs, train_loss, '-o', color=NATURE_ACCENT_PALETTE[0], 
            linewidth=2, markersize=6, label='Train Loss')
    ax.plot(epochs, test_loss, '-s', color=NATURE_ACCENT_PALETTE[1], 
            linewidth=2, markersize=6, label='Test Loss')
    
    # Find best test loss epoch
    if not all(np.isnan(test_loss)):
        best_idx = np.nanargmin(test_loss)
        best_epoch = epochs[best_idx]
        best_loss = test_loss[best_idx]
        ax.axvline(best_epoch, color=NATURE_ACCENT_PALETTE[3], 
                   linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'Best Test (Epoch {best_epoch})')
        ax.plot(best_epoch, best_loss, 'r*', markersize=15, zorder=5)
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.set_title('Decision Transformer: Train vs Test Loss', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    print(f"Saved loss curve to: {out_path}")
    
    # Also save PNG for quick viewing
    png_path = out_path.with_suffix('.png')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f"Saved PNG to: {png_path}")
    plt.close()


def plot_accuracy_curves(history, out_path: Path):
    """Plot train vs test accuracy curves."""
    if not history:
        return
    
    epochs = [h['epoch'] for h in history]
    train_acc_e = [h.get('train_acc_e', np.nan) for h in history]
    test_acc_e = [h.get('test_acc_e', np.nan) for h in history]
    train_acc_a = [h.get('train_acc_a', np.nan) for h in history]
    test_acc_a = [h.get('test_acc_a', np.nan) for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Expert accuracy
    ax1.plot(epochs, train_acc_e, '-o', color=NATURE_ACCENT_PALETTE[0], 
             linewidth=2, markersize=6, label='Train Expert Acc')
    ax1.plot(epochs, test_acc_e, '-s', color=NATURE_ACCENT_PALETTE[1], 
             linewidth=2, markersize=6, label='Test Expert Acc')
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Expert Accuracy', fontsize=14, fontweight='bold')
    ax1.set_title('Expert Classification Accuracy', fontsize=16, fontweight='bold')
    ax1.legend(loc='best', fontsize=12)
    ax1.yaxis.grid(True, color="#E6E6E6", linewidth=0.6)
    ax1.set_axisbelow(True)
    ax1.set_ylim([0, 1])
    
    # Atom accuracy
    ax2.plot(epochs, train_acc_a, '-o', color=NATURE_ACCENT_PALETTE[0], 
             linewidth=2, markersize=6, label='Train Atom Acc')
    ax2.plot(epochs, test_acc_a, '-s', color=NATURE_ACCENT_PALETTE[1], 
             linewidth=2, markersize=6, label='Test Atom Acc')
    ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Atom Accuracy', fontsize=14, fontweight='bold')
    ax2.set_title('Atom Selection Accuracy', fontsize=16, fontweight='bold')
    ax2.legend(loc='best', fontsize=12)
    ax2.yaxis.grid(True, color="#E6E6E6", linewidth=0.6)
    ax2.set_axisbelow(True)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    print(f"Saved accuracy curves to: {out_path}")
    
    png_path = out_path.with_suffix('.png')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f"Saved PNG to: {png_path}")
    plt.close()


def print_summary(history):
    """Print summary statistics."""
    if not history:
        print("No history data!")
        return
    
    print("\n" + "="*80)
    print("Training Summary")
    print("="*80)
    
    last = history[-1]
    print(f"\nFinal Epoch: {last['epoch']}")
    print(f"  Train Loss: {last.get('train_loss_eval', last.get('train_loss', 'N/A')):.4f}")
    print(f"  Test Loss:  {last.get('test_loss', 'N/A'):.4f}")
    print(f"  Train Acc:  expert={last.get('train_acc_e', 'N/A'):.3f}, atom={last.get('train_acc_a', 'N/A'):.3f}")
    print(f"  Test Acc:   expert={last.get('test_acc_e', 'N/A'):.3f}, atom={last.get('test_acc_a', 'N/A'):.3f}")
    
    # Check for overfitting
    test_losses = [h.get('test_loss', np.nan) for h in history]
    if not all(np.isnan(test_losses)):
        best_idx = np.nanargmin(test_losses)
        best_epoch = history[best_idx]['epoch']
        best_loss = test_losses[best_idx]
        
        print(f"\nBest Test Loss: {best_loss:.4f} at epoch {best_epoch}")
        
        if best_epoch < last['epoch']:
            gap = last['epoch'] - best_epoch
            print(f"⚠️  Warning: Test loss has not improved for {gap} epochs")
            print(f"   This suggests potential overfitting!")
        else:
            print("✓ Model is still improving on test set")
    
    # Check train-test gap
    if 'train_loss_eval' in last and 'test_loss' in last:
        gap = last['test_loss'] - last['train_loss_eval']
        gap_pct = gap / last['train_loss_eval'] * 100
        print(f"\nTrain-Test Loss Gap: {gap:+.4f} ({gap_pct:+.1f}%)")
        if gap_pct > 20:
            print(f"⚠️  Large gap suggests overfitting")
        elif gap_pct > 10:
            print(f"⚠️  Moderate gap - monitor closely")
        else:
            print(f"✓ Reasonable generalization gap")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze DT training curves')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to checkpoint file (ckpt_latest.pth)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory for plots (default: same as checkpoint dir)')
    args = parser.parse_args()
    
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found: {ckpt_path}")
        return
    
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading checkpoint: {ckpt_path}")
    history = load_checkpoint_history(ckpt_path)
    
    if not history:
        print("No training history found in checkpoint!")
        return
    
    print(f"Found {len(history)} epochs of training history")
    
    # Print summary
    print_summary(history)
    
    # Generate plots
    plot_loss_curves(history, out_dir / 'loss_curves.pdf')
    plot_accuracy_curves(history, out_dir / 'accuracy_curves.pdf')
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
