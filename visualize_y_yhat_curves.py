#!/usr/bin/env python3
"""
Simple Y vs Y_hat Curve Comparison Visualization
Focuses specifically on comparing actual vs reconstructed signals
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
import sys
from pathlib import Path

# Configure matplotlib for English display
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_reconstruction_data(json_file):
    """Load reconstruction data from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Convert lists back to numpy arrays
    arrays = {}
    for key, array_list in data['arrays'].items():
        arrays[key] = np.array(array_list)
    
    return data, arrays

def create_curve_comparison(data, arrays, n_freq_samples=3, n_time_samples=3):
    """Create focused Y vs Y_hat curve comparison plots with separate panels"""
    Y_data = arrays['Y_data']
    Y_direct_log_best = arrays['Y_direct_log_best']
    
    corr_original = data['correlations']['original_method']
    corr_log_best = data['correlations']['log_space_method']
    improvement_percent = data['correlations']['improvement_percent']
    
    n_freq_bins, n_time_steps = Y_data.shape
    
    # Create figure with 2 rows x 3 columns = 6 panels
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Y vs Y_hat Curve Comparison Analysis (Separate Panels)\n'
                 f'Correlation Improvement: {corr_original:.4f} → {corr_log_best:.4f} (+{improvement_percent:.1f}%)', 
                 fontsize=16, fontweight='bold')
    
    time_axis = np.arange(n_time_steps)
    freq_axis = np.arange(n_freq_bins)
    
    # Select representative frequency bins and time steps
    freq_indices = np.linspace(0, n_freq_bins-1, n_freq_samples, dtype=int)
    time_indices = np.linspace(0, n_time_steps-1, n_time_samples, dtype=int)
    
    # Top row: Time domain comparison (separate panel for each frequency)
    for i, freq_idx in enumerate(freq_indices):
        ax = axes[0, i]
        
        y_freq = Y_data[freq_idx, :]
        yhat_freq = Y_direct_log_best[freq_idx, :]
        
        # Check if there's a large magnitude difference
        y_max = np.max(y_freq)
        yhat_max = np.max(yhat_freq)
        magnitude_ratio = yhat_max / max(y_max, 1e-12)
        
        # If magnitude difference is large (>10x), use dual y-axis
        if magnitude_ratio < 0.1 or magnitude_ratio > 10:
            # Create secondary y-axis
            ax2 = ax.twinx()
            
            # Plot Y_actual on primary axis (left)
            line1 = ax.plot(time_axis, y_freq, 
                           color='blue', linestyle='-', linewidth=3, alpha=0.8,
                           label='Y_actual', marker='o', markersize=2)[0]
            
            # Plot Y_reconstructed on secondary axis (right)
            line2 = ax2.plot(time_axis, yhat_freq,
                            color='red', linestyle='--', linewidth=2.5, alpha=0.9,
                            label='Y_reconstructed', marker='s', markersize=1.5)[0]
            
            # Set labels and colors
            ax.set_ylabel('Y_actual Amplitude', fontsize=11, color='blue')
            ax2.set_ylabel('Y_reconstructed Amplitude', fontsize=11, color='red')
            ax.tick_params(axis='y', labelcolor='blue')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Create combined legend
            lines = [line1, line2]
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper right', fontsize=10)
            
        else:
            # Use single axis when magnitudes are similar
            ax.plot(time_axis, y_freq, 
                   color='blue', linestyle='-', linewidth=3, alpha=0.8,
                   label='Y_actual', marker='o', markersize=2)
            
            ax.plot(time_axis, yhat_freq, 
                   color='red', linestyle='--', linewidth=2.5, alpha=0.9,
                   label='Y_reconstructed', marker='s', markersize=1.5)
            
            ax.set_ylabel('Amplitude', fontsize=11)
            ax.legend(fontsize=10)
        
        # Calculate correlation for this frequency
        if np.std(y_freq) > 1e-12 and np.std(yhat_freq) > 1e-12:
            freq_corr = np.corrcoef(y_freq, yhat_freq)[0, 1]
        else:
            freq_corr = 0.0
            
        ax.set_xlabel('Time Steps', fontsize=11)
        ax.set_title(f'Frequency bin {freq_idx} Time Series\nr = {freq_corr:.3f} (ratio={magnitude_ratio:.3f})', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text with magnitude info
        stats_text = f'Mean Y: {np.mean(y_freq):.2e}\nMean Y_hat: {np.mean(yhat_freq):.2e}\nMax ratio: {magnitude_ratio:.3f}\n{"Dual-axis" if (magnitude_ratio < 0.1 or magnitude_ratio > 10) else "Single-axis"}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Bottom row: Frequency domain comparison (separate panel for each time step)
    for i, time_idx in enumerate(time_indices):
        ax = axes[1, i]
        
        y_time = Y_data[:, time_idx]
        yhat_time = Y_direct_log_best[:, time_idx]
        
        # Check if there's a large magnitude difference
        y_max = np.max(y_time)
        yhat_max = np.max(yhat_time)
        magnitude_ratio = yhat_max / max(y_max, 1e-12)
        
        # If magnitude difference is large (>10x), use dual y-axis
        if magnitude_ratio < 0.1 or magnitude_ratio > 10:
            # Create secondary y-axis
            ax2 = ax.twinx()
            
            # Plot Y_actual on primary axis (left)
            line1 = ax.plot(freq_axis, y_time, 
                           color='green', linestyle='-', linewidth=3, alpha=0.8,
                           label='Y_actual', marker='o', markersize=1)[0]
            
            # Plot Y_reconstructed on secondary axis (right)  
            line2 = ax2.plot(freq_axis, yhat_time,
                            color='orange', linestyle='--', linewidth=2.5, alpha=0.9,
                            label='Y_reconstructed', marker='s', markersize=0.8)[0]
            
            # Set labels and colors
            ax.set_ylabel('Y_actual Amplitude', fontsize=11, color='green')
            ax2.set_ylabel('Y_reconstructed Amplitude', fontsize=11, color='orange')
            ax.tick_params(axis='y', labelcolor='green')
            ax2.tick_params(axis='y', labelcolor='orange')
            
            # Create combined legend
            lines = [line1, line2]
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper right', fontsize=10)
            
        else:
            # Use single axis when magnitudes are similar
            ax.plot(freq_axis, y_time, 
                   color='green', linestyle='-', linewidth=3, alpha=0.8,
                   label='Y_actual', marker='o', markersize=1)
            
            ax.plot(freq_axis, yhat_time, 
                   color='orange', linestyle='--', linewidth=2.5, alpha=0.9,
                   label='Y_reconstructed', marker='s', markersize=0.8)
            
            ax.set_ylabel('Amplitude', fontsize=11)
            ax.legend(fontsize=10)
        
        # Calculate correlation for this time step
        if np.std(y_time) > 1e-12 and np.std(yhat_time) > 1e-12:
            time_corr = np.corrcoef(y_time, yhat_time)[0, 1]
        else:
            time_corr = 0.0
            
        ax.set_xlabel('Frequency Bin', fontsize=11)
        ax.set_title(f'Time step {time_idx} Spectrum\nr = {time_corr:.3f} (ratio={magnitude_ratio:.3f})', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text with magnitude info
        stats_text = f'Mean Y: {np.mean(y_time):.2e}\nMean Y_hat: {np.mean(yhat_time):.2e}\nMax ratio: {magnitude_ratio:.3f}\n{"Dual-axis" if (magnitude_ratio < 0.1 or magnitude_ratio > 10) else "Single-axis"}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize Y vs Y_hat curve comparison')
    parser.add_argument('json_file', nargs='?', help='JSON data file to visualize')
    parser.add_argument('--output', '-o', default='y_yhat_curves_comparison.png', 
                       help='Output plot filename')
    parser.add_argument('--show', action='store_true', help='Show plot interactively')
    parser.add_argument('--n-freq', type=int, default=3, 
                       help='Number of frequency samples to display')
    parser.add_argument('--n-time', type=int, default=3, 
                       help='Number of time samples to display')
    
    args = parser.parse_args()
    
    # If no file specified, look for the most recent one
    if not args.json_file:
        json_files = list(Path('.').glob('nmf_reconstruction_data_*.json'))
        if not json_files:
            print("❌ No reconstruction data files found.")
            print("   Run debug_nmf_with_learned_sources.py first to generate data.")
            return 1
        
        # Get the most recent file
        args.json_file = max(json_files, key=lambda x: x.stat().st_mtime)
        print(f"📁 Using most recent data file: {args.json_file}")
    
    # Check if file exists
    if not Path(args.json_file).exists():
        print(f"❌ File not found: {args.json_file}")
        return 1
    
    try:
        # Load data
        print(f"📊 Loading reconstruction data from {args.json_file}...")
        data, arrays = load_reconstruction_data(args.json_file)
        
        # Display key metrics
        print(f"✅ Data loaded successfully")
        print(f"   Test angle: {data['metadata']['test_angle']}")
        print(f"   Data shape: {data['metadata']['data_shapes']['Y_data']}")
        print(f"   Original correlation: {data['correlations']['original_method']:.4f}")
        print(f"   Log-space correlation: {data['correlations']['log_space_method']:.4f}")
        print(f"   Improvement: +{data['correlations']['improvement_percent']:.1f}%")
        
        # Create visualization
        print(f"🎨 Creating Y vs Y_hat curve comparison...")
        print(f"   Frequency samples: {args.n_freq}")
        print(f"   Time samples: {args.n_time}")
        
        fig = create_curve_comparison(data, arrays, args.n_freq, args.n_time)
        
        # Save plot
        fig.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"✅ Curve comparison plot saved as: {args.output}")
        
        # Show plot if requested
        if args.show:
            plt.show()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())