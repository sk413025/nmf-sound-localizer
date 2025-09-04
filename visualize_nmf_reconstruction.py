#!/usr/bin/env python3
"""
NMF Reconstruction Visualization Tool
Reads JSON data from debug_nmf_with_learned_sources.py and creates comparison plots
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
import sys
from pathlib import Path

# Configure matplotlib for Chinese characters
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
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

def create_comparison_plot(data, arrays):
    """Create comprehensive comparison plot"""
    Y_data = arrays['Y_data']
    Y_direct_original = arrays['Y_direct_original']
    Y_direct_log_best = arrays['Y_direct_log_best']
    
    corr_original = data['correlations']['original_method']
    corr_log_best = data['correlations']['log_space_method']
    improvement_percent = data['correlations']['improvement_percent']
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'NMF 重建品質分析 (λ_group=0.1, γ_sparse=0.01)\n'
                 f'原始方法 vs 對數空間改進方法', fontsize=16, fontweight='bold')
    
    # Plot 1: Scatter plot - Original method
    ax1 = axes[0, 0]
    ax1.scatter(Y_data.flatten(), Y_direct_original.flatten(), 
               alpha=data['visualization_params']['scatter_alpha'], 
               s=data['visualization_params']['scatter_size'], c='red')
    ax1.plot([Y_data.min(), Y_data.max()], [Y_data.min(), Y_data.max()], 
             'k--', alpha=0.8, linewidth=2)
    ax1.set_xlabel('Y (實際值)')
    ax1.set_ylabel('Y_hat (重建值)')
    ax1.set_title(f'Y vs Y_hat 數據點對比\n理想對照 (r=Y vs Y_hat)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot - Log space method  
    ax2 = axes[0, 1]
    ax2.scatter(Y_data.flatten(), Y_direct_log_best.flatten(),
               alpha=data['visualization_params']['scatter_alpha'],
               s=data['visualization_params']['scatter_size'], c='blue')
    ax2.plot([Y_data.min(), Y_data.max()], [Y_data.min(), Y_data.max()],
             'k--', alpha=0.8, linewidth=2)
    ax2.set_xlabel('Y (實際值)')
    ax2.set_ylabel('Y_hat (重建值)')  
    ax2.set_title(f'對角線對比 Y vs Y_hat\n— Y (原值)\n— Y_hat (重建值)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Statistical comparison
    ax3 = axes[1, 0]
    methods = ['Y (原值)', 'Y_hat (重建值)']
    correlations = [corr_original, corr_log_best]
    colors = ['red', 'blue']
    
    bars = ax3.bar(methods, correlations, color=colors, alpha=0.7)
    ax3.set_ylabel('相關性')
    ax3.set_title('統計數據對比')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{corr:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_ylim(0, max(correlations) * 1.1)
    
    # Add statistics text
    stats_text = f"""數據相關性總結：
相關性 r_分析: {corr_original:.4f}
相關性 r_理想: {corr_log_best:.4f}
差異相關性標準 {corr_log_best:.4f} vs {corr_original:.4f}
分析標準值: Div率 = {improvement_percent:.1f}%"""
    
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    # Plot 4: IS Divergence analysis  
    ax4 = axes[1, 1]
    methods_div = ['數據相關性標準', 'IS Divergence分析結果']
    div_values = [46.50, 46.07]  # Values from the analysis
    colors_div = ['blue', 'red']
    
    bars_div = ax4.bar(methods_div, div_values, color=colors_div, alpha=0.7)
    ax4.set_ylabel('IS Divergence分母 (越高越好)')
    ax4.set_title('IS Divergence分析 分母 (基礎測試分析)')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add IS divergence labels
    for bar, div_val in zip(bars_div, div_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{div_val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize NMF reconstruction results')
    parser.add_argument('json_file', nargs='?', help='JSON data file to visualize')
    parser.add_argument('--output', '-o', help='Output plot filename (optional)')
    parser.add_argument('--show', action='store_true', help='Show plot interactively')
    
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
        print(f"   Analysis timestamp: {data['metadata']['timestamp']}")
        print(f"   Original method correlation: {data['correlations']['original_method']:.4f}")
        print(f"   Log space method correlation: {data['correlations']['log_space_method']:.4f}")
        print(f"   Improvement: +{data['correlations']['improvement_percent']:.1f}%")
        
        # Create visualization
        print(f"🎨 Creating comparison plots...")
        fig = create_comparison_plot(data, arrays)
        
        # Save plot
        if args.output:
            output_filename = args.output
        else:
            timestamp = data['metadata']['timestamp']
            output_filename = f'nmf_reconstruction_analysis_{timestamp}.png'
        
        fig.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved as: {output_filename}")
        
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