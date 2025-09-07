#!/usr/bin/env python3
"""
Multi-Angle Comparison Visualization
Creates comprehensive comparison plots for multiple angle test results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
import sys
from pathlib import Path

# Configure matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_summary_data(json_file):
    """Load angle test summary data"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def create_multi_angle_comparison(summary_data):
    """Create comprehensive multi-angle comparison visualization"""
    
    # Extract data for plotting
    results = [r for r in summary_data['detailed_results'] if r['success']]
    angles = [r['angle'] for r in results]
    original_corrs = [r['original_correlation'] for r in results]
    log_corrs = [r['log_space_correlation'] for r in results]
    improvements = [r['improvement_percent'] for r in results]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'NMF Multi-Angle Performance Analysis (100°-130°)\n'
                 f'Log-Space vs Original Method Comparison', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Correlation comparison by angle
    ax1 = axes[0, 0]
    x_pos = np.arange(len(angles))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, original_corrs, width, 
                   label='Original Method', color='red', alpha=0.7)
    bars2 = ax1.bar(x_pos + width/2, log_corrs, width,
                   label='Log-Space Method', color='blue', alpha=0.7)
    
    ax1.set_xlabel('Angle (degrees)')
    ax1.set_ylabel('Correlation Coefficient')
    ax1.set_title('Correlation Performance by Angle')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{a}°' for a in angles])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    def add_value_labels(ax, bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    add_value_labels(ax1, bars1)
    add_value_labels(ax1, bars2)
    
    # Plot 2: Improvement percentage by angle
    ax2 = axes[0, 1]
    bars = ax2.bar(angles, improvements, color='green', alpha=0.7)
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Log-Space Method Improvement by Angle')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{imp:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Highlight best and worst performing angles
    best_idx = np.argmax(improvements)
    worst_idx = np.argmin(improvements)
    bars[best_idx].set_color('darkgreen')
    bars[worst_idx].set_color('orange')
    
    # Plot 3: Correlation scatter plot
    ax3 = axes[1, 0]
    scatter = ax3.scatter(original_corrs, log_corrs, c=angles, s=100, 
                         cmap='viridis', alpha=0.8, edgecolors='black')
    
    # Add diagonal line for reference
    min_corr = min(min(original_corrs), min(log_corrs))
    max_corr = max(max(original_corrs), max(log_corrs))
    ax3.plot([min_corr, max_corr], [min_corr, max_corr], 'k--', alpha=0.5, label='Equal performance')
    
    ax3.set_xlabel('Original Method Correlation')
    ax3.set_ylabel('Log-Space Method Correlation')
    ax3.set_title('Method Performance Correlation')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Angle (degrees)')
    
    # Annotate points with angles
    for i, angle in enumerate(angles):
        ax3.annotate(f'{angle}°', (original_corrs[i], log_corrs[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Plot 4: Performance statistics summary
    ax4 = axes[1, 1]
    
    # Create summary statistics
    stats_data = summary_data['performance_statistics']
    
    categories = ['Original\nMethod', 'Log-Space\nMethod', 'Improvement\n(%)']
    means = [
        stats_data['original_method']['mean_correlation'],
        stats_data['log_space_method']['mean_correlation'], 
        stats_data['improvement_analysis']['mean_improvement'] / 100  # Scale down for visualization
    ]
    stds = [
        stats_data['original_method']['std_correlation'],
        stats_data['log_space_method']['std_correlation'],
        stats_data['improvement_analysis']['std_improvement'] / 100  # Scale down
    ]
    
    bars = ax4.bar(categories[:2], means[:2], yerr=stds[:2], 
                  color=['red', 'blue'], alpha=0.7, capsize=5)
    
    ax4.set_ylabel('Correlation Coefficient')
    ax4.set_title('Overall Performance Statistics')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add mean value labels
    for bar, mean, std in zip(bars, means[:2], stds[:2]):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add improvement info as text
    improvement_text = f"""Average Improvement: +{stats_data['improvement_analysis']['mean_improvement']:.1f}%
Best Angle: {stats_data['improvement_analysis']['best_angle']}° (+{stats_data['improvement_analysis']['max_improvement']:.1f}%)
Worst Angle: {stats_data['improvement_analysis']['worst_angle']}° (+{stats_data['improvement_analysis']['min_improvement']:.1f}%)
Standard Deviation: ±{stats_data['improvement_analysis']['std_improvement']:.1f}%"""
    
    ax4.text(0.02, 0.98, improvement_text, transform=ax4.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
            fontsize=10)
    
    plt.tight_layout()
    return fig

def create_detailed_angle_progression(summary_data):
    """Create detailed progression analysis across angles"""
    
    results = [r for r in summary_data['detailed_results'] if r['success']]
    angles = [r['angle'] for r in results]
    original_corrs = [r['original_correlation'] for r in results]
    log_corrs = [r['log_space_correlation'] for r in results]
    improvements = [r['improvement_percent'] for r in results]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Detailed Angle Progression Analysis (100°-130°)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Raw correlation trends
    ax1 = axes[0]
    ax1.plot(angles, original_corrs, 'ro-', linewidth=2, markersize=8, 
             label='Original Method', alpha=0.8)
    ax1.plot(angles, log_corrs, 'bo-', linewidth=2, markersize=8,
             label='Log-Space Method', alpha=0.8)
    
    ax1.set_xlabel('Angle (degrees)')
    ax1.set_ylabel('Correlation Coefficient')
    ax1.set_title('Correlation Trends Across Angles')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Annotate min/max points
    min_orig_idx = np.argmin(original_corrs)
    max_orig_idx = np.argmax(original_corrs)
    min_log_idx = np.argmin(log_corrs)
    max_log_idx = np.argmax(log_corrs)
    
    ax1.annotate(f'Min: {angles[min_orig_idx]}°', 
                xy=(angles[min_orig_idx], original_corrs[min_orig_idx]),
                xytext=(10, 10), textcoords='offset points', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3))
    
    ax1.annotate(f'Max: {angles[max_log_idx]}°', 
                xy=(angles[max_log_idx], log_corrs[max_log_idx]),
                xytext=(10, -15), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.3))
    
    # Plot 2: Improvement progression
    ax2 = axes[1]
    bars = ax2.bar(angles, improvements, color='green', alpha=0.7, width=2)
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Log-Space Improvement Progression')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add trend line
    z = np.polyfit(angles, improvements, 2)  # 2nd order polynomial
    p = np.poly1d(z)
    ax2.plot(angles, p(angles), 'r--', alpha=0.8, linewidth=2, label='Trend (2nd order)')
    ax2.legend()
    
    # Highlight extremes
    best_idx = np.argmax(improvements)
    worst_idx = np.argmin(improvements)
    bars[best_idx].set_color('darkgreen')
    bars[worst_idx].set_color('orange')
    
    # Plot 3: Performance stability analysis
    ax3 = axes[2]
    
    # Calculate moving statistics (window size 3)
    window_size = 3
    if len(angles) >= window_size:
        moving_mean_orig = np.convolve(original_corrs, np.ones(window_size)/window_size, mode='valid')
        moving_mean_log = np.convolve(log_corrs, np.ones(window_size)/window_size, mode='valid')
        moving_angles = angles[window_size-1:]
        
        ax3.plot(moving_angles, moving_mean_orig, 's--', linewidth=2, markersize=6,
                label=f'Original (MA-{window_size})', alpha=0.8, color='red')
        ax3.plot(moving_angles, moving_mean_log, 's--', linewidth=2, markersize=6,
                label=f'Log-Space (MA-{window_size})', alpha=0.8, color='blue')
    
    # Add raw data as background
    ax3.plot(angles, original_corrs, 'ro', alpha=0.3, markersize=4, label='Original (raw)')
    ax3.plot(angles, log_corrs, 'bo', alpha=0.3, markersize=4, label='Log-Space (raw)')
    
    ax3.set_xlabel('Angle (degrees)')
    ax3.set_ylabel('Correlation Coefficient')
    ax3.set_title('Performance Stability Analysis (Moving Average)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize multi-angle comparison results')
    parser.add_argument('--summary', default='angle_test_summary.json',
                       help='JSON summary file from batch testing')
    parser.add_argument('--output-comparison', default='multi_angle_comparison.png',
                       help='Output filename for comparison plot')
    parser.add_argument('--output-progression', default='angle_progression_analysis.png', 
                       help='Output filename for progression plot')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    
    args = parser.parse_args()
    
    # Check if summary file exists
    if not Path(args.summary).exists():
        print(f"❌ Summary file not found: {args.summary}")
        print("   Run batch_angle_test.py first to generate summary data.")
        return 1
    
    try:
        # Load summary data
        print(f"📊 Loading summary data from {args.summary}...")
        summary_data = load_summary_data(args.summary)
        
        # Display key info
        stats = summary_data['performance_statistics']
        print(f"✅ Loaded data for {summary_data['test_summary']['successful_tests']} angles")
        print(f"   Angle range: {summary_data['test_summary']['angle_range']}")
        print(f"   Mean improvement: +{stats['improvement_analysis']['mean_improvement']:.1f}%")
        print(f"   Best angle: {stats['improvement_analysis']['best_angle']}°")
        print(f"   Worst angle: {stats['improvement_analysis']['worst_angle']}°")
        
        # Create comparison visualization
        print("🎨 Creating multi-angle comparison plot...")
        fig1 = create_multi_angle_comparison(summary_data)
        fig1.savefig(args.output_comparison, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison plot saved: {args.output_comparison}")
        
        # Create progression analysis
        print("🎨 Creating angle progression analysis...")
        fig2 = create_detailed_angle_progression(summary_data)
        fig2.savefig(args.output_progression, dpi=300, bbox_inches='tight')
        print(f"✅ Progression plot saved: {args.output_progression}")
        
        # Show plots if requested
        if args.show:
            plt.show()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())