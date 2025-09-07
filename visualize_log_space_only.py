#!/usr/bin/env python3
"""
Log-Space Only Visualization
Focuses exclusively on log-space method results without comparison to original method
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

def create_log_space_performance_overview(summary_data):
    """Create comprehensive log-space performance visualization"""
    
    # Extract data for plotting
    results = [r for r in summary_data['detailed_results'] if r['success']]
    angles = [r['angle'] for r in results]
    log_corrs = [r['log_space_correlation'] for r in results]
    improvements = [r['improvement_percent'] for r in results]
    
    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Log-Space NMF Performance Analysis ({min(angles)}°-{max(angles)}°)\n'
                 f'Advanced Transfer Function Estimation Results', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Log-space correlation by angle
    ax1 = axes[0, 0]
    bars = ax1.bar(angles, log_corrs, color='blue', alpha=0.8, width=2.5)
    ax1.set_xlabel('Angle (degrees)')
    ax1.set_ylabel('Log-Space Correlation')
    ax1.set_title('Log-Space Method Correlation Performance')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1)
    
    # Color-code bars by performance
    max_corr = max(log_corrs)
    min_corr = min(log_corrs)
    for bar, corr in zip(bars, log_corrs):
        if corr == max_corr:
            bar.set_color('darkgreen')
        elif corr == min_corr:
            bar.set_color('orange')
    
    # Add value labels
    for bar, corr in zip(bars, log_corrs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{corr:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Performance improvement distribution
    ax2 = axes[0, 1]
    bars2 = ax2.bar(angles, improvements, color='green', alpha=0.8, width=2.5)
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Log-Space Method Improvement by Angle')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Color-code by improvement level
    max_imp = max(improvements)
    min_imp = min(improvements)
    for bar, imp in zip(bars2, improvements):
        if imp == max_imp:
            bar.set_color('darkgreen')
        elif imp == min_imp:
            bar.set_color('orange')
        else:
            # Gradient coloring based on improvement level
            normalized_imp = (imp - min_imp) / (max_imp - min_imp)
            bar.set_color(plt.cm.RdYlGn(0.3 + 0.7 * normalized_imp))
    
    # Add value labels
    for bar, imp in zip(bars2, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
               f'{imp:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: Correlation trend analysis
    ax3 = axes[1, 0]
    
    # Plot correlation trend
    ax3.plot(angles, log_corrs, 'bo-', linewidth=3, markersize=8, alpha=0.8, label='Log-Space Correlation')
    
    # Add trend line
    z = np.polyfit(angles, log_corrs, 2)  # 2nd order polynomial
    p = np.poly1d(z)
    trend_x = np.linspace(min(angles), max(angles), 100)
    ax3.plot(trend_x, p(trend_x), 'r--', alpha=0.8, linewidth=2, label='Polynomial Trend')
    
    # Add horizontal line for mean
    mean_corr = np.mean(log_corrs)
    ax3.axhline(y=mean_corr, color='gray', linestyle=':', alpha=0.8, label=f'Mean: {mean_corr:.3f}')
    
    ax3.set_xlabel('Angle (degrees)')
    ax3.set_ylabel('Log-Space Correlation')
    ax3.set_title('Correlation Trend Analysis')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(0, 1)
    
    # Annotate peak performance
    best_idx = np.argmax(log_corrs)
    ax3.annotate(f'Peak: {angles[best_idx]}°\n{log_corrs[best_idx]:.3f}', 
                xy=(angles[best_idx], log_corrs[best_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot 4: Performance statistics and angular sectors
    ax4 = axes[1, 1]
    
    # Define angular sectors
    sectors = {
        'Side (90-100°)': [a for a in angles if 90 <= a <= 100],
        'Rear-Side (105-115°)': [a for a in angles if 105 <= a <= 115],
        'Rear (120-130°)': [a for a in angles if 120 <= a <= 130]
    }
    
    sector_means = []
    sector_stds = []
    sector_labels = []
    
    for sector_name, sector_angles in sectors.items():
        if sector_angles:
            sector_corrs = [log_corrs[angles.index(a)] for a in sector_angles]
            sector_means.append(np.mean(sector_corrs))
            sector_stds.append(np.std(sector_corrs))
            sector_labels.append(sector_name)
    
    if sector_means:
        bars4 = ax4.bar(range(len(sector_labels)), sector_means, yerr=sector_stds,
                       color=['blue', 'green', 'red'], alpha=0.7, capsize=5)
        ax4.set_xlabel('Angular Sectors')
        ax4.set_ylabel('Mean Log-Space Correlation')
        ax4.set_title('Performance by Angular Sector')
        ax4.set_xticks(range(len(sector_labels)))
        ax4.set_xticklabels(sector_labels, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, 1)
        
        # Add value labels
        for bar, mean, std in zip(bars4, sector_means, sector_stds):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                   f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
    
    # Add overall statistics text
    stats = summary_data['performance_statistics']
    stats_text = f"""Overall Statistics:
Mean Correlation: {stats['log_space_method']['mean_correlation']:.3f}
Std Deviation: {stats['log_space_method']['std_correlation']:.3f}
Range: {stats['log_space_method']['min_correlation']:.3f} - {stats['log_space_method']['max_correlation']:.3f}
Mean Improvement: {stats['improvement_analysis']['mean_improvement']:.1f}%
Best Angle: {stats['improvement_analysis']['best_angle']}°
Success Rate: {summary_data['test_summary']['success_rate']:.0f}%"""
    
    ax4.text(0.02, 0.02, stats_text, transform=ax4.transAxes,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
            fontsize=9)
    
    plt.tight_layout()
    return fig

def create_log_space_detailed_analysis(summary_data):
    """Create detailed log-space analysis with polar plot"""
    
    results = [r for r in summary_data['detailed_results'] if r['success']]
    angles = [r['angle'] for r in results]
    log_corrs = [r['log_space_correlation'] for r in results]
    improvements = [r['improvement_percent'] for r in results]
    
    # Create figure with polar and cartesian plots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Detailed Log-Space Analysis: Angular Performance Distribution\n'
                 f'Advanced NMF Transfer Function Estimation', 
                 fontsize=16, fontweight='bold')
    
    # Create polar plot
    ax1 = plt.subplot(221, projection='polar')
    
    # Convert angles to radians for polar plot
    angles_rad = np.radians(angles)
    
    # Create polar bar plot
    bars = ax1.bar(angles_rad, log_corrs, width=np.radians(5), 
                   alpha=0.8, color=plt.cm.viridis(np.array(log_corrs)/max(log_corrs)))
    
    ax1.set_title('Polar Performance Map\n(Correlation by Direction)', pad=20)
    ax1.set_ylim(0, 1)
    ax1.set_theta_zero_location('N')  # 0° at top
    ax1.set_theta_direction(-1)  # Clockwise
    
    # Add angle labels
    for angle, angle_rad, corr in zip(angles, angles_rad, log_corrs):
        ax1.text(angle_rad, corr + 0.05, f'{angle}°', ha='center', va='center', fontsize=8)
    
    # Correlation evolution plot
    ax2 = plt.subplot(222)
    ax2.plot(angles, log_corrs, 'bo-', linewidth=3, markersize=10, alpha=0.8)
    ax2.fill_between(angles, log_corrs, alpha=0.3, color='blue')
    
    # Add performance zones
    ax2.axhspan(0.8, 1.0, alpha=0.2, color='green', label='Excellent (>0.8)')
    ax2.axhspan(0.6, 0.8, alpha=0.2, color='yellow', label='Good (0.6-0.8)')
    ax2.axhspan(0.4, 0.6, alpha=0.2, color='orange', label='Fair (0.4-0.6)')
    ax2.axhspan(0.0, 0.4, alpha=0.2, color='red', label='Poor (<0.4)')
    
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Log-Space Correlation')
    ax2.set_title('Correlation Evolution Across Angles')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_ylim(0, 1)
    
    # Improvement distribution
    ax3 = plt.subplot(223)
    
    # Create histogram of improvements
    n_bins = min(5, len(improvements))
    counts, bin_edges, patches = ax3.hist(improvements, bins=n_bins, alpha=0.8, color='green', edgecolor='black')
    
    # Color bars based on improvement level
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        if left_edge > 300:
            patch.set_facecolor('darkgreen')
        elif left_edge > 200:
            patch.set_facecolor('green')
        else:
            patch.set_facecolor('lightgreen')
    
    ax3.set_xlabel('Improvement (%)')
    ax3.set_ylabel('Number of Angles')
    ax3.set_title('Distribution of Performance Improvements')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    mean_imp = np.mean(improvements)
    std_imp = np.std(improvements)
    ax3.axvline(mean_imp, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_imp:.1f}%')
    ax3.axvline(mean_imp + std_imp, color='red', linestyle=':', alpha=0.7,
               label=f'+1σ: {mean_imp+std_imp:.1f}%')
    ax3.axvline(mean_imp - std_imp, color='red', linestyle=':', alpha=0.7,
               label=f'-1σ: {mean_imp-std_imp:.1f}%')
    ax3.legend()
    
    # Performance ranking
    ax4 = plt.subplot(224)
    
    # Sort by correlation for ranking
    sorted_indices = np.argsort(log_corrs)[::-1]  # Descending order
    sorted_angles = [angles[i] for i in sorted_indices]
    sorted_corrs = [log_corrs[i] for i in sorted_indices]
    
    bars = ax4.barh(range(len(sorted_angles)), sorted_corrs, 
                   color=plt.cm.RdYlGn([0.3 + 0.7 * c / max(log_corrs) for c in sorted_corrs]),
                   alpha=0.8)
    
    ax4.set_yticks(range(len(sorted_angles)))
    ax4.set_yticklabels([f'{a}°' for a in sorted_angles])
    ax4.set_xlabel('Log-Space Correlation')
    ax4.set_title('Performance Ranking (Best to Worst)')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.set_xlim(0, 1)
    
    # Add correlation values
    for bar, corr in zip(bars, sorted_corrs):
        width = bar.get_width()
        ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2,
               f'{corr:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize log-space only results')
    parser.add_argument('--summary', default='angle_test_summary.json',
                       help='JSON summary file from batch testing')
    parser.add_argument('--output-overview', default='log_space_performance_overview.png',
                       help='Output filename for performance overview')
    parser.add_argument('--output-detailed', default='log_space_detailed_analysis.png', 
                       help='Output filename for detailed analysis')
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
        print(f"   Log-space mean correlation: {stats['log_space_method']['mean_correlation']:.3f}")
        print(f"   Best angle: {stats['improvement_analysis']['best_angle']}°")
        
        # Create overview visualization
        print("🎨 Creating log-space performance overview...")
        fig1 = create_log_space_performance_overview(summary_data)
        fig1.savefig(args.output_overview, dpi=300, bbox_inches='tight')
        print(f"✅ Overview plot saved: {args.output_overview}")
        
        # Create detailed analysis
        print("🎨 Creating detailed log-space analysis...")
        fig2 = create_log_space_detailed_analysis(summary_data)
        fig2.savefig(args.output_detailed, dpi=300, bbox_inches='tight')
        print(f"✅ Detailed plot saved: {args.output_detailed}")
        
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