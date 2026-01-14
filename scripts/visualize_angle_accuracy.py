#!/usr/bin/env python3
"""
Visualize per-angle accuracy results from NMF DOA localization experiments.

This script creates comprehensive visualizations comparing the breakthrough 
300-3000 Hz frequency range results with baseline performance, highlighting
the dramatic improvement and the 90° training angle anomaly.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
import json

def load_evaluation_results(results_path):
    """Load evaluation results from .pth file"""
    results = torch.load(results_path, map_location='cpu', weights_only=False)
    return results

def create_angle_accuracy_plot(results_data, output_dir, title_suffix=""):
    """Create bar plot showing per-angle accuracy"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract per-angle data
    per_angle = results_data['evaluation_results']['per_angle_analysis']
    angles = sorted(per_angle.keys())
    accuracies = [per_angle[angle]['accuracy'] * 100 for angle in angles]  # Convert to percentage
    
    # Color bars - highlight 90° in red, others in blue
    colors = ['red' if angle == 90 else 'steelblue' for angle in angles]
    
    # Create bar plot
    bars = ax.bar([f"{angle}°" for angle in angles], accuracies, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Formatting
    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Per-Angle DOA Accuracy{title_suffix}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add overall accuracy as horizontal line
    overall_acc = results_data['evaluation_results']['accuracy'] * 100
    ax.axhline(y=overall_acc, color='orange', linestyle='--', alpha=0.8, 
               label=f'Overall Accuracy: {overall_acc:.1f}%')
    
    # Add legend explaining colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.7, label='Test Angles'),
        Patch(facecolor='red', alpha=0.7, label='Training Angle (90°)'),
        plt.Line2D([0], [0], color='orange', linestyle='--', alpha=0.8, 
                   label=f'Overall: {overall_acc:.1f}%')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / f'angle_accuracy{title_suffix.lower().replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved angle accuracy plot: {output_path}")
    
    return fig

def create_comparison_plot(breakthrough_results, baseline_results, output_dir):
    """Create side-by-side comparison of breakthrough vs baseline results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Extract data for both experiments
    breakthrough_per_angle = breakthrough_results['evaluation_results']['per_angle_analysis']
    baseline_per_angle = baseline_results['evaluation_results']['per_angle_analysis']
    
    # Get common angles (baseline may have different angle coverage)
    breakthrough_angles = set(breakthrough_per_angle.keys())
    baseline_angles = set(baseline_per_angle.keys())
    common_angles = sorted(breakthrough_angles.intersection(baseline_angles))
    
    # Plot breakthrough results
    breakthrough_accs = [breakthrough_per_angle[angle]['accuracy'] * 100 for angle in common_angles]
    colors_breakthrough = ['red' if angle == 90 else 'steelblue' for angle in common_angles]
    
    bars1 = ax1.bar([f"{angle}°" for angle in common_angles], breakthrough_accs, 
                    color=colors_breakthrough, alpha=0.7)
    
    # Add value labels
    for bar, acc in zip(bars1, breakthrough_accs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax1.set_title('Breakthrough: 300-3000 Hz\n94.1% Overall Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Angle (degrees)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_ylim(0, 110)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot baseline results  
    baseline_accs = [baseline_per_angle[angle]['accuracy'] * 100 for angle in common_angles]
    colors_baseline = ['red' if angle == 90 else 'lightcoral' for angle in common_angles]
    
    bars2 = ax2.bar([f"{angle}°" for angle in common_angles], baseline_accs, 
                    color=colors_baseline, alpha=0.7)
    
    # Add value labels
    for bar, acc in zip(bars2, baseline_accs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax2.set_title('Baseline: 500-1500 Hz\n33.3% Overall Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Angle (degrees)', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-axis labels
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save comparison plot
    output_path = output_dir / 'frequency_range_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {output_path}")
    
    return fig

def create_improvement_analysis(breakthrough_results, baseline_results, output_dir):
    """Create analysis showing improvement per angle"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract common angles and calculate improvements
    breakthrough_per_angle = breakthrough_results['evaluation_results']['per_angle_analysis']
    baseline_per_angle = baseline_results['evaluation_results']['per_angle_analysis']
    
    breakthrough_angles = set(breakthrough_per_angle.keys())
    baseline_angles = set(baseline_per_angle.keys())
    common_angles = sorted(breakthrough_angles.intersection(baseline_angles))
    
    # Calculate absolute improvement (percentage points)
    improvements = []
    for angle in common_angles:
        breakthrough_acc = breakthrough_per_angle[angle]['accuracy'] * 100
        baseline_acc = baseline_per_angle[angle]['accuracy'] * 100
        improvement = breakthrough_acc - baseline_acc
        improvements.append(improvement)
    
    # Color bars based on improvement magnitude
    colors = []
    for improvement in improvements:
        if improvement >= 80:
            colors.append('darkgreen')  # Massive improvement
        elif improvement >= 50:
            colors.append('green')      # Large improvement
        elif improvement >= 20:
            colors.append('orange')     # Moderate improvement
        else:
            colors.append('red')        # Small/negative improvement
    
    # Create bar plot
    bars = ax.bar([f"{angle}°" for angle in common_angles], improvements, 
                  color=colors, alpha=0.7)
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        y_pos = height + 2 if height >= 0 else height - 5
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'+{imp:.1f}%' if imp >= 0 else f'{imp:.1f}%', 
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    # Formatting
    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Accuracy Improvement (percentage points)', fontsize=12)
    ax.set_title('Per-Angle Improvement: 300-3000 Hz vs 500-1500 Hz', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkgreen', alpha=0.7, label='Massive Improvement (≥80%)'),
        Patch(facecolor='green', alpha=0.7, label='Large Improvement (≥50%)'),
        Patch(facecolor='orange', alpha=0.7, label='Moderate Improvement (≥20%)'),
        Patch(facecolor='red', alpha=0.7, label='Small Improvement (<20%)')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save improvement analysis
    output_path = output_dir / 'angle_improvement_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved improvement analysis: {output_path}")
    
    return fig

def create_summary_statistics_plot(breakthrough_results, baseline_results, output_dir):
    """Create summary statistics comparison"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Overall accuracy comparison
    breakthrough_acc = breakthrough_results['evaluation_results']['accuracy'] * 100
    baseline_acc = baseline_results['evaluation_results']['accuracy'] * 100
    
    ax1.bar(['Baseline\n(500-1500 Hz)', 'Breakthrough\n(300-3000 Hz)'], 
            [baseline_acc, breakthrough_acc], 
            color=['lightcoral', 'steelblue'], alpha=0.7)
    ax1.set_ylabel('Overall Accuracy (%)')
    ax1.set_title('Overall Accuracy Comparison')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add improvement annotation
    improvement = breakthrough_acc - baseline_acc
    ax1.annotate(f'+{improvement:.1f}%\nImprovement', 
                xy=(1, breakthrough_acc), xytext=(0.5, breakthrough_acc + 10),
                ha='center', fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # Success rate comparison
    breakthrough_success = breakthrough_results['evaluation_results']['success_rate'] * 100
    baseline_success = baseline_results['evaluation_results']['success_rate'] * 100
    
    ax2.bar(['Baseline', 'Breakthrough'], 
            [baseline_success, breakthrough_success],
            color=['lightcoral', 'steelblue'], alpha=0.7)
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Processing Success Rate')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Processing time comparison
    breakthrough_time = breakthrough_results['evaluation_results']['mean_processing_time']
    baseline_time = baseline_results['evaluation_results']['mean_processing_time']
    
    ax3.bar(['Baseline', 'Breakthrough'], 
            [baseline_time * 1000, breakthrough_time * 1000],  # Convert to ms
            color=['lightcoral', 'steelblue'], alpha=0.7)
    ax3.set_ylabel('Mean Processing Time (ms)')
    ax3.set_title('Processing Time Comparison')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Number of perfect angles (100% accuracy)
    breakthrough_perfect = sum(1 for angle_data in breakthrough_results['evaluation_results']['per_angle_analysis'].values() 
                              if angle_data['accuracy'] == 1.0)
    baseline_perfect = sum(1 for angle_data in baseline_results['evaluation_results']['per_angle_analysis'].values() 
                          if angle_data['accuracy'] == 1.0)
    
    ax4.bar(['Baseline', 'Breakthrough'], 
            [baseline_perfect, breakthrough_perfect],
            color=['lightcoral', 'steelblue'], alpha=0.7)
    ax4.set_ylabel('Number of Perfect Angles (100%)')
    ax4.set_title('Perfect Angle Performance')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save summary plot
    output_path = output_dir / 'summary_statistics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary statistics: {output_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="Visualize NMF DOA angle accuracy results")
    parser.add_argument('--breakthrough-results', required=True, type=str,
                       help='Path to breakthrough experiment results (.pth file)')
    parser.add_argument('--baseline-results', type=str, default=None,
                       help='Path to baseline experiment results (.pth file) for comparison')
    parser.add_argument('--output-dir', required=True, type=str,
                       help='Output directory for visualization plots')
    
    args = parser.parse_args()
    
    # Setup paths
    breakthrough_path = Path(args.breakthrough_results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load breakthrough results
    print(f"Loading breakthrough results from: {breakthrough_path}")
    breakthrough_results = load_evaluation_results(breakthrough_path)
    
    # Create breakthrough results visualization
    print("Creating breakthrough results visualization...")
    create_angle_accuracy_plot(breakthrough_results, output_dir, 
                              title_suffix=" - 300-3000 Hz Breakthrough")
    
    # If baseline results provided, create comparison
    if args.baseline_results:
        baseline_path = Path(args.baseline_results)
        print(f"Loading baseline results from: {baseline_path}")
        baseline_results = load_evaluation_results(baseline_path)
        
        print("Creating comparison visualizations...")
        create_comparison_plot(breakthrough_results, baseline_results, output_dir)
        create_improvement_analysis(breakthrough_results, baseline_results, output_dir)
        create_summary_statistics_plot(breakthrough_results, baseline_results, output_dir)
    
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated plots:")
    for plot_file in sorted(output_dir.glob("*.png")):
        print(f"  - {plot_file.name}")

if __name__ == '__main__':
    main()