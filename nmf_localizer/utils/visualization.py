"""
Visualization utilities for NMF localization.
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class Visualizer:
    """Visualization utilities for NMF localization analysis."""
    
    @staticmethod
    def plot_transfer_functions(
        H: torch.Tensor,
        angles: torch.Tensor,
        freqs: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (12, 10),
        title: str = "Transfer Function Magnitudes"
    ):
        """
        Visualize transfer functions as heatmap and polar plot.
        
        Args:
            H: Transfer function matrix (F x D)
            angles: Angle array (D,)
            freqs: Frequency array (optional)
            save_path: Path to save figure
            figsize: Figure size
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Convert to numpy for plotting
        H_np = H.cpu().numpy() if torch.is_tensor(H) else H
        angles_np = angles.cpu().numpy() if torch.is_tensor(angles) else angles
        
        # Heatmap
        im = ax1.imshow(H_np, aspect='auto', origin='lower', cmap='viridis')
        ax1.set_xlabel('Direction Index')
        ax1.set_ylabel('Frequency Bin')
        ax1.set_title(title)
        
        # Add colorbar
        plt.colorbar(im, ax=ax1, label='Magnitude')
        
        # Set angle labels if available
        if len(angles_np) <= 20:  # Only show labels if not too many
            tick_positions = np.arange(len(angles_np))
            ax1.set_xticks(tick_positions)
            ax1.set_xticklabels([f"{int(a)}°" for a in angles_np], rotation=45)
        
        # Polar plot for selected frequencies
        n_freq_samples = min(10, H_np.shape[0])
        freq_indices = np.linspace(0, H_np.shape[0]-1, n_freq_samples, dtype=int)
        
        ax2 = plt.subplot(2, 1, 2, projection='polar')
        
        for i, freq_idx in enumerate(freq_indices):
            response = H_np[freq_idx, :]
            # Convert angles to radians
            theta = angles_np * np.pi / 180
            
            # Close the loop for smooth plotting
            theta_closed = np.append(theta, theta[0])
            response_closed = np.append(response, response[0])
            
            color = plt.cm.viridis(i / n_freq_samples)
            ax2.plot(theta_closed, response_closed, color=color, alpha=0.7, 
                    label=f'Freq bin {freq_idx}')
        
        ax2.set_title('Directional Response at Different Frequencies')
        if n_freq_samples <= 5:  # Only show legend if not too crowded
            ax2.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Transfer functions plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_usm_dictionary(
        W: torch.Tensor,
        freqs: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (15, 8),
        max_atoms: int = 20
    ):
        """
        Visualize USM dictionary atoms.
        
        Args:
            W: USM dictionary matrix (F x K)
            freqs: Frequency array (optional)
            save_path: Path to save figure
            figsize: Figure size
            max_atoms: Maximum number of atoms to display
        """
        W_np = W.cpu().numpy() if torch.is_tensor(W) else W
        n_freq, n_atoms = W_np.shape
        
        # Limit number of atoms for visualization
        display_atoms = min(max_atoms, n_atoms)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Heatmap of all atoms
        im1 = ax1.imshow(W_np[:, :display_atoms], aspect='auto', origin='lower', cmap='viridis')
        ax1.set_xlabel('Atom Index')
        ax1.set_ylabel('Frequency Bin')
        ax1.set_title(f'USM Dictionary Atoms (showing first {display_atoms}/{n_atoms})')
        plt.colorbar(im1, ax=ax1, label='Magnitude')
        
        # Line plot of selected atoms
        sample_atoms = np.linspace(0, display_atoms-1, min(10, display_atoms), dtype=int)
        
        x_axis = freqs if freqs is not None else np.arange(n_freq)
        
        for i, atom_idx in enumerate(sample_atoms):
            color = plt.cm.tab10(i % 10)
            ax2.plot(x_axis, W_np[:, atom_idx], color=color, 
                    label=f'Atom {atom_idx}', alpha=0.7)
        
        ax2.set_xlabel('Frequency (Hz)' if freqs is not None else 'Frequency Bin')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Sample Dictionary Atoms')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"USM dictionary plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_localization_results(
        group_norms: torch.Tensor,
        angles: torch.Tensor,
        true_directions: Optional[List[int]] = None,
        estimated_directions: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (12, 8)
    ):
        """
        Visualize localization results.
        
        Args:
            group_norms: Group norms for each direction (D,)
            angles: Angle array (D,)
            true_directions: True source directions
            estimated_directions: Estimated source directions
            save_path: Path to save figure
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Convert to numpy
        group_norms_np = group_norms.cpu().numpy() if torch.is_tensor(group_norms) else group_norms
        angles_np = angles.cpu().numpy() if torch.is_tensor(angles) else angles
        
        # Bar plot of group norms
        ax1.bar(range(len(group_norms_np)), group_norms_np, alpha=0.7)
        ax1.set_xlabel('Direction Index')
        ax1.set_ylabel('Group Norm')
        ax1.set_title('NMF Group Norms by Direction')
        ax1.grid(True, alpha=0.3)
        
        # Mark estimated directions
        if estimated_directions is not None:
            est_dirs_np = estimated_directions.cpu().numpy() if torch.is_tensor(estimated_directions) else estimated_directions
            for est_dir in est_dirs_np:
                # Find closest angle index
                closest_idx = np.argmin(np.abs(angles_np - est_dir))
                ax1.axvline(closest_idx, color='red', linestyle='--', alpha=0.8, 
                           label=f'Estimated: {est_dir:.0f}°')
        
        # Mark true directions
        if true_directions is not None:
            for true_dir in true_directions:
                if isinstance(true_dir, (int, float)):
                    # Convert to angle if it's an index
                    if true_dir < len(angles_np):
                        true_angle = angles_np[true_dir]
                        closest_idx = true_dir
                    else:
                        true_angle = true_dir
                        closest_idx = np.argmin(np.abs(angles_np - true_angle))
                else:
                    true_angle = true_dir
                    closest_idx = np.argmin(np.abs(angles_np - true_angle))
                    
                ax1.axvline(closest_idx, color='green', linestyle='-', alpha=0.8,
                           label=f'True: {true_angle:.0f}°')
        
        ax1.legend()
        
        # Polar plot
        ax2 = plt.subplot(1, 2, 2, projection='polar')
        
        # Plot group norms in polar coordinates
        theta = angles_np * np.pi / 180
        theta_closed = np.append(theta, theta[0])
        norms_closed = np.append(group_norms_np, group_norms_np[0])
        
        ax2.plot(theta_closed, norms_closed, 'b-', alpha=0.7, linewidth=2)
        ax2.fill(theta_closed, norms_closed, alpha=0.2)
        
        # Mark estimated directions
        if estimated_directions is not None:
            for est_dir in est_dirs_np:
                est_theta = est_dir * np.pi / 180
                # Find corresponding norm
                closest_idx = np.argmin(np.abs(angles_np - est_dir))
                est_norm = group_norms_np[closest_idx]
                ax2.scatter(est_theta, est_norm, color='red', s=100, marker='*',
                           label=f'Est: {est_dir:.0f}°', zorder=5)
        
        # Mark true directions
        if true_directions is not None:
            for true_dir in true_directions:
                if isinstance(true_dir, (int, float)) and true_dir < len(angles_np):
                    true_angle = angles_np[true_dir]
                    true_norm = group_norms_np[true_dir]
                else:
                    true_angle = true_dir
                    closest_idx = np.argmin(np.abs(angles_np - true_angle))
                    true_norm = group_norms_np[closest_idx]
                    
                true_theta = true_angle * np.pi / 180
                ax2.scatter(true_theta, true_norm, color='green', s=100, marker='o',
                           label=f'True: {true_angle:.0f}°', zorder=5)
        
        ax2.set_title('Polar View of Group Norms')
        if estimated_directions is not None or true_directions is not None:
            ax2.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Localization results plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_parameter_sweep_results(
        comparison_results: Dict[str, Any],
        parameter_name: str,
        metric: str = 'accuracy',
        save_path: Optional[str] = None,
        figsize: tuple = (10, 6)
    ):
        """
        Plot parameter sweep results.
        
        Args:
            comparison_results: Results from ExperimentRunner.compare_results()
            parameter_name: Name of parameter that was swept
            metric: Metric to plot ('accuracy', 'mean_error', 'processing_time')
            save_path: Path to save figure
            figsize: Figure size
        """
        if 'parameter_analysis' not in comparison_results:
            raise ValueError("No parameter analysis found in comparison results")
        
        param_data = comparison_results['parameter_analysis'].get(parameter_name)
        if not param_data:
            raise ValueError(f"Parameter '{parameter_name}' not found in analysis")
        
        # Extract data
        param_values = []
        metric_values = []
        error_bars = []
        
        for value, stats in param_data.items():
            try:
                param_values.append(float(value))
            except:
                param_values.append(value)
                
            if metric == 'accuracy':
                metric_values.append(stats.get('mean_accuracy', 0))
                error_bars.append(stats.get('std_accuracy', 0))
            elif metric == 'mean_error':
                metric_values.append(stats.get('mean_error', float('inf')))
                error_bars.append(0)  # No error bars for error metric
            elif metric == 'processing_time':
                metric_values.append(stats.get('mean_processing_time', 0) * 1000)  # Convert to ms
                error_bars.append(0)
        
        # Sort by parameter values if numeric
        if all(isinstance(v, (int, float)) for v in param_values):
            sorted_data = sorted(zip(param_values, metric_values, error_bars))
            param_values, metric_values, error_bars = zip(*sorted_data)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        if all(isinstance(v, (int, float)) for v in param_values):
            # Line plot for numeric parameters
            if metric == 'accuracy' and any(e > 0 for e in error_bars):
                ax.errorbar(param_values, metric_values, yerr=error_bars, 
                           marker='o', capsize=5, capthick=2, linewidth=2)
            else:
                ax.plot(param_values, metric_values, 'o-', linewidth=2, markersize=8)
        else:
            # Bar plot for categorical parameters
            bars = ax.bar(range(len(param_values)), metric_values, alpha=0.7)
            ax.set_xticks(range(len(param_values)))
            ax.set_xticklabels(param_values, rotation=45)
            
            # Add error bars for accuracy
            if metric == 'accuracy' and any(e > 0 for e in error_bars):
                ax.errorbar(range(len(param_values)), metric_values, yerr=error_bars,
                           fmt='none', ecolor='black', capsize=5)
        
        ax.set_xlabel(parameter_name)
        
        if metric == 'accuracy':
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'Accuracy vs {parameter_name}')
        elif metric == 'mean_error':
            ax.set_ylabel('Mean Error (degrees)')
            ax.set_title(f'Mean Error vs {parameter_name}')
        elif metric == 'processing_time':
            ax.set_ylabel('Processing Time (ms)')
            ax.set_title(f'Processing Time vs {parameter_name}')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Parameter sweep plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def create_experiment_dashboard(
        results: Dict[str, Any],
        output_dir: str,
        experiment_name: str = "NMF_Experiment"
    ):
        """
        Create comprehensive dashboard for experiment results.
        
        Args:
            results: Experiment results
            output_dir: Output directory for plots
            experiment_name: Name of experiment
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating experiment dashboard in: {output_path}")
        
        # Create multiple visualizations based on available data
        dashboard_files = []
        
        try:
            # Plot transfer functions if available
            if 'data_pack' in results and hasattr(results['data_pack'], 'transfer_functions'):
                tf_plot_path = output_path / f"{experiment_name}_transfer_functions.png"
                Visualizer.plot_transfer_functions(
                    results['data_pack'].transfer_functions,
                    results['data_pack'].angles,
                    save_path=str(tf_plot_path),
                    title=f"{experiment_name}: Transfer Functions"
                )
                dashboard_files.append(tf_plot_path)
        except Exception as e:
            logger.warning(f"Failed to create transfer function plot: {e}")
        
        # Create dashboard summary
        summary_path = output_path / f"{experiment_name}_dashboard_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"NMF LOCALIZATION EXPERIMENT DASHBOARD\n")
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Generated: {np.datetime64('now')}\n\n")
            
            if dashboard_files:
                f.write("Generated Plots:\n")
                for file_path in dashboard_files:
                    f.write(f"- {file_path.name}\n")
            else:
                f.write("No plots generated (insufficient data)\n")
        
        logger.info(f"Dashboard created with {len(dashboard_files)} plots")