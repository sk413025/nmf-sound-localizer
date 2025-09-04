#!/usr/bin/env python3
"""
Debug IS divergence calculation for Y vs Y_hat reconstruction quality issue.
Analyze the actual numerical values and identify problem sources.
"""

import numpy as np
import torch
import os
import json
from pathlib import Path

# Import nmf_localizer modules
import sys
sys.path.append('/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/development-workspace')

from nmf_localizer.core.data_processor import DataProcessor
from nmf_localizer.core.localizer import NMFSoundLocalizer
from nmf_localizer.config import NMFConfig
from nmf_localizer.utils.audio_utils import AudioProcessor
import datetime


class IterationTracker:
    """Track NMF optimization iterations to analyze sparsification process."""
    
    def __init__(self):
        self.iterations = []
        self.iteration_count = 0
        
    def track_iteration(self, iter_num, A, X, loss_components, Y, Y_hat):
        """Record state of a single iteration."""
        # Convert tensors to numpy if needed
        A_np = A.detach().cpu().numpy() if torch.is_tensor(A) else A
        X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else X
        Y_np = Y.detach().cpu().numpy() if torch.is_tensor(Y) else Y
        Y_hat_np = Y_hat.detach().cpu().numpy() if torch.is_tensor(Y_hat) else Y_hat
        
        # Calculate sparsity metrics
        sparsity_stats = self.calculate_sparsity_metrics(X_np)
        
        # Calculate numerical stability metrics
        stability_stats = self.analyze_numerical_stability(A_np, X_np)
        
        # Calculate reconstruction quality
        reconstruction_stats = self.analyze_reconstruction_quality(Y_np, Y_hat_np)
        
        # Calculate loss breakdown
        loss_breakdown = self.calculate_loss_breakdown(Y_np, Y_hat_np, X_np, loss_components)
        
        iteration_data = {
            'iteration': iter_num,
            'sparsity': sparsity_stats,
            'stability': stability_stats,
            'reconstruction': reconstruction_stats,
            'loss': loss_breakdown,
            'X_stats': {
                'mean': np.mean(X_np),
                'max': np.max(X_np),
                'min': np.min(X_np),
                'std': np.std(X_np)
            },
            'A_stats': {
                'mean': np.mean(A_np),
                'max': np.max(A_np),
                'min': np.min(A_np),
                'std': np.std(A_np)
            },
            # Store actual data for visualization
            'Y_data': Y_np.copy(),
            'Y_hat_data': Y_hat_np.copy(),
            'X_factors': X_np.copy()
        }
        
        self.iterations.append(iteration_data)
        print(f"  Iteration {iter_num:2d}: loss={loss_components:.2e}, X_sparsity={sparsity_stats['sparsity']:.3f}, "
              f"X_mean={np.mean(X_np):.2e}, Y_hat_mean={np.mean(Y_hat_np):.2e}")
        
    def calculate_sparsity_metrics(self, X):
        """Calculate detailed sparsity metrics."""
        total_elements = X.size
        zero_elements = np.sum(X < 1e-10)
        near_zero_elements = np.sum(X < 1e-6)
        
        sparsity = zero_elements / total_elements
        near_zero_ratio = near_zero_elements / total_elements
        
        # Distribution analysis
        non_zero_values = X[X > 1e-10]
        if len(non_zero_values) > 0:
            nz_mean = np.mean(non_zero_values)
            nz_std = np.std(non_zero_values)
            nz_max = np.max(non_zero_values)
            nz_min = np.min(non_zero_values)
        else:
            nz_mean = nz_std = nz_max = nz_min = 0.0
        
        return {
            'sparsity': sparsity,
            'near_zero_ratio': near_zero_ratio,
            'total_elements': total_elements,
            'zero_elements': zero_elements,
            'non_zero_stats': {
                'mean': nz_mean,
                'std': nz_std,
                'max': nz_max,
                'min': nz_min,
                'count': len(non_zero_values)
            }
        }
        
    def analyze_numerical_stability(self, A, X):
        """Analyze numerical stability indicators."""
        # Condition numbers
        A_cond = np.linalg.cond(A) if A.shape[0] <= A.shape[1] else np.linalg.cond(A.T)
        
        # Dynamic range (max/min ratios)
        A_dynamic_range = np.max(A) / (np.min(A[A > 0]) + 1e-12) if np.any(A > 0) else float('inf')
        X_dynamic_range = np.max(X) / (np.min(X[X > 0]) + 1e-12) if np.any(X > 0) else float('inf')
        
        return {
            'A_condition_number': A_cond,
            'A_dynamic_range': A_dynamic_range,
            'X_dynamic_range': X_dynamic_range,
            'A_rank': np.linalg.matrix_rank(A),
            'numerical_warnings': {
                'A_ill_conditioned': A_cond > 1e12,
                'extreme_dynamic_range': max(A_dynamic_range, X_dynamic_range) > 1e10
            }
        }
        
    def analyze_reconstruction_quality(self, Y, Y_hat):
        """Analyze reconstruction quality metrics."""
        mse = np.mean((Y - Y_hat) ** 2)
        
        # Scale matching
        y_mean = np.mean(Y)
        y_hat_mean = np.mean(Y_hat)
        scale_ratio = y_hat_mean / (y_mean + 1e-12)
        
        # Correlation
        Y_flat = Y.flatten()
        Y_hat_flat = Y_hat.flatten()
        
        if np.std(Y_flat) > 1e-10 and np.std(Y_hat_flat) > 1e-10:
            correlation = np.corrcoef(Y_flat, Y_hat_flat)[0, 1]
            correlation = correlation if not np.isnan(correlation) else 0.0
        else:
            correlation = 0.0
            
        return {
            'mse': mse,
            'scale_ratio': scale_ratio,
            'correlation': correlation,
            'y_mean': y_mean,
            'y_hat_mean': y_hat_mean
        }
        
    def calculate_loss_breakdown(self, Y, Y_hat, X, total_loss):
        """Break down loss components."""
        epsilon = 1e-12
        Y_safe = np.maximum(Y, epsilon)
        Y_hat_safe = np.maximum(Y_hat, epsilon)
        
        # Calculate data fitting term based on beta (this should match what the tracker is initialized with)
        # For β=2 (Euclidean): ||Y - Y_hat||_F^2
        # For β=0 (IS): sum(Y/Y_hat - log(Y/Y_hat) - 1)
        # For β=1 (KL): sum(Y*log(Y/Y_hat) - Y + Y_hat)
        
        # We'll calculate Euclidean since that's what the test uses
        euclidean_loss = np.sum((Y_safe - Y_hat_safe) ** 2)
        
        # Also calculate IS divergence for comparison
        ratio = Y_safe / Y_hat_safe
        is_divergence = np.sum(ratio - np.log(ratio) - 1)
        
        return {
            'total_loss': total_loss,
            'euclidean_loss': euclidean_loss,
            'is_divergence': is_divergence,
            'data_fit_ratio': euclidean_loss / (total_loss + 1e-12) if total_loss > 0 else 0.0,
            'is_divergence_value': is_divergence  # For comparison
        }
        
    def get_sparsification_timeline(self):
        """Analyze when and how sparsification occurs."""
        if not self.iterations:
            return {}
            
        sparsities = [iter_data['sparsity']['sparsity'] for iter_data in self.iterations]
        x_means = [iter_data['X_stats']['mean'] for iter_data in self.iterations]
        losses = [iter_data['loss']['total_loss'] for iter_data in self.iterations]
        
        # Find key transition points
        sparsity_jumps = []
        for i in range(1, len(sparsities)):
            sparsity_change = sparsities[i] - sparsities[i-1]
            if sparsity_change > 0.1:  # Significant sparsity increase
                sparsity_jumps.append({
                    'iteration': i,
                    'sparsity_increase': sparsity_change,
                    'from': sparsities[i-1],
                    'to': sparsities[i]
                })
        
        return {
            'initial_sparsity': sparsities[0] if sparsities else 0.0,
            'final_sparsity': sparsities[-1] if sparsities else 0.0,
            'max_sparsity_change': max([abs(sparsities[i] - sparsities[i-1]) 
                                       for i in range(1, len(sparsities))]) if len(sparsities) > 1 else 0.0,
            'sparsity_jumps': sparsity_jumps,
            'sparsity_timeline': sparsities,
            'x_means_timeline': x_means,
            'loss_timeline': losses
        }
        
    def print_summary(self):
        """Print analysis summary."""
        if not self.iterations:
            print("No iteration data recorded.")
            return
            
        timeline = self.get_sparsification_timeline()
        
        print(f"\n{'='*60}")
        print("ITERATION TRACKING SUMMARY")
        print(f"{'='*60}")
        print(f"Total iterations tracked: {len(self.iterations)}")
        print(f"Initial sparsity: {timeline['initial_sparsity']:.3f}")
        print(f"Final sparsity: {timeline['final_sparsity']:.3f}")
        print(f"Max sparsity change between iterations: {timeline['max_sparsity_change']:.3f}")
        
        if timeline['sparsity_jumps']:
            print(f"\nSignificant sparsity increases detected:")
            for jump in timeline['sparsity_jumps']:
                print(f"  Iteration {jump['iteration']}: {jump['from']:.3f} → {jump['to']:.3f} "
                      f"(+{jump['sparsity_increase']:.3f})")
        else:
            print("\nNo significant sparsity jumps detected.")
            
        # Show first and last few iterations
        print(f"\nFirst 3 iterations:")
        for i in range(min(3, len(self.iterations))):
            iter_data = self.iterations[i]
            print(f"  Iter {i}: sparsity={iter_data['sparsity']['sparsity']:.3f}, "
                  f"X_mean={iter_data['X_stats']['mean']:.2e}, "
                  f"loss={iter_data['loss']['total_loss']:.2e}")
        
        if len(self.iterations) > 6:
            print(f"\nLast 3 iterations:")
            for i in range(max(0, len(self.iterations)-3), len(self.iterations)):
                iter_data = self.iterations[i]
                print(f"  Iter {i}: sparsity={iter_data['sparsity']['sparsity']:.3f}, "
                      f"X_mean={iter_data['X_stats']['mean']:.2e}, "
                      f"loss={iter_data['loss']['total_loss']:.2e}")


class TrackedNMFLocalizer(NMFSoundLocalizer):
    """NMFSoundLocalizer with iteration tracking capability."""
    
    def __init__(self, config: NMFConfig, tracker: IterationTracker):
        super().__init__(config)
        self.tracker = tracker
        
    def factorize(self, Y: torch.Tensor):
        """Override factorize to add tracking."""
        if self.A is None:
            raise ValueError("Must load source dictionary and transfer functions first")
        
        F, N = Y.shape
        if F != self.n_freq:
            raise ValueError(f"Expected {self.n_freq} frequency bins, got {F}")
        
        print(f"Starting TRACKED NMF factorization with iteration monitoring...")
        print(f"Input Y shape: {Y.shape}")
        print(f"A shape: {self.A.shape}")

        # Apply frequency weights to Y consistently with A
        if self.freq_weights is not None:
            Y = self.freq_weights.view(-1, 1) * Y
        
        # Initialize X (same as parent class)
        Y_mean = Y.mean()
        X = torch.zeros(self.A.shape[1], N, device=self.device)
        
        # Initialize each group independently with different random seeds
        for d in range(self.n_directions):
            start_idx = d * self.n_components
            end_idx = (d + 1) * self.n_components
            
            # Each group gets different initialization strength
            group_strength = torch.rand(1, device=self.device) * 0.5 + 0.1  # 0.1 to 0.6
            X[start_idx:end_idx, :] = torch.rand(
                self.n_components, N, device=self.device
            ) * Y_mean * group_strength
        
        X = torch.clamp(X, min=self.epsilon)
        
        # Better initialization check
        if X.mean() < 1e-6:
            Y_mean = Y.mean()
            X = torch.rand(self.A.shape[1], N, device=self.device) * Y_mean * 0.5
            X = torch.clamp(X, min=self.epsilon)
        
        print(f"X initialized: shape={X.shape}, min={X.min():.6f}, max={X.max():.6f}, mean={X.mean():.6f}")
        print(f"Tracking iterations:")
        
        losses = []
        
        for iteration in range(self.max_iter):
            # Update X
            X = self._multiplicative_update(Y, X)
            
            # Compute loss
            Y_hat = self.A @ X
            data_fit = self._beta_divergence(Y, Y_hat)
            
            # Group sparsity penalty: count number of active groups
            active_groups = 0
            group_norms = []
            for d in range(self.n_directions):
                start_idx = d * self.n_components
                end_idx = (d + 1) * self.n_components
                X_d = X[start_idx:end_idx, :]
                group_norm = torch.sum(torch.abs(X_d))
                group_norms.append(group_norm)
                if group_norm > 0.1:  # Threshold for "active" group
                    active_groups += 1
                    
            # Sparsity penalty (l1)
            sparse_penalty = torch.sum(torch.abs(X))
            
            total_loss = data_fit + self.lambda_group * active_groups + \
                        self.gamma_sparse * sparse_penalty
            losses.append(total_loss.item())
            
            # Track this iteration
            self.tracker.track_iteration(iteration, self.A, X, total_loss.item(), Y, Y_hat)
            
            # Check convergence
            if iteration > 0 and abs(losses[-1] - losses[-2]) < self.tol:
                print(f"NMF converged at iteration {iteration}")
                break
                
        info = {
            'final_loss': losses[-1] if losses else float('inf'),
            'n_iter': len(losses),
            'converged': len(losses) < self.max_iter,
            'losses': losses
        }
        
        print(f"Factorization complete: {info['n_iter']} iterations, "
               f"final loss: {info['final_loss']:.6f}")
        
        return X, info

def calculate_is_divergence_beta0(Y, Y_hat, epsilon=1e-12):
    """
    Calculate IS divergence with β=0 (Itakura-Saito)
    D_IS(Y||Y_hat) = Σ [Y/Y_hat - log(Y/Y_hat) - 1]
    """
    # Add small epsilon to avoid division by zero
    Y_hat_safe = np.maximum(Y_hat, epsilon)
    Y_safe = np.maximum(Y, epsilon)
    
    # Calculate ratio
    ratio = Y_safe / Y_hat_safe
    
    # IS divergence components
    ratio_term = ratio
    log_term = np.log(ratio)
    constant_term = 1
    
    # IS divergence per element
    divergence_per_element = ratio_term - log_term - constant_term
    
    # Total divergence
    total_divergence = np.sum(divergence_per_element)
    
    return {
        'total_divergence': total_divergence,
        'mean_divergence': np.mean(divergence_per_element),
        'max_divergence': np.max(divergence_per_element),
        'min_divergence': np.min(divergence_per_element),
        'ratio_stats': {
            'mean': np.mean(ratio),
            'max': np.max(ratio),
            'min': np.min(ratio),
            'std': np.std(ratio)
        },
        'Y_stats': {
            'mean': np.mean(Y_safe),
            'max': np.max(Y_safe),
            'min': np.min(Y_safe),
            'std': np.std(Y_safe),
            'shape': Y_safe.shape
        },
        'Y_hat_stats': {
            'mean': np.mean(Y_hat_safe),
            'max': np.max(Y_hat_safe),
            'min': np.min(Y_hat_safe),
            'std': np.std(Y_hat_safe),
            'shape': Y_hat_safe.shape
        }
    }


def calculate_is_divergence_matrix(Y, Y_hat):
    """Calculate IS divergence element-wise matrix for visualization."""
    # Add small epsilon to avoid division by zero
    epsilon = 1e-12
    Y_safe = Y + epsilon
    Y_hat_safe = Y_hat + epsilon
    
    # IS divergence per element: Y_safe/Y_hat_safe - log(Y_safe/Y_hat_safe) - 1
    ratio = Y_safe / Y_hat_safe
    is_div_matrix = ratio - np.log(ratio) - 1
    
    return is_div_matrix


def save_visualization_data(Y, Y_hat, is_div_matrix, tracker, scale_ratio, x_sparsity, 
                           lambda_group=None, gamma_sparse=None, output_file=None):
    """Save Y, Y_hat, and IS divergence data for visualization."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_file is None:
        output_file = f"nmf_reconstruction_data_{timestamp}.json"
    
    # Sample data points for visualization (reduce size for JSON)
    # Take every 10th point to keep file manageable but representative
    freq_step = max(1, Y.shape[0] // 100)  # Max 100 frequency points
    time_step = max(1, Y.shape[1] // 200)  # Max 200 time points
    
    Y_sampled = Y[::freq_step, ::time_step]
    Y_hat_sampled = Y_hat[::freq_step, ::time_step]
    is_div_sampled = is_div_matrix[::freq_step, ::time_step]
    
    # Flatten for easier plotting
    Y_flat = Y_sampled.flatten()
    Y_hat_flat = Y_hat_sampled.flatten()
    is_div_flat = is_div_sampled.flatten()
    
    # Get iteration progression data
    iteration_data = []
    if tracker and tracker.iterations:
        for i in range(0, len(tracker.iterations), max(1, len(tracker.iterations) // 20)):  # Max 20 points
            iter_info = tracker.iterations[i]
            iteration_data.append({
                'iteration': int(iter_info['iteration']),
                'loss': float(iter_info['loss']['total_loss']) if 'total_loss' in iter_info['loss'] else 0.0,
                'X_sparsity': float(iter_info['sparsity']['sparsity']),
                'X_mean': float(iter_info['X_stats']['mean']),
                'Y_hat_mean': float(np.mean(iter_info['Y_hat_data'])) if 'Y_hat_data' in iter_info else 0.0,
                'scale_ratio': float(iter_info['reconstruction']['scale_ratio']) if 'reconstruction' in iter_info else 1.0
            })
    
    visualization_data = {
        'metadata': {
            'timestamp': timestamp,
            'lambda_group': lambda_group,
            'gamma_sparse': gamma_sparse,
            'Y_shape': Y.shape,
            'Y_hat_shape': Y_hat.shape,
            'sampling_info': {
                'freq_step': freq_step,
                'time_step': time_step,
                'sampled_shape': Y_sampled.shape
            }
        },
        'reconstruction_quality': {
            'scale_ratio': float(scale_ratio),
            'x_sparsity': float(x_sparsity),
            'Y_stats': {
                'mean': float(np.mean(Y)),
                'std': float(np.std(Y)),
                'min': float(np.min(Y)),
                'max': float(np.max(Y))
            },
            'Y_hat_stats': {
                'mean': float(np.mean(Y_hat)),
                'std': float(np.std(Y_hat)),
                'min': float(np.min(Y_hat)),
                'max': float(np.max(Y_hat))
            },
            'is_divergence_stats': {
                'total': float(np.sum(is_div_matrix)),
                'mean': float(np.mean(is_div_matrix)),
                'std': float(np.std(is_div_matrix)),
                'min': float(np.min(is_div_matrix)),
                'max': float(np.max(is_div_matrix))
            }
        },
        'comparison_data': {
            'Y_values': [float(x) for x in Y_flat],
            'Y_hat_values': [float(x) for x in Y_hat_flat],
            'is_divergence_values': [float(x) for x in is_div_flat],
            'data_points': int(len(Y_flat))
        },
        'iteration_progression': iteration_data,
        'curve_comparison': {
            # For plotting Y vs Y_hat curves - take diagonal slice
            'diagonal_Y': [float(x) for x in np.diag(Y_sampled)],
            'diagonal_Y_hat': [float(x) for x in np.diag(Y_hat_sampled)],
            'diagonal_is_div': [float(x) for x in np.diag(is_div_sampled)],
            'diagonal_length': int(len(np.diag(Y_sampled)))
        }
    }
    
    print(f"Saving visualization data to: {output_file}")
    print(f"Data points for visualization: {len(Y_flat)}")
    print(f"Iteration progression points: {len(iteration_data)}")
    
    with open(output_file, 'w') as f:
        json.dump(visualization_data, f, indent=2)
    
    return output_file


def analyze_nmf_reconstruction():
    """
    Reproduce the NMF reconstruction and analyze IS divergence
    """
    # Create configuration for IS divergence optimization (β=0)
    config = NMFConfig(
        sample_rate=16000,
        n_fft=2048,
        hop_length=512,
        freq_min=500.0,
        freq_max=3000.0,
        n_files_per_angle=1,
        max_iter=50,
        beta=0,  # IS divergence - our target optimization
        lambda_group=0.1,  # Default value for grid search
        gamma_sparse=0.01,  # Default value for grid search
        tolerance=1e-6
    )
    
    # Set up data paths (using synchronized VAD data)
    x_root = "/Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original/white_noise_original_data_no_edge_sync_vad"
    y_root = "/Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original/white_noise_box_data_no_edge_sync_vad"
    
    # Test angle
    test_angle = "angle_90"
    
    # Load data
    x_dir = os.path.join(x_root, test_angle)
    y_dir = os.path.join(y_root, test_angle)
    
    print(f"Loading data from:")
    print(f"X: {x_dir}")
    print(f"Y: {y_dir}")
    
    # Get first available file
    x_files = [f for f in os.listdir(x_dir) if f.endswith('.npy')]
    y_files = [f for f in os.listdir(y_dir) if f.endswith('.npy')]
    
    if not x_files or not y_files:
        raise FileNotFoundError(f"No .npy files found in {x_dir} or {y_dir}")
    
    # Use first file for analysis
    x_file = x_files[0]
    y_file = y_files[0]
    
    print(f"Analyzing files: {x_file}, {y_file}")
    
    # Load audio data
    x_audio = np.load(os.path.join(x_dir, x_file))
    y_audio = np.load(os.path.join(y_dir, y_file))
    
    print(f"Audio shapes: X={x_audio.shape}, Y={y_audio.shape}")
    print(f"Audio stats: X mean={np.mean(x_audio):.2e}, Y mean={np.mean(y_audio):.2e}")
    
    # Initialize audio processor
    audio_processor = AudioProcessor()
    
    # Process to get magnitude spectrograms using STFT
    print("Processing X audio to spectrogram...")
    freqs_x, times_x, stft_x, magnitude_x = audio_processor.compute_stft_spectrogram(
        x_audio.astype(np.float32),
        fs=config.sample_rate,
        nperseg=config.n_fft,
        noverlap=config.n_fft - config.hop_length  # Convert hop_length to overlap
    )
    
    print("Processing Y audio to spectrogram...")  
    freqs_y, times_y, stft_y, magnitude_y = audio_processor.compute_stft_spectrogram(
        y_audio.astype(np.float32),
        fs=config.sample_rate,
        nperseg=config.n_fft,
        noverlap=config.n_fft - config.hop_length  # Convert hop_length to overlap
    )
    
    # Apply frequency filtering to match test configuration
    X_tensor = torch.from_numpy(magnitude_x).float()
    Y_tensor = torch.from_numpy(magnitude_y).float()
    
    X_spec, freqs_filtered_x = audio_processor.apply_frequency_filter(
        X_tensor, freqs_x, config.freq_min, config.freq_max
    )
    Y_spec, freqs_filtered_y = audio_processor.apply_frequency_filter(
        Y_tensor, freqs_y, config.freq_min, config.freq_max
    )
    
    print(f"Spectrogram shapes: X={X_spec.shape}, Y={Y_spec.shape}")
    print(f"Spectrogram stats: X mean={torch.mean(X_spec):.2e}, Y mean={torch.mean(Y_spec):.2e}")
    
    # Convert to numpy for simpler NMF processing
    X_np = X_spec.detach().cpu().numpy()
    Y_np = Y_spec.detach().cpu().numpy()
    
    print(f"NumPy arrays: X={X_np.shape}, Y={Y_np.shape}")
    print(f"NumPy stats: X mean={np.mean(X_np):.2e}, Y mean={np.mean(Y_np):.2e}")
    
    # Real NMF using NMFSoundLocalizer (same as test setup)
    print("\nPerforming REAL NMF using NMFSoundLocalizer...")
    
    # Initialize iteration tracker
    tracker = IterationTracker()
    
    # Initialize TRACKED NMF localizer
    localizer = TrackedNMFLocalizer(config, tracker)
    
    # Create source dictionary W (same as test)
    n_components = 15
    F = Y_np.shape[0]
    
    W = torch.randn(F, n_components) * 0.1
    W = torch.abs(W) + 0.01
    W = W / W.sum(dim=0, keepdim=True)
    
    print(f"Created source dictionary W: {W.shape}")
    print(f"W stats: mean={W.mean():.4f}, max={W.max():.4f}, min={W.min():.4f}")
    
    # Create simple transfer function H (single direction for this analysis)
    H = torch.randn(F, 1) * 0.1 + 1.0  # Single direction
    H = torch.abs(H)
    
    print(f"Created transfer function H: {H.shape}")
    print(f"H stats: mean={H.mean():.4f}, max={H.max():.4f}, min={H.min():.4f}")
    
    # Load into localizer
    localizer.load_source_dictionary(W)
    localizer.load_transfer_functions(H)
    
    # Convert Y to tensor
    Y_tensor = torch.from_numpy(Y_np).float()
    
    print(f"Input Y_tensor stats: mean={Y_tensor.mean():.2e}, max={Y_tensor.max():.2e}, min={Y_tensor.min():.2e}")
    print(f"A matrix shape: {localizer.A.shape}")
    print(f"A matrix stats: mean={localizer.A.mean():.2e}, max={localizer.A.max():.2e}, min={localizer.A.min():.2e}")
    
    # Perform NMF factorization
    print("Starting NMF factorization...")
    X_factors_tensor, result = localizer.factorize(Y_tensor)
    
    # Calculate Y_hat reconstruction
    Y_hat_tensor = localizer.A @ X_factors_tensor
    
    # Convert back to numpy for analysis
    A = localizer.A.detach().cpu().numpy()
    X_factors = X_factors_tensor.detach().cpu().numpy() 
    Y_hat = Y_hat_tensor.detach().cpu().numpy()
    
    print(f"Real NMF results:")
    print(f"  Converged: {result['converged']}")
    print(f"  Iterations: {result['n_iter']}")
    print(f"  Final loss: {result['final_loss']:.2e}")
    print(f"  A shape: {A.shape}")
    print(f"  X_factors shape: {X_factors.shape}")
    print(f"  X_factors sparsity: {1.0 - np.count_nonzero(X_factors > 1e-10) / X_factors.size:.3f}")
    print(f"  A mean: {np.mean(A):.2e}")
    print(f"  X_factors mean: {np.mean(X_factors):.2e}")
    print(f"  Loss history: {result.get('losses', [])[-5:] if 'losses' in result else 'N/A'}")  # Last 5 losses
    
    print(f"Y_hat reconstruction:")
    print(f"  Shape: {Y_hat.shape}")
    print(f"  Mean: {np.mean(Y_hat):.2e}")
    print(f"  Max: {np.max(Y_hat):.2e}")
    print(f"  Min: {np.min(Y_hat):.2e}")
    print(f"  Std: {np.std(Y_hat):.2e}")
    
    print(f"Y original:")
    print(f"  Shape: {Y_np.shape}")
    print(f"  Mean: {np.mean(Y_np):.2e}")
    print(f"  Max: {np.max(Y_np):.2e}")
    print(f"  Min: {np.min(Y_np):.2e}")
    print(f"  Std: {np.std(Y_np):.2e}")
    
    # Calculate IS divergence
    print(f"\n{'='*50}")
    print("IS DIVERGENCE ANALYSIS")
    print(f"{'='*50}")
    
    is_div_results = calculate_is_divergence_beta0(Y_np, Y_hat)
    
    print(f"IS Divergence Results:")
    print(f"  Total divergence: {is_div_results['total_divergence']:.2e}")
    print(f"  Mean divergence per element: {is_div_results['mean_divergence']:.2e}")
    print(f"  Max divergence per element: {is_div_results['max_divergence']:.2e}")
    print(f"  Min divergence per element: {is_div_results['min_divergence']:.2e}")
    
    print(f"\nY/Y_hat Ratio Statistics:")
    ratio_stats = is_div_results['ratio_stats']
    print(f"  Mean ratio: {ratio_stats['mean']:.2e}")
    print(f"  Max ratio: {ratio_stats['max']:.2e}")
    print(f"  Min ratio: {ratio_stats['min']:.2e}")
    print(f"  Std ratio: {ratio_stats['std']:.2e}")
    
    # Problem source analysis
    print(f"\n{'='*50}")
    print("PROBLEM SOURCE ANALYSIS")
    print(f"{'='*50}")
    
    # Check X_factors sparsity
    x_nonzero = np.count_nonzero(X_factors)
    x_total = X_factors.size
    x_sparsity = 1.0 - (x_nonzero / x_total)
    
    print(f"X_factors sparsity analysis:")
    print(f"  Non-zero elements: {x_nonzero}/{x_total}")
    print(f"  Sparsity: {x_sparsity:.3f} ({x_sparsity*100:.1f}%)")
    print(f"  X_factors mean: {np.mean(X_factors):.2e}")
    print(f"  X_factors max: {np.max(X_factors):.2e}")
    
    # Check A matrix condition
    print(f"\nA matrix analysis:")
    print(f"  A mean: {np.mean(A):.2e}")
    print(f"  A max: {np.max(A):.2e}")
    print(f"  A condition number: {np.linalg.cond(A):.2e}")
    
    # Check if Y_hat is dominated by very small values
    y_hat_histogram = np.histogram(Y_hat.flatten(), bins=50)
    print(f"\nY_hat value distribution:")
    print(f"  Values near zero (<1e-10): {np.sum(Y_hat < 1e-10)}/{Y_hat.size}")
    print(f"  Values near zero (<1e-8): {np.sum(Y_hat < 1e-8)}/{Y_hat.size}")
    print(f"  Values near zero (<1e-6): {np.sum(Y_hat < 1e-6)}/{Y_hat.size}")
    
    # Compare scale difference
    scale_ratio = np.mean(Y_np) / np.mean(Y_hat)
    print(f"\nScale analysis:")
    print(f"  Y/Y_hat scale ratio: {scale_ratio:.2e}")
    print(f"  Expected based on test results: ~28,000")
    
    # Print iteration tracking summary
    tracker.print_summary()
    
    # Calculate IS divergence matrix for visualization
    is_div_matrix = calculate_is_divergence_matrix(Y_np, Y_hat)
    
    # Save visualization data for plotting
    save_visualization_data(Y_np, Y_hat, is_div_matrix, tracker, 
                           scale_ratio, x_sparsity, lambda_group=0.01, gamma_sparse=0.05)
    
    return {
        'is_divergence': is_div_results,
        'Y': Y_np,
        'Y_hat': Y_hat,
        'A': A,
        'X_factors': X_factors,
        'scale_ratio': scale_ratio,
        'x_sparsity': 1.0 - np.count_nonzero(X_factors > 1e-10) / X_factors.size,
        'tracker': tracker,
        'nmf_result': result
    }

def create_config_with_parameters(lambda_group, gamma_sparse, tolerance=1e-6):
    """Create NMF config with specified parameters."""
    return NMFConfig(
        sample_rate=16000,
        n_fft=2048,
        hop_length=512,
        freq_min=500.0,
        freq_max=3000.0,
        n_files_per_angle=1,
        max_iter=50,
        beta=0,  # IS divergence
        lambda_group=lambda_group,
        gamma_sparse=gamma_sparse,
        tolerance=tolerance
    )


def run_single_configuration(lambda_group, gamma_sparse, tolerance=1e-6):
    """Run NMF with specific parameters and return results."""
    print(f"\n" + "="*80)
    print(f"TESTING: λ_group={lambda_group}, γ_sparse={gamma_sparse}, tolerance={tolerance}")
    print("="*80)
    
    # Create config with specified parameters
    config = create_config_with_parameters(lambda_group, gamma_sparse, tolerance)
    
    # Use the same data loading logic as analyze_nmf_reconstruction
    x_root = "/Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original/white_noise_original_data_no_edge_sync_vad"
    y_root = "/Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original/white_noise_box_data_no_edge_sync_vad"
    test_angle = "angle_90"
    
    # Load and process data
    x_dir = os.path.join(x_root, test_angle)
    y_dir = os.path.join(y_root, test_angle)
    
    x_files = [f for f in os.listdir(x_dir) if f.endswith('.npy')]
    y_files = [f for f in os.listdir(y_dir) if f.endswith('.npy')]
    
    if not x_files or not y_files:
        return {'success': False, 'error': 'No data files found'}
    
    # Use first file
    x_audio = np.load(os.path.join(x_dir, x_files[0]))
    y_audio = np.load(os.path.join(y_dir, y_files[0]))
    
    # Process to spectrograms
    audio_processor = AudioProcessor()
    
    freqs_x, times_x, stft_x, magnitude_x = audio_processor.compute_stft_spectrogram(
        x_audio.astype(np.float32),
        fs=config.sample_rate,
        nperseg=config.n_fft,
        noverlap=config.n_fft - config.hop_length
    )
    
    freqs_y, times_y, stft_y, magnitude_y = audio_processor.compute_stft_spectrogram(
        y_audio.astype(np.float32),
        fs=config.sample_rate,
        nperseg=config.n_fft,
        noverlap=config.n_fft - config.hop_length
    )
    
    X_tensor = torch.from_numpy(magnitude_x).float()
    Y_tensor = torch.from_numpy(magnitude_y).float()
    
    X_spec, freqs_filtered_x = audio_processor.apply_frequency_filter(
        X_tensor, freqs_x, config.freq_min, config.freq_max
    )
    Y_spec, freqs_filtered_y = audio_processor.apply_frequency_filter(
        Y_tensor, freqs_y, config.freq_min, config.freq_max
    )
    
    Y_np = Y_spec.detach().cpu().numpy()
    
    try:
        # Initialize tracker and NMF
        tracker = IterationTracker()
        localizer = TrackedNMFLocalizer(config, tracker)
        
        # Create W and H (same as main function)
        n_components = 15
        F = Y_np.shape[0]
        
        W = torch.randn(F, n_components) * 0.1
        W = torch.abs(W) + 0.01
        W = W / W.sum(dim=0, keepdim=True)
        
        H = torch.randn(F, 1) * 0.1 + 1.0
        H = torch.abs(H)
        
        localizer.load_source_dictionary(W)
        localizer.load_transfer_functions(H)
        
        # Run NMF
        Y_tensor = torch.from_numpy(Y_np).float()
        X_factors_tensor, result = localizer.factorize(Y_tensor)
        
        # Calculate reconstruction
        Y_hat_tensor = localizer.A @ X_factors_tensor
        
        # Convert to numpy
        A = localizer.A.detach().cpu().numpy()
        X_factors = X_factors_tensor.detach().cpu().numpy()
        Y_hat = Y_hat_tensor.detach().cpu().numpy()
        
        # Calculate metrics
        is_div_results = calculate_is_divergence_beta0(Y_np, Y_hat)
        scale_ratio = np.mean(Y_np) / np.mean(Y_hat)
        x_sparsity = 1.0 - np.count_nonzero(X_factors > 1e-10) / X_factors.size
        
        # Calculate success score
        score = evaluate_configuration_beta0(result, is_div_results, scale_ratio, x_sparsity)
        
        print(f"RESULT: Score={score:.1f}, Sparsity={x_sparsity:.3f}, Scale={scale_ratio:.2e}, "
              f"Iterations={result['n_iter']}, IS_div={is_div_results['total_divergence']:.2e}")
        
        return {
            'success': True,
            'lambda_group': lambda_group,
            'gamma_sparse': gamma_sparse,
            'tolerance': tolerance,
            'score': score,
            'x_sparsity': x_sparsity,
            'scale_ratio': scale_ratio,
            'n_iterations': result['n_iter'],
            'converged': result['converged'],
            'final_loss': result['final_loss'],
            'is_divergence': is_div_results['total_divergence'],
            'tracker': tracker,
            'Y_mean': np.mean(Y_np),
            'Y_hat_mean': np.mean(Y_hat),
            'A_condition_number': np.linalg.cond(A)
        }
        
    except Exception as e:
        print(f"ERROR in configuration: {e}")
        return {'success': False, 'error': str(e), 'lambda_group': lambda_group, 'gamma_sparse': gamma_sparse}


def evaluate_configuration_beta0(nmf_result, is_div_results, scale_ratio, x_sparsity):
    """Evaluate configuration performance for IS divergence (β=0) - REVISED."""
    score = 0
    
    # Scale ratio evaluation (MOST IMPORTANT for reconstruction quality)
    if 0.8 <= scale_ratio <= 1.2:
        score += 50  # Excellent scale matching (near perfect)
    elif 0.5 <= scale_ratio <= 2.0:
        score += 35  # Very good scale matching
    elif 0.2 <= scale_ratio <= 5.0:
        score += 20  # Acceptable scale matching
    elif scale_ratio > 100:
        score -= 50  # Scale disaster (reconstruction failure)
    elif scale_ratio > 10:
        score -= 30  # Poor scale matching
    else:
        score += 10  # Other cases
    
    # Sparsity evaluation (less critical for IS divergence, focus on avoiding over-sparsity)
    if x_sparsity < 0.1:
        score += 20  # Good - not over-sparse
    elif 0.1 <= x_sparsity <= 0.5:
        score += 15  # Moderate sparsity is acceptable
    elif 0.5 < x_sparsity <= 0.8:
        score += 5   # Getting sparse but not terrible
    elif x_sparsity > 0.95:
        score -= 40  # Over-sparsification disaster
    else:
        score -= 10  # High sparsity but not disaster
    
    # IS divergence evaluation (lower is better, but be realistic about IS divergence levels)
    is_div = is_div_results['total_divergence']
    if is_div < 1e6:
        score += 25  # Excellent IS divergence
    elif is_div < 1e7:
        score += 20  # Very good IS divergence  
    elif is_div < 5e7:
        score += 15  # Good IS divergence (realistic for this problem)
    elif is_div < 1e8:
        score += 5   # Acceptable IS divergence
    elif is_div > 1e10:
        score -= 30  # Terrible IS divergence
    else:
        score -= 10  # High but not terrible
    
    # Convergence evaluation (less important than quality)
    n_iter = nmf_result['n_iter']
    if n_iter < 5:
        score -= 20  # Premature convergence is concerning
    elif n_iter >= 50:
        score -= 5   # Slow convergence but not critical if quality is good
    else:
        score += 5   # Normal convergence behavior
    
    # Convergence success bonus (small bonus)
    if nmf_result['converged']:
        score += 5
    
    return score


def parameter_grid_search_beta0():
    """Systematic parameter search for IS divergence (β=0)."""
    print("="*80)
    print("PARAMETER GRID SEARCH FOR IS DIVERGENCE (β=0)")
    print("="*80)
    
    # Parameter ranges optimized for IS divergence
    lambda_groups = [0.01, 0.05, 0.1, 0.2, 0.5]
    gamma_sparses = [0.001, 0.01, 0.05, 0.1]
    tolerances = [1e-8, 1e-6]
    
    all_results = []
    total_configs = len(lambda_groups) * len(gamma_sparses) * len(tolerances)
    current_config = 0
    
    print(f"Testing {total_configs} parameter combinations...")
    
    for tolerance in tolerances:
        for lambda_group in lambda_groups:
            for gamma_sparse in gamma_sparses:
                current_config += 1
                print(f"\n[{current_config}/{total_configs}] Configuration:")
                
                result = run_single_configuration(lambda_group, gamma_sparse, tolerance)
                result['config_id'] = current_config
                all_results.append(result)
                
                if result['success']:
                    print(f"✓ Success: Score {result['score']:.1f}")
                else:
                    print(f"✗ Failed: {result['error']}")
    
    return analyze_grid_search_results(all_results)


def analyze_grid_search_results(all_results):
    """Analyze grid search results and provide recommendations."""
    print("\n" + "="*80)
    print("GRID SEARCH ANALYSIS RESULTS")
    print("="*80)
    
    # Filter successful results
    successful_results = [r for r in all_results if r['success']]
    failed_results = [r for r in all_results if not r['success']]
    
    print(f"Successful configurations: {len(successful_results)}/{len(all_results)}")
    print(f"Failed configurations: {len(failed_results)}")
    
    if not successful_results:
        print("❌ No successful configurations found!")
        return {'success': False, 'all_results': all_results}
    
    # Sort by score
    successful_results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n🏆 TOP 5 CONFIGURATIONS:")
    print("-" * 100)
    print(f"{'Rank':<4} {'λ_group':<8} {'γ_sparse':<10} {'Tol':<6} {'Score':<6} {'Sparsity':<9} {'Scale':<10} {'Iter':<4} {'IS_div':<10}")
    print("-" * 100)
    
    for i, result in enumerate(successful_results[:5]):
        print(f"{i+1:<4} {result['lambda_group']:<8} {result['gamma_sparse']:<10} "
              f"{result['tolerance']:<6} {result['score']:<6.1f} {result['x_sparsity']:<9.3f} "
              f"{result['scale_ratio']:<10.2e} {result['n_iterations']:<4} {result['is_divergence']:<10.2e}")
    
    # Best configuration analysis
    best = successful_results[0]
    print(f"\n🎯 BEST CONFIGURATION:")
    print(f"   λ_group = {best['lambda_group']}")
    print(f"   γ_sparse = {best['gamma_sparse']}")  
    print(f"   tolerance = {best['tolerance']}")
    print(f"   Score = {best['score']:.1f}")
    print(f"   X sparsity = {best['x_sparsity']:.3f}")
    print(f"   Scale ratio = {best['scale_ratio']:.2e}")
    print(f"   Iterations = {best['n_iterations']}")
    print(f"   IS divergence = {best['is_divergence']:.2e}")
    
    # Parameter sensitivity analysis
    print(f"\n📊 PARAMETER SENSITIVITY ANALYSIS:")
    analyze_parameter_sensitivity(successful_results)
    
    # Failure analysis
    if failed_results:
        print(f"\n❌ FAILURE ANALYSIS:")
        failure_reasons = {}
        for result in failed_results:
            reason = result.get('error', 'Unknown')
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        for reason, count in failure_reasons.items():
            print(f"   {reason}: {count} failures")
    
    return {
        'success': True,
        'best_config': best,
        'top_5': successful_results[:5],
        'all_successful': successful_results,
        'all_results': all_results,
        'failure_count': len(failed_results)
    }


def analyze_parameter_sensitivity(results):
    """Analyze how different parameters affect performance."""
    # Group by lambda_group
    lambda_scores = {}
    for result in results:
        lg = result['lambda_group']
        if lg not in lambda_scores:
            lambda_scores[lg] = []
        lambda_scores[lg].append(result['score'])
    
    print(f"   λ_group sensitivity:")
    for lg in sorted(lambda_scores.keys()):
        scores = lambda_scores[lg]
        avg_score = np.mean(scores)
        print(f"     λ_group={lg}: avg_score={avg_score:.1f} ({len(scores)} configs)")
    
    # Group by gamma_sparse
    gamma_scores = {}
    for result in results:
        gs = result['gamma_sparse']
        if gs not in gamma_scores:
            gamma_scores[gs] = []
        gamma_scores[gs].append(result['score'])
    
    print(f"   γ_sparse sensitivity:")
    for gs in sorted(gamma_scores.keys()):
        scores = gamma_scores[gs]
        avg_score = np.mean(scores)
        print(f"     γ_sparse={gs}: avg_score={avg_score:.1f} ({len(scores)} configs)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "grid_search":
        # Run grid search
        try:
            results = parameter_grid_search_beta0()
            if results['success']:
                print(f"\n✅ Grid search completed successfully!")
                print(f"🏆 Best configuration: λ_group={results['best_config']['lambda_group']}, "
                      f"γ_sparse={results['best_config']['gamma_sparse']}")
            else:
                print(f"\n❌ Grid search failed!")
        except Exception as e:
            print(f"Grid search error: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Run single analysis (original behavior)
        try:
            results = analyze_nmf_reconstruction()
            print(f"\n{'='*50}")
            print("SUMMARY")
            print(f"{'='*50}")
            print(f"IS Divergence: {results['is_divergence']['total_divergence']:.2e}")
            print(f"Scale ratio: {results['scale_ratio']:.2e}")
            print(f"X sparsity: {results['x_sparsity']:.3f}")
            print(f"\nTo run parameter grid search, use: python debug_is_divergence.py grid_search")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()