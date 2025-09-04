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
            }
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

def analyze_nmf_reconstruction():
    """
    Reproduce the NMF reconstruction and analyze IS divergence
    """
    # Create configuration EXACTLY matching the test
    config = NMFConfig(
        sample_rate=16000,
        n_fft=2048,
        hop_length=512,
        freq_min=500.0,
        freq_max=3000.0,
        n_files_per_angle=1,
        max_iter=50,
        beta=2.0,  # Euclidean distance (same as test)
        lambda_group=1.0,  # Same as test (10x higher than before)
        gamma_sparse=0.01,  # Same as test (10x lower than before)  
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

if __name__ == "__main__":
    try:
        results = analyze_nmf_reconstruction()
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        print(f"IS Divergence: {results['is_divergence']['total_divergence']:.2e}")
        print(f"Scale ratio: {results['scale_ratio']:.2e}")
        print(f"X sparsity: {results['x_sparsity']:.3f}")
        print(f"Simulated NMF with extreme sparsity (100%)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()