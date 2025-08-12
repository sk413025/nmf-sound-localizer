"""
NMF-based sound source localizer.
Extracted and refactored from src/models/nmf_sound_localizer.py.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, List
from itertools import permutations
import logging

from ..config.defaults import NMFConfig

logger = logging.getLogger(__name__)


class NMFSoundLocalizer(nn.Module):
    """
    Non-negative Matrix Factorization for Sound Source Localization.
    
    Implements the complete algorithm from:
    "Direction of Arrival with One Microphone, a few LEGOs, and Non-Negative Matrix Factorization"
    
    Key components:
    - W: Source dictionary (F x K) - learned spectral templates
    - H_d: Direction-dependent transfer functions (F x 1) for each direction d
    - A: Mixing matrix [diag(H_1)W, ..., diag(H_D)W] (F x KD)
    - X: Coefficient matrix (KD x N) with group structure
    
    The model solves: Y ≈ AX with sparse-group sparsity regularization
    """
    
    def __init__(self, config: NMFConfig):
        super().__init__()
        
        self.config = config
        self.n_freq = None  # Will be set when loading data
        self.n_components = None  # Will be set when loading W
        self.n_directions = None  # Will be set when loading H
        
        self.beta = config.beta
        self.lambda_group = config.lambda_group
        self.gamma_sparse = config.gamma_sparse
        self.max_iter = config.max_iter
        self.tol = config.tolerance
        self.epsilon = config.epsilon
        self.device = config.device
        self.normalize_blocks = config.normalize_blocks
        
        # Components (will be loaded)
        self.W = None  # Source dictionary
        self.H = None  # Transfer functions
        self.A = None  # Mixing matrix
        self.actual_angles = None  # Actual angle values
        self.freq_weights = None  # Frequency weights
        
    def load_source_dictionary(self, W: torch.Tensor):
        """Load pre-trained source dictionary (e.g., USM)."""
        if W.dim() != 2:
            raise ValueError(f"Expected 2D tensor for W, got shape {W.shape}")
        
        self.n_freq = W.shape[0]
        self.n_components = W.shape[1]
        self.W = W.to(self.device)
        
        logger.info(f"Loaded source dictionary W: {W.shape}")
        logger.info(f"Updated n_freq={self.n_freq}, n_components={self.n_components}")
        
        # Initialize frequency weights if not set
        if self.freq_weights is None:
            self.freq_weights = torch.ones(self.n_freq, device=self.device)
        
        self._construct_mixing_matrix()
        
    def load_transfer_functions(self, H: torch.Tensor, angles: Optional[torch.Tensor] = None):
        """Load measured directional transfer functions."""
        if H.dim() != 2:
            raise ValueError(f"Expected 2D tensor for H, got shape {H.shape}")
        
        if self.n_freq is not None and H.shape[0] != self.n_freq:
            raise ValueError(f"H frequency dimension {H.shape[0]} doesn't match W frequency dimension {self.n_freq}")
        
        if self.n_freq is None:
            self.n_freq = H.shape[0]
            
        self.n_directions = H.shape[1]
        self.H = H.to(self.device)
        
        logger.info(f"Loaded transfer functions H: {H.shape}")
        logger.info(f"Updated n_directions={self.n_directions}")
        
        # Store actual angles if provided
        if angles is not None:
            self.actual_angles = angles.to(self.device)
            logger.info(f"Loaded actual angles: {angles.tolist()}")
        else:
            # Default to uniform distribution
            self.actual_angles = torch.linspace(0, 360, self.n_directions + 1)[:-1].to(self.device)
            logger.info("Using uniform angle distribution")
            
        # Initialize frequency weights if not set
        if self.freq_weights is None:
            self.freq_weights = torch.ones(self.n_freq, device=self.device)
            
        self._construct_mixing_matrix()
    
    def set_frequency_weights(self, weights: torch.Tensor):
        """
        Set per-frequency weights w (F,) to reweight A and Y consistently.
        """
        if weights.shape[0] != self.n_freq:
            raise ValueError(f"Expected weights length {self.n_freq}, got {weights.shape}")
        
        w = weights.to(self.device)
        # Avoid zeros to keep updates well-defined
        w = torch.clamp(w, min=self.epsilon)
        self.freq_weights = w
        
        logger.info(f"Set frequency weights: min={w.min():.6f}, max={w.max():.6f}, mean={w.mean():.6f}")
        
        # Rebuild A to apply new weights
        if self.W is not None and self.H is not None:
            self._construct_mixing_matrix()
            
    def _construct_mixing_matrix(self):
        """Construct mixing matrix A = [diag(H_1)W, ..., diag(H_D)W]."""
        if self.W is None or self.H is None:
            return
            
        logger.info("=== Constructing Mixing Matrix A ===")
        logger.info(f"W shape: {self.W.shape}, H shape: {self.H.shape}")
        
        # A will be F x KD
        A_blocks = []
        block_norms = []
        
        for d in range(self.n_directions):
            # diag(H_d) @ W
            H_d_diag = torch.diag(self.H[:, d])
            A_d = H_d_diag @ self.W
            
            if d == 0:  # Log first block
                logger.info(f"H_d_diag shape: {H_d_diag.shape}")
                logger.info(f"A_d (block {d}) shape: {A_d.shape}")
            
            # Apply frequency weights (left-multiply by diag(w))
            if self.freq_weights is not None:
                A_d = self.freq_weights.view(-1, 1) * A_d
                
            if self.normalize_blocks:
                # Equalize Frobenius norm across direction blocks
                bn = torch.linalg.norm(A_d, ord='fro') + self.epsilon
                A_d = A_d / bn
                block_norms.append(bn.item())
                
            A_blocks.append(A_d)
            
        self.A = torch.cat(A_blocks, dim=1)  # F x KD
        logger.info(f"Final A matrix shape: {self.A.shape} "
                   f"(expected: {self.n_freq} x {self.n_components * self.n_directions})")
        
        if self.normalize_blocks and block_norms:
            logger.info(f"Applied block normalization; sample norms: {block_norms[:5]}")

    def _beta_divergence(self, Y: torch.Tensor, Y_hat: torch.Tensor) -> torch.Tensor:
        """Compute beta divergence between Y and Y_hat."""
        # Replace nan and inf values with epsilon before clamping
        Y = torch.where(torch.isnan(Y) | torch.isinf(Y), self.epsilon, Y)
        Y_hat = torch.where(torch.isnan(Y_hat) | torch.isinf(Y_hat), self.epsilon, Y_hat)
        
        Y = torch.clamp(Y, min=self.epsilon)
        Y_hat = torch.clamp(Y_hat, min=self.epsilon)
        
        if self.beta == 0:  # Itakura-Saito divergence
            ratio = Y / Y_hat
            log_ratio = torch.log(ratio)
            div = torch.sum(ratio - log_ratio - 1)
            
        elif self.beta == 1:  # KL divergence
            div = torch.sum(Y * torch.log(Y / Y_hat) - Y + Y_hat)
        elif self.beta == 2:  # Euclidean distance
            div = 0.5 * torch.sum((Y - Y_hat) ** 2)
        else:  # General beta
            div = torch.sum(
                (Y.pow(self.beta) - Y_hat.pow(self.beta)) / self.beta +
                (Y_hat.pow(self.beta - 1) - Y.pow(self.beta - 1)) * Y / (self.beta - 1)
            )
        
        return div
        
    def _compute_group_penalty_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute group penalty matrix P where P_d = 1/(ε + ||vec(X_d)||_1).
        X shape: (KD x N)
        Returns P shape: (KD x N)
        """
        K = self.n_components
        D = self.n_directions
        N = X.shape[1]
        
        P = torch.zeros_like(X)
        
        for d in range(D):
            # Extract group X_d (K x N)
            start_idx = d * K
            end_idx = (d + 1) * K
            X_d = X[start_idx:end_idx, :]
            
            # Compute group norm
            group_norm = torch.sum(torch.abs(X_d)) + self.epsilon
            
            # Set penalty for this group
            P[start_idx:end_idx, :] = 1.0 / group_norm
            
        return P
        
    def _multiplicative_update(self, Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Multiplicative update for X with sparse-group sparsity.
        Implements Algorithm 2 from the paper.
        """
        Y_hat = self.A @ X
        
        # Compute group penalty matrix
        P = self._compute_group_penalty_matrix(X)
        
        if self.beta == 0:  # Itakura-Saito
            y_over_yhat2 = Y / (Y_hat ** 2 + self.epsilon)
            inv_yhat = 1.0 / (Y_hat + self.epsilon)
            
            numerator = self.A.T @ y_over_yhat2
            denominator = self.A.T @ inv_yhat + self.lambda_group * P + self.gamma_sparse
            
            ratio = numerator / (denominator + self.epsilon)
            
            # Ensure ratio is non-negative before taking square root
            ratio = torch.clamp(ratio, min=self.epsilon)
            X_new = X * torch.sqrt(ratio)
            
        elif self.beta == 2:  # Euclidean
            numerator = self.A.T @ Y - self.lambda_group * P - self.gamma_sparse
            denominator = self.A.T @ Y_hat
            X_new = X * torch.clamp(numerator / (denominator + self.epsilon), min=self.epsilon)
            
        else:  # General beta (including KL for beta=1)
            numerator = self.A.T @ (Y * Y_hat.pow(self.beta - 2))
            denominator = self.A.T @ Y_hat.pow(self.beta - 1) + \
                          self.lambda_group * P + self.gamma_sparse
            exponent = 1.0 if self.beta == 1 else 0.5
            X_new = X * (numerator / (denominator + self.epsilon)) ** exponent
        
        # Replace nan and inf values with epsilon
        X_new = torch.where(torch.isnan(X_new) | torch.isinf(X_new), self.epsilon, X_new)
        X_clamped = torch.clamp(X_new, min=self.epsilon)
        
        return X_clamped
        
    def factorize(self, Y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, any]]:
        """
        Factorize Y ≈ AX using multiplicative updates.
        
        Args:
            Y: Input magnitude spectrogram (F x N)
            
        Returns:
            X: Coefficient matrix (KD x N)
            info: Dictionary with convergence info
        """
        if self.A is None:
            raise ValueError("Must load source dictionary and transfer functions first")
        
        F, N = Y.shape
        if F != self.n_freq:
            raise ValueError(f"Expected {self.n_freq} frequency bins, got {F}")
        
        logger.info(f"=== NMF Factorization ===")
        logger.info(f"Input Y shape: {Y.shape}")
        logger.info(f"A shape: {self.A.shape}")

        # Apply frequency weights to Y consistently with A
        if self.freq_weights is not None:
            Y = self.freq_weights.view(-1, 1) * Y
        
        # Initialize X = A^T @ Y
        X = self.A.T @ Y
        
        # Better initialization: use random positive values if too small
        if X.mean() < 1e-6:
            logger.info("X too small, using random initialization")
            X = torch.rand_like(X) * Y.mean() * 0.1 + self.epsilon
        else:
            X = torch.clamp(X, min=self.epsilon)
        
        # Add small random noise to avoid getting stuck
        noise = torch.rand_like(X) * X.mean() * 0.01
        X = X + noise
        logger.info(f"X initialized: shape={X.shape}, min={X.min():.6f}, max={X.max():.6f}")
        
        losses = []
        
        for iteration in range(self.max_iter):
            # Update X
            X = self._multiplicative_update(Y, X)
            
            # Compute loss
            Y_hat = self.A @ X
            data_fit = self._beta_divergence(Y, Y_hat)
            
            # Group sparsity penalty (log/l1)
            group_penalty = 0
            for d in range(self.n_directions):
                start_idx = d * self.n_components
                end_idx = (d + 1) * self.n_components
                X_d = X[start_idx:end_idx, :]
                group_norm = torch.sum(torch.abs(X_d))
                group_penalty += torch.log(self.epsilon + group_norm)
                
            # Sparsity penalty (l1)
            sparse_penalty = torch.sum(torch.abs(X))
            
            total_loss = data_fit + self.lambda_group * group_penalty + \
                        self.gamma_sparse * sparse_penalty
            losses.append(total_loss.item())
            
            # Check convergence
            if iteration > 0 and abs(losses[-1] - losses[-2]) < self.tol:
                logger.info(f"NMF converged at iteration {iteration}")
                break
                
        info = {
            'final_loss': losses[-1] if losses else float('inf'),
            'n_iter': len(losses),
            'converged': len(losses) < self.max_iter,
            'losses': losses
        }
        
        logger.info(f"Factorization complete: {info['n_iter']} iterations, "
                   f"final loss: {info['final_loss']:.6f}")
        
        return X, info
        
    def localize(
        self, 
        Y: torch.Tensor, 
        n_sources: int = 1,
        multiresolution: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Localize sound sources from magnitude spectrogram.
        
        Args:
            Y: Input magnitude spectrogram (F x N)
            n_sources: Number of sources to localize
            multiresolution: Whether to use coarse-to-fine approach
            
        Returns:
            Dictionary with:
                - directions: Estimated directions (n_sources,)
                - group_norms: Norms of coefficient groups (n_directions,)
                - coefficients: Full coefficient matrix X
                - info: Factorization info
        """
        # Initial factorization
        X, info = self.factorize(Y)
        
        # Calculate group norms
        group_norms = torch.zeros(self.n_directions, device=self.device)
        logger.info("=== Computing Group Norms ===")
        
        for d in range(self.n_directions):
            start_idx = d * self.n_components
            end_idx = (d + 1) * self.n_components
            X_d = X[start_idx:end_idx, :]
            group_norm = torch.sum(torch.abs(X_d))
            group_norms[d] = group_norm
            
            if d < 3:  # Log first few groups
                logger.info(f"Group {d}: indices [{start_idx}:{end_idx}], "
                           f"norm: {group_norm:.6f}")
            
        # Select top n_sources directions
        _, source_indices = torch.topk(group_norms, n_sources)
        
        # Convert indices to actual angles
        if self.actual_angles is not None:
            angles = self.actual_angles[source_indices]
        else:
            # Fallback to uniform spacing
            angles = source_indices.float() * (360.0 / self.n_directions)
        
        logger.info(f"Localization result: {angles.tolist()} degrees")
        
        return {
            'directions': angles,
            'group_norms': group_norms,
            'coefficients': X,
            'info': info
        }
        
    def compute_accuracy(
        self, 
        estimated: torch.Tensor, 
        ground_truth: List[int],
        tolerance: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Compute localization accuracy and mean error.
        
        Args:
            estimated: Estimated directions (n_sources,)
            ground_truth: True directions (list of angle indices or degrees)
            tolerance: Tolerance in degrees (default from config)
            
        Returns:
            accuracy: Percentage of sources within tolerance
            mean_error: Mean angular error for accurate localizations
        """
        tolerance = tolerance or self.config.tolerance_degrees
        n_sources = len(ground_truth)
        
        # Convert ground truth to angles if needed
        if isinstance(ground_truth[0], int) and max(ground_truth) < self.n_directions:
            # Assume these are indices
            gt_angles = [self.actual_angles[i].item() if self.actual_angles is not None 
                        else i * (360.0 / self.n_directions) for i in ground_truth]
        else:
            # Assume these are already angles
            gt_angles = ground_truth
        
        # Find best permutation matching
        min_error = float('inf')
        best_errors = None
        
        for perm in permutations(range(n_sources)):
            errors = []
            for i, j in enumerate(perm):
                # Compute angular difference (handle wraparound)
                if j < len(estimated):
                    diff = abs(estimated[j].item() - gt_angles[i])
                    diff = min(diff, 360 - diff)
                else:
                    diff = 180  # Maximum error if not enough estimates
                errors.append(diff)
                
            total_error = sum(errors)
            if total_error < min_error:
                min_error = total_error
                best_errors = errors
                
        # Compute accuracy and mean error
        accurate = [e <= tolerance for e in best_errors]
        accuracy = sum(accurate) / n_sources * 100
        
        if sum(accurate) > 0:
            mean_error = sum(e for e, a in zip(best_errors, accurate) if a) / sum(accurate)
        else:
            mean_error = float('inf')
            
        return accuracy, mean_error
    
    def save_model(self, path: str):
        """Save the complete localization model."""
        if self.W is None or self.H is None:
            raise ValueError("Model components not loaded")
        
        save_dict = {
            'W': self.W,
            'H': self.H,
            'actual_angles': self.actual_angles,
            'freq_weights': self.freq_weights,
            'config': self.config.to_dict(),
            'model_info': {
                'n_freq': self.n_freq,
                'n_components': self.n_components,
                'n_directions': self.n_directions
            }
        }
        
        torch.save(save_dict, path)
        logger.info(f"NMF localizer saved to {path}")
    
    def load_model(self, path: str):
        """Load a complete localization model."""
        data = torch.load(path, map_location=self.device, weights_only=False)
        
        # Load components
        self.load_source_dictionary(data['W'])
        self.load_transfer_functions(data['H'], data.get('actual_angles'))
        
        if 'freq_weights' in data and data['freq_weights'] is not None:
            self.set_frequency_weights(data['freq_weights'])
        
        # Update config if available
        if 'config' in data:
            saved_config = data['config']
            for key, value in saved_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        logger.info(f"NMF localizer loaded from {path}")
        logger.info(f"Model: {self.n_freq} freq, {self.n_components} components, "
                   f"{self.n_directions} directions")
    
    def forward(self, Y: torch.Tensor, n_sources: int = 1) -> Dict[str, torch.Tensor]:
        """Forward pass: factorize and localize."""
        return self.localize(Y, n_sources)