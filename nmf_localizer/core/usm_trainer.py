"""
Universal Speech Model (USM) trainer module.
Extracted and refactored from src/models/usm_trainer.py.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
# from tqdm import tqdm  # Temporarily disabled for testing
import logging

from ..config.defaults import NMFConfig

logger = logging.getLogger(__name__)


class USMTrainer(nn.Module):
    """
    Train a Universal Speech Model (USM) using NMF.
    
    The USM is a dictionary W that contains spectral templates
    learned from multiple speakers. Each speaker contributes K atoms.
    """
    
    def __init__(self, config: NMFConfig):
        super().__init__()
        
        self.config = config
        self.n_freq = None  # Will be set when training
        self.n_atoms_per_speaker = config.n_atoms_per_speaker
        self.beta = config.beta
        self.sparsity_weight = config.usm_sparsity_weight
        self.max_iter = config.usm_max_iter
        self.tol = config.tolerance
        self.epsilon = config.epsilon
        self.device = config.device
        
        self.W = None  # Will be initialized during training
        
    def _beta_divergence(self, X: torch.Tensor, X_hat: torch.Tensor) -> torch.Tensor:
        """Compute beta divergence."""
        X = torch.clamp(X, min=self.epsilon)
        X_hat = torch.clamp(X_hat, min=self.epsilon)
        
        if self.beta == 0:  # Itakura-Saito
            div = torch.sum(X / X_hat - torch.log(X / X_hat) - 1)
        elif self.beta == 1:  # KL
            div = torch.sum(X * torch.log(X / X_hat) - X + X_hat)
        elif self.beta == 2:  # Euclidean
            div = 0.5 * torch.sum((X - X_hat) ** 2)
        else:
            div = torch.sum(
                (X.pow(self.beta) - X_hat.pow(self.beta)) / self.beta +
                (X_hat.pow(self.beta - 1) - X.pow(self.beta - 1)) * X / (self.beta - 1)
            )
        return div
        
    def _nmf_fit(self, V: torch.Tensor, n_components: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Stable NMF using sklearn implementation (matches debug_nmf_with_learned_sources.py success).
        
        Args:
            V: Input matrix (n_freq x n_samples)
            n_components: Number of components
            
        Returns:
            W: Dictionary matrix (n_freq x n_components)
            H: Activation matrix (n_components x n_samples)
            info: Training information
        """
        n_freq, n_samples = V.shape
        logger.info(f"NMF fitting: V shape {V.shape}, n_components {n_components}")
        
        # Convert to numpy for sklearn - NO SCALING (debug script doesn't scale either)
        V_np = V.detach().cpu().numpy()
        
        logger.info(f"Input data statistics: mean={V_np.mean():.2e}, std={V_np.std():.2e}, max={V_np.max():.2e}")
        
        # Use sklearn NMF with exact same parameters as successful debug script
        from sklearn.decomposition import NMF as sklearn_NMF
        
        nmf = sklearn_NMF(
            n_components=n_components,
            init='nndsvd',              # Stable initialization
            max_iter=1000,              # Sufficient iterations
            random_state=42,            # Reproducible results
            beta_loss='frobenius',      # Stable loss (equivalent to beta=2)
            alpha_W=0.0,               # No L1 regularization on W
            alpha_H=0.0,  # NO L1 regularization on H (causes zero matrices!)
            l1_ratio=1.0               # Pure L1 regularization
        )
        
        try:
            # Fit NMF: V.T ≈ H_sklearn @ W_sklearn.T (same as debug script)
            H_sklearn = nmf.fit_transform(V_np.T)      # (n_samples, n_components)
            W_sklearn = nmf.components_                # (n_components, n_freq)
            
            # Convert back to our convention: V ≈ W @ H
            W = torch.from_numpy(W_sklearn.T).float().to(self.device)  # (n_freq, n_components)
            H = torch.from_numpy(H_sklearn.T).float().to(self.device)  # (n_components, n_samples)
            
            # Ensure non-negative and non-zero
            W = torch.clamp(W, min=self.epsilon)
            H = torch.clamp(H, min=self.epsilon)
            
            info = {
                'final_loss': float(nmf.reconstruction_err_),
                'n_iterations': nmf.n_iter_,
                'converged': nmf.n_iter_ < 1000,
                'sklearn_method': True,
                'losses': [float(nmf.reconstruction_err_)],  # Single final loss
                'input_data_mean': float(V_np.mean())  # Record input data characteristics
            }
            
            # Verification
            nonzero_W = (W > self.epsilon).sum().item()
            total_W = W.numel()
            logger.info(f"NMF completed: {info['n_iterations']} iterations, final loss: {info['final_loss']:.6f}")
            logger.info(f"W non-zero elements: {nonzero_W}/{total_W} ({100*nonzero_W/total_W:.1f}%)")
            logger.info(f"W statistics: min={W.min():.6f}, max={W.max():.6f}, mean={W.mean():.6f}")
            
            if nonzero_W == 0:
                raise ValueError("sklearn NMF produced zero W matrix - check input data")
            
        except Exception as e:
            logger.error(f"sklearn NMF failed: {e}")
            # Fallback to simple random initialization if sklearn fails
            logger.warning("Falling back to random initialization")
            avg = V.mean()
            W = torch.rand(n_freq, n_components, device=self.device) * avg * 0.1
            H = torch.rand(n_components, n_samples, device=self.device) * avg * 0.1
            W = torch.clamp(W, min=self.epsilon)
            H = torch.clamp(H, min=self.epsilon)
            
            info = {
                'final_loss': float('inf'),
                'n_iterations': 0,
                'converged': False,
                'sklearn_method': False,
                'fallback': True,
                'losses': []
            }
        
        return W, H, info
        
    def train_speaker_model(self, speaker_spectrograms: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Train NMF model for a single speaker.
        
        Args:
            speaker_spectrograms: Concatenated spectrograms (n_freq x total_frames)
            
        Returns:
            W_speaker: Dictionary for this speaker (n_freq x n_atoms_per_speaker)
            info: Training information
        """
        logger.info(f"Training speaker model: input shape {speaker_spectrograms.shape}")
        W_speaker, _, info = self._nmf_fit(speaker_spectrograms, self.n_atoms_per_speaker)
        return W_speaker, info
        
    def train_usm(self, speaker_data_list: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Train Universal Speech Model from multiple speakers.
        
        Args:
            speaker_data_list: List of spectrograms, one per speaker
            
        Returns:
            W: Universal dictionary (n_freq x (n_speakers * n_atoms_per_speaker))
            info: Training information
        """
        if not speaker_data_list:
            raise ValueError("Empty speaker data list")
        
        # Set frequency dimension from first speaker
        self.n_freq = speaker_data_list[0].shape[0]
        
        # Ensure all speakers have same frequency dimension
        for i, speaker_data in enumerate(speaker_data_list):
            if speaker_data.shape[0] != self.n_freq:
                raise ValueError(f"Speaker {i} has different frequency dimension: "
                               f"{speaker_data.shape[0]} vs expected {self.n_freq}")
        
        W_blocks = []
        all_info = {}
        
        logger.info(f"Training USM with {len(speaker_data_list)} speakers...")
        logger.info(f"Frequency dimension: {self.n_freq}")
        logger.info(f"Atoms per speaker: {self.n_atoms_per_speaker}")
        
        for i, speaker_data in enumerate(speaker_data_list):
            # Move to device
            speaker_data = speaker_data.to(self.device)
            
            W_speaker, speaker_info = self.train_speaker_model(speaker_data)
            W_blocks.append(W_speaker)
            all_info[f'speaker_{i}'] = speaker_info
            
            logger.info(f"Speaker {i}: W shape {W_speaker.shape}, "
                       f"loss {speaker_info['final_loss']:.6f}")
            
        # Concatenate all speaker dictionaries
        self.W = torch.cat(W_blocks, dim=1)
        
        # Overall training info
        final_info = {
            'n_speakers': len(speaker_data_list),
            'total_atoms': self.W.shape[1],
            'n_freq': self.n_freq,
            'speaker_info': all_info,
            'config': self.config.to_dict()
        }
        
        logger.info(f"USM training complete: {self.W.shape[0]} frequencies x {self.W.shape[1]} atoms")
        
        return self.W, final_info
    
    def test_multiple_betas(
        self, 
        speaker_data_list: List[torch.Tensor],
        beta_values: List[float] = [0.5, 1.0, 1.5, 2.0]
    ) -> Tuple[torch.Tensor, float, Dict[float, Dict[str, Any]]]:
        """
        Test multiple beta values and select the best one.
        
        Args:
            speaker_data_list: List of speaker spectrograms
            beta_values: List of beta values to test
            
        Returns:
            best_W: Best USM dictionary
            best_beta: Best beta value
            all_results: Results for all beta values
        """
        if not speaker_data_list:
            raise ValueError("Empty speaker data list")
        
        original_beta = self.config.beta
        best_beta = original_beta
        best_W = None
        best_error = float('inf')
        all_results = {}
        
        logger.info(f"Testing multiple beta values: {beta_values}")
        
        # Use first speaker's data for quick reconstruction test
        test_data = speaker_data_list[0][:, :100].to(self.device)  # First 100 frames
        
        for beta in beta_values:
            logger.info(f"Testing beta = {beta}")
            
            # Update config temporarily
            self.config.beta = beta
            self.beta = beta
            
            try:
                W_beta, train_info = self.train_usm(speaker_data_list)
                
                # Quick reconstruction test
                reconstructed = W_beta @ torch.linalg.pinv(W_beta) @ test_data
                reconstruction_error = torch.mean((test_data - reconstructed) ** 2).item()
                
                result_info = {
                    'W': W_beta,
                    'reconstruction_error': reconstruction_error,
                    'train_info': train_info
                }
                all_results[beta] = result_info
                
                logger.info(f"Beta {beta}: reconstruction error = {reconstruction_error:.6f}")
                
                # Update best if this is better
                if reconstruction_error < best_error:
                    best_error = reconstruction_error
                    best_beta = beta
                    best_W = W_beta
                    
            except Exception as e:
                logger.warning(f"Beta {beta} failed: {e}")
                all_results[beta] = {'error': str(e)}
        
        # Restore original beta
        self.config.beta = original_beta
        self.beta = original_beta
        self.W = best_W
        
        logger.info(f"Best beta value: {best_beta} (error: {best_error:.6f})")
        
        return best_W, best_beta, all_results
        
    def save_model(self, path: str, include_info: bool = True):
        """Save the trained USM dictionary."""
        if self.W is None:
            raise ValueError("No model to save. Train the USM first.")
        
        save_dict = {
            'W': self.W,
            'n_freq': self.n_freq,
            'n_atoms_per_speaker': self.n_atoms_per_speaker,
            'beta': self.beta,
            'config': self.config.to_dict()
        }
        
        if include_info:
            save_dict['training_config'] = self.config.to_dict()
        
        torch.save(save_dict, path)
        logger.info(f"USM model saved to {path}")
        
    def load_model(self, path: str):
        """Load a pre-trained USM dictionary."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.W = checkpoint['W'].to(self.device)
        self.n_freq = checkpoint.get('n_freq', self.W.shape[0])
        self.n_atoms_per_speaker = checkpoint.get('n_atoms_per_speaker', self.n_atoms_per_speaker)
        self.beta = checkpoint.get('beta', self.beta)
        
        # Update config if available
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            # Update only training-related parameters
            for key in ['n_atoms_per_speaker', 'beta', 'usm_sparsity_weight', 'usm_max_iter']:
                if key in saved_config:
                    setattr(self.config, key, saved_config[key])
        
        logger.info(f"USM model loaded from {path}")
        logger.info(f"Model shape: {self.W.shape}, beta: {self.beta}")
        
        return self.W

    # === Content spectrum estimation (ŝ) from precomputed W ===
    @staticmethod
    def compute_content_s_hat(
        Y: "np.ndarray",
        W: "np.ndarray",
        mode: str = "S1",
        H: Optional["np.ndarray"] = None,
        n_iter: int = 50,
        l1: float = 0.0,
        eps: float = 1e-12,
    ) -> "np.ndarray":
        """
        Estimate content spectrum ŝ(F,) from a precomputed dictionary W(F,K) and a spectrogram Y(F,N).

        Geometry: IS (β=0) multiplicative updates for z with global normalization; ŝ = W z.
        Modes:
          - S1: Ybar = mean_t(Y)
          - S2: Ybar = mean_t(Y) / mean_d(H_d)  (requires H (F,D) or (D,F))
        """
        import numpy as np

        W_np = np.asarray(W)
        Y_np = np.asarray(Y)
        if W_np.ndim != 2:
            raise ValueError(f"W must be 2D (F,K), got shape {W_np.shape}")
        if Y_np.ndim != 2:
            raise ValueError(f"Y must be 2D (F,N), got shape {Y_np.shape}")
        if W_np.shape[0] != Y_np.shape[0]:
            raise ValueError(f"W and Y must have same F, got W {W_np.shape}, Y {Y_np.shape}")

        def _safe(x: "np.ndarray") -> "np.ndarray":
            return np.maximum(x, eps)

        Ybar = _safe(Y_np.mean(axis=1))
        if (mode or "S1").upper() == "S2":
            if H is None:
                raise ValueError("S2 mode requires H (transfer functions)")
            Hm = np.asarray(H)
            if Hm.ndim != 2:
                raise ValueError(f"H must be 2D, got {Hm.shape}")
            if Hm.shape[0] < Hm.shape[1]:  # (D,F) -> (F,D)
                Hm = Hm.T
            Hbar = _safe(Hm.mean(axis=1))
            Ybar = _safe(Ybar / Hbar)

        # IS-MU for z
        F, K = W_np.shape
        z = np.ones(K, dtype=W_np.dtype) / max(K, 1)
        Wp = _safe(W_np)
        for _ in range(max(int(n_iter), 0)):
            Yhat = _safe(Wp @ z)
            num = (Wp.T @ (Ybar / (Yhat ** 2)))
            den = (Wp.T @ (1.0 / Yhat)) + float(l1)
            z *= _safe(num / _safe(den))
            z /= _safe(z.sum())
        s_hat = _safe(Wp @ z)
        return s_hat.astype(np.float32)
    
    def get_reconstruction_quality(
        self, 
        test_data: torch.Tensor,
        n_test_frames: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate reconstruction quality of the trained USM.
        
        Args:
            test_data: Test spectrogram data
            n_test_frames: Number of frames to use for testing
            
        Returns:
            Quality metrics
        """
        if self.W is None:
            raise ValueError("No trained model available")
        
        test_data = test_data.to(self.device)
        
        # Use subset for testing
        if test_data.shape[1] > n_test_frames:
            test_subset = test_data[:, :n_test_frames]
        else:
            test_subset = test_data
        
        # Reconstruct
        W_pinv = torch.linalg.pinv(self.W)
        H_estimated = W_pinv @ test_subset
        reconstructed = self.W @ H_estimated
        
        # Compute metrics
        mse = torch.mean((test_subset - reconstructed) ** 2).item()
        rmse = torch.sqrt(torch.tensor(mse)).item()
        
        # Signal-to-noise ratio
        signal_power = torch.mean(test_subset ** 2).item()
        noise_power = mse
        snr = 10 * torch.log10(torch.tensor(signal_power / (noise_power + self.epsilon))).item()
        
        # Sparsity of coefficients
        sparsity = (H_estimated == 0).float().mean().item()
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'snr_db': snr,
            'sparsity': sparsity,
            'n_atoms': self.W.shape[1],
            'n_freq': self.W.shape[0]
        }
        
        logger.info(f"Reconstruction quality: MSE={mse:.6f}, RMSE={rmse:.6f}, SNR={snr:.2f}dB")
        
        return metrics
