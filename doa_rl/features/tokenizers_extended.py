"""
Extended tokenizers for multi-modal In-Context Learning (ICL).

This module provides advanced tokenizers that encode physical and structural
information from spectrograms:
- NMFAtomTokenizer: Decomposes spectrogram into NMF atoms (phonetic/resonance structure)
- DirectionProjectionTokenizer: Computes direction-wise correlations (physical prior)

Token formats:
- NMF Atom: <AT_atom_id:level> where atom_id ∈ [0, K-1], level ∈ [0, 15]
- Direction: <R_angle:level> where angle ∈ {000, 005, ..., 180}, level ∈ [0, 15]
"""

import logging
import numpy as np
from typing import List, Optional, Literal

from .nmf_utils import estimate_z_is

logger = logging.getLogger(__name__)


class NMFAtomTokenizer:
    """Generate tokens from NMF atom activations.
    
    This tokenizer decomposes the input spectrogram using a pre-trained NMF
    dictionary W to extract atom activations z, then generates tokens
    representing the top-k most active atoms and their intensity levels.
    
    Token format: <AT_atom_id:level>
    - atom_id: Index of the NMF atom (0 to K-1)
    - level: Quantized activation strength (0 to n_levels-1)
    
    Example:
        >>> W = np.random.rand(116, 64)  # F=116, K=64
        >>> tokenizer = NMFAtomTokenizer(W, top_k=5, n_levels=16)
        >>> Y = np.random.rand(116, 189)
        >>> tokens = tokenizer(Y)
        >>> print(tokens)
        ['<AT_5:14>', '<AT_23:12>', '<AT_41:10>', '<AT_8:9>', '<AT_15:7>']
    """
    
    def __init__(
        self,
        W: np.ndarray,
        top_k: int = 8,
        n_levels: int = 16,
        n_iter: int = 50,
        l1: float = 0.0,
    ):
        """Initialize NMF Atom Tokenizer.
        
        Args:
            W: NMF dictionary matrix (F, K) where F is frequency bins, K is atoms
            top_k: Number of top atoms to select (by activation strength)
            n_levels: Number of quantization levels for activation strength
            n_iter: Number of IS-MU iterations for z estimation
            l1: L1 regularization weight for z estimation
        """
        if W.ndim != 2:
            raise ValueError(f"W must be 2D (F, K), got shape {W.shape}")
        
        self.W = W
        self.K = W.shape[1]
        self.top_k = min(top_k, self.K)  # Can't select more than available atoms
        self.n_levels = n_levels
        self.n_iter = n_iter
        self.l1 = l1
        
        logger.info(
            "NMFAtomTokenizer initialized: K=%d, top_k=%d, n_levels=%d, n_iter=%d",
            self.K, self.top_k, self.n_levels, self.n_iter
        )
    
    def __call__(self, Y: np.ndarray) -> List[str]:
        """Generate NMF atom tokens from spectrogram.
        
        Args:
            Y: Input spectrogram (F, N) where F is frequency bins, N is time frames
        
        Returns:
            List of atom tokens, e.g., ['<AT_5:14>', '<AT_23:12>', ...]
        """
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2D (F, N), got shape {Y.shape}")
        
        if Y.shape[0] != self.W.shape[0]:
            raise ValueError(
                f"Y frequency dimension {Y.shape[0]} doesn't match W {self.W.shape[0]}"
            )
        
        # Average spectrogram over time to get mean spectrum
        Ybar = Y.mean(axis=1)  # (F,)
        
        # Estimate atom activations using IS-MU
        z = estimate_z_is(Ybar, self.W, n_iter=self.n_iter, l1=self.l1)  # (K,)
        
        # Select top-k atoms by activation strength
        top_indices = np.argsort(-z)[:self.top_k]
        
        tokens = []
        for idx in top_indices:
            # Quantize activation to [0, n_levels-1]
            # Scale z values (typically in [0, 1] after normalization) to level range
            # Use 2*K as a heuristic upper bound for unnormalized z before normalization
            level = int(np.clip(z[idx] * self.K * 2, 0, self.n_levels - 1))
            tokens.append(f"<AT_{idx}:{level}>")
        
        logger.debug(
            "NMFAtomTokenizer: Y shape=%s, z range=[%.4f, %.4f], emitted %d tokens",
            Y.shape, z.min(), z.max(), len(tokens)
        )
        
        return tokens


class DirectionProjectionTokenizer:
    """Generate tokens from direction transfer function projections.
    
    This tokenizer computes the similarity between the input spectrogram and
    direction-specific transfer functions H, generating tokens that encode
    which directions have the strongest correlation with the input.
    
    Token format: <R_angle:level>
    - angle: Direction angle in degrees (000, 005, 010, ..., 180)
    - level: Quantized similarity score (0 to n_levels-1)
    
    Example:
        >>> H = np.random.rand(116, 37)  # F=116, D=37 (0° to 180° in 5° steps)
        >>> angles = list(range(0, 181, 5))
        >>> tokenizer = DirectionProjectionTokenizer(H, angles, top_m=3)
        >>> Y = np.random.rand(116, 189)
        >>> tokens = tokenizer(Y)
        >>> print(tokens)
        ['<R_090:15>', '<R_085:14>', '<R_095:13>']
    """
    
    def __init__(
        self,
        H: np.ndarray,
        angles: List[int],
        top_m: int = 5,
        n_levels: int = 16,
        metric: Literal["correlation", "is_divergence"] = "correlation",
    ):
        """Initialize Direction Projection Tokenizer.
        
        Args:
            H: Transfer function matrix (F, D) where F is frequency bins, D is directions
            angles: List of direction angles in degrees corresponding to H columns
            top_m: Number of top directions to select
            n_levels: Number of quantization levels for similarity scores
            metric: Similarity metric - 'correlation' or 'is_divergence'
        """
        if H.ndim != 2:
            raise ValueError(f"H must be 2D (F, D), got shape {H.shape}")
        
        if len(angles) != H.shape[1]:
            raise ValueError(
                f"Number of angles {len(angles)} doesn't match H columns {H.shape[1]}"
            )
        
        self.H = H
        self.angles = angles
        self.D = H.shape[1]
        self.top_m = min(top_m, self.D)
        self.n_levels = n_levels
        self.metric = metric
        
        logger.info(
            "DirectionProjectionTokenizer initialized: D=%d, top_m=%d, n_levels=%d, metric=%s",
            self.D, self.top_m, self.n_levels, self.metric
        )
    
    def __call__(self, Y: np.ndarray, top_m: Optional[int] = None) -> List[str]:
        """Generate direction projection tokens from spectrogram.
        
        Args:
            Y: Input spectrogram (F, N) where F is frequency bins, N is time frames
            top_m: Override the default top_m (optional)
        
        Returns:
            List of direction tokens, e.g., ['<R_090:15>', '<R_085:14>', ...]
        """
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2D (F, N), got shape {Y.shape}")
        
        if Y.shape[0] != self.H.shape[0]:
            raise ValueError(
                f"Y frequency dimension {Y.shape[0]} doesn't match H {self.H.shape[0]}"
            )
        
        # Average spectrogram over time
        Ybar = Y.mean(axis=1)  # (F,)
        
        # Compute similarity scores for each direction
        scores = []
        for d in range(self.D):
            H_d = self.H[:, d]
            
            if self.metric == "correlation":
                # Normalized correlation (cosine similarity)
                numerator = np.dot(Ybar, H_d)
                denominator = (
                    np.linalg.norm(Ybar) * np.linalg.norm(H_d) + 1e-12
                )
                score = numerator / denominator
            
            elif self.metric == "is_divergence":
                # Negative IS divergence (higher is better)
                # IS divergence: D_IS(Y||H) = sum(Y/H - log(Y/H) - 1)
                ratio = Ybar / (H_d + 1e-12)
                divergence = np.sum(ratio - np.log(ratio + 1e-12) - 1.0)
                score = -divergence  # Negate so higher is better
            
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
            
            scores.append((d, score))
        
        # Sort by score (descending) and select top-m
        scores.sort(key=lambda x: -x[1])
        top_m_use = top_m if top_m is not None else self.top_m
        top_dirs = scores[:top_m_use]
        
        # Generate tokens
        tokens = []
        for dir_idx, score in top_dirs:
            # Quantize score to [0, n_levels-1]
            # For correlation: score ∈ [-1, 1], map to [0, n_levels-1]
            # For IS divergence: score can be any value, use adaptive scaling
            if self.metric == "correlation":
                # Map [-1, 1] -> [0, n_levels-1]
                level = int(np.clip((score + 1) / 2 * (self.n_levels - 1), 0, self.n_levels - 1))
            else:
                # For IS divergence, use min-max normalization across all scores
                all_scores = [s[1] for s in scores]
                score_min, score_max = min(all_scores), max(all_scores)
                if score_max > score_min:
                    normalized = (score - score_min) / (score_max - score_min)
                else:
                    normalized = 0.5
                level = int(np.clip(normalized * (self.n_levels - 1), 0, self.n_levels - 1))
            
            angle = self.angles[dir_idx]
            # Convert to int (handle both tensor and numpy/python types)
            angle_int = int(angle.item() if hasattr(angle, 'item') else angle)
            tokens.append(f"<R_{angle_int:03d}:{level}>")
        
        logger.debug(
            "DirectionProjectionTokenizer: Y shape=%s, score range=[%.4f, %.4f], emitted %d tokens",
            Y.shape, min(s[1] for s in scores), max(s[1] for s in scores), len(tokens)
        )
        
        return tokens


__all__ = [
    "NMFAtomTokenizer",
    "DirectionProjectionTokenizer",
]
