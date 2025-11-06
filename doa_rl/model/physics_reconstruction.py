#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics Reconstruction Head for DTMin

Purpose: Enforce physical interpretability in token embeddings by requiring
         them to reconstruct physical quantities (residuals, directions).

Key idea: In addition to predicting actions, the model must also:
  1. Reconstruct the next residual r_{t+1} from current token embedding
  2. Identify the dominant direction (expert) in current residual r_t
  3. Maintain spectral coherence between predicted and true residuals

This encourages token embeddings to capture physically meaningful features
rather than just memorizing trajectory patterns.

Design Principles (AGENTS.md):
- First principles: Residual r_t is defined by acoustic physics (Y - H·W·ŝ)
- No fallbacks: All quantities must be physically valid (no NaN, finite)
- Real data only: Reconstruction targets come from actual trajectories
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsReconstructionHead(nn.Module):
    """
    Multi-task physics reconstruction head.
    
    Tasks:
      1. Residual prediction: Given H_t, predict r_{t+1}
      2. Direction classification: Identify dominant expert in r_t
      3. Spectral coherence: Ensure predictions maintain spectral structure
    
    Architecture:
      H_t (B, K, d_model) → [Residual Decoder] → r̂_{t+1} (B, K-1, F)
                          → [Direction Decoder] → θ̂_t (B, K, E)
    
    Loss components:
      L_physics = w_r * L_residual + w_d * L_direction + w_c * L_coherence
    
    Args:
        d_model: Hidden dimension of token embeddings
        F: Frequency bins (STFT dimension)
        E: Number of experts (directions)
        reconstruction_weight: Overall weight for physics loss
        residual_weight: Weight for residual reconstruction (default: 1.0)
        direction_weight: Weight for direction classification (default: 0.5)
        coherence_weight: Weight for spectral coherence (default: 0.1)
    """
    
    def __init__(
        self,
        d_model: int,
        F: int,
        E: int,
        reconstruction_weight: float = 0.1,
        residual_weight: float = 1.0,
        direction_weight: float = 0.5,
        coherence_weight: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.F = F
        self.E = E
        self.reconstruction_weight = reconstruction_weight
        self.w_r = residual_weight
        self.w_d = direction_weight
        self.w_c = coherence_weight
        
        # Task 1: Residual decoder (next-step prediction)
        # Input: H_t → Output: r_{t+1}
        self.residual_decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, F)
        )
        
        # Task 2: Direction decoder (current expert identification)
        # Input: H_t → Output: expert logits
        self.direction_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, E)
        )
        
        # Task 3: Coherence monitoring (no trainable params, just diagnostic)
        # Computed in forward pass
    
    def forward(
        self,
        H_t: torch.Tensor,
        r_seq: torch.Tensor,
        expert_gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass with physics reconstruction.
        
        Args:
            H_t: (B, K, d_model) - Token embeddings from Transformer
            r_seq: (B, K, F) - Ground truth residual sequence
            expert_gt: (B, K) - Ground truth expert labels
            mask: (B, K) - Optional mask for valid timesteps
        
        Returns:
            loss_physics: Scalar physics reconstruction loss
            metrics: Dict with diagnostic metrics
                - 'loss_residual': Residual prediction MSE
                - 'loss_direction': Direction classification CE
                - 'coherence_mean': Mean spectral coherence
                - 'residual_mse_db': Residual error in dB
                - 'direction_acc': Direction classification accuracy
        """
        B, K, _ = H_t.shape
        
        # ===================================================================
        # Task 1: Residual Prediction (predict r_{t+1} from H_t)
        # ===================================================================
        
        # Decode next residual for t=0..K-2
        r_pred = self.residual_decoder(H_t[:, :-1])  # (B, K-1, F)
        r_target = r_seq[:, 1:]  # (B, K-1, F) - next step ground truth
        
        # MSE loss in linear scale
        if mask is not None:
            # Apply mask to valid timesteps only
            mask_valid = mask[:, :-1]  # (B, K-1)
            loss_residual = (
                ((r_pred - r_target) ** 2).mean(dim=-1) * mask_valid
            ).sum() / (mask_valid.sum() + 1e-8)
        else:
            loss_residual = F.mse_loss(r_pred, r_target)
        
        # Diagnostic: MSE in dB scale
        with torch.no_grad():
            mse_linear = ((r_pred - r_target) ** 2).mean()
            residual_mse_db = 10 * torch.log10(mse_linear + 1e-12)
        
        # ===================================================================
        # Task 2: Direction Classification (identify expert in r_t)
        # ===================================================================
        
        # Classify current expert for all t=0..K-1
        direction_logits = self.direction_decoder(H_t)  # (B, K, E)
        
        # Cross-entropy loss
        if mask is not None:
            # Masked CE loss
            loss_direction_per_step = F.cross_entropy(
                direction_logits.view(-1, self.E),
                expert_gt.view(-1),
                reduction='none'
            ).view(B, K)
            loss_direction = (
                loss_direction_per_step * mask
            ).sum() / (mask.sum() + 1e-8)
        else:
            loss_direction = F.cross_entropy(
                direction_logits.view(-1, self.E),
                expert_gt.view(-1)
            )
        
        # Diagnostic: classification accuracy
        with torch.no_grad():
            pred_experts = direction_logits.argmax(dim=-1)  # (B, K)
            if mask is not None:
                correct = ((pred_experts == expert_gt) * mask).sum()
                total = mask.sum()
            else:
                correct = (pred_experts == expert_gt).sum()
                total = B * K
            direction_acc = correct.float() / (total + 1e-8)
        
        # ===================================================================
        # Task 3: Spectral Coherence (frequency structure preservation)
        # ===================================================================
        
        # Compute coherence between predicted and target residuals
        # This ensures predictions maintain physical spectral structure
        coherence = self._compute_spectral_coherence(r_pred, r_target)
        
        # Coherence loss: maximize mean coherence (minimize negative)
        loss_coherence = -coherence.mean()
        
        # ===================================================================
        # Combined Physics Loss
        # ===================================================================
        
        loss_physics = (
            self.w_r * loss_residual +
            self.w_d * loss_direction +
            self.w_c * loss_coherence
        ) * self.reconstruction_weight
        
        # ===================================================================
        # Diagnostic Metrics
        # ===================================================================
        
        metrics = {
            'loss_residual': float(loss_residual.item()),
            'loss_direction': float(loss_direction.item()),
            'loss_coherence': float(loss_coherence.item()),
            'coherence_mean': float(coherence.mean().item()),
            'residual_mse_db': float(residual_mse_db.item()),
            'direction_acc': float(direction_acc.item()),
        }
        
        return loss_physics, metrics
    
    def _compute_spectral_coherence(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spectral coherence γ²(f) between predicted and target residuals.
        
        Coherence definition (from acoustic physics):
            γ²(f) = |S_xy(f)|² / (S_xx(f) · S_yy(f))
        
        Where:
            S_xy = Cross-spectral density
            S_xx = Auto-spectral density of prediction
            S_yy = Auto-spectral density of target
        
        Args:
            pred: (B, K, F) - Predicted residuals
            target: (B, K, F) - Target residuals
        
        Returns:
            coherence: (B, K) - Mean coherence across frequencies
        
        Physical interpretation:
            γ² ≈ 1.0: Predictions preserve spectral structure (good)
            γ² ≈ 0.0: Predictions decorrelated from target (bad)
        """
        # FFT along frequency axis
        P = torch.fft.rfft(pred, dim=-1)  # (B, K, F//2+1)
        T = torch.fft.rfft(target, dim=-1)
        
        # Cross-spectral density
        S_xy = P * T.conj()
        
        # Auto-spectral densities
        S_xx = P * P.conj()
        S_yy = T * T.conj()
        
        # Coherence (magnitude squared)
        gamma_sq = (S_xy.abs() ** 2) / (S_xx.real * S_yy.real + 1e-12)
        
        # Average across frequencies
        coherence = gamma_sq.mean(dim=-1)  # (B, K)
        
        return coherence
    
    def predict_next_residual(
        self,
        H_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Inference mode: predict next residual from current token embedding.
        
        Args:
            H_t: (B, d_model) or (B, 1, d_model) - Current token embedding
        
        Returns:
            r_next: (B, F) - Predicted next residual
        """
        if H_t.dim() == 2:
            H_t = H_t.unsqueeze(1)  # (B, 1, d_model)
        
        r_next = self.residual_decoder(H_t).squeeze(1)  # (B, F)
        return r_next
    
    def identify_direction(
        self,
        H_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference mode: identify dominant expert from token embedding.
        
        Args:
            H_t: (B, d_model) or (B, 1, d_model) - Current token embedding
        
        Returns:
            expert_pred: (B,) - Predicted expert index
            expert_probs: (B, E) - Expert probability distribution
        """
        if H_t.dim() == 2:
            H_t = H_t.unsqueeze(1)  # (B, 1, d_model)
        
        logits = self.direction_decoder(H_t).squeeze(1)  # (B, E)
        expert_probs = F.softmax(logits, dim=-1)
        expert_pred = logits.argmax(dim=-1)
        
        return expert_pred, expert_probs


# =====================================================================
# Warmup Scheduler for Physics Loss
# =====================================================================

class PhysicsLossScheduler:
    """
    Gradually increase physics loss weight during training warmup.
    
    Strategy:
        - Early epochs: Focus on physics reconstruction (high weight)
        - Later epochs: Balance action prediction and physics (lower weight)
    
    This encourages learning physically meaningful representations first.
    """
    
    def __init__(
        self,
        warmup_epochs: int = 50,
        max_weight: float = 0.7,
        min_weight: float = 0.1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_weight = max_weight
        self.min_weight = min_weight
    
    def get_weight(self, epoch: int) -> float:
        """
        Get physics loss weight for current epoch.
        
        Schedule:
            epoch < warmup_epochs: max_weight (focus on physics)
            epoch >= warmup_epochs: min_weight (focus on action)
        """
        if epoch < self.warmup_epochs:
            return self.max_weight
        else:
            return self.min_weight
    
    def get_action_weight(self, epoch: int) -> float:
        """
        Get action loss weight (complementary to physics weight).
        """
        physics_weight = self.get_weight(epoch)
        return 1.0 - physics_weight
