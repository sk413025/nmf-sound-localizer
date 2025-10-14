"""
Multi-modal prompt builder for In-Context Learning (ICL).

This module provides a flexible prompt building system that combines multiple
tokenizers (patch, atom, direction) into a unified prompt string, with support
for configurable ordering, weighting, and ICL context management.

Key classes:
- PromptConfig: Configuration for prompt building behavior
- MultiModalPromptBuilder: Main prompt builder combining multiple tokenizers
"""

import logging
import random
from dataclasses import dataclass
from typing import List, Optional, Literal
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PromptConfig:
    """Configuration for multi-modal prompt building.
    
    Attributes:
        use_patches: Include patch tokens in prompt
        use_atoms: Include NMF atom tokens in prompt
        use_directions: Include direction projection tokens in prompt
        ordering: Token ordering strategy
            - "physics_first": [Direction] → [Atom] → [Patch]
            - "structure_first": [Atom] → [Direction] → [Patch]
            - "patch_first": [Patch] → [Atom] → [Direction]
            - "interleaved": Mix tokens in alternating pattern
        max_tokens: Maximum total tokens in prompt (None = unlimited)
        max_patch_tokens: Maximum patch tokens (None = unlimited)
        max_atom_tokens: Maximum atom tokens (None = unlimited)
        max_direction_tokens: Maximum direction tokens (None = unlimited)
        atom_top_k: Number of top atoms to use (overrides tokenizer default)
        direction_top_m: Number of top directions to use (overrides tokenizer default)
    """
    use_patches: bool = True
    use_atoms: bool = True
    use_directions: bool = True
    ordering: Literal["physics_first", "structure_first", "patch_first", "interleaved"] = "physics_first"
    max_tokens: Optional[int] = None
    max_patch_tokens: Optional[int] = None
    max_atom_tokens: Optional[int] = None
    max_direction_tokens: Optional[int] = None
    atom_top_k: Optional[int] = None
    direction_top_m: Optional[int] = None


class MultiModalPromptBuilder:
    """Build multi-modal prompts by combining multiple tokenizers.
    
    This class orchestrates multiple tokenizers (patch, atom, direction) to create
    a unified prompt string. It supports flexible token ordering strategies and
    token budget management for efficiency.
    
    Example:
        >>> from doa_rl.features import PatchTokenizer, NMFAtomTokenizer, DirectionProjectionTokenizer
        >>> patch_tok = PatchTokenizer(Fp=16, Np=10, n_levels=16)
        >>> atom_tok = NMFAtomTokenizer(W, top_k=8)
        >>> dir_tok = DirectionProjectionTokenizer(H, angles, top_m=5)
        >>> config = PromptConfig(ordering="physics_first")
        >>> builder = MultiModalPromptBuilder(patch_tok, atom_tok, dir_tok, config)
        >>> Y = np.random.rand(116, 189)
        >>> prompt = builder.build_prompt(Y)
        >>> print(prompt)
        '<R_090:14> <R_085:12> <AT_5:12> <AT_23:8> <P_0_0_5> <P_0_1_8> ...'
    """
    
    def __init__(
        self,
        patch_tokenizer,
        atom_tokenizer=None,
        direction_tokenizer=None,
        config: Optional[PromptConfig] = None,
    ):
        """Initialize the prompt builder.
        
        Args:
            patch_tokenizer: PatchTokenizer instance (required)
            atom_tokenizer: NMFAtomTokenizer instance (optional)
            direction_tokenizer: DirectionProjectionTokenizer instance (optional)
            config: PromptConfig instance (defaults to PromptConfig())
        """
        self.patch_tok = patch_tokenizer
        self.atom_tok = atom_tokenizer
        self.dir_tok = direction_tokenizer
        self.config = config if config is not None else PromptConfig()
        
        # Validate configuration
        if self.config.use_atoms and self.atom_tok is None:
            logger.warning("use_atoms=True but atom_tokenizer is None, disabling atoms")
            self.config.use_atoms = False
        
        if self.config.use_directions and self.dir_tok is None:
            logger.warning("use_directions=True but direction_tokenizer is None, disabling directions")
            self.config.use_directions = False
        
        logger.info(
            "MultiModalPromptBuilder initialized: ordering=%s, use_patches=%s, use_atoms=%s, use_directions=%s",
            self.config.ordering,
            self.config.use_patches,
            self.config.use_atoms,
            self.config.use_directions,
        )
    
    def build_prompt(self, Y: np.ndarray) -> str:
        """Build a multi-modal prompt from spectrogram.
        
        Args:
            Y: Input spectrogram (F, N)
        
        Returns:
            Prompt string combining all enabled token types
        """
        # Generate tokens from each tokenizer
        patch_tokens = self._get_patch_tokens(Y) if self.config.use_patches else []
        atom_tokens = self._get_atom_tokens(Y) if self.config.use_atoms else []
        direction_tokens = self._get_direction_tokens(Y) if self.config.use_directions else []
        
        # Combine tokens according to ordering strategy
        if self.config.ordering == "physics_first":
            all_tokens = direction_tokens + atom_tokens + patch_tokens
        elif self.config.ordering == "structure_first":
            all_tokens = atom_tokens + direction_tokens + patch_tokens
        elif self.config.ordering == "patch_first":
            all_tokens = patch_tokens + atom_tokens + direction_tokens
        elif self.config.ordering == "interleaved":
            all_tokens = self._interleave_tokens(direction_tokens, atom_tokens, patch_tokens)
        else:
            raise ValueError(f"Unknown ordering strategy: {self.config.ordering}")
        
        # Apply global token limit if specified
        if self.config.max_tokens is not None:
            all_tokens = all_tokens[:self.config.max_tokens]
        
        prompt = " ".join(all_tokens)
        
        logger.debug(
            "build_prompt: Y shape=%s, tokens: dir=%d, atom=%d, patch=%d, total=%d",
            Y.shape,
            len(direction_tokens),
            len(atom_tokens),
            len(patch_tokens),
            len(all_tokens),
        )
        
        return prompt
    
    def _get_patch_tokens(self, Y: np.ndarray) -> List[str]:
        """Get patch tokens with optional limit."""
        tokens = self.patch_tok(Y)
        if self.config.max_patch_tokens is not None:
            tokens = tokens[:self.config.max_patch_tokens]
        return tokens
    
    def _get_atom_tokens(self, Y: np.ndarray) -> List[str]:
        """Get atom tokens with optional limit and top_k override."""
        # If top_k override is specified in config, we need to modify the call
        # Since NMFAtomTokenizer doesn't take top_k as argument, we apply limit after
        tokens = self.atom_tok(Y)
        
        # Apply config override for top_k (limit number of atoms)
        if self.config.atom_top_k is not None:
            tokens = tokens[:self.config.atom_top_k]
        
        # Apply max_atom_tokens limit
        if self.config.max_atom_tokens is not None:
            tokens = tokens[:self.config.max_atom_tokens]
        
        return tokens
    
    def _get_direction_tokens(self, Y: np.ndarray) -> List[str]:
        """Get direction tokens with optional limit and top_m override."""
        # DirectionProjectionTokenizer accepts top_m as parameter
        top_m = self.config.direction_top_m if self.config.direction_top_m is not None else 5
        tokens = self.dir_tok(Y, top_m=top_m)
        
        # Apply max_direction_tokens limit
        if self.config.max_direction_tokens is not None:
            tokens = tokens[:self.config.max_direction_tokens]
        
        return tokens
    
    def _interleave_tokens(
        self,
        direction_tokens: List[str],
        atom_tokens: List[str],
        patch_tokens: List[str],
    ) -> List[str]:
        """Interleave tokens from different sources.
        
        Strategy: Cycle through non-empty token lists, taking one token at a time
        from each until all are exhausted.
        """
        result = []
        max_len = max(len(direction_tokens), len(atom_tokens), len(patch_tokens))
        
        for i in range(max_len):
            if i < len(direction_tokens):
                result.append(direction_tokens[i])
            if i < len(atom_tokens):
                result.append(atom_tokens[i])
            if i < len(patch_tokens):
                result.append(patch_tokens[i])
        
        return result
    
    def build_icl_prompt(
        self,
        query_Y: np.ndarray,
        context_examples: List[tuple],
        target_token: Optional[str] = None,
    ) -> str:
        """Build an In-Context Learning prompt with few-shot examples.
        
        Format:
            [BOS] <context_1> <target_1> <context_2> <target_2> ... <query> [<target>]
        
        Args:
            query_Y: Query spectrogram (F, N)
            context_examples: List of (Y, target_token) tuples for few-shot context
            target_token: Target token for query (optional, for teacher forcing)
        
        Returns:
            ICL prompt string with context examples and query
        """
        prompt_parts = []
        
        # Add context examples
        for ctx_Y, ctx_target in context_examples:
            ctx_prompt = self.build_prompt(ctx_Y)
            prompt_parts.append(ctx_prompt)
            prompt_parts.append(ctx_target)
        
        # Add query
        query_prompt = self.build_prompt(query_Y)
        prompt_parts.append(query_prompt)
        
        # Optionally add target for query
        if target_token is not None:
            prompt_parts.append(target_token)
        
        return " ".join(prompt_parts)
    
    def sample_context_examples(
        self,
        dataset,
        query_idx: int,
        n_shots: int = 3,
        strategy: Literal["random", "nearest_angle", "diverse_angle"] = "random",
        exclude_same_angle: bool = True,
    ) -> List[tuple]:
        """Sample context examples from dataset for ICL.
        
        Args:
            dataset: DoADataset or similar with __getitem__ returning {"Y": ..., "angle_deg": ...}
            query_idx: Index of the query sample (to avoid including it in context)
            n_shots: Number of context examples to sample
            strategy: Sampling strategy
                - "random": Uniformly random from dataset
                - "nearest_angle": Select closest angles to query
                - "diverse_angle": Maximize angle diversity in context
            exclude_same_angle: If True, exclude examples with same angle as query
        
        Returns:
            List of (Y, target_token) tuples
        """
        query_sample = dataset[query_idx]
        query_angle = query_sample.get("angle_deg")
        
        # Get all candidate indices
        candidates = list(range(len(dataset)))
        candidates.remove(query_idx)  # Don't include query itself
        
        # Filter by angle if requested
        if exclude_same_angle and query_angle is not None:
            candidates = [
                idx for idx in candidates
                if dataset[idx].get("angle_deg") != query_angle
            ]
        
        # Apply sampling strategy
        if strategy == "random":
            selected_indices = random.sample(candidates, min(n_shots, len(candidates)))
        
        elif strategy == "nearest_angle":
            if query_angle is None:
                # Fall back to random if no angle info
                selected_indices = random.sample(candidates, min(n_shots, len(candidates)))
            else:
                # Sort by angle distance
                angle_distances = [
                    (idx, abs(dataset[idx].get("angle_deg", float('inf')) - query_angle))
                    for idx in candidates
                ]
                angle_distances.sort(key=lambda x: x[1])
                selected_indices = [idx for idx, _ in angle_distances[:n_shots]]
        
        elif strategy == "diverse_angle":
            # Greedy selection to maximize minimum angle distance
            selected_indices = []
            remaining = candidates.copy()
            
            # Start with a random sample
            if remaining:
                first_idx = random.choice(remaining)
                selected_indices.append(first_idx)
                remaining.remove(first_idx)
            
            # Greedily add samples that maximize min distance to selected
            while len(selected_indices) < n_shots and remaining:
                best_idx = None
                best_min_dist = -1
                
                for idx in remaining:
                    idx_angle = dataset[idx].get("angle_deg", 0)
                    min_dist = min(
                        abs(idx_angle - dataset[sel_idx].get("angle_deg", 0))
                        for sel_idx in selected_indices
                    )
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_idx = idx
                
                if best_idx is not None:
                    selected_indices.append(best_idx)
                    remaining.remove(best_idx)
        
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        
        # Build context examples
        context_examples = []
        for idx in selected_indices:
            sample = dataset[idx]
            Y = sample["Y"]
            if isinstance(Y, np.ndarray):
                Y_np = Y
            else:
                # Assume torch tensor
                Y_np = Y.numpy()
            
            # Create target token (direction token)
            angle = sample.get("angle_deg")
            if angle is not None:
                target_token = f"<D_{int(angle):03d}>"
            else:
                target_token = "<D_UNK>"
            
            context_examples.append((Y_np, target_token))
        
        logger.debug(
            "sample_context_examples: query_idx=%d, n_shots=%d, strategy=%s, selected=%s",
            query_idx, n_shots, strategy, selected_indices
        )
        
        return context_examples
