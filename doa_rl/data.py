import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from nmf_localizer.utils.audio_utils import AudioProcessor


logger = logging.getLogger(__name__)


class DoADataset(Dataset):
    """Dataset reading normalized Box test clips and producing Y (F,N).

    Matches docs/rl_data_spec.md (Config C): yields dict with
    - Y: (F,N) float32 magnitude spectrogram (band 300–3000 Hz)
    - angle_deg: float
    - angle_index: int (index into angles array)
    - path: str
    """

    def __init__(self, root: str, angles: List[float], fs: int = 16000,
                 n_fft: int = 2048, window: str = "hann",
                 freq_min: float = 300.0, freq_max: float = 3000.0):
        self.root = Path(root)
        self.angles = [float(a) for a in angles]
        self.fs = fs
        self.n_fft = n_fft
        self.window = window
        self.freq_min = freq_min
        self.freq_max = freq_max

        self.index: List[Tuple[Path, float, int]] = []
        for angle_deg in self.angles:
            angle_dir = self.root / f"angle_{int(angle_deg)}"
            if not angle_dir.exists():
                continue
            for f in sorted(angle_dir.glob("*.npy")):
                self.index.append((f, angle_deg, int(angle_deg)))

    def __len__(self) -> int:
        size = len(self.index)
        logger.info("DoADataset.__len__: %d samples in %s", size, self.root)
        return size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path, angle_deg, _ = self.index[idx]
        wav = np.load(path)
        logger.info("DoADataset.__getitem__: idx=%d wav_path=%s wav_shape=%s", idx, path, wav.shape)
        assert wav.ndim == 1, "Expected mono waveform"
        freqs, times, stft, magnitude = AudioProcessor.compute_stft_spectrogram(
            wav, fs=self.fs, nperseg=self.n_fft, window=self.window
        )
        mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
        mag_band = magnitude[mask, :].astype(np.float32)
        Y = torch.from_numpy(mag_band)
        logger.info("DoADataset.__getitem__: spectrogram shape=%s (masked from %s)",
                     Y.shape, magnitude.shape)
        return {
            "Y": Y,
            "angle_deg": float(angle_deg),
            "angle_index": self.angles.index(float(angle_deg)),
            "path": str(path)
        }


class DoAICLDataset(DoADataset):
    """Dataset with In-Context Learning (ICL) multi-modal prompt support.
    
    Extends DoADataset to generate multi-modal prompts combining:
    - Patch tokens (fine-grained spectral details)
    - NMF Atom tokens (structural/phonetic components)
    - Direction Projection tokens (physical direction prior)
    
    Supports two modes:
    1. Basic mode: Single prompt per sample
    2. ICL mode: Few-shot prompts with context examples
    
    Args:
        root: Path to data directory (angle_* subdirs with .npy clips)
        angles: List of direction angles (e.g., [0, 5, 10, ..., 180])
        prompt_builder: MultiModalPromptBuilder instance for prompt generation
        icl_mode: Enable ICL few-shot prompts (default: False)
        n_shots: Number of context examples for ICL (default: 3)
        context_strategy: ICL sampling strategy - "random", "nearest", "diverse" (default: "random")
        exclude_query_angle: Exclude query angle from ICL context (default: True)
        **kwargs: Additional arguments passed to DoADataset
    
    Output batch dict contains:
        - Y: (F, N) spectrogram tensor
        - angle_deg: float angle in degrees
        - angle_index: int index into angles list
        - path: str path to .npy file
        - prompt: str multi-modal prompt (basic or ICL format)
        - (ICL mode only) context_indices: List[int] indices of context examples
    
    Example:
        >>> from doa_rl.features import MultiModalPromptBuilder, PromptConfig
        >>> builder = MultiModalPromptBuilder(patch_tok, atom_tok, dir_tok, config)
        >>> dataset = DoAICLDataset(
        ...     root="data/angle_*",
        ...     angles=[0, 5, 10, ..., 180],
        ...     prompt_builder=builder,
        ...     icl_mode=True,
        ...     n_shots=3
        ... )
        >>> batch = dataset[0]
        >>> print(batch["prompt"])
        # ICL format: [ctx1_prompt] <D_030> [ctx2_prompt] <D_060> [query_prompt]
    """
    
    def __init__(
        self,
        root: str,
        angles: List[float],
        prompt_builder: Any,  # MultiModalPromptBuilder
        icl_mode: bool = False,
        n_shots: int = 3,
        context_strategy: str = "random",
        exclude_query_angle: bool = True,
        **kwargs
    ):
        super().__init__(root, angles, **kwargs)
        self.prompt_builder = prompt_builder
        self.icl_mode = icl_mode
        self.n_shots = n_shots
        self.context_strategy = context_strategy
        self.exclude_query_angle = exclude_query_angle
        
        logger.info(
            "DoAICLDataset initialized: root=%s, angles=%d, icl_mode=%s, n_shots=%d, strategy=%s",
            root, len(angles), icl_mode, n_shots, context_strategy
        )
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample with multi-modal prompt.
        
        Returns:
            Dict with Y, angle_deg, angle_index, path, and prompt.
            If ICL mode, also includes context_indices.
        """
        # Get base data from parent class
        batch = super().__getitem__(idx)
        Y = batch["Y"]
        angle_deg = batch["angle_deg"]
        
        # Generate prompt based on mode
        if self.icl_mode:
            # Build ICL prompt with context examples
            prompt, context_indices = self._build_icl_prompt(idx, Y.numpy(), angle_deg)
            batch["prompt"] = prompt
            batch["context_indices"] = context_indices
        else:
            # Build single multi-modal prompt
            prompt = self.prompt_builder.build_prompt(Y.numpy())
            batch["prompt"] = prompt
        
        logger.debug(
            "DoAICLDataset.__getitem__: idx=%d, angle=%.1f, prompt_len=%d tokens, icl=%s",
            idx, angle_deg, len(prompt.split()), self.icl_mode
        )
        
        return batch
    
    def _build_icl_prompt(
        self, 
        query_idx: int, 
        query_Y: np.ndarray, 
        query_angle: float
    ) -> Tuple[str, List[int]]:
        """Build In-Context Learning prompt with few-shot examples.
        
        Format: [ctx1_prompt] <D_angle1> [ctx2_prompt] <D_angle2> ... [query_prompt]
        
        Args:
            query_idx: Index of query sample
            query_Y: Query spectrogram (F, N)
            query_angle: Query angle in degrees
        
        Returns:
            Tuple of (icl_prompt_string, context_indices_list)
        """
        # Sample context examples
        context_indices = self._sample_context_examples(query_idx, query_angle)
        
        # Build ICL prompt parts
        prompt_parts = []
        
        for ctx_idx in context_indices:
            # Get context sample
            ctx_batch = super().__getitem__(ctx_idx)
            ctx_Y = ctx_batch["Y"].numpy()
            ctx_angle = ctx_batch["angle_deg"]
            
            # Generate prompt for context
            ctx_prompt = self.prompt_builder.build_prompt(ctx_Y)
            
            # Format: [prompt] <D_angle>
            angle_int = int(ctx_angle)
            direction_token = f"<D_{angle_int:03d}>"
            prompt_parts.append(f"{ctx_prompt} {direction_token}")
        
        # Add query prompt (no direction token)
        query_prompt = self.prompt_builder.build_prompt(query_Y)
        prompt_parts.append(query_prompt)
        
        # Join all parts
        icl_prompt = " ".join(prompt_parts)
        
        return icl_prompt, context_indices
    
    def _sample_context_examples(
        self, 
        query_idx: int, 
        query_angle: float
    ) -> List[int]:
        """Sample context examples for ICL using specified strategy.
        
        Args:
            query_idx: Index of query sample
            query_angle: Query angle in degrees
        
        Returns:
            List of indices for context examples
        """
        # Build candidate pool (exclude query sample)
        candidates = [i for i in range(len(self)) if i != query_idx]
        
        # Optionally exclude same angle as query
        if self.exclude_query_angle:
            candidates = [
                i for i in candidates 
                if self.index[i][1] != query_angle
            ]
        
        # Check we have enough candidates
        if len(candidates) < self.n_shots:
            logger.warning(
                "Not enough candidates (%d) for n_shots=%d, using all available",
                len(candidates), self.n_shots
            )
            return candidates[:self.n_shots]
        
        # Apply sampling strategy
        if self.context_strategy == "random":
            return random.sample(candidates, self.n_shots)
        
        elif self.context_strategy == "nearest":
            # Select samples with closest angles to query
            angle_dists = [
                (i, abs(self.index[i][1] - query_angle))
                for i in candidates
            ]
            angle_dists.sort(key=lambda x: x[1])
            return [idx for idx, _ in angle_dists[:self.n_shots]]
        
        elif self.context_strategy == "diverse":
            # Maximize minimum pairwise angle distance
            selected = [random.choice(candidates)]
            remaining = [i for i in candidates if i not in selected]
            
            for _ in range(self.n_shots - 1):
                # For each candidate, compute min distance to selected
                best_idx = None
                best_min_dist = -1
                
                for cand_idx in remaining:
                    cand_angle = self.index[cand_idx][1]
                    min_dist = min(
                        abs(cand_angle - self.index[sel_idx][1])
                        for sel_idx in selected
                    )
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_idx = cand_idx
                
                if best_idx is not None:
                    selected.append(best_idx)
                    remaining.remove(best_idx)
            
            return selected
        
        else:
            raise ValueError(
                f"Unknown context_strategy: {self.context_strategy}. "
                f"Choose from: random, nearest, diverse"
            )


def create_dataloader(dataset: DoADataset, batch_size: int = 1, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
