"""
Audio processing utilities for NMF localization.
"""

import numpy as np
import torch
from scipy import signal
from typing import Tuple, Optional


class AudioProcessor:
    """Audio processing utilities."""
    
    @staticmethod
    def compute_stft_spectrogram(
        waveform: np.ndarray, 
        fs: int = 16000, 
        nperseg: int = 2048,
        noverlap: Optional[int] = None, 
        window: str = 'hann'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram using STFT to preserve temporal information.
        
        Args:
            waveform: Input signal
            fs: Sampling frequency
            nperseg: Length of each segment
            noverlap: Number of points to overlap (default: nperseg * 3/4)
            window: Window function
        
        Returns:
            freqs: Frequency array
            times: Time array
            spectrogram: Complex spectrogram (freq x time)
            magnitude: Magnitude spectrogram (freq x time)
        """
        if noverlap is None:
            noverlap = int(nperseg * 0.75)  # 75% overlap for better time resolution
        
        # Remove DC offset
        waveform = waveform - np.mean(waveform)
        
        # Compute STFT
        freqs, times, stft = signal.stft(
            waveform, fs=fs, window=window, 
            nperseg=nperseg, noverlap=noverlap,
            detrend='constant'
        )
        
        # Get magnitude spectrogram
        magnitude = np.abs(stft)
        
        return freqs, times, stft, magnitude
    
    @staticmethod
    def apply_frequency_filter(
        spectrogram: torch.Tensor,
        freqs: np.ndarray, 
        freq_min: float,
        freq_max: float
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Apply frequency band filtering to spectrogram.
        
        Args:
            spectrogram: Input spectrogram (F, T) or (F, D)
            freqs: Frequency array
            freq_min: Minimum frequency
            freq_max: Maximum frequency
            
        Returns:
            Filtered spectrogram
            Filtered frequency array
        """
        # Find frequency indices
        idx_min = np.argmin(np.abs(freqs - freq_min))
        idx_max = np.argmin(np.abs(freqs - freq_max))
        
        # Apply filtering
        filtered_spec = spectrogram[idx_min:idx_max+1, :]
        filtered_freqs = freqs[idx_min:idx_max+1]
        
        return filtered_spec, filtered_freqs
    
    @staticmethod
    def normalize_spectrogram(
        spectrogram: torch.Tensor,
        method: str = 'none'
    ) -> torch.Tensor:
        """
        Normalize spectrogram.
        
        Args:
            spectrogram: Input spectrogram
            method: Normalization method ('none', 'global', 'per_freq')
            
        Returns:
            Normalized spectrogram
        """
        if method == 'none':
            return spectrogram
        elif method == 'global':
            return spectrogram / (spectrogram.max() + 1e-8)
        elif method == 'per_freq':
            # Normalize per frequency bin
            norm_spec = spectrogram.clone()
            for f in range(spectrogram.shape[0]):
                max_val = spectrogram[f, :].max()
                if max_val > 0:
                    norm_spec[f, :] = spectrogram[f, :] / max_val
            return norm_spec
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod 
    def compute_magnitude_response(
        waveform: np.ndarray,
        fs: int = 16000,
        method: str = 'stft'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute magnitude response from waveform.
        
        Args:
            waveform: Input waveform
            fs: Sample rate
            method: Method ('stft' or 'spectrogram')
            
        Returns:
            freqs: Frequency array
            magnitude: Magnitude response
        """
        if method == 'stft':
            # Use STFT method
            freqs, times, stft, magnitude = AudioProcessor.compute_stft_spectrogram(
                waveform, fs=fs, nperseg=2048, window='hann'
            )
            # Average over time to get transfer function estimate
            avg_magnitude = np.mean(magnitude, axis=1)
            return freqs, avg_magnitude
            
        elif method == 'spectrogram':
            # Original spectrogram method
            _, _, Sxx = signal.spectrogram(
                waveform, 
                fs=fs,
                nperseg=1024,
                noverlap=512,
                nfft=1024
            )
            # Average over time
            avg_spec = np.mean(Sxx, axis=1)
            freqs = np.linspace(0, fs/2, len(avg_spec))
            return freqs, avg_spec
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def enhance_contrast(
        transfer_functions: torch.Tensor,
        reference_idx: Optional[int] = None,
        enhancement_factor: float = 2.0
    ) -> torch.Tensor:
        """
        Apply per-frequency contrast enhancement.
        
        Args:
            transfer_functions: Input H matrix (F, D)
            reference_idx: Reference direction index (e.g., 90 degrees)
            enhancement_factor: Enhancement factor
            
        Returns:
            Enhanced transfer functions
        """
        enhanced = transfer_functions.clone()
        
        for f_idx in range(enhanced.shape[0]):
            freq_values = enhanced[f_idx, :]
            min_val = freq_values.min()
            max_val = freq_values.max()
            range_val = max_val - min_val
            
            if range_val > 1e-6:  # Only enhance if there's meaningful variation
                if reference_idx is not None:
                    # Enhanced around reference
                    ref_val = enhanced[f_idx, reference_idx]
                    centered = freq_values - ref_val
                    enhanced_centered = centered * enhancement_factor
                    enhanced[f_idx, :] = enhanced_centered + ref_val
                    # Ensure non-negative
                    enhanced[f_idx, :] = torch.clamp(enhanced[f_idx, :], min=0.01)
                else:
                    # Global enhancement
                    enhanced[f_idx, :] = (freq_values - min_val) / range_val * enhancement_factor + min_val
                    
        return enhanced
    
    @staticmethod
    def reconstruct_audio_from_spectrogram(
        magnitude_spec: np.ndarray,
        phase_spec: Optional[np.ndarray] = None,
        fs: int = 16000,
        nperseg: int = 2048,
        noverlap: Optional[int] = None,
        window: str = 'hann'
    ) -> np.ndarray:
        """
        Reconstruct audio waveform from magnitude spectrogram using ISTFT.
        
        Args:
            magnitude_spec: Magnitude spectrogram (freq x time)
            phase_spec: Phase spectrogram (optional, if None uses random phase)
            fs: Sampling frequency
            nperseg: Length of each segment
            noverlap: Number of points to overlap
            window: Window function
            
        Returns:
            Reconstructed audio waveform
        """
        if noverlap is None:
            noverlap = int(nperseg * 0.75)
        
        # If no phase provided, use random phase or Griffin-Lim algorithm
        if phase_spec is None:
            # Simple random phase initialization
            phase_spec = np.random.uniform(0, 2*np.pi, magnitude_spec.shape)
        
        # Reconstruct complex spectrogram
        complex_spec = magnitude_spec * np.exp(1j * phase_spec)
        
        # Inverse STFT
        _, reconstructed_audio = signal.istft(
            complex_spec,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        return reconstructed_audio
    
    @staticmethod
    def griffin_lim_reconstruction(
        magnitude_spec: np.ndarray,
        fs: int = 16000,
        nperseg: int = 2048,
        noverlap: Optional[int] = None,
        window: str = 'hann',
        n_iter: int = 50
    ) -> np.ndarray:
        """
        Reconstruct audio from magnitude spectrogram using Griffin-Lim algorithm.
        
        Args:
            magnitude_spec: Magnitude spectrogram (freq x time)
            fs: Sampling frequency
            nperseg: Length of each segment
            noverlap: Number of points to overlap
            window: Window function
            n_iter: Number of Griffin-Lim iterations
            
        Returns:
            Reconstructed audio waveform
        """
        if noverlap is None:
            noverlap = int(nperseg * 0.75)
        
        # Initialize with random phase
        phase = np.random.uniform(0, 2*np.pi, magnitude_spec.shape)
        
        for i in range(n_iter):
            # Reconstruct complex spectrogram
            complex_spec = magnitude_spec * np.exp(1j * phase)
            
            # ISTFT to get audio
            _, audio = signal.istft(
                complex_spec,
                fs=fs,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap
            )
            
            # STFT to get new phase
            _, _, new_complex_spec = signal.stft(
                audio,
                fs=fs,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap
            )
            
            # Update phase while keeping magnitude fixed
            phase = np.angle(new_complex_spec)
        
        # Final reconstruction
        final_complex_spec = magnitude_spec * np.exp(1j * phase)
        _, reconstructed_audio = signal.istft(
            final_complex_spec,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        return reconstructed_audio