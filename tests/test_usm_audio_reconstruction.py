"""
Tests for USM Trainer audio reconstruction capabilities.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from scipy.io import wavfile
import os

from nmf_localizer.config import NMFConfig
from nmf_localizer.core.usm_trainer import USMTrainer
from nmf_localizer.utils.audio_utils import AudioProcessor


class TestUSMReconstruction:
    """Test USM audio reconstruction capabilities."""
    
    @pytest.fixture
    def audio_file_path(self):
        """Path to test audio file."""
        test_data_dir = Path(__file__).parent / "data"
        audio_path = test_data_dir / "original_clip_000.wav"
        
        if not audio_path.exists():
            pytest.skip(f"Test audio file not found: {audio_path}")
        
        return str(audio_path)
    
    @pytest.fixture
    def config(self):
        """Default configuration for testing."""
        return NMFConfig(
            n_atoms_per_speaker=20,
            usm_max_iter=50,
            beta=2.0,  # Euclidean distance
            usm_sparsity_weight=0.01,
            device='cpu',
            sample_rate=16000,
            n_fft=2048,
            hop_length=512,
            freq_min=100,
            freq_max=4000
        )
    
    @pytest.fixture 
    def audio_processor(self):
        """Audio processor instance."""
        return AudioProcessor()
    
    def save_reconstructed_audio(
        self, 
        reconstructed_magnitude: torch.Tensor,
        W: torch.Tensor,
        freqs: np.ndarray,
        config: NMFConfig,
        output_path: str,
        original_stft: np.ndarray = None
    ):
        """
        Save reconstructed audio from NMF reconstruction.
        
        Args:
            reconstructed_magnitude: Reconstructed magnitude spectrogram
            W: NMF dictionary
            freqs: Frequency array  
            config: Configuration
            output_path: Output file path
            original_stft: Original STFT for phase information
        """
        # Perform NMF reconstruction
        W_pinv = torch.linalg.pinv(W)
        H_estimated = W_pinv @ reconstructed_magnitude
        reconstructed_spec = W @ H_estimated
        
        # Convert to numpy
        reconstructed_magnitude_np = reconstructed_spec.detach().cpu().numpy()
        
        # Use original phase if available, otherwise Griffin-Lim
        audio_processor = AudioProcessor()
        
        if original_stft is not None:
            # For phase reconstruction, we need to reconstruct the full-band spectrogram
            # then apply ISTFT to get audio
            
            # Create full-band reconstruction by zero-padding filtered reconstruction
            full_band_magnitude = np.zeros_like(np.abs(original_stft))
            
            # Find frequency indices for filtered band
            freq_idx_min = np.argmin(np.abs(freqs - config.freq_min))
            freq_idx_max = np.argmin(np.abs(freqs - config.freq_max))
            
            # Insert reconstructed magnitude into correct frequency range
            reconst_freq_range = freq_idx_max - freq_idx_min + 1
            if reconstructed_magnitude_np.shape[0] <= reconst_freq_range:
                end_idx = freq_idx_min + reconstructed_magnitude_np.shape[0]
                time_frames = min(reconstructed_magnitude_np.shape[1], full_band_magnitude.shape[1])
                full_band_magnitude[freq_idx_min:end_idx, :time_frames] = reconstructed_magnitude_np[:, :time_frames]
            
            # Use original phase for full reconstruction
            original_phase = np.angle(original_stft)
            full_complex_spec = full_band_magnitude * np.exp(1j * original_phase)
            
            # Reconstruct audio using ISTFT with exact same parameters
            reconstructed_audio = audio_processor.reconstruct_audio_from_spectrogram(
                full_band_magnitude,
                phase_spec=original_phase,
                fs=config.sample_rate,
                nperseg=config.n_fft,
                noverlap=config.hop_length
            )
        else:
            # Use Griffin-Lim algorithm with same STFT parameters
            reconstructed_audio = audio_processor.griffin_lim_reconstruction(
                reconstructed_magnitude_np,
                fs=config.sample_rate,
                nperseg=config.n_fft,
                noverlap=config.hop_length,  # Use same overlap as original
                n_iter=30
            )
        
        # Normalize audio
        reconstructed_audio = reconstructed_audio / np.max(np.abs(reconstructed_audio))
        
        # Convert to 16-bit integer
        audio_int16 = (reconstructed_audio * 32767).astype(np.int16)
        
        # Save to file
        wavfile.write(output_path, config.sample_rate, audio_int16)
    
    def load_audio_file(self, file_path: str, target_sr: int = 16000) -> np.ndarray:
        """
        Load audio file using scipy.io.wavfile.
        
        Args:
            file_path: Path to audio file
            target_sr: Target sample rate
            
        Returns:
            Audio waveform as numpy array
        """
        try:
            sample_rate, audio_data = wavfile.read(file_path)
            
            # Convert to float32 and normalize
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Handle stereo by taking first channel
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
            
            # Basic resampling if needed (simple decimation/interpolation)
            if sample_rate != target_sr:
                # Simple resampling by factor
                factor = target_sr / sample_rate
                if factor < 1:
                    # Downsample by taking every nth sample
                    step = int(1 / factor)
                    audio_data = audio_data[::step]
                else:
                    # Upsample by repeating samples (basic interpolation)
                    audio_data = np.repeat(audio_data, int(factor))
            
            return audio_data
            
        except Exception as e:
            pytest.fail(f"Failed to load audio file {file_path}: {e}")
    
    def test_usm_audio_reconstruction_basic(self, audio_file_path, config, audio_processor):
        """Test basic audio reconstruction using USM."""
        # Load audio file
        audio_waveform = self.load_audio_file(audio_file_path, config.sample_rate)
        
        # Ensure minimum length for processing
        min_length = config.n_fft * 2
        if len(audio_waveform) < min_length:
            # Repeat audio if too short
            repeats = (min_length // len(audio_waveform)) + 1
            audio_waveform = np.tile(audio_waveform, repeats)[:min_length]
        
        # Compute spectrogram
        freqs, times, stft, magnitude = audio_processor.compute_stft_spectrogram(
            audio_waveform, 
            fs=config.sample_rate,
            nperseg=config.n_fft,
            noverlap=config.hop_length
        )
        
        # Convert to torch tensor
        magnitude_tensor = torch.from_numpy(magnitude).float()
        
        # Apply frequency filtering
        filtered_magnitude, filtered_freqs = audio_processor.apply_frequency_filter(
            magnitude_tensor, freqs, config.freq_min, config.freq_max
        )
        
        # Create USM trainer
        trainer = USMTrainer(config)
        
        # Train USM with the audio data (treating as single speaker)
        speaker_data_list = [filtered_magnitude]
        
        # Train the model
        W, train_info = trainer.train_usm(speaker_data_list)
        
        # Test reconstruction quality
        reconstruction_metrics = trainer.get_reconstruction_quality(
            filtered_magnitude, n_test_frames=50
        )
        
        # Assertions
        assert W is not None, "USM dictionary should be trained"
        assert W.shape[0] == filtered_magnitude.shape[0], f"Frequency dimension mismatch: {W.shape[0]} vs {filtered_magnitude.shape[0]}"
        assert W.shape[1] == config.n_atoms_per_speaker, f"Atoms mismatch: {W.shape[1]} vs {config.n_atoms_per_speaker}"
        
        # Check training info
        assert 'n_speakers' in train_info
        assert 'total_atoms' in train_info
        assert train_info['n_speakers'] == 1
        assert train_info['total_atoms'] == config.n_atoms_per_speaker
        
        # Check reconstruction metrics
        assert 'mse' in reconstruction_metrics
        assert 'rmse' in reconstruction_metrics
        assert 'snr_db' in reconstruction_metrics
        assert 'sparsity' in reconstruction_metrics
        
        # Quality thresholds (adjust based on expected performance)
        assert reconstruction_metrics['mse'] < 1.0, f"MSE too high: {reconstruction_metrics['mse']}"
        assert reconstruction_metrics['snr_db'] > -10, f"SNR too low: {reconstruction_metrics['snr_db']} dB"
        
        # Save reconstructed audio
        self.save_reconstructed_audio(
            filtered_magnitude, W, freqs, config, 
            output_path="tests/data/reconstructed_basic.wav",
            original_stft=stft
        )
        
        print(f"Reconstruction Results:")
        print(f"  MSE: {reconstruction_metrics['mse']:.6f}")
        print(f"  RMSE: {reconstruction_metrics['rmse']:.6f}")
        print(f"  SNR: {reconstruction_metrics['snr_db']:.2f} dB")
        print(f"  Sparsity: {reconstruction_metrics['sparsity']:.3f}")
        print(f"  Dictionary shape: {W.shape}")
        print(f"  Reconstructed audio saved to: tests/data/reconstructed_basic.wav")
        if 'speaker_info' in train_info and 'speaker_0' in train_info['speaker_info']:
            speaker_0_info = train_info['speaker_info']['speaker_0']
            print(f"  Training iterations: {speaker_0_info['n_iterations']}")
            print(f"  Final loss: {speaker_0_info['final_loss']:.6f}")
        else:
            print(f"  Training info keys: {list(train_info.keys())}")
    
    def test_usm_audio_reconstruction_with_beta_search(self, audio_file_path, config, audio_processor):
        """Test audio reconstruction with multiple beta values."""
        # Load and process audio (same as basic test)
        audio_waveform = self.load_audio_file(audio_file_path, config.sample_rate)
        
        min_length = config.n_fft * 2
        if len(audio_waveform) < min_length:
            repeats = (min_length // len(audio_waveform)) + 1
            audio_waveform = np.tile(audio_waveform, repeats)[:min_length]
        
        freqs, times, stft, magnitude = audio_processor.compute_stft_spectrogram(
            audio_waveform, fs=config.sample_rate, nperseg=config.n_fft, noverlap=config.hop_length
        )
        
        magnitude_tensor = torch.from_numpy(magnitude).float()
        filtered_magnitude, filtered_freqs = audio_processor.apply_frequency_filter(
            magnitude_tensor, freqs, config.freq_min, config.freq_max
        )
        
        # Create USM trainer and test multiple beta values
        trainer = USMTrainer(config)
        speaker_data_list = [filtered_magnitude]
        
        # Test with different beta values
        beta_values = [0.5, 1.0, 2.0]
        best_W, best_beta, all_results = trainer.test_multiple_betas(
            speaker_data_list, beta_values
        )
        
        # Assertions
        assert best_W is not None, "Best USM dictionary should be found"
        assert best_beta in beta_values, f"Best beta {best_beta} should be in tested values {beta_values}"
        assert len(all_results) <= len(beta_values), "Should have results for tested betas"
        
        # Check that we have valid results for at least one beta
        valid_results = [r for r in all_results.values() if 'reconstruction_error' in r]
        assert len(valid_results) > 0, "Should have at least one valid beta result"
        
        # Test final reconstruction quality with best beta
        final_metrics = trainer.get_reconstruction_quality(filtered_magnitude, n_test_frames=50)
        
        print(f"Beta Search Results:")
        print(f"  Best beta: {best_beta}")
        print(f"  Best dictionary shape: {best_W.shape}")
        for beta, result in all_results.items():
            if 'reconstruction_error' in result:
                print(f"  Beta {beta}: reconstruction error = {result['reconstruction_error']:.6f}")
        # Save reconstructed audio with best beta
        self.save_reconstructed_audio(
            filtered_magnitude, best_W, freqs, config,
            output_path="tests/data/reconstructed_beta_search.wav",
            original_stft=stft
        )
        
        print(f"Final reconstruction with best beta:")
        print(f"  MSE: {final_metrics['mse']:.6f}")
        print(f"  SNR: {final_metrics['snr_db']:.2f} dB")
        print(f"  Reconstructed audio saved to: tests/data/reconstructed_beta_search.wav")
    
    def test_usm_save_load_reconstruction(self, audio_file_path, config, audio_processor, tmp_path):
        """Test save/load functionality with reconstruction verification."""
        # Process audio
        audio_waveform = self.load_audio_file(audio_file_path, config.sample_rate)
        
        min_length = config.n_fft * 2
        if len(audio_waveform) < min_length:
            repeats = (min_length // len(audio_waveform)) + 1
            audio_waveform = np.tile(audio_waveform, repeats)[:min_length]
        
        freqs, times, stft, magnitude = audio_processor.compute_stft_spectrogram(
            audio_waveform, fs=config.sample_rate, nperseg=config.n_fft, noverlap=config.hop_length
        )
        
        magnitude_tensor = torch.from_numpy(magnitude).float()
        filtered_magnitude, filtered_freqs = audio_processor.apply_frequency_filter(
            magnitude_tensor, freqs, config.freq_min, config.freq_max
        )
        
        # Train original model
        trainer1 = USMTrainer(config)
        speaker_data_list = [filtered_magnitude]
        W1, train_info1 = trainer1.train_usm(speaker_data_list)
        
        # Get reconstruction quality from original model
        metrics1 = trainer1.get_reconstruction_quality(filtered_magnitude, n_test_frames=50)
        
        # Save model
        model_path = tmp_path / "test_usm_model.pt"
        trainer1.save_model(str(model_path))
        
        # Load model in new trainer
        trainer2 = USMTrainer(config)
        W2 = trainer2.load_model(str(model_path))
        
        # Get reconstruction quality from loaded model
        metrics2 = trainer2.get_reconstruction_quality(filtered_magnitude, n_test_frames=50)
        
        # Assertions
        assert torch.allclose(W1, W2, rtol=1e-5), "Loaded model should match original"
        assert abs(metrics1['mse'] - metrics2['mse']) < 1e-6, "Reconstruction quality should be identical"
        assert abs(metrics1['snr_db'] - metrics2['snr_db']) < 1e-3, "SNR should be identical"
        
        # Save reconstructed audio from both models
        self.save_reconstructed_audio(
            filtered_magnitude, W1, freqs, config,
            output_path="tests/data/reconstructed_original.wav",
            original_stft=stft
        )
        
        self.save_reconstructed_audio(
            filtered_magnitude, W2, freqs, config,
            output_path="tests/data/reconstructed_loaded.wav",
            original_stft=stft
        )
        
        print(f"Save/Load Test Results:")
        print(f"  Original MSE: {metrics1['mse']:.6f}")
        print(f"  Loaded MSE: {metrics2['mse']:.6f}")
        print(f"  MSE difference: {abs(metrics1['mse'] - metrics2['mse']):.8f}")
        print(f"  Model file: {model_path}")
        print(f"  Original reconstruction saved to: tests/data/reconstructed_original.wav")
        print(f"  Loaded reconstruction saved to: tests/data/reconstructed_loaded.wav")