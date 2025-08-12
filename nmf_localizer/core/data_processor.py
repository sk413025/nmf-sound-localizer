"""
Data processing module for NMF localization.
Extracted and refactored from prepare_real_data.py.
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from tqdm import tqdm
import logging
from scipy import signal

from ..config.defaults import NMFConfig, DataPack
from ..utils.audio_utils import AudioProcessor

logger = logging.getLogger(__name__)


class DataProcessor:
    """Main data processor for NMF localization."""
    
    def __init__(self, config: NMFConfig):
        self.config = config
        self.audio_processor = AudioProcessor()
    
    def load_npy_files(self, folder_path: Path, max_files: Optional[int] = None) -> List[np.ndarray]:
        """Load .npy files from a folder."""
        npy_files = sorted(folder_path.glob('*.npy'))
        if max_files:
            npy_files = npy_files[:max_files]
        
        logger.info(f"Loading {len(npy_files)} files from {folder_path.name}")
        
        data = []
        for i, file_path in enumerate(npy_files):
            arr = np.load(file_path)
            if i == 0:
                logger.info(f"First file shape in {folder_path.name}: {arr.shape}")
            data.append(arr)
        
        return data
    
    def estimate_transfer_functions(
        self, 
        root_path: Path,
        method: str = 'improved'
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Path], Dict[str, Any]]:
        """
        Estimate transfer functions from angle folders.
        
        Args:
            root_path: Path to root directory
            method: 'simple' (original) or 'improved' (Welch method)
        
        Returns:
            H: Transfer functions (linear or normalized)
            angles: Angle array
            angle_folders: List of angle folders
            metadata: Additional information (freqs, etc.)
        """
        # Get all angle folders
        angle_folders = sorted([d for d in root_path.iterdir() 
                               if d.is_dir() and d.name.startswith('angle_')])
        logger.info(f"Found {len(angle_folders)} angle folders")
        
        # Map angle names to degrees
        angle_mapping = {}
        for folder in angle_folders:
            angle_str = folder.name.replace('angle_', '')
            angle_mapping[folder.name] = int(angle_str)
        
        # Sort by angle value
        angle_folders = sorted(angle_folders, key=lambda x: angle_mapping[x.name])
        n_directions = len(angle_folders)
        
        # Load sample file to get dimensions
        sample_files = list(angle_folders[0].glob('*.npy'))
        if not sample_files:
            raise ValueError(f"No .npy files found in {angle_folders[0]}")
        
        sample_data = np.load(sample_files[0])
        logger.info(f"Sample data shape: {sample_data.shape}")
        logger.info(f"Number of angle folders: {n_directions}")
        
        # Check if data is waveform or spectrogram
        if sample_data.ndim == 1:
            is_waveform = True
            if method == 'improved':
                n_freq = 1025  # Default for Welch with nperseg=2048
            else:
                n_freq = 513   # Default for STFT with n_fft=1024
        else:
            # Assume data is already spectrograms (F x T)
            n_freq = sample_data.shape[0]
            is_waveform = False
            method = 'simple'  # Force simple method for pre-computed spectrograms
        
        # Initialize transfer functions
        all_magnitudes_linear = []
        freqs = None
        
        logger.info(f"Estimating transfer functions using {method} method...")
        logger.info(f"Expected freq bins: {n_freq}")
        
        for i, folder in enumerate(tqdm(angle_folders)):
            # Load files from this angle
            data = self.load_npy_files(folder, max_files=self.config.n_files_per_angle)
            
            if not data:
                logger.warning(f"No data found for {folder.name}")
                continue
                
            # Compute magnitude response for each file
            angle_magnitudes_linear = []
            
            for waveform in data:
                if is_waveform:
                    if method == 'improved':
                        # Use STFT method to preserve temporal information
                        f, times, stft, magnitude = self.audio_processor.compute_stft_spectrogram(
                            waveform, fs=self.config.sample_rate, 
                            nperseg=self.config.n_fft, window=self.config.window
                        )
                        # Average over time to get transfer function estimate
                        avg_magnitude = np.mean(magnitude, axis=1)
                    else:
                        # Original method
                        f, avg_magnitude = self.audio_processor.compute_magnitude_response(
                            waveform, fs=self.config.sample_rate, method='spectrogram'
                        )
                        
                    if freqs is None:
                        freqs = f
                        logger.info(f"Frequency array shape: {len(freqs)}, range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")
                        
                    if i == 0 and len(angle_magnitudes_linear) == 0:
                        logger.info(f"Magnitude response shape for angle {i}: {len(avg_magnitude)}")
                        
                    angle_magnitudes_linear.append(avg_magnitude)
                else:
                    # Already a spectrogram - use simple averaging
                    if np.iscomplexobj(waveform):
                        waveform = np.abs(waveform)
                    if waveform.ndim > 1:
                        avg_spec = np.mean(waveform, axis=1)
                    else:
                        avg_spec = waveform
                    angle_magnitudes_linear.append(avg_spec)
            
            # Average in linear domain
            avg_linear = np.mean(angle_magnitudes_linear, axis=0)
            all_magnitudes_linear.append(avg_linear)
        
        # Convert to tensors
        H_linear = torch.tensor(np.array(all_magnitudes_linear).T, dtype=torch.float32)
        logger.info(f"Raw H shape - linear: {H_linear.shape}")
        
        # Apply processing based on method
        if method == 'improved':
            H_linear = self._apply_improved_processing(H_linear, angle_mapping, angle_folders, freqs)
        else:
            # Simple normalization: global normalization
            H_max = H_linear.max()
            H_linear = H_linear / H_max
            logger.info(f"H max value before normalization: {H_max:.4f}")
        
        # Create angle array (in degrees)
        angles = torch.tensor([angle_mapping[f.name] for f in angle_folders], dtype=torch.float32)
        logger.info(f"Angles array: {angles.tolist()}")
        
        # Create metadata
        metadata = {
            'method': method,
            'freqs': freqs,
            'n_files_per_angle': self.config.n_files_per_angle
        }
        
        return H_linear, angles, angle_folders, metadata
    
    def _apply_improved_processing(
        self, 
        H_linear: torch.Tensor,
        angle_mapping: Dict[str, int],
        angle_folders: List[Path],
        freqs: np.ndarray
    ) -> torch.Tensor:
        """Apply improved processing with 90-degree reference and frequency filtering."""
        
        # Find 90-degree reference
        angles_degrees = [angle_mapping[folder.name] for folder in angle_folders]
        logger.info(f"Available angles: {angles_degrees}")
        
        if 90 in angles_degrees:
            ref_angle_idx = angles_degrees.index(90)
            reference_spectrum = H_linear[:, ref_angle_idx:ref_angle_idx+1]
            logger.info(f"Using angle 90° (index {ref_angle_idx}) as reference")
            
            # Compute relative transfer functions
            H_relative = H_linear / (reference_spectrum + 1e-10)
            
            # Apply contrast enhancement if enabled
            if self.config.apply_contrast_enhancement:
                logger.info("Applying per-frequency contrast enhancement...")
                H_enhanced = self.audio_processor.enhance_contrast(
                    H_relative, reference_idx=ref_angle_idx, enhancement_factor=2.0
                )
                
                # Log enhancement effect
                original_range = torch.mean(torch.max(H_relative, dim=1)[0] - torch.min(H_relative, dim=1)[0])
                enhanced_range = torch.mean(torch.max(H_enhanced, dim=1)[0] - torch.min(H_enhanced, dim=1)[0])
                logger.info(f"Contrast enhancement: mean range {original_range:.4f} → {enhanced_range:.4f}")
                
                H_linear = H_enhanced
            else:
                H_linear = H_relative
                
        else:
            logger.warning("90° angle not found, falling back to per-frequency max normalization")
            H_linear = self.audio_processor.normalize_spectrogram(H_linear, method='per_freq')
        
        # Apply frequency band limit
        if freqs is not None and self.config.freq_min and self.config.freq_max:
            H_linear, freqs_limited = self.audio_processor.apply_frequency_filter(
                H_linear, freqs, self.config.freq_min, self.config.freq_max
            )
            logger.info(f"Applied {self.config.freq_min}-{self.config.freq_max}Hz band limit: {H_linear.shape[0]} freq bins")
        
        return H_linear
    
    def prepare_speech_data(
        self, 
        root_path: Path,
        freq_limit: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        Prepare speech data for USM training.
        
        Args:
            root_path: Root data path
            freq_limit: Frequency dimension limit
            
        Returns:
            List of speaker spectrograms
        """
        angle_folders = sorted([d for d in root_path.iterdir() 
                               if d.is_dir() and d.name.startswith('angle_')])
        
        if self.config.use_90deg_only:
            # Find 90-degree folder for physical consistency
            angle_90_folder = None
            for folder in angle_folders:
                angle_str = folder.name.replace('angle_', '')
                if int(angle_str) == 90:
                    angle_90_folder = folder
                    break
            
            if angle_90_folder is None:
                logger.warning("90-degree folder not found! Falling back to all angles.")
                speaker_folders = angle_folders[:min(self.config.n_speakers, len(angle_folders))]
            else:
                logger.info(f"Using only 90-degree data ({angle_90_folder.name}) for USM training")
                speaker_folders = [angle_90_folder]
        else:
            # Use subset of angles as different speakers
            speaker_folders = angle_folders[:min(self.config.n_speakers, len(angle_folders))]
            logger.info(f"Using multiple angles as speakers: {[f.name for f in speaker_folders]}")
        
        speaker_data = []
        logger.info("=== Speech Data Preparation ===")
        
        if self.config.use_90deg_only and len(speaker_folders) == 1:
            # Special handling for 90-degree only
            folder = speaker_folders[0]
            logger.info(f"Using only 90-degree data ({folder.name}) for USM training")
            
            # Load all available files from 90-degree folder
            files = self.load_npy_files(folder, max_files=100)  # Use more files for better representation
            
            if not files:
                logger.warning(f"No files found in {folder.name}")
                return speaker_data
            
            logger.info(f"Loading {len(files)} files from {folder.name}")
            
            # Process all files and concatenate
            specs = []
            for waveform in files:
                if waveform.ndim == 1:
                    # Use STFT to preserve real temporal information
                    freqs_speech, times, stft, magnitude = self.audio_processor.compute_stft_spectrogram(
                        waveform, fs=self.config.sample_rate, 
                        nperseg=self.config.n_fft, window=self.config.window
                    )
                    specs.append(magnitude)
                    
                    if len(specs) == 1:  # Log first spectrogram
                        logger.info(f"First real spectrogram shape: {magnitude.shape}")
                        logger.info(f"Time frames: {len(times)}, duration: {times[-1]:.2f}s")
                else:
                    if np.iscomplexobj(waveform):
                        waveform = np.abs(waveform)
                    specs.append(waveform)
            
            # Concatenate all 90-degree data along time axis
            concatenated = np.concatenate(specs, axis=1)
            
            # Apply frequency limit if specified
            if freq_limit is not None and concatenated.shape[0] > freq_limit:
                concatenated = concatenated[:freq_limit, :]
            
            logger.info(f"90° Speaker concatenated shape: {concatenated.shape}")
            speaker_data.append(torch.from_numpy(concatenated).float())
        
        else:
            # Original behavior: different folders as different speakers
            logger.info(f"Loading speech data from {len(speaker_folders)} folders...")
            
            for folder in tqdm(speaker_folders):
                # Load files
                files = self.load_npy_files(folder, max_files=self.config.n_files_per_speaker)
                
                if not files:
                    continue
                    
                # Concatenate spectrograms
                specs = []
                for waveform in files:
                    if waveform.ndim == 1:
                        freqs_speech, times, stft, magnitude = self.audio_processor.compute_stft_spectrogram(
                            waveform, fs=self.config.sample_rate, 
                            nperseg=self.config.n_fft, window=self.config.window
                        )
                        specs.append(magnitude)
                    else:
                        if np.iscomplexobj(waveform):
                            waveform = np.abs(waveform)
                        specs.append(waveform)
                
                # Concatenate along time axis
                concatenated = np.concatenate(specs, axis=1)
                
                # Apply frequency limit if specified
                if freq_limit is not None and concatenated.shape[0] > freq_limit:
                    concatenated = concatenated[:freq_limit, :]
                    
                logger.info(f"Speaker {len(speaker_data)} concatenated shape: {concatenated.shape}")
                speaker_data.append(torch.from_numpy(concatenated).float())
        
        return speaker_data
    
    def load_real_angle_test_data(
        self, 
        test_data_root: str, 
        n_test_examples: int,
        device: str = 'cpu'
    ) -> List[Dict]:
        """
        Load REAL multi-angle audio test data.
        
        Args:
            test_data_root: Path to test data directory
            n_test_examples: Number of test examples to load
            device: Device to use for tensors
            
        Returns:
            List of test examples with 'mixture' and 'directions' keys
        """
        test_data = []
        data_root = Path(test_data_root)
        
        if not data_root.exists():
            logger.warning(f"Test data directory does not exist: {data_root}")
            logger.warning("Creating dummy test data for compatibility...")
            
            # Create dummy test data for compatibility
            for i in range(min(n_test_examples, 10)):
                dummy_mixture = torch.randn(129, 100, device=device) * 0.1
                dummy_mixture = torch.abs(dummy_mixture)
                dummy_direction = np.random.randint(0, 11)
                
                test_data.append({
                    'mixture': dummy_mixture,
                    'directions': [dummy_direction]
                })
                
            logger.info(f"Created {len(test_data)} dummy test examples")
            return test_data
        
        logger.info(f"Loading REAL audio test data from: {data_root}")
        
        # Collect all angle directories
        angle_dirs = sorted([d for d in data_root.iterdir() 
                           if d.is_dir() and d.name.startswith('angle_')])
        
        if not angle_dirs:
            raise FileNotFoundError(f"No angle directories found in {data_root}")
        
        logger.info(f"Found {len(angle_dirs)} angle directories")
        
        # Load examples from each angle directory
        examples_per_angle = max(1, n_test_examples // len(angle_dirs))
        
        for angle_dir in angle_dirs:
            # Extract angle number from directory name
            angle_str = angle_dir.name.split('_')[-1]
            angle_idx = int(angle_str)
            
            # Find .npy files in the angle directory
            npy_files = list(angle_dir.glob('*.npy'))
            
            if not npy_files:
                logger.warning(f"No .npy files found in {angle_dir}")
                continue
                
            # Load up to examples_per_angle files from this angle
            selected_files = npy_files[:examples_per_angle]
            
            for file_path in selected_files:
                try:
                    # Load real waveform data
                    audio_data = np.load(file_path)
                    
                    if len(audio_data.shape) > 1:
                        # Take first channel if stereo
                        audio_data = audio_data[0]
                    
                    # Convert to spectrogram using STFT
                    f, t, Zxx = signal.stft(audio_data, fs=self.config.sample_rate, 
                                          nperseg=512, noverlap=256)
                    magnitude_spec = np.abs(Zxx)
                    
                    # Convert to torch tensor
                    mixture_tensor = torch.from_numpy(magnitude_spec).float().to(device)
                    
                    # Apply frequency filtering to match transfer function dimensions (129 bins for 500-1500Hz)
                    if mixture_tensor.shape[0] > 129:
                        freq_start = max(0, int(500 * 512 / self.config.sample_rate))
                        freq_end = min(mixture_tensor.shape[0], freq_start + 129)
                        mixture_tensor = mixture_tensor[freq_start:freq_end, :]
                    
                    # Ensure we have exactly 129 frequency bins
                    if mixture_tensor.shape[0] != 129:
                        if mixture_tensor.shape[0] < 129:
                            padding = torch.zeros(129 - mixture_tensor.shape[0], 
                                                mixture_tensor.shape[1], device=device)
                            mixture_tensor = torch.cat([mixture_tensor, padding], dim=0)
                        else:
                            mixture_tensor = mixture_tensor[:129, :]
                    
                    test_data.append({
                        'mixture': mixture_tensor,
                        'directions': [angle_idx],
                        'source_file': str(file_path)
                    })
                    
                    # Stop if we have enough examples
                    if len(test_data) >= n_test_examples:
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    continue
                    
            # Stop if we have enough examples
            if len(test_data) >= n_test_examples:
                break
        
        if not test_data:
            raise RuntimeError(f"No valid test data loaded from {data_root}")
        
        logger.info(f"Successfully loaded {len(test_data)} REAL test examples")
        logger.info(f"Example dimensions: {test_data[0]['mixture'].shape}")
        logger.info(f"Angle range: {min(ex['directions'][0] for ex in test_data)} - {max(ex['directions'][0] for ex in test_data)}")
        
        return test_data
    
    def process_full_dataset(
        self, 
        data_root: str,
        output_dir: Optional[str] = None,
        speech_data_root: Optional[str] = None
    ) -> DataPack:
        """
        Process complete dataset for NMF localization.
        
        Args:
            data_root: Root data directory (for transfer function estimation)
            output_dir: Optional output directory to save results
            speech_data_root: Optional separate path for speech data (USM training and testing)
            
        Returns:
            Complete DataPack with all processed data
        """
        root_path = Path(data_root)
        
        if not root_path.exists():
            raise FileNotFoundError(f"Data root not found: {root_path}")
        
        logger.info(f"Processing dataset from: {root_path}")
        
        # Determine speech data path
        if speech_data_root is not None:
            speech_path = Path(speech_data_root)
            if not speech_path.exists():
                raise FileNotFoundError(f"Speech data root not found: {speech_path}")
            logger.info(f"Using separate speech data from: {speech_path}")
        else:
            speech_path = root_path
            logger.info("Using same path for transfer functions and speech data")
        
        # Create data pack
        data_pack = DataPack()
        data_pack.config = self.config
        
        # Estimate transfer functions
        logger.info("Estimating transfer functions...")
        H, angles, angle_folders, tf_metadata = self.estimate_transfer_functions(
            root_path, method=self.config.tf_method
        )
        
        data_pack.transfer_functions = H
        data_pack.angles = angles
        data_pack.angle_names = [f.name for f in angle_folders]
        data_pack.metadata.update(tf_metadata)
        
        # Prepare speech data
        logger.info("Preparing speech data...")
        freq_limit = H.shape[0] if H is not None else None
        speaker_data = self.prepare_speech_data(speech_path, freq_limit=freq_limit)
        data_pack.speaker_data = speaker_data
        
        # Load test data if available
        # First try speech data split, then fall back to root data split
        speech_test_path = speech_path.parent / "root_split" / "test" if speech_data_root else None
        root_test_path = root_path.parent / "root_split" / "test"
        
        if speech_test_path and speech_test_path.exists():
            logger.info(f"Loading test data from speech data split: {speech_test_path}")
            test_data = self.load_real_angle_test_data(
                str(speech_test_path), 
                self.config.n_test_examples
            )
            data_pack.test_data = test_data
        elif root_test_path.exists():
            logger.info(f"Loading test data from root data split: {root_test_path}")
            test_data = self.load_real_angle_test_data(
                str(root_test_path), 
                self.config.n_test_examples
            )
            data_pack.test_data = test_data
        else:
            # If no separate test data, create test data from speech directory
            logger.info("No separate test data found. Creating test data from speech data directory...")
            test_data = self.load_real_angle_test_data(
                str(speech_path), 
                min(self.config.n_test_examples, 20)  # Use fewer samples from training data
            )
            data_pack.test_data = test_data
        
        # Validate data pack
        if not data_pack.validate():
            logger.warning("Data pack validation failed!")
        
        # Save if output directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            data_pack.save(str(output_path / "data_pack.pth"))
            logger.info(f"Saved data pack to: {output_path / 'data_pack.pth'}")
        
        logger.info(f"Dataset processing complete: {data_pack}")
        return data_pack