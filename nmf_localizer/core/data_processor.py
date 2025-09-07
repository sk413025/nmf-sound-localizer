"""
Data processing module for NMF localization.
Extracted and refactored from prepare_real_data.py.
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
# from tqdm import tqdm  # Temporarily disabled for testing
import logging
import warnings
from scipy import signal

from ..config.defaults import NMFConfig, DataPack
from ..utils.audio_utils import AudioProcessor
from .stft_unified_processor import STFTUnifiedProcessor

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
        original_root: Path,
        box_root: Optional[Path] = None,
        *,
        method: Optional[str] = None,
        time_pooling: str = 'linear'
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Path], Dict[str, Any]]:
        """
        Estimate transfer functions using STFT-unified approach.
        
        This method uses consistent STFT processing to fix the previous
        Welch vs STFT scale/units mismatch issue.
        
        Args:
            original_root: Path to original data (X), or unified data root
            box_root: Optional path to box recordings (Y). If None, assumes
                     original_root contains both X and Y data (unified structure)
        
        Returns:
            H: Transfer functions [freq × directions] (magnitude-based units)
            angles: Angle array
            angle_folders: List of angle folders
            metadata: Additional information (freqs, coherence stats, etc.)
        """
        # Handle optional box_root parameter
        if box_root is None:
            box_root = original_root
            logger.info(f"Using unified data root for both X and Y: {original_root}")
        else:
            logger.info(f"Using separate paths - X: {original_root}, Y: {box_root}")
        
        # Use STFT-unified approach (the only correct method)
        logger.info("Using STFT-unified transfer function estimation")
        stft_processor = STFTUnifiedProcessor(self.config)
        # 'method' is accepted for backward API compatibility but ignored here
        chosen_method = 'stft_unified'
        return stft_processor.estimate_transfer_functions_stft(
            Path(original_root), Path(box_root), method=chosen_method, time_pooling=time_pooling
        )
    
    def _apply_improved_processing(
        self,
        H_linear: torch.Tensor,
        angle_mapping: Dict[str, int],
        angle_folders: List[Path],
        freqs: np.ndarray
    ) -> torch.Tensor:
        """Deprecated: normalization/contrast removed; return input unchanged."""
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
                    # Apply frequency band mask consistent with config
                    if self.config.freq_min is not None and self.config.freq_max is not None:
                        mask = (freqs_speech >= self.config.freq_min) & (freqs_speech <= self.config.freq_max)
                        magnitude = magnitude[mask, :]
                    specs.append(magnitude)
                    
                    if len(specs) == 1:  # Log first spectrogram
                        logger.info(f"First real spectrogram shape (band-limited): {magnitude.shape}")
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
            
            for folder in speaker_folders:
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
                        if self.config.freq_min is not None and self.config.freq_max is not None:
                            mask = (freqs_speech >= self.config.freq_min) & (freqs_speech <= self.config.freq_max)
                            magnitude = magnitude[mask, :]
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
                    
                    # Convert to spectrogram using STFT consistent with config
                    f, t, Zxx = signal.stft(
                        audio_data,
                        fs=self.config.sample_rate,
                        nperseg=self.config.n_fft,
                        noverlap=self.config.n_fft - self.config.hop_length,
                        window=self.config.window
                    )
                    magnitude_spec = np.abs(Zxx)
                    # Apply frequency band-limiting using freqs from STFT
                    mask = (f >= (self.config.freq_min or 0.0)) & (f <= (self.config.freq_max or f.max()))
                    magnitude_spec = magnitude_spec[mask, :]

                    # Convert to torch tensor
                    mixture_tensor = torch.from_numpy(magnitude_spec).float().to(device)
                    
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
            root_path
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
