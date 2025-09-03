"""
Generate comprehensive dataset card for the fixed dataset in conftest.py.
Uses existing nmf_localizer modules to ensure consistency with actual tests.
"""
import numpy as np
import json
import pytest
import torch
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Import existing modules to ensure consistency
from nmf_localizer.config import NMFConfig
from nmf_localizer.core.data_processor import DataProcessor
from nmf_localizer.core.transfer_functions import TransferFunctionProcessor
from nmf_localizer.utils.audio_utils import AudioProcessor


class DatasetCardGenerator:
    """Generates comprehensive statistics using existing nmf_localizer modules."""
    
    def __init__(self, x_root: str, y_root: str, angles: List[str], n_files: int):
        self.x_root = Path(x_root)
        self.y_root = Path(y_root)
        self.angles = angles
        self.n_files = n_files
        
        # Use existing configuration for consistency
        self.config = NMFConfig(
            sample_rate=16000,
            n_fft=2048, 
            hop_length=512,
            freq_min=500.0,
            freq_max=3000.0,
            n_files_per_angle=n_files
        )
        
        # Initialize processors using existing modules
        self.data_processor = DataProcessor(self.config)
        self.transfer_processor = TransferFunctionProcessor(self.config)
        self.audio_processor = AudioProcessor()
        
    def analyze_signals_with_existing_modules(self, root_path: Path, signal_type: str) -> Dict[str, Any]:
        """Analyze signals using existing nmf_localizer modules for consistency."""
        print(f"\nAnalyzing {signal_type} signals using existing modules: {root_path}")
        
        analysis = {
            "overall": {},
            "per_angle": {},
            "processing_info": {
                "config_used": {
                    "sample_rate": self.config.sample_rate,
                    "n_fft": self.config.n_fft,
                    "hop_length": self.config.hop_length,
                    "freq_min": self.config.freq_min,
                    "freq_max": self.config.freq_max
                }
            }
        }
        
        all_signals = []
        all_spectrograms = []
        
        for angle in self.angles:
            angle_dir = root_path / angle
            if not angle_dir.exists():
                print(f"Warning: {angle_dir} not found")
                continue
            
            print(f"  Processing {angle}...")
            
            # Use existing data_processor to load files 
            try:
                angle_data = self.data_processor.load_npy_files(angle_dir, max_files=self.n_files)
                
                if not angle_data:
                    print(f"    No data loaded for {angle}")
                    continue
                
                # Process each file using existing audio processing
                angle_signals = []
                angle_spectrograms = []
                file_stats = []
                
                for i, signal_data in enumerate(angle_data):
                    # Flatten and convert to proper format
                    if signal_data.ndim > 1:
                        signal_data = signal_data.flatten()
                    signal_data = signal_data.astype(np.float32)
                    
                    angle_signals.append(signal_data)
                    all_signals.append(signal_data)
                    
                    # Use AudioProcessor for spectrogram computation (same as tests)
                    try:
                        freqs, times, stft_complex, magnitude = self.audio_processor.compute_stft_spectrogram(
                            signal_data,
                            fs=self.config.sample_rate,
                            nperseg=self.config.n_fft,
                            noverlap=self.config.hop_length
                        )
                        
                        angle_spectrograms.append(magnitude)
                        all_spectrograms.append(magnitude)
                        
                        # Apply frequency filtering (same as in tests)
                        magnitude_tensor = torch.from_numpy(magnitude).float()
                        filtered_magnitude, filtered_freqs = self.audio_processor.apply_frequency_filter(
                            magnitude_tensor, freqs, self.config.freq_min, self.config.freq_max
                        )
                        
                    except Exception as e:
                        print(f"      Error processing spectrogram for file {i}: {e}")
                        continue
                    
                    # File statistics
                    file_stat = {
                        "file_index": i,
                        "length": len(signal_data),
                        "duration_sec": len(signal_data) / self.config.sample_rate,
                        "rms": float(np.sqrt(np.mean(signal_data**2))),
                        "peak": float(np.max(np.abs(signal_data))),
                        "std": float(np.std(signal_data)),
                        "spectrogram_shape": magnitude.shape,
                        "filtered_spectrogram_shape": filtered_magnitude.shape
                    }
                    file_stats.append(file_stat)
                
                # Per-angle analysis
                if angle_signals:
                    angle_concat = np.concatenate(angle_signals)
                    
                    analysis["per_angle"][angle] = {
                        "n_files_processed": len(angle_signals),
                        "total_samples": len(angle_concat),
                        "duration_sec": len(angle_concat) / self.config.sample_rate,
                        "amplitude_stats": {
                            "mean": float(np.mean(angle_concat)),
                            "std": float(np.std(angle_concat)),
                            "min": float(np.min(angle_concat)),
                            "max": float(np.max(angle_concat)),
                            "rms": float(np.sqrt(np.mean(angle_concat**2))),
                            "percentiles": {
                                p: float(np.percentile(angle_concat, p)) 
                                for p in [1, 5, 25, 50, 75, 95, 99]
                            }
                        },
                        "file_details": file_stats,
                        "spectrogram_info": {
                            "n_spectrograms": len(angle_spectrograms),
                            "avg_shape": tuple(np.mean([s.shape for s in angle_spectrograms], axis=0).astype(int)) if angle_spectrograms else None
                        }
                    }
                    
                    # Dynamic range
                    magnitude_vals = np.abs(angle_concat)
                    max_mag = np.max(magnitude_vals)
                    min_mag = np.min(magnitude_vals[magnitude_vals > 0])
                    if min_mag > 0:
                        analysis["per_angle"][angle]["dynamic_range_db"] = float(20 * np.log10(max_mag / min_mag))
                    else:
                        analysis["per_angle"][angle]["dynamic_range_db"] = float('inf')
                        
            except Exception as e:
                print(f"    Error processing {angle}: {e}")
                continue
        
        # Overall statistics
        if all_signals:
            all_concat = np.concatenate(all_signals)
            
            analysis["overall"] = {
                "total_files_processed": len(all_signals),
                "total_samples": len(all_concat),
                "total_duration_sec": len(all_concat) / self.config.sample_rate,
                "amplitude_distribution": {
                    "mean": float(np.mean(all_concat)),
                    "std": float(np.std(all_concat)),
                    "min": float(np.min(all_concat)),
                    "max": float(np.max(all_concat)),
                    "rms": float(np.sqrt(np.mean(all_concat**2))),
                    "skewness": float(np.mean(((all_concat - np.mean(all_concat)) / (np.std(all_concat) + 1e-10))**3)),
                    "kurtosis": float(np.mean(((all_concat - np.mean(all_concat)) / (np.std(all_concat) + 1e-10))**4))
                },
                "signal_quality": {
                    "zero_crossing_rate": float(np.mean(np.abs(np.diff(np.sign(all_concat))) > 0)),
                    "spectrograms_computed": len(all_spectrograms)
                }
            }
            
            # Overall dynamic range
            magnitude_vals = np.abs(all_concat)
            max_mag = np.max(magnitude_vals)
            min_mag = np.min(magnitude_vals[magnitude_vals > 0])
            if min_mag > 0:
                analysis["overall"]["dynamic_range_db"] = float(20 * np.log10(max_mag / min_mag))
            else:
                analysis["overall"]["dynamic_range_db"] = float('inf')
        
        return analysis
    
    def compute_transfer_functions_with_existing_modules(self) -> Dict[str, Any]:
        """Compute H = Y/X transfer functions using existing data_processor."""
        print("\nComputing transfer functions using existing DataProcessor...")
        
        try:
            # Use existing data_processor.estimate_transfer_functions method
            H, angles_tensor, angle_folders, metadata = self.data_processor.estimate_transfer_functions(
                original_root=self.x_root,
                box_root=self.y_root,
                method='xy_correspondence'
            )
            
            print(f"Transfer functions computed successfully:")
            print(f"  H shape: {H.shape}")
            print(f"  Angles: {angles_tensor}")
            print(f"  Metadata: {list(metadata.keys())}")
            
            # Analyze the computed transfer functions
            H_numpy = H.detach().cpu().numpy()
            angles_numpy = angles_tensor.detach().cpu().numpy()
            
            transfer_analysis = {
                "computation_method": "existing_data_processor",
                "method_params": metadata,
                "overall_statistics": {},
                "per_angle_statistics": {},
                "frequency_analysis": {}
            }
            
            # Overall transfer function statistics
            H_magnitude = np.abs(H_numpy)
            transfer_analysis["overall_statistics"] = {
                "h_shape": [int(H.shape[0]), int(H.shape[1])],
                "n_frequencies": int(H.shape[0]),
                "n_angles": int(H.shape[1]),
                "magnitude_stats": {
                    "mean": float(np.mean(H_magnitude)),
                    "std": float(np.std(H_magnitude)),
                    "min": float(np.min(H_magnitude)),
                    "max": float(np.max(H_magnitude)),
                    "median": float(np.median(H_magnitude)),
                    "percentiles": {
                        p: float(np.percentile(H_magnitude, p)) 
                        for p in [5, 25, 50, 75, 95]
                    }
                },
                "phase_stats": {
                    "phase_range": [float(np.min(np.angle(H_numpy))), float(np.max(np.angle(H_numpy)))],
                    "phase_std": float(np.std(np.angle(H_numpy)))
                }
            }
            
            # Dynamic range
            max_mag = np.max(H_magnitude)
            min_mag = np.min(H_magnitude[H_magnitude > 0])
            if min_mag > 0:
                transfer_analysis["overall_statistics"]["dynamic_range_db"] = float(20 * np.log10(max_mag / min_mag))
            else:
                transfer_analysis["overall_statistics"]["dynamic_range_db"] = float('inf')
            
            # Per-angle statistics
            for i, angle_deg in enumerate(angles_numpy):
                angle_h = H_magnitude[:, i]
                
                transfer_analysis["per_angle_statistics"][f"angle_{int(angle_deg)}"] = {
                    "angle_degrees": float(angle_deg),
                    "magnitude_stats": {
                        "mean": float(np.mean(angle_h)),
                        "std": float(np.std(angle_h)),
                        "min": float(np.min(angle_h)),
                        "max": float(np.max(angle_h)),
                        "median": float(np.median(angle_h))
                    },
                    "dynamic_range_db": float(20 * np.log10(np.max(angle_h) / (np.min(angle_h[angle_h > 0]) + 1e-12)))
                }
            
            # Frequency analysis
            if 'freqs' in metadata:
                freqs = metadata['freqs']
                transfer_analysis["frequency_analysis"] = {
                    "frequency_range_hz": [float(freqs[0]), float(freqs[-1])],
                    "n_frequency_bins": len(freqs),
                    "frequency_resolution_hz": float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0,
                    "filtered_range_hz": [self.config.freq_min, self.config.freq_max]
                }
            
            return transfer_analysis
            
        except Exception as e:
            print(f"Error computing transfer functions: {e}")
            return {
                "error": str(e),
                "computation_method": "existing_data_processor_failed",
                "fallback_attempted": False
            }
        
        transfer_analysis = {
            "overall": {},
            "per_angle": {},
            "computation_params": {
                "method": "Welch PSD",
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "window": "hann",
                "sample_rate": self.config.sample_rate
            }
        }
        
        all_h_magnitudes = []
        all_coherences = []
        freq_range = None
        
        for angle in self.angles:
            x_angle_dir = self.x_root / angle
            y_angle_dir = self.y_root / angle
            
            if not (x_angle_dir.exists() and y_angle_dir.exists()):
                print(f"Warning: Missing angle directories for {angle}")
                continue
            
            x_files = sorted([f for f in x_angle_dir.iterdir() if f.suffix == '.npy'])
            y_files = sorted([f for f in y_angle_dir.iterdir() if f.suffix == '.npy'])
            
            angle_h_magnitudes = []
            angle_coherences = []
            file_transfer_stats = []
            
            n_pairs = min(len(x_files), len(y_files), self.n_files)
            print(f"  {angle}: Computing H for {n_pairs} file pairs")
            
            for i in range(n_pairs):
                try:
                    # Load X and Y signals
                    x_signal = np.load(x_files[i]).astype(np.float32).flatten()
                    y_signal = np.load(y_files[i]).astype(np.float32).flatten()
                    
                    # Ensure same length
                    min_length = min(len(x_signal), len(y_signal))
                    x_signal = x_signal[:min_length]
                    y_signal = y_signal[:min_length]
                    
                    # Compute cross-power spectral density using Welch method
                    freqs, Pxx = signal.welch(x_signal, fs=self.config.sample_rate, 
                                            nperseg=self.n_fft, 
                                            noverlap=self.n_fft - self.hop_length,
                                            window='hann')
                    
                    _, Pyy = signal.welch(y_signal, fs=self.config.sample_rate,
                                        nperseg=self.n_fft, 
                                        noverlap=self.n_fft - self.hop_length,
                                        window='hann')
                    
                    _, Pxy = signal.csd(x_signal, y_signal, fs=self.config.sample_rate,
                                       nperseg=self.n_fft, 
                                       noverlap=self.n_fft - self.hop_length,
                                       window='hann')
                    
                    # Transfer function H = Pxy / Pxx
                    # Regularize to avoid division by zero
                    Pxx_reg = Pxx + 1e-12
                    H = Pxy / Pxx_reg
                    
                    # Coherence function
                    Pyy_reg = Pyy + 1e-12
                    coherence = np.abs(Pxy)**2 / (Pxx_reg * Pyy_reg)
                    
                    # Store frequency range (should be same for all)
                    if freq_range is None:
                        freq_range = freqs
                    
                    H_magnitude = np.abs(H)
                    H_phase = np.angle(H)
                    
                    angle_h_magnitudes.append(H_magnitude)
                    angle_coherences.append(coherence)
                    all_h_magnitudes.append(H_magnitude)
                    all_coherences.append(coherence)
                    
                    # File-level transfer function statistics
                    file_stat = {
                        "x_file": x_files[i].name,
                        "y_file": y_files[i].name,
                        "h_magnitude": {
                            "mean": float(np.mean(H_magnitude)),
                            "std": float(np.std(H_magnitude)),
                            "min": float(np.min(H_magnitude)),
                            "max": float(np.max(H_magnitude)),
                            "median": float(np.median(H_magnitude))
                        },
                        "h_phase": {
                            "mean": float(np.mean(H_phase)),
                            "std": float(np.std(H_phase)),
                            "wrapped_range": [float(np.min(H_phase)), float(np.max(H_phase))]
                        },
                        "coherence": {
                            "mean": float(np.mean(coherence)),
                            "std": float(np.std(coherence)),
                            "min": float(np.min(coherence)),
                            "max": float(np.max(coherence)),
                            "median": float(np.median(coherence))
                        }
                    }
                    file_transfer_stats.append(file_stat)
                    
                except Exception as e:
                    print(f"    Error computing H for {angle} file {i}: {e}")
                    continue
            
            # Per-angle transfer function statistics
            if angle_h_magnitudes:
                angle_h_concat = np.concatenate(angle_h_magnitudes)
                angle_coherence_concat = np.concatenate(angle_coherences)
                
                transfer_analysis["per_angle"][angle] = {
                    "n_files": n_pairs,
                    "h_magnitude": {
                        "mean": float(np.mean(angle_h_concat)),
                        "std": float(np.std(angle_h_concat)),
                        "min": float(np.min(angle_h_concat)),
                        "max": float(np.max(angle_h_concat)),
                        "median": float(np.median(angle_h_concat)),
                        "percentiles": {
                            "5": float(np.percentile(angle_h_concat, 5)),
                            "25": float(np.percentile(angle_h_concat, 25)),
                            "75": float(np.percentile(angle_h_concat, 75)),
                            "95": float(np.percentile(angle_h_concat, 95))
                        }
                    },
                    "coherence": {
                        "mean": float(np.mean(angle_coherence_concat)),
                        "std": float(np.std(angle_coherence_concat)),
                        "min": float(np.min(angle_coherence_concat)),
                        "max": float(np.max(angle_coherence_concat)),
                        "median": float(np.median(angle_coherence_concat))
                    },
                    "file_statistics": file_transfer_stats
                }
                
                # Dynamic range of H magnitude
                max_h = np.max(angle_h_concat)
                min_h = np.min(angle_h_concat[angle_h_concat > 0])
                if min_h > 0:
                    transfer_analysis["per_angle"][angle]["h_dynamic_range_db"] = float(20 * np.log10(max_h / min_h))
                else:
                    transfer_analysis["per_angle"][angle]["h_dynamic_range_db"] = float('inf')
        
        # Overall transfer function statistics
        if all_h_magnitudes:
            all_h_concat = np.concatenate(all_h_magnitudes)
            all_coherence_concat = np.concatenate(all_coherences)
            
            transfer_analysis["overall"] = {
                "frequency_range": {
                    "min_hz": float(freq_range[0]),
                    "max_hz": float(freq_range[-1]),
                    "n_bins": len(freq_range),
                    "resolution_hz": float(freq_range[1] - freq_range[0])
                },
                "h_magnitude": {
                    "mean": float(np.mean(all_h_concat)),
                    "std": float(np.std(all_h_concat)),
                    "min": float(np.min(all_h_concat)),
                    "max": float(np.max(all_h_concat)),
                    "median": float(np.median(all_h_concat)),
                    "geometric_mean": float(np.exp(np.mean(np.log(all_h_concat + 1e-12)))),
                },
                "coherence": {
                    "mean": float(np.mean(all_coherence_concat)),
                    "std": float(np.std(all_coherence_concat)),
                    "min": float(np.min(all_coherence_concat)),
                    "max": float(np.max(all_coherence_concat)),
                    "median": float(np.median(all_coherence_concat))
                }
            }
            
            # Overall H dynamic range
            max_h = np.max(all_h_concat)
            min_h = np.min(all_h_concat[all_h_concat > 0])
            if min_h > 0:
                transfer_analysis["overall"]["h_dynamic_range_db"] = float(20 * np.log10(max_h / min_h))
            else:
                transfer_analysis["overall"]["h_dynamic_range_db"] = float('inf')
        
        return transfer_analysis
    
    def generate_dataset_card(self) -> Dict[str, Any]:
        """Generate comprehensive dataset card using existing modules."""
        print("\n" + "="*80)
        print("GENERATING DATASET CARD WITH EXISTING NMF_LOCALIZER MODULES")
        print("="*80)
        
        # Analyze X data using existing modules
        print("\n" + "="*60)
        print("ANALYZING X DATA (Original Signals) - Using Existing Modules")
        print("="*60)
        x_analysis = self.analyze_signals_with_existing_modules(self.x_root, "X_DATA")
        
        # Analyze Y data using existing modules 
        print("\n" + "="*60)
        print("ANALYZING Y DATA (Box Response Signals) - Using Existing Modules")
        print("="*60)
        y_analysis = self.analyze_signals_with_existing_modules(self.y_root, "Y_DATA")
        
        # Compute transfer functions using existing modules
        print("\n" + "="*60)
        print("COMPUTING TRANSFER FUNCTIONS H = Y/X - Using Existing DataProcessor")
        print("="*60)
        h_analysis = self.compute_transfer_functions_with_existing_modules()
        
        # Create comprehensive dataset card
        dataset_card = {
            "metadata": {
                "dataset_name": "Multi-Angle NMF Optimization Dataset",
                "commit_reference": "83c959735d2f959240e764a20ba92c1cf64424eb",
                "description": "Fixed test dataset with X (original) and Y (box response) signals",
                "generation_timestamp": datetime.now().isoformat(),
                "angles": self.angles,
                "files_per_angle": self.n_files,
                "sample_rate_hz": self.config.sample_rate,
                "analysis_parameters": {
                    "n_fft": self.config.n_fft,
                    "hop_length": self.config.hop_length,
                    "freq_min": self.config.freq_min,
                    "freq_max": self.config.freq_max,
                    "window": "hann"
                }
            },
            "x_data_analysis": x_analysis,
            "y_data_analysis": y_analysis, 
            "transfer_function_analysis": h_analysis,
            "summary": {
                "x_signal_characteristics": {
                    "amplitude_range": [x_analysis["overall"]["amplitude_distribution"]["min"], 
                                      x_analysis["overall"]["amplitude_distribution"]["max"]],
                    "rms_global": x_analysis["overall"]["amplitude_distribution"]["rms"],
                    "dynamic_range_db": x_analysis["overall"]["dynamic_range_db"],
                    "total_duration_minutes": x_analysis["overall"]["total_duration_sec"] / 60
                },
                "y_signal_characteristics": {
                    "amplitude_range": [y_analysis["overall"]["amplitude_distribution"]["min"], 
                                      y_analysis["overall"]["amplitude_distribution"]["max"]],
                    "rms_global": y_analysis["overall"]["amplitude_distribution"]["rms"],
                    "dynamic_range_db": y_analysis["overall"]["dynamic_range_db"],
                    "total_duration_minutes": y_analysis["overall"]["total_duration_sec"] / 60
                },
                "transfer_function_characteristics": {
                    "magnitude_range": [h_analysis["overall_statistics"]["magnitude_stats"]["min"], 
                                      h_analysis["overall_statistics"]["magnitude_stats"]["max"]],
                    "magnitude_mean": h_analysis["overall_statistics"]["magnitude_stats"]["mean"],
                    "coherence_range": "N/A - computed from existing modules without coherence",
                    "coherence_mean": "N/A - computed from existing modules without coherence",
                    "frequency_range_hz": [h_analysis["frequency_analysis"]["frequency_range_hz"][0],
                                         h_analysis["frequency_analysis"]["frequency_range_hz"][1]] if "frequency_analysis" in h_analysis else ["N/A", "N/A"]
                },
                "signal_relationships": {
                    "x_to_y_rms_ratio": x_analysis["overall"]["amplitude_distribution"]["rms"] / (y_analysis["overall"]["amplitude_distribution"]["rms"] + 1e-12),
                    "x_to_y_dynamic_range_ratio": x_analysis["overall"]["dynamic_range_db"] / (y_analysis["overall"]["dynamic_range_db"] + 1e-12),
                    "mean_transfer_function_gain": h_analysis["overall_statistics"]["magnitude_stats"]["mean"],
                    "mean_coherence_quality": "N/A - computed from existing modules without coherence"
                }
            }
        }
        
        return dataset_card


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif torch.is_tensor(obj):
        return convert_numpy_types(obj.detach().cpu().numpy().tolist())
    elif hasattr(obj, 'item'):  # Handle numpy scalars with .item() method
        return convert_numpy_types(obj.item())
    else:
        return obj


def test_generate_dataset_card(test_data_paths):
    """Generate comprehensive dataset card for the fixed dataset."""
    # Get paths from conftest
    x_root = test_data_paths["x_root"]
    y_root = test_data_paths["y_root"]
    angles = test_data_paths["validated_angles"]
    n_files = test_data_paths["n_files"]
    
    # Create generator
    generator = DatasetCardGenerator(x_root, y_root, angles, n_files)
    
    # Generate dataset card
    dataset_card = generator.generate_dataset_card()
    
    # Save to file
    output_dir = Path("tests/data")
    output_dir.mkdir(exist_ok=True)
    
    card_file = output_dir / "comprehensive_dataset_card.json"
    with open(card_file, "w") as f:
        # Convert numpy types to Python types for JSON serialization
        dataset_card_serializable = convert_numpy_types(dataset_card)
        json.dump(dataset_card_serializable, f, indent=2)
    
    print(f"\n" + "="*80)
    print("COMPREHENSIVE DATASET CARD GENERATED")
    print("="*80)
    print(f"Saved to: {card_file}")
    
    # Print key summary statistics
    summary = dataset_card["summary"]
    
    print(f"\nX DATA (Original Signals):")
    x_char = summary["x_signal_characteristics"]
    print(f"  Amplitude range: [{x_char['amplitude_range'][0]:.6f}, {x_char['amplitude_range'][1]:.6f}]")
    print(f"  RMS (global): {x_char['rms_global']:.6f}")
    print(f"  Dynamic range: {x_char['dynamic_range_db']:.1f} dB")
    print(f"  Total duration: {x_char['total_duration_minutes']:.1f} minutes")
    
    print(f"\nY DATA (Box Response Signals):")
    y_char = summary["y_signal_characteristics"]
    print(f"  Amplitude range: [{y_char['amplitude_range'][0]:.6f}, {y_char['amplitude_range'][1]:.6f}]")
    print(f"  RMS (global): {y_char['rms_global']:.6f}")
    print(f"  Dynamic range: {y_char['dynamic_range_db']:.1f} dB")
    print(f"  Total duration: {y_char['total_duration_minutes']:.1f} minutes")
    
    print(f"\nTRANSFER FUNCTIONS H = Y/X:")
    h_char = summary["transfer_function_characteristics"]
    print(f"  Magnitude range: [{h_char['magnitude_range'][0]:.6f}, {h_char['magnitude_range'][1]:.6f}]")
    print(f"  Mean magnitude: {h_char['magnitude_mean']:.6f}")
    # Handle coherence values that may be strings
    if isinstance(h_char['coherence_range'], str):
        print(f"  Coherence range: {h_char['coherence_range']}")
    else:
        print(f"  Coherence range: [{h_char['coherence_range'][0]:.3f}, {h_char['coherence_range'][1]:.3f}]")
    
    if isinstance(h_char['coherence_mean'], str):
        print(f"  Mean coherence: {h_char['coherence_mean']}")
    else:
        print(f"  Mean coherence: {h_char['coherence_mean']:.3f}")
    # Handle frequency range that may contain strings
    if isinstance(h_char['frequency_range_hz'][0], str):
        print(f"  Frequency range: {h_char['frequency_range_hz'][0]} - {h_char['frequency_range_hz'][1]} Hz")
    else:
        print(f"  Frequency range: {h_char['frequency_range_hz'][0]:.0f} - {h_char['frequency_range_hz'][1]:.0f} Hz")
    
    print(f"\nSIGNAL RELATIONSHIPS:")
    rel = summary["signal_relationships"]
    print(f"  X/Y RMS ratio: {rel['x_to_y_rms_ratio']:.2f}")
    print(f"  X/Y dynamic range ratio: {rel['x_to_y_dynamic_range_ratio']:.2f}")
    print(f"  Mean transfer gain: {rel['mean_transfer_function_gain']:.6f}")
    # Handle coherence quality that may be a string
    if isinstance(rel['mean_coherence_quality'], str):
        print(f"  Mean coherence quality: {rel['mean_coherence_quality']}")
    else:
        print(f"  Mean coherence quality: {rel['mean_coherence_quality']:.3f}")
    
    # Basic assertions to ensure data is reasonable
    assert dataset_card["metadata"]["angles"] == angles
    assert dataset_card["metadata"]["files_per_angle"] == n_files
    assert "x_data_analysis" in dataset_card
    assert "y_data_analysis" in dataset_card
    assert "transfer_function_analysis" in dataset_card
    
    print(f"\n✅ Dataset card generation completed successfully!")
    return dataset_card


if __name__ == "__main__":
    # Allow running this script directly with pytest
    pytest.main([__file__, "-v", "-s"])