"""
Data loading utilities for NMF localization.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging

from ..config.defaults import DataPack, NMFConfig

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loading utilities."""
    
    @staticmethod
    def load_data_pack(path: Union[str, Path]) -> DataPack:
        """Load a complete data pack from file."""
        return DataPack.load(str(path))
    
    @staticmethod
    def load_config(path: Union[str, Path]) -> NMFConfig:
        """Load configuration from file."""
        data = torch.load(path, weights_only=False)
        
        if isinstance(data, dict) and 'config' in data:
            return NMFConfig.from_dict(data['config'])
        elif isinstance(data, dict):
            return NMFConfig.from_dict(data)
        else:
            raise ValueError(f"Invalid config file format: {path}")
    
    @staticmethod
    def load_angle_data(
        root_path: Union[str, Path],
        angles: Optional[List[int]] = None,
        max_files_per_angle: Optional[int] = None
    ) -> Dict[int, List[np.ndarray]]:
        """
        Load data from angle directories.
        
        Args:
            root_path: Root directory containing angle folders
            angles: Specific angles to load (loads all if None)
            max_files_per_angle: Maximum files to load per angle
            
        Returns:
            Dictionary mapping angle -> list of data arrays
        """
        root_path = Path(root_path)
        
        if not root_path.exists():
            raise FileNotFoundError(f"Root path not found: {root_path}")
        
        # Find angle directories
        angle_dirs = sorted([d for d in root_path.iterdir() 
                           if d.is_dir() and d.name.startswith('angle_')])
        
        if not angle_dirs:
            raise FileNotFoundError(f"No angle directories found in {root_path}")
        
        angle_data = {}
        
        for angle_dir in angle_dirs:
            # Extract angle number
            angle_str = angle_dir.name.replace('angle_', '')
            angle_num = int(angle_str)
            
            # Skip if not in requested angles
            if angles is not None and angle_num not in angles:
                continue
            
            # Load .npy files from this angle
            npy_files = sorted(list(angle_dir.glob('*.npy')))
            
            if max_files_per_angle:
                npy_files = npy_files[:max_files_per_angle]
            
            angle_data[angle_num] = []
            for file_path in npy_files:
                try:
                    data = np.load(file_path)
                    angle_data[angle_num].append(data)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    continue
            
            logger.info(f"Loaded {len(angle_data[angle_num])} files for angle {angle_num}")
        
        return angle_data


class ResultLoader:
    """Result loading utilities."""
    
    @staticmethod
    def load_experiment_results(path: Union[str, Path]) -> Dict[str, Any]:
        """Load experiment results from file."""
        return torch.load(path, weights_only=False)
    
    @staticmethod
    def load_evaluation_results(path: Union[str, Path]) -> Dict[str, Any]:
        """Load evaluation results from file."""
        data = torch.load(path, weights_only=False)
        return data.get('evaluation_results', data)
    
    @staticmethod
    def load_batch_results(
        batch_root: Union[str, Path],
        load_failed: bool = False
    ) -> Dict[str, Any]:
        """
        Load all results from a batch experiment directory.
        
        Args:
            batch_root: Root directory of batch experiment
            load_failed: Whether to load failed experiment info
            
        Returns:
            Dictionary with batch results and summary
        """
        batch_root = Path(batch_root)
        
        if not batch_root.exists():
            raise FileNotFoundError(f"Batch root not found: {batch_root}")
        
        # Load batch summary if available
        summary_file = batch_root / "batch_summary.json"
        batch_summary = None
        if summary_file.exists():
            import json
            with open(summary_file, 'r') as f:
                batch_summary = json.load(f)
        
        # Find experiment directories
        exp_dirs = sorted([d for d in batch_root.iterdir() 
                          if d.is_dir() and d.name.startswith('exp_')])
        
        experiment_results = []
        failed_experiments = []
        
        for exp_dir in exp_dirs:
            exp_id = exp_dir.name
            
            # Try to load successful results
            results_file = exp_dir / "experiment_results.pth"
            if results_file.exists():
                try:
                    results = torch.load(results_file, weights_only=False)
                    results['experiment_id'] = exp_id
                    experiment_results.append(results)
                except Exception as e:
                    logger.warning(f"Failed to load results for {exp_id}: {e}")
                    
            elif load_failed:
                # Try to load failed experiment info
                failed_file = exp_dir / "experiment_failed.json"
                if failed_file.exists():
                    try:
                        import json
                        with open(failed_file, 'r') as f:
                            failed_info = json.load(f)
                        failed_experiments.append(failed_info)
                    except Exception as e:
                        logger.warning(f"Failed to load failed info for {exp_id}: {e}")
        
        batch_results = {
            'batch_summary': batch_summary,
            'experiment_results': experiment_results,
            'failed_experiments': failed_experiments,
            'n_successful': len(experiment_results),
            'n_failed': len(failed_experiments),
            'n_total': len(exp_dirs)
        }
        
        logger.info(f"Loaded batch results: {len(experiment_results)} successful, "
                   f"{len(failed_experiments)} failed experiments")
        
        return batch_results
    
    @staticmethod
    def extract_performance_summary(
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract performance summary from experiment results.
        
        Args:
            results: List of experiment results
            
        Returns:
            Performance summary statistics
        """
        if not results:
            return {'error': 'No results provided'}
        
        # Extract performance metrics
        accuracies = []
        errors = []
        processing_times = []
        parameters = []
        
        for result in results:
            if not result.get('success', False):
                continue
                
            eval_results = result.get('stages', {}).get('evaluation', {}).get('results', {})
            if not eval_results:
                continue
            
            accuracies.append(eval_results.get('accuracy', 0))
            mean_error = eval_results.get('mean_error', float('inf'))
            if mean_error != float('inf'):
                errors.append(mean_error)
            processing_times.append(eval_results.get('mean_processing_time', 0))
            
            # Extract parameter combination
            param_combo = result.get('parameter_combination', {})
            parameters.append(param_combo)
        
        # Compute statistics
        summary = {
            'n_results': len(accuracies),
            'accuracy_stats': {
                'mean': np.mean(accuracies) if accuracies else 0,
                'std': np.std(accuracies) if accuracies else 0,
                'min': np.min(accuracies) if accuracies else 0,
                'max': np.max(accuracies) if accuracies else 0,
                'median': np.median(accuracies) if accuracies else 0
            },
            'error_stats': {
                'mean': np.mean(errors) if errors else float('inf'),
                'std': np.std(errors) if errors else 0,
                'min': np.min(errors) if errors else float('inf'),
                'max': np.max(errors) if errors else float('inf'),
                'median': np.median(errors) if errors else float('inf')
            },
            'processing_time_stats': {
                'mean': np.mean(processing_times) if processing_times else 0,
                'std': np.std(processing_times) if processing_times else 0,
                'min': np.min(processing_times) if processing_times else 0,
                'max': np.max(processing_times) if processing_times else 0
            }
        }
        
        # Find best performing experiment
        if accuracies:
            best_idx = np.argmax(accuracies)
            summary['best_experiment'] = {
                'experiment_id': results[best_idx].get('experiment_id', 'unknown'),
                'accuracy': accuracies[best_idx],
                'parameters': parameters[best_idx] if best_idx < len(parameters) else {}
            }
        
        return summary