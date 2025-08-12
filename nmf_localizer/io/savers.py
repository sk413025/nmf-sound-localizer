"""
Data saving utilities for NMF localization.
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

from ..config.defaults import DataPack, NMFConfig

logger = logging.getLogger(__name__)


class DataSaver:
    """Data saving utilities."""
    
    @staticmethod
    def save_data_pack(
        data_pack: DataPack,
        path: Union[str, Path],
        include_test_data: bool = True
    ):
        """
        Save data pack to file.
        
        Args:
            data_pack: Data pack to save
            path: Output file path
            include_test_data: Whether to include test data (can be large)
        """
        if not include_test_data and data_pack.test_data:
            # Create copy without test data
            temp_pack = DataPack()
            temp_pack.transfer_functions = data_pack.transfer_functions
            temp_pack.angles = data_pack.angles
            temp_pack.angle_names = data_pack.angle_names
            temp_pack.freqs = data_pack.freqs
            temp_pack.speaker_data = data_pack.speaker_data
            temp_pack.metadata = data_pack.metadata
            temp_pack.config = data_pack.config
            temp_pack.save(str(path))
        else:
            data_pack.save(str(path))
        
        logger.info(f"Data pack saved to: {path}")
    
    @staticmethod
    def save_config(config: NMFConfig, path: Union[str, Path]):
        """Save configuration to file."""
        config_dict = config.to_dict()
        
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            torch.save({'config': config_dict}, path)
        
        logger.info(f"Configuration saved to: {path}")
    
    @staticmethod
    def save_model_checkpoint(
        model_state: Dict[str, Any],
        config: NMFConfig,
        path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save model checkpoint with config and metadata.
        
        Args:
            model_state: Model state dictionary
            config: Model configuration
            path: Output file path
            metadata: Additional metadata to save
        """
        checkpoint = {
            'model_state': model_state,
            'config': config.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        if metadata:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, path)
        logger.info(f"Model checkpoint saved to: {path}")


class ResultSaver:
    """Result saving utilities."""
    
    @staticmethod
    def save_experiment_results(
        results: Dict[str, Any],
        output_dir: Union[str, Path],
        experiment_name: Optional[str] = None,
        save_formats: List[str] = ['pth', 'json']
    ):
        """
        Save experiment results in multiple formats.
        
        Args:
            results: Experiment results
            output_dir: Output directory
            experiment_name: Name of experiment (uses timestamp if None)
            save_formats: List of formats to save ('pth', 'json', 'txt')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        saved_files = []
        
        # Save as PyTorch file (complete)
        if 'pth' in save_formats:
            pth_path = output_dir / f"{experiment_name}_results.pth"
            torch.save(results, pth_path)
            saved_files.append(pth_path)
        
        # Save as JSON (readable, but may lose some data types)
        if 'json' in save_formats:
            json_path = output_dir / f"{experiment_name}_results.json"
            
            # Create JSON-serializable version
            json_results = ResultSaver._make_json_serializable(results)
            
            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            saved_files.append(json_path)
        
        # Save as text report (summary only)
        if 'txt' in save_formats:
            txt_path = output_dir / f"{experiment_name}_report.txt"
            report = ResultSaver._create_text_report(results)
            
            with open(txt_path, 'w') as f:
                f.write(report)
            saved_files.append(txt_path)
        
        logger.info(f"Experiment results saved in {len(saved_files)} formats to: {output_dir}")
        return saved_files
    
    @staticmethod
    def save_comparison_results(
        comparison: Dict[str, Any],
        output_dir: Union[str, Path],
        comparison_name: Optional[str] = None
    ):
        """
        Save comparison results.
        
        Args:
            comparison: Comparison results
            output_dir: Output directory  
            comparison_name: Name for comparison files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if comparison_name is None:
            comparison_name = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save complete comparison
        pth_path = output_dir / f"{comparison_name}.pth"
        torch.save(comparison, pth_path)
        
        # Save JSON version
        json_path = output_dir / f"{comparison_name}.json"
        json_comparison = ResultSaver._make_json_serializable(comparison)
        
        with open(json_path, 'w') as f:
            json.dump(json_comparison, f, indent=2)
        
        # Create comparison report
        report_path = output_dir / f"{comparison_name}_report.txt"
        report = ResultSaver._create_comparison_report(comparison)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Comparison results saved to: {output_dir}")
    
    @staticmethod
    def save_batch_summary(
        batch_results: Dict[str, Any],
        output_dir: Union[str, Path]
    ):
        """
        Save batch experiment summary.
        
        Args:
            batch_results: Batch results
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary as JSON
        summary_path = output_dir / "batch_summary.json"
        
        # Extract summary data
        summary = {
            'n_successful': batch_results.get('n_successful', 0),
            'n_failed': batch_results.get('n_failed', 0),
            'n_total': batch_results.get('n_total', 0),
            'success_rate': (batch_results.get('n_successful', 0) / 
                           max(1, batch_results.get('n_total', 1))) * 100,
            'timestamp': datetime.now().isoformat()
        }
        
        if 'batch_summary' in batch_results and batch_results['batch_summary']:
            summary.update(batch_results['batch_summary'])
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create performance summary
        if batch_results.get('experiment_results'):
            perf_summary = ResultSaver._extract_performance_metrics(
                batch_results['experiment_results']
            )
            
            perf_path = output_dir / "performance_summary.json"
            with open(perf_path, 'w') as f:
                json.dump(perf_summary, f, indent=2)
        
        logger.info(f"Batch summary saved to: {output_dir}")
    
    @staticmethod
    def _make_json_serializable(obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if torch.is_tensor(obj):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: ResultSaver._make_json_serializable(value) 
                   for key, value in obj.items()}
        elif isinstance(obj, list):
            return [ResultSaver._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif obj is None or isinstance(obj, (str, bool)):
            return obj
        else:
            return str(obj)  # Fallback to string representation
    
    @staticmethod
    def _create_text_report(results: Dict[str, Any]) -> str:
        """Create human-readable text report."""
        report = []
        report.append("NMF LOCALIZATION EXPERIMENT REPORT")
        report.append("=" * 50)
        report.append(f"Timestamp: {results.get('timestamp', 'unknown')}")
        report.append(f"Success: {results.get('success', False)}")
        
        if 'config' in results:
            config = results['config']
            report.append(f"\nConfiguration:")
            report.append(f"  Beta: {config.get('beta', 'N/A')}")
            report.append(f"  Lambda Group: {config.get('lambda_group', 'N/A')}")
            report.append(f"  Device: {config.get('device', 'N/A')}")
        
        if 'stages' in results and 'evaluation' in results['stages']:
            eval_results = results['stages']['evaluation']['results']
            report.append(f"\nResults:")
            report.append(f"  Accuracy: {eval_results.get('accuracy', 0):.1f}%")
            report.append(f"  Mean Error: {eval_results.get('mean_error', 0):.1f}°")
            report.append(f"  Processing Time: {eval_results.get('mean_processing_time', 0)*1000:.1f}ms")
        
        total_duration = results.get('total_duration', 0)
        report.append(f"\nTotal Duration: {total_duration:.1f}s")
        
        return "\n".join(report)
    
    @staticmethod
    def _create_comparison_report(comparison: Dict[str, Any]) -> str:
        """Create comparison report."""
        report = []
        report.append("NMF LOCALIZATION COMPARISON REPORT")
        report.append("=" * 50)
        
        summary = comparison.get('summary', {})
        
        report.append(f"Number of Experiments: {summary.get('n_experiments', 0)}")
        
        if 'best_accuracy' in summary:
            best = summary['best_accuracy']
            report.append(f"\nBest Accuracy: {best.get('accuracy', 0):.1f}% ({best.get('experiment_id', 'unknown')})")
            
        if 'parameter_analysis' in comparison:
            report.append(f"\nParameter Analysis:")
            for param, analysis in comparison['parameter_analysis'].items():
                report.append(f"  {param}:")
                for value, stats in analysis.items():
                    acc = stats.get('mean_accuracy', 0)
                    report.append(f"    {value}: {acc:.1f}% accuracy")
        
        return "\n".join(report)
    
    @staticmethod
    def _extract_performance_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract performance metrics from experiment results."""
        accuracies = []
        errors = []
        times = []
        
        for result in results:
            if not result.get('success', False):
                continue
                
            eval_results = result.get('stages', {}).get('evaluation', {}).get('results', {})
            if eval_results:
                accuracies.append(eval_results.get('accuracy', 0))
                mean_error = eval_results.get('mean_error', float('inf'))
                if mean_error != float('inf'):
                    errors.append(mean_error)
                times.append(eval_results.get('mean_processing_time', 0))
        
        metrics = {
            'accuracy': {
                'mean': float(np.mean(accuracies)) if accuracies else 0,
                'std': float(np.std(accuracies)) if accuracies else 0,
                'min': float(np.min(accuracies)) if accuracies else 0,
                'max': float(np.max(accuracies)) if accuracies else 0
            },
            'error': {
                'mean': float(np.mean(errors)) if errors else float('inf'),
                'std': float(np.std(errors)) if errors else 0,
                'min': float(np.min(errors)) if errors else float('inf'),
                'max': float(np.max(errors)) if errors else float('inf')
            },
            'processing_time': {
                'mean': float(np.mean(times)) if times else 0,
                'std': float(np.std(times)) if times else 0,
                'min': float(np.min(times)) if times else 0,
                'max': float(np.max(times)) if times else 0
            },
            'n_experiments': len(accuracies)
        }
        
        return metrics