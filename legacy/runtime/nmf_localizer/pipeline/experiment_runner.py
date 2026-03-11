"""
Experiment runner for automated parameter sweeps and batch experiments.
"""

import torch
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple
import logging
import time
import json
from datetime import datetime
from itertools import product
import copy

from ..config.defaults import NMFConfig
from .full_pipeline import NMFLocalizationPipeline

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Automated experiment runner for NMF localization parameter sweeps.
    
    Supports running multiple experiments with different parameter combinations,
    comparing results, and generating comprehensive analysis reports.
    """
    
    def __init__(self, base_config: Optional[NMFConfig] = None):
        """
        Initialize experiment runner.
        
        Args:
            base_config: Base configuration for experiments
        """
        self.base_config = base_config or NMFConfig()
        self.parameter_sweeps = {}
        self.fixed_parameters = {}
        self.experiment_results = []
        
        logger.info("Experiment runner initialized")
        logger.info(f"Base config device: {self.base_config.device}")
    
    def add_parameter_sweep(self, parameter_name: str, values: List[Any]):
        """
        Add parameter sweep for experiment grid.
        
        Args:
            parameter_name: Name of parameter to sweep
            values: List of values to test
        """
        if not hasattr(self.base_config, parameter_name):
            raise ValueError(f"Parameter '{parameter_name}' not found in config")
        
        self.parameter_sweeps[parameter_name] = values
        logger.info(f"Added parameter sweep: {parameter_name} = {values}")
    
    def set_fixed_parameter(self, parameter_name: str, value: Any):
        """
        Set fixed parameter value for all experiments.
        
        Args:
            parameter_name: Name of parameter to fix
            value: Fixed value
        """
        if not hasattr(self.base_config, parameter_name):
            raise ValueError(f"Parameter '{parameter_name}' not found in config")
        
        self.fixed_parameters[parameter_name] = value
        logger.info(f"Set fixed parameter: {parameter_name} = {value}")
    
    def generate_experiment_configs(self) -> List[Tuple[str, NMFConfig]]:
        """
        Generate all experiment configurations from parameter sweeps.
        
        Returns:
            List of (experiment_id, config) tuples
        """
        if not self.parameter_sweeps:
            logger.warning("No parameter sweeps defined, using base config only")
            return [("base_config", self.base_config)]
        
        # Get parameter names and values
        param_names = list(self.parameter_sweeps.keys())
        param_values = list(self.parameter_sweeps.values())
        
        # Generate all combinations
        experiment_configs = []
        for i, combination in enumerate(product(*param_values)):
            # Create experiment config
            config = copy.deepcopy(self.base_config)
            
            # Apply parameter combination
            param_dict = {}
            for param_name, value in zip(param_names, combination):
                setattr(config, param_name, value)
                param_dict[param_name] = value
            
            # Apply fixed parameters
            for param_name, value in self.fixed_parameters.items():
                setattr(config, param_name, value)
                param_dict[param_name] = value
            
            # Generate experiment ID
            param_str = "_".join(f"{k}-{v}" for k, v in param_dict.items() if k in param_names)
            experiment_id = f"exp_{i:03d}_{param_str}"
            
            experiment_configs.append((experiment_id, config))
        
        logger.info(f"Generated {len(experiment_configs)} experiment configurations")
        return experiment_configs
    
    def run_experiments(
        self,
        data_root: Union[str, Path],
        output_root: Union[str, Path],
        n_sources: int = 1,
        max_experiments: Optional[int] = None,
        save_models: bool = False,
        force_rerun: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Run all experiment configurations.
        
        Args:
            data_root: Root directory containing data
            output_root: Root output directory for all experiments
            n_sources: Number of sources to localize
            max_experiments: Maximum number of experiments to run
            save_models: Whether to save trained models
            force_rerun: Whether to rerun existing experiments
            
        Returns:
            List of experiment results
        """
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        
        # Generate experiment configurations
        experiment_configs = self.generate_experiment_configs()
        
        if max_experiments:
            experiment_configs = experiment_configs[:max_experiments]
        
        logger.info("=" * 80)
        logger.info("STARTING BATCH EXPERIMENT RUN")
        logger.info("=" * 80)
        logger.info(f"Data root: {data_root}")
        logger.info(f"Output root: {output_root}")
        logger.info(f"Number of experiments: {len(experiment_configs)}")
        logger.info(f"Parameter sweeps: {list(self.parameter_sweeps.keys())}")
        
        batch_start_time = time.time()
        results = []
        failed_experiments = []
        
        for exp_idx, (experiment_id, config) in enumerate(experiment_configs):
            logger.info(f"\n--- EXPERIMENT {exp_idx + 1}/{len(experiment_configs)}: {experiment_id} ---")
            
            # Check if experiment already exists
            exp_output_dir = output_root / experiment_id
            results_file = exp_output_dir / "experiment_results.pth"
            
            if results_file.exists() and not force_rerun:
                logger.info(f"Experiment {experiment_id} already exists, loading results...")
                try:
                    existing_results = torch.load(results_file, weights_only=False)
                    existing_results['experiment_id'] = experiment_id
                    results.append(existing_results)
                    logger.info(f"Loaded existing results: {existing_results.get('stages', {}).get('evaluation', {}).get('results', {}).get('accuracy', 'N/A')}% accuracy")
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load existing results: {e}, rerunning...")
            
            try:
                exp_start_time = time.time()
                
                # Create pipeline with experiment config
                pipeline = NMFLocalizationPipeline(config)
                
                # Run experiment
                exp_results = pipeline.run_full_experiment(
                    data_root=data_root,
                    output_dir=exp_output_dir,
                    n_sources=n_sources,
                    save_models=save_models,
                    force_reprocess=False  # Only reprocess data once
                )
                
                # Add experiment metadata
                exp_results['experiment_id'] = experiment_id
                exp_results['experiment_index'] = exp_idx
                exp_results['experiment_duration'] = time.time() - exp_start_time
                exp_results['parameter_combination'] = {
                    param: getattr(config, param) for param in self.parameter_sweeps.keys()
                }
                
                results.append(exp_results)
                
                # Log experiment completion
                accuracy = exp_results.get('stages', {}).get('evaluation', {}).get('results', {}).get('accuracy', 0)
                logger.info(f"Experiment {experiment_id} completed: {accuracy:.1f}% accuracy "
                           f"in {exp_results['experiment_duration']:.1f}s")
                
            except Exception as e:
                logger.error(f"Experiment {experiment_id} failed: {e}")
                
                failed_exp = {
                    'experiment_id': experiment_id,
                    'experiment_index': exp_idx,
                    'error': str(e),
                    'parameter_combination': {
                        param: getattr(config, param) for param in self.parameter_sweeps.keys()
                    },
                    'timestamp': datetime.now().isoformat()
                }
                failed_experiments.append(failed_exp)
                
                # Save failed experiment info
                failed_file = exp_output_dir / "experiment_failed.json"
                exp_output_dir.mkdir(parents=True, exist_ok=True)
                with open(failed_file, 'w') as f:
                    json.dump(failed_exp, f, indent=2)
                
                continue
        
        # Store results
        self.experiment_results = results
        
        batch_duration = time.time() - batch_start_time
        
        # Create batch summary
        batch_summary = self._create_batch_summary(
            results, failed_experiments, batch_duration, output_root
        )
        
        logger.info("=" * 80)
        logger.info("BATCH EXPERIMENT COMPLETED")
        logger.info(f"Successful experiments: {len(results)}")
        logger.info(f"Failed experiments: {len(failed_experiments)}")
        logger.info(f"Total duration: {batch_duration:.1f}s")
        logger.info(f"Results saved to: {output_root}")
        logger.info("=" * 80)
        
        return results
    
    def compare_results(
        self, 
        results: Optional[List[Dict[str, Any]]] = None,
        save_comparison: bool = True,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare results across experiments.
        
        Args:
            results: Experiment results (uses self.experiment_results if None)
            save_comparison: Whether to save comparison results
            output_path: Output path for comparison results
            
        Returns:
            Comparison analysis
        """
        results = results or self.experiment_results
        if not results:
            raise ValueError("No experiment results available for comparison")
        
        logger.info(f"Comparing {len(results)} experiment results")
        
        # Extract key metrics
        comparison_data = []
        for result in results:
            if not result.get('success', False):
                continue
                
            eval_results = result.get('stages', {}).get('evaluation', {}).get('results', {})
            if not eval_results:
                continue
            
            param_combo = result.get('parameter_combination', {})
            
            data_point = {
                'experiment_id': result.get('experiment_id', 'unknown'),
                'accuracy': eval_results.get('accuracy', 0),
                'mean_error': eval_results.get('mean_error', float('inf')),
                'processing_time': eval_results.get('mean_processing_time', 0),
                'total_duration': result.get('total_duration', 0),
                **param_combo
            }
            
            comparison_data.append(data_point)
        
        if not comparison_data:
            logger.warning("No valid results for comparison")
            return {'error': 'No valid results'}
        
        # Find best and worst performers
        best_accuracy = max(comparison_data, key=lambda x: x['accuracy'])
        worst_accuracy = min(comparison_data, key=lambda x: x['accuracy'])
        
        best_error = min(comparison_data, key=lambda x: x['mean_error'] if x['mean_error'] != float('inf') else float('inf'))
        fastest = min(comparison_data, key=lambda x: x['processing_time'])
        
        # Parameter analysis
        param_analysis = self._analyze_parameter_effects(comparison_data)
        
        # Create comparison summary
        comparison = {
            'summary': {
                'n_experiments': len(comparison_data),
                'best_accuracy': {
                    'experiment_id': best_accuracy['experiment_id'],
                    'accuracy': best_accuracy['accuracy'],
                    'parameters': {k: v for k, v in best_accuracy.items() 
                                 if k in self.parameter_sweeps.keys()}
                },
                'worst_accuracy': {
                    'experiment_id': worst_accuracy['experiment_id'],
                    'accuracy': worst_accuracy['accuracy'],
                    'parameters': {k: v for k, v in worst_accuracy.items() 
                                 if k in self.parameter_sweeps.keys()}
                },
                'best_error': {
                    'experiment_id': best_error['experiment_id'],
                    'mean_error': best_error['mean_error'],
                    'parameters': {k: v for k, v in best_error.items() 
                                 if k in self.parameter_sweeps.keys()}
                },
                'fastest_processing': {
                    'experiment_id': fastest['experiment_id'],
                    'processing_time': fastest['processing_time'],
                    'parameters': {k: v for k, v in fastest.items() 
                                 if k in self.parameter_sweeps.keys()}
                }
            },
            'parameter_analysis': param_analysis,
            'all_results': comparison_data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save comparison if requested
        if save_comparison and output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save(comparison, output_file)
            
            # Also save as JSON for readability
            json_file = output_file.with_suffix('.json')
            with open(json_file, 'w') as f:
                json.dump(comparison, f, indent=2, default=str)
            
            logger.info(f"Comparison results saved to: {output_file}")
        
        # Log comparison summary
        logger.info("=== EXPERIMENT COMPARISON SUMMARY ===")
        logger.info(f"Best accuracy: {best_accuracy['accuracy']:.1f}% ({best_accuracy['experiment_id']})")
        logger.info(f"Worst accuracy: {worst_accuracy['accuracy']:.1f}% ({worst_accuracy['experiment_id']})")
        if best_error['mean_error'] != float('inf'):
            logger.info(f"Best error: {best_error['mean_error']:.1f}° ({best_error['experiment_id']})")
        logger.info(f"Fastest processing: {fastest['processing_time']*1000:.1f}ms ({fastest['experiment_id']})")
        
        return comparison
    
    def _analyze_parameter_effects(self, comparison_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the effect of different parameters on performance."""
        param_analysis = {}
        
        for param_name in self.parameter_sweeps.keys():
            param_values = list(set(data[param_name] for data in comparison_data if param_name in data))
            param_effects = {}
            
            for value in param_values:
                # Find all experiments with this parameter value
                matching_exps = [data for data in comparison_data if data.get(param_name) == value]
                
                if matching_exps:
                    accuracies = [exp['accuracy'] for exp in matching_exps]
                    errors = [exp['mean_error'] for exp in matching_exps if exp['mean_error'] != float('inf')]
                    times = [exp['processing_time'] for exp in matching_exps]
                    
                    param_effects[str(value)] = {
                        'mean_accuracy': sum(accuracies) / len(accuracies) if accuracies else 0,
                        'std_accuracy': torch.std(torch.tensor(accuracies)).item() if len(accuracies) > 1 else 0,
                        'mean_error': sum(errors) / len(errors) if errors else float('inf'),
                        'mean_processing_time': sum(times) / len(times) if times else 0,
                        'n_experiments': len(matching_exps)
                    }
            
            param_analysis[param_name] = param_effects
        
        return param_analysis
    
    def _create_batch_summary(
        self,
        results: List[Dict[str, Any]], 
        failed_experiments: List[Dict[str, Any]],
        batch_duration: float,
        output_root: Path
    ) -> Dict[str, Any]:
        """Create comprehensive batch summary."""
        
        # Calculate statistics
        successful_results = [r for r in results if r.get('success', False)]
        
        if successful_results:
            accuracies = [r['stages']['evaluation']['results']['accuracy'] 
                         for r in successful_results 
                         if 'stages' in r and 'evaluation' in r['stages']]
            
            durations = [r.get('total_duration', 0) for r in successful_results]
            
            best_result = max(successful_results, 
                            key=lambda x: x.get('stages', {}).get('evaluation', {}).get('results', {}).get('accuracy', 0))
        else:
            accuracies = []
            durations = []
            best_result = None
        
        # Create summary
        summary = {
            'batch_info': {
                'total_experiments': len(results) + len(failed_experiments),
                'successful_experiments': len(successful_results),
                'failed_experiments': len(failed_experiments),
                'success_rate': len(successful_results) / (len(results) + len(failed_experiments)) * 100 if (results or failed_experiments) else 0,
                'batch_duration': batch_duration,
                'timestamp': datetime.now().isoformat()
            },
            'performance_statistics': {
                'mean_accuracy': sum(accuracies) / len(accuracies) if accuracies else 0,
                'std_accuracy': torch.std(torch.tensor(accuracies)).item() if len(accuracies) > 1 else 0,
                'max_accuracy': max(accuracies) if accuracies else 0,
                'min_accuracy': min(accuracies) if accuracies else 0,
                'mean_duration': sum(durations) / len(durations) if durations else 0
            },
            'parameter_sweeps': self.parameter_sweeps,
            'fixed_parameters': self.fixed_parameters,
            'best_experiment': {
                'experiment_id': best_result.get('experiment_id', 'none') if best_result else 'none',
                'accuracy': best_result.get('stages', {}).get('evaluation', {}).get('results', {}).get('accuracy', 0) if best_result else 0,
                'parameters': best_result.get('parameter_combination', {}) if best_result else {}
            } if best_result else None,
            'failed_experiments': failed_experiments
        }
        
        # Save summary
        summary_file = output_root / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create human-readable report
        self._create_batch_report(summary, output_root)
        
        logger.info(f"Batch summary saved to: {summary_file}")
        
        return summary
    
    def _create_batch_report(self, summary: Dict[str, Any], output_root: Path):
        """Create human-readable batch report."""
        report = []
        report.append("NMF LOCALIZATION BATCH EXPERIMENT REPORT")
        report.append("=" * 60)
        
        # Batch info
        batch_info = summary['batch_info']
        report.append(f"Timestamp: {batch_info['timestamp']}")
        report.append(f"Total Experiments: {batch_info['total_experiments']}")
        report.append(f"Successful: {batch_info['successful_experiments']}")
        report.append(f"Failed: {batch_info['failed_experiments']}")
        report.append(f"Success Rate: {batch_info['success_rate']:.1f}%")
        report.append(f"Batch Duration: {batch_info['batch_duration']:.1f}s")
        
        # Parameter sweeps
        report.append(f"\nParameter Sweeps:")
        for param, values in self.parameter_sweeps.items():
            report.append(f"  {param}: {values}")
        
        if self.fixed_parameters:
            report.append(f"\nFixed Parameters:")
            for param, value in self.fixed_parameters.items():
                report.append(f"  {param}: {value}")
        
        # Performance statistics
        perf_stats = summary['performance_statistics']
        report.append(f"\nPerformance Statistics:")
        report.append(f"  Mean Accuracy: {perf_stats['mean_accuracy']:.1f}% (±{perf_stats['std_accuracy']:.1f}%)")
        report.append(f"  Max Accuracy: {perf_stats['max_accuracy']:.1f}%")
        report.append(f"  Min Accuracy: {perf_stats['min_accuracy']:.1f}%")
        report.append(f"  Mean Duration: {perf_stats['mean_duration']:.1f}s")
        
        # Best experiment
        if summary['best_experiment']:
            best_exp = summary['best_experiment']
            report.append(f"\nBest Experiment: {best_exp['experiment_id']}")
            report.append(f"  Accuracy: {best_exp['accuracy']:.1f}%")
            report.append(f"  Parameters:")
            for param, value in best_exp['parameters'].items():
                report.append(f"    {param}: {value}")
        
        # Failed experiments
        if summary['failed_experiments']:
            report.append(f"\nFailed Experiments:")
            for failed in summary['failed_experiments'][:5]:  # Show first 5
                report.append(f"  {failed['experiment_id']}: {failed['error']}")
            if len(summary['failed_experiments']) > 5:
                report.append(f"  ... and {len(summary['failed_experiments']) - 5} more")
        
        report.append("\n" + "=" * 60)
        
        # Save report
        with open(output_root / "batch_report.txt", 'w') as f:
            f.write("\n".join(report))
        
    def clear_experiments(self):
        """Clear stored experiment results."""
        self.experiment_results = []
        self.parameter_sweeps = {}
        self.fixed_parameters = {}
        logger.info("Experiment runner cleared")