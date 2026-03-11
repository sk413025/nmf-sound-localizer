"""
Evaluation module for NMF localization performance.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import logging
import time
from pathlib import Path

from ..config.defaults import NMFConfig
from .localizer import NMFSoundLocalizer

logger = logging.getLogger(__name__)


class Evaluator:
    """Comprehensive evaluator for NMF localization performance."""
    
    def __init__(self, config: NMFConfig):
        self.config = config
        self.device = config.device
        
    def evaluate_localization(
        self,
        model: NMFSoundLocalizer,
        test_data: List[Dict],
        n_sources: int = 1,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate localization performance on test data.
        
        Args:
            model: Trained NMF localizer
            test_data: List of test examples with 'mixture' and 'directions' keys
            n_sources: Number of sources to localize
            verbose: Whether to show progress bar
            
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        if not test_data:
            raise ValueError("Empty test data")
        
        logger.info(f"Evaluating localization performance on {len(test_data)} examples")
        logger.info(f"Target sources: {n_sources}")
        
        # Initialize metrics storage
        accuracies = []
        errors = []
        per_source_correct = []
        processing_times = []
        failed_examples = []
        
        # Detailed per-angle analysis
        angle_stats = {}  # angle -> {'correct': count, 'total': count, 'errors': []}
        
        iterator = tqdm(test_data, desc="Evaluating") if verbose else test_data
        
        for idx, example in enumerate(iterator):
            try:
                # Extract test example
                mixture = example['mixture'].to(self.device)
                true_directions = example['directions']
                
                # Ensure true_directions is a list
                if not isinstance(true_directions, list):
                    true_directions = [true_directions]
                
                # Measure processing time
                start_time = time.time()
                
                # Localize
                results = model.localize(mixture, n_sources=n_sources)
                estimated_directions = results['directions']
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # Convert estimated directions to list for analysis
                if torch.is_tensor(estimated_directions):
                    est_dirs_list = estimated_directions.cpu().tolist()
                else:
                    est_dirs_list = list(estimated_directions)
                
                # Compute accuracy and error
                accuracy, mean_error = model.compute_accuracy(
                    estimated_directions, 
                    true_directions,
                    tolerance=self.config.tolerance_degrees
                )
                
                accuracies.append(accuracy)
                if mean_error != float('inf'):
                    errors.append(mean_error)
                    
                # Per-source accuracy analysis
                for est_dir in est_dirs_list:
                    # Check if any true direction is within tolerance
                    min_diff = min(
                        min(abs(est_dir - true_dir), 360 - abs(est_dir - true_dir))
                        for true_dir in true_directions
                    )
                    per_source_correct.append(min_diff <= self.config.tolerance_degrees)
                
                # Per-angle statistics
                for true_dir in true_directions:
                    angle_key = int(round(true_dir))
                    if angle_key not in angle_stats:
                        angle_stats[angle_key] = {'correct': 0, 'total': 0, 'errors': []}
                    
                    angle_stats[angle_key]['total'] += 1
                    
                    # Find closest estimate
                    if est_dirs_list:
                        closest_est = min(est_dirs_list, 
                                        key=lambda x: min(abs(x - true_dir), 360 - abs(x - true_dir)))
                        est_error = min(abs(closest_est - true_dir), 360 - abs(closest_est - true_dir))
                        angle_stats[angle_key]['errors'].append(est_error)
                        
                        if est_error <= self.config.tolerance_degrees:
                            angle_stats[angle_key]['correct'] += 1
                    else:
                        angle_stats[angle_key]['errors'].append(180.0)  # Maximum error
                        
            except Exception as e:
                logger.warning(f"Failed to evaluate example {idx}: {e}")
                failed_examples.append({
                    'index': idx,
                    'error': str(e),
                    'true_directions': example.get('directions', [])
                })
                continue
        
        # Compute overall metrics
        overall_accuracy = float(np.mean(accuracies)) if accuracies else 0.0
        mean_error = float(np.mean(errors)) if errors else float('inf')
        median_error = float(np.median(errors)) if errors else float('inf')
        std_error = float(np.std(errors)) if errors else 0.0
        
        # Per-source accuracy
        if per_source_correct:
            correct_count = sum(1 if bool(x) else 0 for x in per_source_correct)
            per_source_accuracy = 100.0 * correct_count / len(per_source_correct)
        else:
            per_source_accuracy = 0.0
        
        # Processing time statistics
        mean_processing_time = np.mean(processing_times) if processing_times else 0.0
        
        # Per-angle analysis
        angle_analysis = {}
        for angle, stats in angle_stats.items():
            if stats['total'] > 0:
                angle_acc = 100.0 * stats['correct'] / stats['total']
                angle_mean_error = np.mean(stats['errors']) if stats['errors'] else float('inf')
                angle_analysis[angle] = {
                    'accuracy': angle_acc,
                    'mean_error': angle_mean_error,
                    'correct': stats['correct'],
                    'total': stats['total']
                }
        
        # Compile results
        evaluation_results = {
            # Overall performance
            'accuracy': overall_accuracy,
            'mean_error': mean_error,
            'median_error': median_error,
            'std_error': std_error,
            'per_source_accuracy': per_source_accuracy,
            
            # Processing performance
            'mean_processing_time': mean_processing_time,
            'total_processing_time': sum(processing_times),
            
            # Data statistics
            'n_evaluated': len(accuracies),
            'n_failed': len(failed_examples),
            'n_total': len(test_data),
            'success_rate': len(accuracies) / len(test_data) * 100 if test_data else 0,
            
            # Detailed analysis
            'per_angle_analysis': angle_analysis,
            'failed_examples': failed_examples,
            
            # Configuration
            'tolerance_degrees': self.config.tolerance_degrees,
            'n_sources': n_sources
        }
        
        # Log summary
        logger.info(f"=== Evaluation Results ===")
        logger.info(f"Overall accuracy: {overall_accuracy:.1f}%")
        logger.info(f"Mean error: {mean_error:.1f}°")
        logger.info(f"Per-source accuracy: {per_source_accuracy:.1f}%")
        logger.info(f"Success rate: {evaluation_results['success_rate']:.1f}%")
        logger.info(f"Mean processing time: {mean_processing_time*1000:.1f}ms")
        
        return evaluation_results
    
    def evaluate_reconstruction_quality(
        self,
        model: NMFSoundLocalizer,
        test_spectrograms: List[torch.Tensor],
        n_samples: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate reconstruction quality of the NMF model.
        
        Args:
            model: Trained NMF localizer
            test_spectrograms: List of test spectrograms
            n_samples: Number of samples to evaluate
            
        Returns:
            Reconstruction quality metrics
        """
        if not test_spectrograms:
            raise ValueError("Empty test spectrograms")
        
        logger.info(f"Evaluating reconstruction quality on {min(n_samples, len(test_spectrograms))} samples")
        
        reconstruction_errors = []
        snr_values = []
        
        # Sample spectrograms to evaluate
        selected_specs = test_spectrograms[:n_samples] if len(test_spectrograms) >= n_samples else test_spectrograms
        
        for spec in tqdm(selected_specs, desc="Reconstructing"):
            try:
                spec = spec.to(self.device)
                
                # Factorize
                X, _ = model.factorize(spec)
                
                # Reconstruct
                reconstructed = model.A @ X
                
                # Compute reconstruction error
                mse = torch.mean((spec - reconstructed) ** 2).item()
                reconstruction_errors.append(mse)
                
                # Compute SNR
                signal_power = torch.mean(spec ** 2).item()
                snr = 10 * np.log10(signal_power / (mse + model.epsilon)) if mse > 0 else float('inf')
                snr_values.append(snr)
                
            except Exception as e:
                logger.warning(f"Failed to evaluate reconstruction: {e}")
                continue
        
        # Compute statistics
        metrics = {
            'mean_mse': np.mean(reconstruction_errors) if reconstruction_errors else float('inf'),
            'std_mse': np.std(reconstruction_errors) if reconstruction_errors else 0.0,
            'mean_snr_db': np.mean(snr_values) if snr_values else -float('inf'),
            'std_snr_db': np.std(snr_values) if snr_values else 0.0,
            'n_evaluated': len(reconstruction_errors)
        }
        
        logger.info(f"Reconstruction quality: MSE={metrics['mean_mse']:.6f}, SNR={metrics['mean_snr_db']:.2f}dB")
        
        return metrics
    
    def compare_methods(
        self,
        models: Dict[str, NMFSoundLocalizer],
        test_data: List[Dict],
        n_sources: int = 1
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple NMF localization methods.
        
        Args:
            models: Dictionary of model_name -> model
            test_data: Test data
            n_sources: Number of sources to localize
            
        Returns:
            Comparison results for each method
        """
        logger.info(f"Comparing {len(models)} methods on {len(test_data)} examples")
        
        comparison_results = {}
        
        for method_name, model in models.items():
            logger.info(f"Evaluating method: {method_name}")
            
            try:
                results = self.evaluate_localization(model, test_data, n_sources, verbose=False)
                comparison_results[method_name] = results
                
            except Exception as e:
                logger.error(f"Failed to evaluate method {method_name}: {e}")
                comparison_results[method_name] = {'error': str(e)}
        
        # Create comparison summary
        summary = {
            'method_comparison': {},
            'best_method': None,
            'best_accuracy': 0.0
        }
        
        for method_name, results in comparison_results.items():
            if 'accuracy' in results:
                summary['method_comparison'][method_name] = {
                    'accuracy': results['accuracy'],
                    'mean_error': results['mean_error'],
                    'processing_time': results['mean_processing_time']
                }
                
                # Track best method
                if results['accuracy'] > summary['best_accuracy']:
                    summary['best_accuracy'] = results['accuracy']
                    summary['best_method'] = method_name
        
        comparison_results['summary'] = summary
        
        logger.info(f"Method comparison complete. Best method: {summary['best_method']} "
                   f"({summary['best_accuracy']:.1f}% accuracy)")
        
        return comparison_results
    
    def evaluate_parameter_sensitivity(
        self,
        base_model: NMFSoundLocalizer,
        test_data: List[Dict],
        parameter_ranges: Dict[str, List[Any]],
        n_sources: int = 1,
        max_examples: int = 100
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate sensitivity to different parameter values.
        
        Args:
            base_model: Base model to modify
            test_data: Test data
            parameter_ranges: Dictionary of parameter_name -> [values]
            n_sources: Number of sources to localize
            max_examples: Maximum number of test examples to use
            
        Returns:
            Parameter sensitivity results
        """
        logger.info(f"Evaluating parameter sensitivity for: {list(parameter_ranges.keys())}")
        
        # Use subset of test data for efficiency
        test_subset = test_data[:max_examples] if len(test_data) > max_examples else test_data
        
        sensitivity_results = {}
        
        # Store original parameter values
        original_params = {}
        for param_name in parameter_ranges.keys():
            if hasattr(base_model.config, param_name):
                original_params[param_name] = getattr(base_model.config, param_name)
                
        # Test each parameter
        for param_name, param_values in parameter_ranges.items():
            if not hasattr(base_model.config, param_name):
                logger.warning(f"Parameter {param_name} not found in config")
                continue
                
            logger.info(f"Testing parameter {param_name}: {param_values}")
            param_results = {}
            
            for param_value in param_values:
                try:
                    # Set parameter value
                    setattr(base_model.config, param_name, param_value)
                    
                    # Update model components that depend on this parameter
                    if param_name in ['beta', 'lambda_group', 'gamma_sparse', 'max_iter']:
                        if hasattr(base_model, param_name):
                            setattr(base_model, param_name, param_value)
                    
                    # Evaluate with this parameter value
                    results = self.evaluate_localization(base_model, test_subset, n_sources, verbose=False)
                    param_results[param_value] = {
                        'accuracy': results['accuracy'],
                        'mean_error': results['mean_error'],
                        'processing_time': results['mean_processing_time']
                    }
                    
                    logger.info(f"  {param_name}={param_value}: {results['accuracy']:.1f}% accuracy")
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate {param_name}={param_value}: {e}")
                    param_results[param_value] = {'error': str(e)}
            
            sensitivity_results[param_name] = param_results
        
        # Restore original parameters
        for param_name, original_value in original_params.items():
            setattr(base_model.config, param_name, original_value)
            if hasattr(base_model, param_name):
                setattr(base_model, param_name, original_value)
        
        logger.info("Parameter sensitivity evaluation complete")
        
        return sensitivity_results
    
    def save_evaluation_results(
        self, 
        results: Dict[str, Any], 
        output_path: str,
        include_details: bool = True
    ):
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results
            output_path: Output file path
            include_details: Whether to include detailed analysis
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare save data
        save_data = {
            'evaluation_results': results,
            'config': self.config.to_dict(),
            'timestamp': time.time()
        }
        
        if not include_details:
            # Remove detailed per-example data
            if 'failed_examples' in save_data['evaluation_results']:
                save_data['evaluation_results']['failed_examples'] = []
        
        # Save results
        torch.save(save_data, output_path)
        logger.info(f"Evaluation results saved to: {output_path}")
    
    def create_evaluation_report(self, results: Dict[str, Any]) -> str:
        """
        Create a human-readable evaluation report.
        
        Args:
            results: Evaluation results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("NMF LOCALIZATION EVALUATION REPORT")
        report.append("=" * 60)
        
        # Overall performance
        report.append("\n--- OVERALL PERFORMANCE ---")
        report.append(f"Accuracy: {results.get('accuracy', 0):.1f}%")
        report.append(f"Mean Error: {results.get('mean_error', 0):.1f}°")
        report.append(f"Median Error: {results.get('median_error', 0):.1f}°")
        report.append(f"Per-Source Accuracy: {results.get('per_source_accuracy', 0):.1f}%")
        
        # Processing performance
        report.append("\n--- PROCESSING PERFORMANCE ---")
        report.append(f"Mean Processing Time: {results.get('mean_processing_time', 0)*1000:.1f}ms")
        report.append(f"Success Rate: {results.get('success_rate', 0):.1f}%")
        report.append(f"Examples Evaluated: {results.get('n_evaluated', 0)}")
        
        # Per-angle analysis
        if 'per_angle_analysis' in results and results['per_angle_analysis']:
            report.append("\n--- PER-ANGLE ANALYSIS ---")
            for angle, stats in sorted(results['per_angle_analysis'].items()):
                report.append(f"Angle {angle:3d}°: {stats['accuracy']:5.1f}% "
                             f"({stats['correct']:2d}/{stats['total']:2d}), "
                             f"Error: {stats['mean_error']:5.1f}°")
        
        # Configuration
        report.append("\n--- CONFIGURATION ---")
        report.append(f"Tolerance: {results.get('tolerance_degrees', 0)}°")
        report.append(f"Sources: {results.get('n_sources', 1)}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)