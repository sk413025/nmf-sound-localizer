#!/usr/bin/env python3
"""
Parameter sweep experiment example.

This script demonstrates how to run automated parameter sweeps
to find optimal NMF localization parameters.
"""

import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from nmf_localizer import ExperimentRunner, NMFConfig
from nmf_localizer.utils import Visualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Run parameter sweep experiment."""
    
    # Base configuration
    base_config = NMFConfig(
        # Audio processing
        sample_rate=16000,
        freq_min=500.0,
        freq_max=1500.0,
        
        # Fixed parameters
        max_iter=100,
        n_atoms_per_speaker=50,
        tolerance_degrees=10.0,
        n_test_examples=50,  # Reduced for faster testing
        
        # Hardware
        device='cpu'  # Adjust as needed
    )
    
    # Create experiment runner
    runner = ExperimentRunner(base_config)
    
    # Define parameter sweeps
    print("Setting up parameter sweeps...")
    
    # Test different beta divergences
    runner.add_parameter_sweep("beta", [0.0, 0.5, 1.0, 2.0])
    
    # Test different group sparsity weights
    runner.add_parameter_sweep("lambda_group", [10.0, 20.0, 40.0])
    
    # Fixed parameters (applied to all experiments)
    runner.set_fixed_parameter("gamma_sparse", 1.0)
    runner.set_fixed_parameter("normalize_blocks", False)
    
    print(f"Total experiments: {len(runner.generate_experiment_configs())}")
    
    # Run experiments
    print("Starting parameter sweep...")
    
    try:
        all_results = runner.run_experiments(
            data_root="root",  # Adjust path as needed
            output_root="outputs/parameter_sweep",
            n_sources=1,
            max_experiments=None,  # Run all combinations
            save_models=False,     # Don't save models for sweep (saves space)
            force_rerun=False      # Skip existing experiments
        )
        
        print(f"\nCompleted {len(all_results)} experiments")
        
        # Compare results
        print("Analyzing results...")
        comparison = runner.compare_results(all_results)
        
        # Print summary
        summary = comparison['summary']
        print("\n" + "="*60)
        print("PARAMETER SWEEP RESULTS")
        print("="*60)
        print(f"Total experiments: {summary['n_experiments']}")
        
        best = summary['best_accuracy']
        print(f"\nBest accuracy: {best['accuracy']:.1f}% ({best['experiment_id']})")
        print("Best parameters:")
        for param, value in best['parameters'].items():
            print(f"  {param}: {value}")
        
        # Parameter analysis
        print(f"\nParameter Analysis:")
        param_analysis = comparison['parameter_analysis']
        
        for param_name, param_results in param_analysis.items():
            print(f"\n{param_name}:")
            for value, stats in param_results.items():
                acc = stats['mean_accuracy']
                std = stats['std_accuracy']
                n_exp = stats['n_experiments']
                print(f"  {value}: {acc:.1f}% (±{std:.1f}%, n={n_exp})")
        
        # Save comparison results
        output_dir = Path("outputs/parameter_sweep")
        runner.compare_results(
            all_results,
            save_comparison=True,
            output_path=str(output_dir / "comparison_results.pth")
        )
        
        # Create visualizations
        print("\nCreating visualizations...")
        
        try:
            # Plot beta parameter effects
            Visualizer.plot_parameter_sweep_results(
                comparison,
                parameter_name="beta",
                metric="accuracy",
                save_path=str(output_dir / "beta_sweep_accuracy.png")
            )
            
            # Plot lambda_group parameter effects  
            Visualizer.plot_parameter_sweep_results(
                comparison,
                parameter_name="lambda_group", 
                metric="accuracy",
                save_path=str(output_dir / "lambda_group_sweep_accuracy.png")
            )
            
            print("Visualizations saved to outputs/parameter_sweep/")
            
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
        
        print(f"\nComplete results saved to: {output_dir}")
        
        # Recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        # Find best beta
        beta_analysis = param_analysis.get('beta', {})
        if beta_analysis:
            best_beta = max(beta_analysis.items(), key=lambda x: x[1]['mean_accuracy'])
            print(f"Best beta divergence: {best_beta[0]} ({best_beta[1]['mean_accuracy']:.1f}% accuracy)")
        
        # Find best lambda_group
        lambda_analysis = param_analysis.get('lambda_group', {})
        if lambda_analysis:
            best_lambda = max(lambda_analysis.items(), key=lambda x: x[1]['mean_accuracy'])
            print(f"Best group sparsity: {best_lambda[0]} ({best_lambda[1]['mean_accuracy']:.1f}% accuracy)")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the 'root' directory exists with angle data.")
        return 1
        
    except Exception as e:
        print(f"Parameter sweep failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())