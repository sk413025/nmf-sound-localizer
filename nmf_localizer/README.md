# NMF Localizer

A modular toolkit for traditional NMF-based sound source localization, designed for large-scale experiments.

## Overview

This package provides a complete pipeline for NMF-based sound source localization, from data processing to evaluation. It's designed to be:
- **Modular**: Each component can be used independently
- **Experimental**: Built for parameter sweeps and batch experiments  
- **Reproducible**: Complete configuration management and result tracking
- **Efficient**: Optimized for large-scale testing

## Quick Start

### Basic Usage

```python
from nmf_localizer import NMFLocalizationPipeline, NMFConfig

# Create pipeline with default configuration
config = NMFConfig(beta=0.0, lambda_group=20.0)
pipeline = NMFLocalizationPipeline(config)

# Run complete experiment
results = pipeline.run_full_experiment(
    data_root="root/",
    output_dir="outputs/experiment_001",
    n_sources=1
)

print(f"Accuracy: {results['stages']['evaluation']['results']['accuracy']:.1f}%")
```

### Batch Parameter Sweeps

```python
from nmf_localizer import ExperimentRunner, NMFConfig

# Create experiment runner
base_config = NMFConfig()
runner = ExperimentRunner(base_config)

# Define parameter sweeps
runner.add_parameter_sweep("beta", [0.0, 0.5, 1.0, 1.5, 2.0])
runner.add_parameter_sweep("lambda_group", [10.0, 20.0, 30.0])

# Run all combinations
all_results = runner.run_experiments(
    data_root="root/",
    output_root="outputs/parameter_sweep"
)

# Compare results
comparison = runner.compare_results(all_results)
print(f"Best method: {comparison['summary']['best_method']}")
```

### Advanced Usage

```python
from nmf_localizer import DataProcessor, USMTrainer, NMFSoundLocalizer, Evaluator

# Manual pipeline construction
config = NMFConfig(beta=0.0, device='cuda')

# 1. Process data
processor = DataProcessor(config)
data_pack = processor.process_full_dataset("root/")

# 2. Train USM
usm_trainer = USMTrainer(config)
W, usm_info = usm_trainer.train_usm(data_pack.speaker_data)

# 3. Initialize localizer
localizer = NMFSoundLocalizer(config)
localizer.load_source_dictionary(W)
localizer.load_transfer_functions(data_pack.transfer_functions, data_pack.angles)

# 4. Evaluate
evaluator = Evaluator(config)
results = evaluator.evaluate_localization(localizer, data_pack.test_data)
```

## Architecture

### Core Components

- **DataProcessor**: Handles raw data processing, transfer function estimation
- **USMTrainer**: Trains Universal Speech Model dictionary
- **NMFSoundLocalizer**: Core NMF localization algorithm
- **Evaluator**: Comprehensive performance evaluation

### Pipeline Components

- **NMFLocalizationPipeline**: High-level interface for complete experiments
- **ExperimentRunner**: Automated parameter sweeps and batch processing

### Configuration

- **NMFConfig**: Centralized configuration management
- **DataPack**: Standardized data container

## Configuration Options

```python
config = NMFConfig(
    # Audio processing
    sample_rate=16000,
    n_fft=2048,
    freq_min=500.0,
    freq_max=1500.0,
    
    # NMF parameters  
    beta=0.0,              # 0: IS divergence, 1: KL, 2: Euclidean
    lambda_group=20.0,     # Group sparsity weight
    gamma_sparse=1.0,      # Sparsity weight
    max_iter=100,
    
    # USM parameters
    n_atoms_per_speaker=50,
    
    # Evaluation
    tolerance_degrees=10.0,
    n_test_examples=1000,
    
    # Hardware
    device='cpu'  # or 'cuda', 'mps'
)
```

## Data Format

### Input Data Structure
```
root/
├─ angle_00/
│    ├─ clip_000.npy
│    ├─ clip_001.npy
│    └─ ...
├─ angle_18/
│    └─ ...
└─ angle_36/
     └─ ...
```

### Output Structure
```
outputs/experiment_001/
├─ processed_data/
│   └─ data_pack.pth
├─ models/
│   ├─ usm.pth
│   └─ localizer.pth
├─ evaluation/
│   ├─ evaluation_results.pth
│   └─ evaluation_report.txt
├─ experiment_results.pth
└─ experiment_summary.txt
```

## Experiment Management

### Single Experiments

```python
# Run with custom configuration
config = NMFConfig(
    beta=1.0,
    lambda_group=30.0,
    max_iter=200,
    device='cuda'
)

pipeline = NMFLocalizationPipeline(config)
results = pipeline.run_full_experiment(
    data_root="root/",
    output_dir="outputs/custom_experiment",
    test_multiple_betas=True,  # Test multiple beta values for USM
    save_models=True
)
```

### Parameter Sweeps

```python
# Comprehensive parameter sweep
runner = ExperimentRunner()

runner.add_parameter_sweep("beta", [0.0, 0.5, 1.0, 2.0])
runner.add_parameter_sweep("lambda_group", [10.0, 20.0, 40.0])
runner.add_parameter_sweep("n_atoms_per_speaker", [25, 50, 100])

# Fixed parameters
runner.set_fixed_parameter("max_iter", 150)
runner.set_fixed_parameter("device", "cuda")

# Run experiments
results = runner.run_experiments(
    data_root="root/",
    output_root="outputs/comprehensive_sweep",
    max_experiments=20  # Limit for testing
)
```

### Result Analysis

```python
# Load and analyze results
from nmf_localizer.io import ResultLoader

batch_results = ResultLoader.load_batch_results("outputs/parameter_sweep/")
performance = ResultLoader.extract_performance_summary(batch_results['experiment_results'])

print(f"Best accuracy: {performance['accuracy_stats']['max']:.1f}%")
print(f"Mean accuracy: {performance['accuracy_stats']['mean']:.1f}%")
```

## Visualization

```python
from nmf_localizer.utils import Visualizer

# Plot transfer functions
Visualizer.plot_transfer_functions(
    H=data_pack.transfer_functions,
    angles=data_pack.angles,
    save_path="transfer_functions.png"
)

# Plot parameter sweep results
Visualizer.plot_parameter_sweep_results(
    comparison_results=comparison,
    parameter_name="beta",
    metric="accuracy",
    save_path="beta_sweep.png"
)

# Create experiment dashboard
Visualizer.create_experiment_dashboard(
    results=experiment_results,
    output_dir="dashboard/",
    experiment_name="beta_sweep"
)
```

## Advanced Features

### Custom Evaluation Metrics

```python
evaluator = Evaluator(config)

# Parameter sensitivity analysis
sensitivity = evaluator.evaluate_parameter_sensitivity(
    base_model=localizer,
    test_data=test_data,
    parameter_ranges={
        'beta': [0.0, 0.5, 1.0, 2.0],
        'lambda_group': [5.0, 10.0, 20.0, 40.0]
    }
)
```

### Method Comparison

```python
# Compare different configurations
models = {
    'IS_divergence': create_localizer(beta=0.0),
    'KL_divergence': create_localizer(beta=1.0), 
    'Euclidean': create_localizer(beta=2.0)
}

comparison = evaluator.compare_methods(models, test_data)
```

### Pipeline State Management

```python
# Save complete pipeline state
pipeline.save_pipeline_state("pipeline_state.pth")

# Load and resume
new_pipeline = NMFLocalizationPipeline()
new_pipeline.load_pipeline_state("pipeline_state.pth")
```

## Performance Tips

1. **GPU Acceleration**: Set `device='cuda'` for GPU acceleration
2. **Data Caching**: Processed data is automatically cached to avoid reprocessing
3. **Batch Processing**: Use ExperimentRunner for efficient parameter sweeps
4. **Memory Management**: Large datasets are processed in chunks
5. **Parallel Processing**: Multiple experiments can run in parallel

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `n_test_examples` or use CPU processing
2. **Convergence Issues**: Try different beta values or increase `max_iter`
3. **Accuracy Problems**: Check transfer function quality and frequency range
4. **Performance Issues**: Use GPU acceleration and optimize data loading

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Detailed logging for debugging
pipeline = NMFLocalizationPipeline(config)
results = pipeline.run_full_experiment(...)
```

## API Reference

### Core Classes

- `NMFConfig`: Configuration management
- `DataPack`: Data container 
- `DataProcessor`: Data processing utilities
- `USMTrainer`: USM training
- `NMFSoundLocalizer`: Core localization algorithm
- `Evaluator`: Performance evaluation

### Pipeline Classes  

- `NMFLocalizationPipeline`: Complete experiment pipeline
- `ExperimentRunner`: Batch experiment management

### Utilities

- `AudioProcessor`: Audio processing utilities
- `Visualizer`: Plotting and visualization
- `MathUtils`: Mathematical utilities

See individual class documentation for detailed API information.

## Examples

Check the `examples/` directory for complete example scripts:

- `basic_experiment.py`: Simple single experiment
- `parameter_sweep.py`: Comprehensive parameter sweep
- `custom_evaluation.py`: Custom evaluation metrics
- `visualization_demo.py`: Visualization examples

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{nmf_localizer,
  title={NMF Localizer: A Modular Toolkit for Sound Source Localization},
  author={Speech Processing Lab},
  year={2024},
  url={https://github.com/your-repo/nmf-localizer}
}
```