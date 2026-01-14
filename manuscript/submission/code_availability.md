# Code Availability Statement

The complete source code for reproducing all experiments and figures is
available under the MIT License at [GitHub URL].

## Code Components

1. **Model implementation**: `doa_rl/omp/soft_omp.py`
2. **Training scripts**: `scripts/omp-transformer-ldv.py`
3. **Evaluation scripts**: `scripts/eval_omp_transformer_split.py`
4. **Figure generation**:
   - `scripts/create_master_figure.py`
   - `scripts/plot_ablation_figure4_speech260.py`
5. **Analysis tools**: `scripts/visualize_omp_transformer_routing.py`

## Reproducibility

All experiments are reproducible using the provided configuration files and
conda environment specifications. Detailed reproduction instructions are
provided in Supplementary Methods.

## Environment Requirements

- Python 3.8+
- PyTorch 1.12+
- Conda environment: `environment.yml` (provided in repository)
- Hardware: Apple M1/M2 (MPS) or CUDA-capable GPU

Exact reproduction commands are documented in each experiment's commit message.
