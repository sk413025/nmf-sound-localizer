# Main Text Figure Captions

## Figure 1: System Overview and Performance
[詳細說明]

## Figure 2: [Title]
[詳細說明]

## Figure 3: [Title]
[詳細說明]

## Figure 4: Mechanism Attribution via Ablation Study
**Panel A**: Transformer contribution. Ablating the transformer (replacing with
identity mapping) reduces accuracy from 93.5% to 63.1%, demonstrating a 30.4%
gap attributable to learned feature extraction.

**Panel B**: Sparsity necessity. Removing sparsity constraint (dense routing)
causes system collapse to 2.7% accuracy, demonstrating that sparsity is not
just beneficial but fundamental to system function.

Error bars represent 95% confidence intervals over 5 random seeds.

**Data source**: commit b9dcafa, `results/ablate_*_speech260_seed42_*/`
