# Fig. 6 Material-Structure Bridge

This note is the paper-facing version of the `results/fig06_material_structure_bridge_20260327_190552/` bundle. It is written for cross-disciplinary readers and is intended to be directly reusable when assembling supplementary evidence for Fig. 6.

## Core message

The five objects in Fig. 6 should be interpreted as **five everyday structural archetypes** spanning broad differences in stiffness, damping, anisotropy, shell or plate geometry, layering, and cavity structure. The goal is not to claim exact constitutive truth for each specimen, but to show that the observed compression and separability indices sit inside a meaningful material-structure design space.

## Figure anchor

![Fig. 6 anchor](assets/fig06_universality.jpg)

## Metric anchor

![Compression and separability anchor](assets/compression_separability_colored_by_metrics.png)

This anchor keeps the existing scorecard language intact:

- `CompressionIndex` asks how compactly the angle-frequency response can be represented.
- `SeparabilityIndex` asks whether directional fingerprints remain distinct after that compression.
- Task outcome still needs to be read alongside geometry; the geometric score is descriptive, not a validated one-number predictor.

## Descriptor matrix

![Descriptor matrix](assets/material_structure_descriptor_matrix.png)

The matrix shows why the set is representative even with only five objects:

- acrylic anchors the near-isotropic flat polymer plate case
- wood anchors the orthotropic solid-board case
- cardboard anchors the corrugated cavity-rich packaging case
- paper cup anchors the coated curved-shell paperboard case
- laptop shell anchors the stiff consumer-shell enclosure case

## Mechanism bridge

![Mechanism bridge](assets/mechanism_bridge_schematic.png)

The physics should therefore be read in three layers:

1. material constants such as stiffness, density, damping, and caliper
2. structure-level modifiers such as curvature, corrugation, layering, anisotropy, and cavity geometry
3. measured encoding outcomes such as compression, separability, and screening accuracy

## Supplementary Table X preview

| Object | Archetype | Stiffness | Damping | Anisotropy | Compression | Separability | Top-1 (%) | Within 10° (%) | MAE (deg) |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cardboard box | Corrugated hollow packaging shell | Low | High | High | 0.0 | 45.8 | 81.1 | 95.5 | 2.61 |
| Wooden board | Orthotropic solid board | Mid-high | Moderate | High | 75.0 | 22.2 | 79.3 | 97.3 | 5.09 |
| Acrylic plate | Homogeneous flat polymer plate | Mid | Moderate | Low | 100.0 | 44.4 | 78.4 | 93.7 | 3.83 |
| Paper cup | Curved coated paperboard shell | Low | Moderate-high | Medium | 37.5 | 69.4 | 77.5 | 91.9 | 5.99 |
| Laptop shell | Thin consumer-device shell with cavity | High | Low | Low-mid | 37.5 | 68.1 | 67.6 | 92.8 | 5.23 |

## Manuscript-safe wording

Results bridge:

These five objects should be read not merely as five material labels but as five structurally distinct everyday object archetypes spanning broad differences in stiffness, damping, anisotropy, shell versus plate geometry, layering, and internal cavity complexity. Representative material-structure descriptors compiled from official technical references support this breadth interpretation and clarify why cross-material performance should be expected to vary even when low-rank continuity persists across the full set (Supplementary Table X and Supplementary Fig. X).

Discussion bridge:

The cross-material result should therefore be interpreted at the level of coupled material and structural descriptors rather than nominal material identity alone. In this view, everyday directional encoding is a generic property of bounded dispersive structures, whereas object choice is modulated by how stiffness, damping, anisotropy, layering, curvature, and cavity geometry jointly shape modal richness, overlap, and the separability of the resulting directional fingerprints.

## Source policy

- official or handbook-style sources only
- exact specimen identity was not inferred where the workspace did not record it
- laptop shell values remain low-confidence and representative only
