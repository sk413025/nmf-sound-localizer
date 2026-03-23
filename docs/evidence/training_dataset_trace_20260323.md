# Training Dataset Trace - 2026-03-23

## Purpose

This note records the two confirmed neural-network training datasets discussed on 2026-03-23, together with their absolute paths, paired original roots, upstream sources, and the key commits that document preprocessing and successful training.

All absolute paths listed here were checked for existence on 2026-03-23.

## Confirmed Training Datasets

### White Noise

- Training dataset root: `/Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized`
- File count: `111` `.npy` files
- Paired original root: `/Users/sbplab/LDV-data-processed/white_noise_original_data_no_edge_sync_vad_normalized`

### Speech

- Training dataset root: `/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized`
- File count: `9620` `.npy` files
- Paired original root: `/Users/sbplab/LDV-data-processed/speech260_original_16k_no_edge_sync_vad_normalized`

## Raw and Upstream Sources

- White-noise laboratory root: `/Users/sbplab/jiawei/datasets/20250709/20250709/`
- Expanded 37-angle source root: `/Users/sbplab/LDV-data`
- Speech extraction source: `/Users/sbplab/LDV-data` segment1

## Key Processing Commits

### White Noise

- `329ae66b33c4f8a2315540db60766ba1da1e2adb`: Stage 0 WAV to NPY conversion
- `b7e1675cd3b8e5ced4c0fdcf2d2b651da6edd44f`: Stage 1-2 synchronized VAD and normalization
- `3f3d8eb3272e3ce39746dca7e31bfd53a56aa9fa`: Stage 3-4 H matrix estimation and USM training

### Speech

- `13dd6e2ca9bd4b6590c51fbb4e6fa88a32443e90`: Speech260 Stage 0-2 processing with 260 clips per angle
- `a7dae2d1c3c3ec03fb7ea2a5f7422d8c4adb1e77`: Resample Speech260 roots to 16 kHz for H/DoADataset alignment

## Successful Training Commits

### White Noise

- `1f6b68c91a09d1ceb4433c7ef0d6d3d598120798`
- Run type: g-routed Transformer on DoADataset
- Dataset root in command: `/Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized`
- Reported result: `100.0%` accuracy

### Speech

- `3785b1fcfed18ad35777b5cc076e4485ab31cd40`
- Run type: Speech260 16 kHz full training
- Dataset root in command: `/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized`
- Reported result: `97.5%` accuracy on the full dataset

- `06bf65de4071bd0cda0211f96e6900442bb67ce7`
- Run type: Speech260 16 kHz train/val split
- Dataset root in command: `/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized`
- Reported result: `94.6%` validation accuracy

## Evidence References

- `docs/evidence/dataset_training_lineage.md`
- `docs/evidence/dataset_creation_pipeline.md`

This note is intentionally compact and should be read alongside the longer lineage documents above when full preprocessing details or exact reproduction commands are needed.
