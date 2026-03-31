# 37-Angle White-Noise LDV Re-Recording Protocol

## Purpose

This document reconstructs the later 37-angle white-noise LDV recording setup as far as the available evidence allows. It is intended as a practical bench protocol for re-recording additional LDV files while keeping provenance explicit.

This is not a claim that the original session can be reproduced with zero guesswork. The protocol distinguishes:

- confirmed facts from raw files
- metadata inherited from prior logs
- unresolved items that must be bench-confirmed before recording

## Canonical Target

Use the later 37-angle dataset under `/Users/sbplab/LDV-data` as the canonical target for re-recording.

The immediate goal is to reproduce the white-noise calibration-style capture that fed the white-noise transfer-function and training pipeline, not the older mixed speech + tone program.

## Evidence Base

Primary sources:

- `/Users/sbplab/LDV-data/LDV-data_test_log.txt`
- `/Users/sbplab/LDV-data/complete/`
- `/Users/sbplab/jiawei/datasets/20250709/20250709/20250709_test_log.txt`
- `/Users/sbplab/jiawei/datasets/20250709/20250709/readme.ipynb`
- [dataset_training_lineage.md](./dataset_training_lineage.md)
- [dataset_creation_pipeline.md](./dataset_creation_pipeline.md)
- [data_provenance_verification.md](./data_provenance_verification.md)

Important provenance caveat:

- The repository evidence shows that `standard_file.wav` and `LDV-data_test_log.txt` in `/Users/sbplab/LDV-data` were prepared later by copying or adapting assets from the older `20250709` dataset for processing.
- Therefore, some metadata in the 37-angle root is inherited metadata, not direct proof that the same file was captured alongside the original 37-angle session.

## Recovered Setup

### High-Confidence Facts

These are corroborated by actual files and filenames.

#### Recording Matrix

- Materials: `box`, `IrregularBox`
- Angles: `000` to `180` in `5` degree steps
- Total angle count: `37`
- Segments: `segment1`, `segment2`

#### Native Raw Recording Inventory

Observed under `/Users/sbplab/LDV-data/complete`:

- `74` files named `box_degXXX_segmentY_complete.wav`
- `74` files named `IrregularBox_degXXX_segmentY_complete.wav`
- `24` files named `micropone_box_degXXX_segmentY_complete.wav`
- `24` files named `micropone_IrregularBox_degXXX_segmentY_complete.wav`

The `micropone_*` files only exist for angles `000-025` and `155-180`. Treat them as partial auxiliary recordings, not the canonical full sweep.

#### Audio Format

Observed from representative files:

- `complete/*.wav`: mono, `48000 Hz`, `32-bit PCM`
- `standard_file.wav`: mono, `48000 Hz`, `16-bit PCM`
- `white_noise_1.wav`, `white_noise_2.wav`, `white_noise_3.wav`: mono, `48000 Hz`, `16-bit PCM`
- `white_noise_all.wav`: mono, `48000 Hz`, `16-bit PCM`

#### Segment Durations

Observed from representative `complete` files:

- `segment1_complete`: `839.563 s` (`00:13:59.563`)
- `segment2_complete`: `1005.288 s` (`00:16:45.288`)

These durations match the values stored in `LDV-data_test_log.txt`.

#### White-Noise Units Present in the 37-Angle Root

- `white_noise_1.wav`: `3.04 s`
- `white_noise_2.wav`: `3.04 s`
- `white_noise_3.wav`: `3.04 s`
- `white_noise_all.wav`: `9.12 s`

### Medium-Confidence Facts

These are present in the 37-angle test log, but the provenance trail indicates that some of the log was adapted from the older `20250709` setup.

- `SNR = 70dB`
- `LDV_to_Obj = 90cm`
- `Obj_to_Sorce = 90cm`
- `segment1 file_tag = white_noise_1-3.wav`
- `segment2 file_tag = white_noise_1-3.wav`
- `segment2 start_pos_onStandard = 00:16:45.288`

Use these values as the current best recovered setup, but mark them as bench-confirm items rather than unquestioned ground truth.

### Historical Cross-Check from the Older 20250709 Dataset

The older `20250709` root confirms the same:

- `SNR = 70dB`
- `LDV_to_Obj = 90cm`
- `Obj_to_Sorce = 90cm`
- materials: `box`, `IrregularBox`
- segment structure: `segment1`, `segment2`

It differs in stimulus content:

- `segment1` used `boy2_29-288.wav`
- `segment2` used `girl7_29-288.wav`, pure tones, and `white_noise_1-3.wav`

Do not use the older mixed-program content as the target protocol for the 37-angle white-noise re-recording. Use it only to understand which metadata fields were inherited.

## Bench Protocol

### 1. Pre-Session Confirmation

Before recording, confirm and write down:

- LDV hardware model
- speaker model and placement
- amplifier or playback gain chain
- microphone chain, if `micropone_*` is to be reproduced
- exact measurement point for the claimed `70dB`
- object mounting method
- angle origin and rotation direction
- room or enclosure conditions

If any of these are unknown, do not silently assume them. Record them as unresolved in the session notes.

### 2. Playback Assets

Prepare the playback package at `48000 Hz`, mono WAV:

- a master reference waveform equivalent to `standard_file.wav`
- the three white-noise units equivalent to `white_noise_1.wav`, `white_noise_2.wav`, `white_noise_3.wav`
- optionally a concatenated `white_noise_all.wav` equivalent for quick verification

Do not downsample for recording. The raw session format is `48000 Hz`.

### 3. Recording Matrix

For each material:

- `box`
- `IrregularBox`

For each angle:

- `000`, `005`, `010`, `015`, `020`, `025`, `030`, `035`, `040`, `045`, `050`, `055`, `060`, `065`, `070`, `075`, `080`, `085`, `090`, `095`, `100`, `105`, `110`, `115`, `120`, `125`, `130`, `135`, `140`, `145`, `150`, `155`, `160`, `165`, `170`, `175`, `180`

Capture both:

- `segment1`
- `segment2`

Use the canonical output naming:

- `box_degXXX_segment1_complete.wav`
- `box_degXXX_segment2_complete.wav`
- `IrregularBox_degXXX_segment1_complete.wav`
- `IrregularBox_degXXX_segment2_complete.wav`

Only create `micropone_*` files if you intentionally reproduce the auxiliary microphone path and document why the coverage is partial.

### 4. Target Raw Export Format

Prefer to match the observed canonical raw export:

- mono WAV
- `48000 Hz`
- `32-bit PCM` for `*_complete.wav`

If the capture software exports a different bit depth, record that explicitly in the session notes and do not overwrite or normalize it before archiving the raw files.

### 5. Immediate Session Verification

After recording, verify before leaving the bench:

- all `148` canonical LDV complete files exist
- filenames follow the exact `material_degXXX_segmentY_complete.wav` pattern
- angle coverage is complete and contiguous in `5` degree steps
- sample rate is `48000 Hz`
- files are mono
- representative `segment1` duration is approximately `839.563 s`
- representative `segment2` duration is approximately `1005.288 s`
- if microphone files were recorded, their intended scope is documented

### 6. Handoff to Processing

Once the raw WAV set is verified:

- keep raw files unchanged under the recording root
- create or update a session log that records the confirmed bench settings
- only then hand off to the Stage 0 conversion path that creates the white-noise `111`-clip processed dataset used downstream

## Bench Confirmation Required

These items are not recoverable with high confidence from the current evidence and must be confirmed manually:

- exact LDV model
- exact speaker model
- exact microphone model and purpose of the `micropone_*` path
- SPL or SNR measurement method and location
- room treatment or ambient-noise controls
- exact physical definition of the angle axis
- whether `segment1` and `segment2` were separate playback sessions or partitions on one master timeline

## Stop Conditions

Do not describe the re-recording as a faithful reproduction of the original session unless the bench-confirm items above are resolved in the new session notes.

Do not replace the canonical `*_complete.wav` naming or raw sample-rate convention, because the downstream conversion and lineage documents assume this structure.
