from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import nmf_localizer
from doa_rl import DoADataset
from doa_rl.omp import TrainableRoutedSoftOMP, build_dictionary


def test_nmf_localizer_exports_only_active_runtime_core() -> None:
    exports = set(nmf_localizer.__all__)
    assert "NMFConfig" in exports
    assert "DataProcessor" in exports
    assert "TransferFunctionProcessor" in exports
    assert "USMTrainer" in exports
    assert "AudioProcessor" in exports
    assert "NMFLocalizationPipeline" not in exports
    assert "ExperimentRunner" not in exports
    assert "NMFSoundLocalizer" not in exports
    assert "Evaluator" not in exports


def test_soft_omp_surface_builds_dictionary_and_model() -> None:
    W = torch.tensor([[1.0, 2.0], [0.5, 1.5], [2.0, 0.5]], dtype=torch.float32)
    H = torch.tensor([[1.0, 0.4], [0.8, 1.2], [0.6, 1.0]], dtype=torch.float32)

    D, idx2 = build_dictionary(W, H)
    model = TrainableRoutedSoftOMP(F=3, E=2, M=2, steps=2, top_e=1, L=1)

    assert D.shape == (3, 4)
    assert idx2 == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert isinstance(model, TrainableRoutedSoftOMP)


def test_doa_dataset_returns_expected_contract(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    angle_dir = root / "angle_00"
    angle_dir.mkdir(parents=True)

    t = np.linspace(0, 0.2, 3200, endpoint=False)
    waveform = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    np.save(angle_dir / "clip_000.npy", waveform)

    dataset = DoADataset(
        root=root,
        angles=[0],
        fs=16000,
        n_fft=256,
        freq_min=300.0,
        freq_max=3000.0,
    )

    sample = dataset[0]
    assert set(sample) == {"Y", "angle_deg", "angle_index", "path"}
    assert sample["Y"].ndim == 2
    assert sample["angle_deg"] == 0.0
    assert sample["angle_index"] == 0
    assert sample["path"].endswith("clip_000.npy")
