import os
from pathlib import Path
import os
import pytest
import torch
import numpy as np

from nmf_localizer.config import NMFConfig
from nmf_localizer.core.data_processor import DataProcessor
from nmf_localizer.core.transfer_functions import TransferFunctionProcessor


def _find_angles(root: Path):
    angles = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.startswith('angle_'):
            try:
                ang = int(d.name.replace('angle_', ''))
                angles.append(ang)
            except ValueError:
                pass
    return sorted(angles)


def _prepare_symlink_subset(x_root: Path, y_root: Path, out_root: Path, angles: list[int], n_files: int):
    x_out = out_root / 'original'
    y_out = out_root / 'box'
    x_out.mkdir(parents=True, exist_ok=True)
    y_out.mkdir(parents=True, exist_ok=True)

    common = []
    for a in angles:
        dx = x_root / f'angle_{a}'
        dy = y_root / f'angle_{a}'
        if not dx.exists() or not dy.exists():
            continue
        xs = sorted(dx.glob('*.npy'))
        ys = sorted(dy.glob('*.npy'))
        k = min(len(xs), len(ys), n_files)
        if k == 0:
            continue
        sx = x_out / f'angle_{a}'
        sy = y_out / f'angle_{a}'
        sx.mkdir(parents=True, exist_ok=True)
        sy.mkdir(parents=True, exist_ok=True)
        for i in range(k):
            srcx = xs[i]
            srcy = ys[i]
            (sx / srcx.name).symlink_to(srcx)
            (sy / srcy.name).symlink_to(srcy)
        common.append(a)
    return x_out, y_out, common


@pytest.mark.integration
def test_real_tf_subset_integration(tmp_path):
    """Integration test on real X/Y folders with conservative thresholds.

    Requires environment variables REAL_TF_X_ROOT and REAL_TF_Y_ROOT to be set.
    Optionally REAL_TF_ANGLE_START/END/STEP and REAL_TF_N_FILES can be used.
    If not set, the test is skipped.
    """
    x_root_env = os.environ.get('REAL_TF_X_ROOT')
    y_root_env = os.environ.get('REAL_TF_Y_ROOT')
    if not x_root_env or not y_root_env:
        pytest.skip('REAL_TF_X_ROOT or REAL_TF_Y_ROOT not set')

    x_root = Path(x_root_env)
    y_root = Path(y_root_env)
    if not x_root.exists() or not y_root.exists():
        pytest.skip('Provided data roots do not exist')

    a_start = int(os.environ.get('REAL_TF_ANGLE_START', '80'))
    a_end = int(os.environ.get('REAL_TF_ANGLE_END', '150'))
    a_step = int(os.environ.get('REAL_TF_ANGLE_STEP', '5'))
    n_files = int(os.environ.get('REAL_TF_N_FILES', '3'))

    target_angles = list(range(a_start, a_end + 1, a_step))

    # Prepare symlink subset under tmp_path
    workdir = tmp_path / 'real_tf_subset'
    x_sub, y_sub, common = _prepare_symlink_subset(x_root, y_root, workdir, target_angles, n_files)
    if not common:
        pytest.skip('No common angles with files in the requested range')

    # Config consistent with docs and earlier analysis
    cfg = NMFConfig(
        sample_rate=16000,
        n_fft=2048,
        hop_length=512,
        freq_min=500.0,
        freq_max=1500.0,
        n_files_per_angle=n_files,
        apply_contrast_enhancement=False,
        device='cpu',
    )
    dp = DataProcessor(cfg)

    # Estimate H
    H, angles, angle_folders, meta = dp.estimate_transfer_functions(x_sub, y_sub, method='xy_correspondence')

    # Shape checks
    assert H.shape[1] == len(angles)
    assert len(angles) == len(common)
    # Expect exactly 129 bins for 500–1500 Hz at fs=16k, n_fft=2048
    assert H.shape[0] == 129

    # Numerical sanity
    assert torch.all(torch.isfinite(H))
    vmin = H.min().item()
    vmax = H.max().item()
    assert vmin >= 0.09 and vmax <= 0.91  # scaled to ~[0.1,0.9]

    # Separability metrics (conservative thresholds)
    tfp = TransferFunctionProcessor(cfg)
    sep = tfp.analyze_separability(H)
    assert sep['mean_correlation'] <= 0.985
    # Angle dependence
    angle_resp = torch.mean(H, dim=0)
    assert torch.std(angle_resp).item() >= 0.05
    # Frequency-wise variation across angles
    assert sep['mean_freq_range'] >= 0.05

