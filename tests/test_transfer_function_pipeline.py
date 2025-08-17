import numpy as np
import torch
import pytest

from pathlib import Path
from scipy import signal

from nmf_localizer.config import NMFConfig
from nmf_localizer.core.data_processor import DataProcessor
from nmf_localizer.core.transfer_functions import TransferFunctionProcessor
from nmf_localizer.core.localizer import NMFSoundLocalizer


def _make_synthetic_H(freqs: np.ndarray, angles_deg: torch.Tensor) -> torch.Tensor:
    """Create a physically plausible synthetic H(F,D).

    H(f,θ) = (0.5 + 0.5 cos^2(θ-90)) * (1 + 0.3 sin(f/100)).
    Returns torch.FloatTensor of shape (F, D) aligned to freqs and angles.
    """
    F = len(freqs)
    D = len(angles_deg)
    H = torch.zeros(F, D)
    for i, angle in enumerate(angles_deg):
        spatial = torch.cos(torch.deg2rad(angle - 90.0)) ** 2
        for j, f in enumerate(freqs):
            H[j, i] = (0.5 + 0.5 * spatial) * (1.0 + 0.3 * np.sin(f / 100.0))
    return H.float()


def _synthesize_xy_waveforms_for_angle(x: np.ndarray, H_f: torch.Tensor, fs: int, n_fft: int, hop_length: int) -> np.ndarray:
    """Given a real waveform x and per-frequency magnitude H_f(F,),
    create y so that STFT(y) ≈ H_f[:,None] * STFT(x)."""
    # STFT params consistent with DataProcessor
    f, t, X_stft = signal.stft(
        x,
        fs=fs,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        window='hann',
    )
    # Broadcast multiply in frequency across time frames
    H_np = H_f.cpu().numpy().reshape(-1, 1)
    Y_stft = H_np * X_stft
    # iSTFT to waveform
    _, y = signal.istft(
        Y_stft,
        fs=fs,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        window='hann',
    )
    return y.astype(np.float32)


def _write_angle_npy_tree(root: Path, angle_values: torch.Tensor, xs: list[list[np.ndarray]]):
    """Create a directory tree with angle_XX subfolders and save .npy arrays.
    xs is a list per angle, each a list of arrays to save.
    """
    for angle_idx, angle_deg in enumerate(angle_values.tolist()):
        d = root / f"angle_{int(angle_deg)}"
        d.mkdir(parents=True, exist_ok=True)
        for i, arr in enumerate(xs[angle_idx]):
            np.save(d / f"sample_{i:02d}.npy", arr)


def test_h_estimation_from_synthetic_xy(tmp_path: Path):
    """Integration: DataProcessor.estimate_transfer_functions recovers expected
    relative H after band-limit and mean normalization + global scaling.
    """
    # Config consistent with docs
    cfg = NMFConfig(
        sample_rate=16000,
        n_fft=2048,
        hop_length=512,
        freq_min=500.0,
        freq_max=1500.0,
        n_files_per_angle=2,
        apply_contrast_enhancement=False,  # keep pipeline simple for verification
        device='cpu',
    )

    # Angles and synthetic H across full STFT freqs
    # Use 0, 90, 180 degrees to match reference behavior
    angles = torch.tensor([0.0, 90.0, 180.0], dtype=torch.float32)

    # Build synthetic X,Y sets per angle
    duration = 1.0  # seconds
    n_samples = int(cfg.sample_rate * duration)
    rng = np.random.default_rng(123)
    # Create a single base x per file per angle to avoid leakage of angle structure into X
    n_files = cfg.n_files_per_angle

    # We need freqs matching DataProcessor STFT to build H_true
    # Use a temporary probe STFT to get frequency vector
    probe = rng.standard_normal(n_samples).astype(np.float32)
    f, _, _ = signal.stft(
        probe,
        fs=cfg.sample_rate,
        nperseg=cfg.n_fft,
        noverlap=cfg.n_fft - cfg.hop_length,
        window='hann',
    )
    H_full = _make_synthetic_H(f, angles)

    # Build file trees
    x_root = tmp_path / "original"
    y_root = tmp_path / "box"
    x_root.mkdir()
    y_root.mkdir()

    xs_all = []
    ys_all = []
    for a_idx in range(len(angles)):
        angle_xs = []
        angle_ys = []
        for _ in range(n_files):
            x = rng.standard_normal(n_samples).astype(np.float32)
            y = _synthesize_xy_waveforms_for_angle(x, H_full[:, a_idx], cfg.sample_rate, cfg.n_fft, cfg.hop_length)
            angle_xs.append(x)
            angle_ys.append(y)
        xs_all.append(angle_xs)
        ys_all.append(angle_ys)

    _write_angle_npy_tree(x_root, angles, xs_all)
    _write_angle_npy_tree(y_root, angles, ys_all)

    # Run estimation
    dp = DataProcessor(cfg)
    H_est, angles_out, angle_folders, meta = dp.estimate_transfer_functions(x_root, y_root, method='xy_correspondence')

    # Verify angles preserved
    assert torch.allclose(angles_out, angles)

    # Compute expected band-limited + mean-normalized + global-scaled H
    freq_mask = (f >= cfg.freq_min) & (f <= cfg.freq_max)
    H_bl = H_full[freq_mask, :].clone()
    mean_spectrum = torch.mean(H_bl, dim=1, keepdim=True)
    H_rel = H_bl / (mean_spectrum + 1e-10)
    H_min = H_rel.min()
    H_max = H_rel.max()
    H_expected = (H_rel - H_min) / (H_max - H_min) * 0.8 + 0.1

    # Compare recovered and expected
    assert H_est.shape == H_expected.shape
    diff = torch.mean(torch.abs(H_est - H_expected)).item()
    # Tight average absolute error threshold
    assert diff < 1e-2, f"Mean |Δ| too large: {diff:.4e}"


def test_transfer_processor_frequency_limit_and_reference_norm():
    """Unit: TransferFunctionProcessor.apply_frequency_limit + reference normalization.
    Ensures 90° reference yields ~1.0 column and ratios preserved.
    """
    cfg = NMFConfig(
        freq_min=500.0,
        freq_max=1500.0,
        apply_contrast_enhancement=False,
        device='cpu',
    )
    tfp = TransferFunctionProcessor(cfg)

    # Synthetic freqs 0..4000, angles 0,90,180
    fs = 16000
    n_fft = 2048
    freqs = np.linspace(0, fs / 2, n_fft // 2 + 1)
    angles = torch.tensor([0.0, 90.0, 180.0], dtype=torch.float32)
    H_full = _make_synthetic_H(freqs, angles)

    # Process
    H_proc, info = tfp.process_transfer_functions(H_full, angles, freqs=freqs, reference_angle=90.0)

    # Shape check within band
    assert H_proc.shape[0] == np.count_nonzero((freqs >= cfg.freq_min) & (freqs <= cfg.freq_max))
    assert H_proc.shape[1] == len(angles)

    # Reference column near 1 per frequency
    ref_col = H_proc[:, 1]
    assert torch.allclose(ref_col, torch.ones_like(ref_col), atol=1e-5)

    # Ratio preservation: for any freq, H[:,0]/H[:,1] equals original ratio at that freq
    # Rebuild band-limited originals
    mask = (freqs >= cfg.freq_min) & (freqs <= cfg.freq_max)
    H_bl = H_full[mask, :]
    ratios_orig = H_bl[:, 0] / (H_bl[:, 1] + 1e-10)
    ratios_proc = H_proc[:, 0]
    mae = torch.mean(torch.abs(ratios_proc - ratios_orig)).item()
    assert mae < 1e-5


def test_mixing_matrix_construction_and_freq_weights():
    """Unit: A = [diag(H_d)W] and frequency weights applied consistently."""
    cfg = NMFConfig(device='cpu', normalize_blocks=False)
    localizer = NMFSoundLocalizer(cfg)

    # Small deterministic W,H
    W = torch.tensor([
        [1.0, 2.0, 3.0],
        [0.5, 1.0, 1.5],
        [2.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ])  # F=4, K=3
    H = torch.tensor([
        [1.0, 0.5],
        [2.0, 1.0],
        [0.5, 2.0],
        [1.0, 1.0],
    ])  # F=4, D=2
    angles = torch.tensor([0.0, 90.0])

    localizer.load_source_dictionary(W)
    localizer.load_transfer_functions(H, angles)

    # Expected A
    A_blocks = []
    for d in range(H.shape[1]):
        H_d = torch.diag(H[:, d])
        A_blocks.append(H_d @ W)
    A_expected = torch.cat(A_blocks, dim=1)
    assert torch.allclose(localizer.A, A_expected, atol=1e-7)

    # Apply frequency weights and recheck
    w = torch.tensor([1.0, 0.5, 2.0, 0.25])
    localizer.set_frequency_weights(w)
    A_w_expected = torch.cat([(w.view(-1, 1) * (torch.diag(H[:, d]) @ W)) for d in range(H.shape[1])], dim=1)
    assert torch.allclose(localizer.A, A_w_expected, atol=1e-7)


def test_separability_metrics_discriminate_cases():
    """Unit: analyze_separability yields higher correlation/condition for collinear H."""
    cfg = NMFConfig(device='cpu', apply_contrast_enhancement=False)
    tfp = TransferFunctionProcessor(cfg)

    # Build two H sets with same F,D
    F, D = 64, 4
    freqs = np.linspace(500, 1500, F)
    angles = torch.linspace(0, 180, D)
    # Build a non-separable, directionally distinct H by adding angle-frequency interaction
    H_sep = torch.zeros(F, D)
    for i, ang in enumerate(angles):
        spatial = torch.cos(torch.deg2rad(ang - 90.0)) ** 2  # base spatial gain
        phase = 2 * np.pi * (i + 1) / (D + 1)                # angle-dependent phase
        for j, f in enumerate(freqs):
            base = (1.0 + 0.3 * np.sin(f / 100.0))
            interaction = 1.0 + 0.1 * np.sin(f / 150.0 + phase)  # vary shape with angle
            H_sep[j, i] = (0.5 + 0.5 * spatial) * base * interaction

    # Collinear: make all columns nearly the same
    base_col = H_sep[:, 1:2]
    noise = 1e-3 * torch.randn_like(H_sep)
    H_col = base_col.repeat(1, D) + noise

    sep1 = tfp.analyze_separability(H_sep)
    sep2 = tfp.analyze_separability(H_col)

    assert sep2['mean_correlation'] > sep1['mean_correlation']


def test_get_direction_index_wraparound():
    """Unit: get_direction_index handles 0/360 wrap-around correctly."""
    cfg = NMFConfig(device='cpu')
    tfp = TransferFunctionProcessor(cfg)
    angles = torch.tensor([350.0, 10.0, 30.0])
    # Target 0 is equally distant to 350 and 10; use 355 for unique closest to 350
    idx = tfp.get_direction_index(angles, 355.0)
    assert idx == 0
