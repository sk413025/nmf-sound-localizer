from pathlib import Path

import pytest

from doa_rl.domain_randomization.generator import AngleRange, AngleRangeShardGenerator, GenerationConfig


def test_stft_mismatch_raises_on_wrong_sample_rate(tmp_path):
    """Generator should fail fast when STFT F dimension no longer matches W/H."""
    data_root = Path("/Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized")
    w_path = Path("/Users/jnrle/Documents/LDVReorientation/worktrees/domain-randomization-AgentV2/doa_normalized_config_c_corrected/models/usm.pth")
    h_path = Path("/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth")
    config = GenerationConfig(
        data_root=data_root,
        w_path=w_path,
        h_path=h_path,
        output_root=tmp_path / "out",
        clips_per_angle=1,
        k=2,
        m=8,
        reduction_seed=1,
        projection_seed=1,
        orig_sample_rate=48000,
        target_sample_rate=8000,  # force STFT grid mismatch
        n_fft=2048,
        freq_min=300.0,
        freq_max=3000.0,
        d_model=16,
    )
    gen = AngleRangeShardGenerator(config)
    with pytest.raises(ValueError):
        gen.generate_shard(AngleRange("smoke", 0, 0), max_samples=1)
