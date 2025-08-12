"""
Tests for configuration management.
"""

import pytest
import torch
from nmf_localizer.config import NMFConfig, DataPack


class TestNMFConfig:
    """Test NMFConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = NMFConfig()
        
        # Audio processing defaults
        assert config.sample_rate == 16000
        assert config.n_fft == 2048
        assert config.hop_length == 512
        
        # NMF defaults
        assert config.beta == 0.0
        assert config.lambda_group == 20.0
        assert config.gamma_sparse == 1.0
        assert config.max_iter == 100
        
        # Device default
        assert config.device == 'cpu'
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = NMFConfig(
            beta=1.0,
            lambda_group=30.0,
            device='cuda',
            n_test_examples=200
        )
        
        assert config.beta == 1.0
        assert config.lambda_group == 30.0
        assert config.device == 'cuda'
        assert config.n_test_examples == 200
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid beta values
        for beta in [0.0, 0.5, 1.0, 2.0]:
            config = NMFConfig(beta=beta)
            assert config.beta == beta
        
        # Test frequency range
        config = NMFConfig(freq_min=300, freq_max=2000)
        assert config.freq_min < config.freq_max
        
        # Test positive values
        config = NMFConfig(max_iter=50, n_atoms_per_speaker=25)
        assert config.max_iter > 0
        assert config.n_atoms_per_speaker > 0


class TestDataPack:
    """Test DataPack class."""
    
    def test_empty_datapack(self):
        """Test empty DataPack initialization."""
        data_pack = DataPack()
        
        assert data_pack.transfer_functions is None
        assert data_pack.angles is None
        assert data_pack.speaker_data is None
        assert data_pack.test_data is None
    
    def test_datapack_with_data(self):
        """Test DataPack with sample data."""
        # Create sample tensors
        H = torch.randn(129, 11)  # Transfer functions
        angles = torch.linspace(0, 180, 11)
        speaker_data = [torch.randn(129, 100)]
        test_data = [{'magnitude_spec': torch.randn(129, 80), 'true_angle': 45}]
        
        data_pack = DataPack(
            transfer_functions=H,
            angles=angles,
            speaker_data=speaker_data,
            test_data=test_data
        )
        
        assert data_pack.transfer_functions.shape == (129, 11)
        assert data_pack.angles.shape == (11,)
        assert len(data_pack.speaker_data) == 1
        assert len(data_pack.test_data) == 1
    
    def test_datapack_validation(self):
        """Test DataPack validation method."""
        # Valid data pack
        data_pack = DataPack(
            transfer_functions=torch.randn(129, 11),
            angles=torch.linspace(0, 180, 11),
            speaker_data=[torch.randn(129, 100)],
            test_data=[{'magnitude_spec': torch.randn(129, 80), 'true_angle': 45}]
        )
        
        assert data_pack.validate() is True
        
        # Invalid data pack (missing required data)
        empty_pack = DataPack()
        assert empty_pack.validate() is False
    
    def test_datapack_properties(self):
        """Test DataPack computed properties."""
        H = torch.randn(64, 8)
        angles = torch.linspace(0, 180, 8)
        
        data_pack = DataPack(transfer_functions=H, angles=angles)
        
        assert data_pack.tf_shape == torch.Size([64, 8])
        assert data_pack.n_angles == 8
        assert data_pack.n_speakers == 0  # No speaker data provided
        assert data_pack.n_test == 0  # No test data provided
    
    def test_datapack_str_representation(self):
        """Test DataPack string representation."""
        data_pack = DataPack(
            transfer_functions=torch.randn(129, 11),
            angles=torch.linspace(0, 180, 11),
            speaker_data=[torch.randn(129, 100)],
            test_data=[{'magnitude_spec': torch.randn(129, 80), 'true_angle': 45}] * 20
        )
        
        str_repr = str(data_pack)
        assert "tf_shape=torch.Size([129, 11])" in str_repr
        assert "n_angles=11" in str_repr
        assert "n_speakers=1" in str_repr
        assert "n_test=20" in str_repr