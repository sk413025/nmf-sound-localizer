"""
Test to verify fixed data source configuration works correctly.
Validates that all tests use the data from commit 83c959735d2f959240e764a20ba92c1cf64424eb.
"""
import os
import pytest


def test_environment_variables_set():
    """Test that required environment variables are automatically set."""
    assert "REAL_TF_X_ROOT" in os.environ
    assert "REAL_TF_Y_ROOT" in os.environ  
    assert "REAL_TF_N_FILES" in os.environ
    
    expected_x_root = "/Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original/white_noise_original_data_no_edge"
    expected_y_root = "/Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original/white_noise_box_data_no_edge"
    
    assert os.environ["REAL_TF_X_ROOT"] == expected_x_root
    assert os.environ["REAL_TF_Y_ROOT"] == expected_y_root
    assert os.environ["REAL_TF_N_FILES"] == "3"


def test_data_paths_fixture(test_data_paths):
    """Test that test_data_paths fixture provides expected structure."""
    assert "x_root" in test_data_paths
    assert "y_root" in test_data_paths
    assert "n_files" in test_data_paths
    assert "validated_angles" in test_data_paths
    assert "commit_reference" in test_data_paths
    
    assert test_data_paths["n_files"] == 3
    assert test_data_paths["commit_reference"] == "83c959735d2f959240e764a20ba92c1cf64424eb"
    assert "angle_90" in test_data_paths["validated_angles"]
    assert "angle_100" in test_data_paths["validated_angles"] 
    assert "angle_110" in test_data_paths["validated_angles"]


def test_data_integrity_check(data_integrity_check):
    """Test that data integrity check validates expected data structure."""
    assert "x_files_found" in data_integrity_check
    assert "y_files_found" in data_integrity_check
    assert "missing_files" in data_integrity_check
    assert "data_valid" in data_integrity_check
    
    # Should find files in validated angles
    for angle in ["angle_90", "angle_100", "angle_110"]:
        assert angle in data_integrity_check["x_files_found"]
        assert angle in data_integrity_check["y_files_found"]
        
    # Data should be valid (no missing files)
    assert data_integrity_check["data_valid"] is True
    assert len(data_integrity_check["missing_files"]) == 0


def test_data_sources_exist(test_data_paths):
    """Test that actual data directories and files exist."""
    x_root = test_data_paths["x_root"]
    y_root = test_data_paths["y_root"]
    
    # Test root directories exist
    assert os.path.exists(x_root), f"X data root not found: {x_root}"
    assert os.path.exists(y_root), f"Y data root not found: {y_root}"
    
    # Test angle directories exist
    for angle in test_data_paths["validated_angles"]:
        x_angle_dir = os.path.join(x_root, angle)
        y_angle_dir = os.path.join(y_root, angle)
        
        assert os.path.exists(x_angle_dir), f"X angle dir not found: {x_angle_dir}"
        assert os.path.exists(y_angle_dir), f"Y angle dir not found: {y_angle_dir}"
        
        # Test that .npy files exist in each angle directory
        x_files = [f for f in os.listdir(x_angle_dir) if f.endswith('.npy')]
        y_files = [f for f in os.listdir(y_angle_dir) if f.endswith('.npy')]
        
        assert len(x_files) >= test_data_paths["n_files"], f"Not enough X files in {angle}"
        assert len(y_files) >= test_data_paths["n_files"], f"Not enough Y files in {angle}"


def test_commit_reference_data_consistency(test_data_paths):
    """
    Test that data characteristics match those documented in the reference commit.
    
    Based on commit 83c959735d2f959240e764a20ba92c1cf64424eb:
    - 3 angles used: 90°, 100°, 110°  
    - 3 files per angle (clip_000, clip_001, clip_002)
    - Multi-angle NMF optimization successful
    """
    x_root = test_data_paths["x_root"]
    y_root = test_data_paths["y_root"]
    
    # Test the specific angles documented in the commit
    documented_angles = ["angle_90", "angle_100", "angle_110"]
    
    for angle in documented_angles:
        x_angle_dir = os.path.join(x_root, angle)
        y_angle_dir = os.path.join(y_root, angle)
        
        # Count .npy files in each directory
        x_files = [f for f in os.listdir(x_angle_dir) if f.endswith('.npy')]
        y_files = [f for f in os.listdir(y_angle_dir) if f.endswith('.npy')]
        
        # Should have at least 3 files as documented
        assert len(x_files) >= 3, f"Expected ≥3 X files in {angle}, found {len(x_files)}"
        assert len(y_files) >= 3, f"Expected ≥3 Y files in {angle}, found {len(y_files)}"
        
        print(f"✅ {angle}: X={len(x_files)} files, Y={len(y_files)} files")


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])