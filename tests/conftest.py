"""
Pytest configuration and fixtures for fixed test data sources.
Based on commit 83c959735d2f959240e764a20ba92c1cf64424eb experiment data.
"""
import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_fixed_test_data():
    """
    Automatically set up fixed test data sources for all tests.
    
    This fixture ensures all pytest sessions use the same data source
    that was validated in commit 83c959735d2f959240e764a20ba92c1cf64424eb.
    
    Data source characteristics:
    - X (original): white_noise_original_data_no_edge_sync_vad (synchronized VAD processed)
    - Y (box response): white_noise_box_data_no_edge_sync_vad (synchronized VAD processed)
    - Files per angle: 3 (clip_000, clip_001, clip_002)
    - Angles available: 90°, 100°, 110° (and others)
    - Processing: Synchronized VAD with same mask applied to X and Y
    - Validation: Time-frequency correspondence maintained
    """
    # Set fixed environment variables for all tests - using synchronized VAD processed data
    os.environ["REAL_TF_X_ROOT"] = "/Users/sbplab/jnrle/datasets/test_nmf_output_no_edge_with_original/white_noise_original_data_no_edge_sync_vad"
    os.environ["REAL_TF_Y_ROOT"] = "/Users/sbplab/jnrle/datasets/test_nmf_output_no_edge_with_original/white_noise_box_data_no_edge_sync_vad"
    os.environ["REAL_TF_N_FILES"] = "3"
    
    # Verify data sources exist
    x_root = os.environ["REAL_TF_X_ROOT"]
    y_root = os.environ["REAL_TF_Y_ROOT"]
    
    if not os.path.exists(x_root):
        raise FileNotFoundError(f"X data source not found: {x_root}")
    if not os.path.exists(y_root):
        raise FileNotFoundError(f"Y data source not found: {y_root}")
    
    # Verify expected structure exists
    required_angles = ["angle_90", "angle_100", "angle_110"]
    for angle in required_angles:
        x_angle_path = os.path.join(x_root, angle)
        y_angle_path = os.path.join(y_root, angle)
        
        if not os.path.exists(x_angle_path):
            raise FileNotFoundError(f"X angle directory not found: {x_angle_path}")
        if not os.path.exists(y_angle_path):
            raise FileNotFoundError(f"Y angle directory not found: {y_angle_path}")
    
    print(f"✅ Fixed test data configured:")
    print(f"   X_ROOT: {x_root}")
    print(f"   Y_ROOT: {y_root}")
    print(f"   N_FILES: {os.environ['REAL_TF_N_FILES']}")
    print(f"   Validated angles: {required_angles}")


@pytest.fixture
def test_data_paths():
    """
    Provide test data paths for individual test functions.
    
    Returns:
        dict: Dictionary with data source paths and parameters
    """
    return {
        "x_root": os.environ["REAL_TF_X_ROOT"],
        "y_root": os.environ["REAL_TF_Y_ROOT"], 
        "n_files": int(os.environ["REAL_TF_N_FILES"]),
        "validated_angles": ["angle_90", "angle_100", "angle_110"],
        "commit_reference": "83c959735d2f959240e764a20ba92c1cf64424eb"
    }


@pytest.fixture
def data_integrity_check(test_data_paths):
    """
    Verify data integrity before running tests.
    
    Returns:
        dict: Data integrity report
    """
    x_root = test_data_paths["x_root"]
    y_root = test_data_paths["y_root"]
    n_files = test_data_paths["n_files"]
    
    integrity_report = {
        "x_files_found": {},
        "y_files_found": {},
        "missing_files": [],
        "data_valid": True
    }
    
    # Check for required files in each angle
    for angle in test_data_paths["validated_angles"]:
        x_angle_dir = os.path.join(x_root, angle)
        y_angle_dir = os.path.join(y_root, angle)
        
        # Count files in each directory
        if os.path.exists(x_angle_dir):
            x_files = [f for f in os.listdir(x_angle_dir) if f.endswith('.npy')]
            integrity_report["x_files_found"][angle] = len(x_files)
        else:
            integrity_report["missing_files"].append(f"X/{angle}")
            integrity_report["data_valid"] = False
            
        if os.path.exists(y_angle_dir):
            y_files = [f for f in os.listdir(y_angle_dir) if f.endswith('.npy')]
            integrity_report["y_files_found"][angle] = len(y_files)
        else:
            integrity_report["missing_files"].append(f"Y/{angle}")
            integrity_report["data_valid"] = False
    
    return integrity_report