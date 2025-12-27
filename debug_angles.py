
import torch
import numpy as np
import sys

def check_data():
    h_path = 'h_matrix_normalized_original_to_box.pth'
    print(f"Loading {h_path}...")
    try:
        h_data = torch.load(h_path, map_location='cpu', weights_only=False)
        angles = h_data['angles']
        H = h_data['H']
        print(f"Angles shape: {angles.shape}")
        print(f"Angles values: {angles}")
        print(f"H shape: {H.shape}")
        
        # Check if angles are 0-180
        if len(angles) > 0:
            print(f"Min angle: {np.min(angles)}")
            print(f"Max angle: {np.max(angles)}")
            
        # Do a quick SVD to see v_vec
        H_np = H.numpy()
        eps = 1e-8
        H_log = np.log(np.clip(np.abs(H_np), eps, None))
        H_log_centered = H_log - H_log.mean(axis=1, keepdims=True)
        U, S, Vt = np.linalg.svd(H_log_centered, full_matrices=False)
        V = Vt.T
        v_vec = V[:, 0]
        print(f"v_vec shape: {v_vec.shape}")
        print(f"v_vec first 5: {v_vec[:5]}")
        print(f"v_vec last 5: {v_vec[-5:]}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_data()
