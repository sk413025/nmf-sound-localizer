#!/usr/bin/env python3
"""
基本一致性測試：驗證 NMF 數學框架
測試邏輯：如果 H 是從 y_data 和 x_data 計算的，且 x_data = W_from_X @ U_from_X
那麼理論上應該能重建 y_data
"""

import numpy as np
import torch
import os
import sys
from pathlib import Path
from sklearn.decomposition import NMF as sklearn_NMF

# Add project path
sys.path.append('/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/development-workspace')

from nmf_localizer.core.data_processor import DataProcessor
from nmf_localizer.core.localizer import NMFSoundLocalizer
from nmf_localizer.config import NMFConfig
from nmf_localizer.utils.audio_utils import AudioProcessor


def learn_W_from_X(X_data, n_components=15):
    """從 X_data 學習源特徵 W_from_X 和係數 U_from_X"""
    print("從 X_data 學習源特徵...")
    
    # 使用 sklearn NMF 分解 X_data ≈ W @ U.T
    nmf_x = sklearn_NMF(n_components=n_components, init='nndsvd', max_iter=1000, random_state=42)
    H_x_time = nmf_x.fit_transform(X_data.T)  # (N, K)
    W_x = nmf_x.components_.T               # (F, K)
    U_from_X = torch.from_numpy(H_x_time.T).float()  # (K, N)
    
    print(f"  W_from_X shape: {W_x.shape}, U_from_X shape: {U_from_X.shape}")
    print(f"  Reconstruction error: {nmf_x.reconstruction_err_:.2e}")
    
    # 驗證分解正確性
    X_recon = W_x @ H_x_time.T
    recon_mse = np.mean((X_data - X_recon) ** 2)
    print(f"  驗證 X_data ≈ W_from_X @ U_from_X: MSE={recon_mse:.2e}")
    
    return torch.from_numpy(W_x).float(), U_from_X


def basic_consistency_test(Y_data, H, W_from_X, U_from_X, direction_idx, config):
    """基本一致性測試：使用擴展的 U_from_X 驗證數學框架"""
    print("\n" + "="*80)
    print("基本一致性測試")
    print("="*80)
    
    # 創建 localizer 獲取 A 矩陣
    localizer = NMFSoundLocalizer(config)
    localizer.load_source_dictionary(W_from_X)
    localizer.load_transfer_functions(H)
    
    print(f"設定檢查：")
    print(f"  A (mixing matrix) shape: {localizer.A.shape}")
    print(f"  H directions: {H.shape[1]}")
    print(f"  W_from_X components: {W_from_X.shape[1]}")
    print(f"  U_from_X shape: {U_from_X.shape}")
    
    # 擴展 U_from_X 到正確的維度 (255, N)
    n_directions = H.shape[1]
    n_components = W_from_X.shape[1]
    expected_sources = n_directions * n_components
    
    X_expanded = torch.zeros(expected_sources, U_from_X.shape[1])
    start_idx = direction_idx * n_components
    end_idx = (direction_idx + 1) * n_components
    
    print(f"擴展邏輯：")
    print(f"  總源數: {expected_sources} = {n_directions} 方向 × {n_components} 成分")
    print(f"  方向 {direction_idx} 對應源索引: {start_idx}-{end_idx}")
    
    # 只在正確方向填入 U_from_X
    X_expanded[start_idx:end_idx, :] = U_from_X
    
    print(f"  非零行數: {torch.count_nonzero(torch.sum(X_expanded, dim=1))}/{X_expanded.shape[0]}")
    print(f"  係數範圍: {X_expanded[start_idx:end_idx, :].min():.3f} - {X_expanded[start_idx:end_idx, :].max():.3f}")
    
    # 用 A @ X_expanded 重建 Y
    print(f"\n重建測試：")
    Y_hat_tensor = localizer.A @ X_expanded
    Y_hat = Y_hat_tensor.detach().cpu().numpy()
    
    # 計算品質指標
    epsilon = 1e-12
    mse = float(np.mean((Y_data - Y_hat) ** 2))
    scale_ratio = float(np.mean(Y_data) / max(np.mean(Y_hat), epsilon))
    corr = float(np.corrcoef(Y_data.flatten(), Y_hat.flatten())[0, 1])
    
    print(f"基本一致性測試結果：")
    print(f"  MSE: {mse:.2e}")
    print(f"  規模比例: {scale_ratio:.4f}")
    print(f"  相關性: {corr:.4f}")
    
    if corr > 0.8:
        print("✓ 高相關性 - 數學框架基本一致！")
    elif corr > 0.5:
        print("⚠ 中等相關性 - 框架部分有效，可能有細節問題")
    else:
        print("✗ 低相關性 - 數學框架可能有根本問題")
    
    return {
        'mse': mse,
        'scale_ratio': scale_ratio,
        'correlation': corr,
        'Y_hat': Y_hat
    }


def main():
    """主函數：執行基本一致性測試"""
    print("基本一致性測試：驗證 NMF 數學框架")
    print("="*80)
    
    # 配置
    config = NMFConfig(
        sample_rate=16000, n_fft=2048, hop_length=512,
        freq_min=500.0, freq_max=3000.0, n_files_per_angle=1,
        max_iter=50, beta=0, tolerance=1e-6
    )
    
    # 數據路徑
    x_root = "/Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original/white_noise_original_data_no_edge_sync_vad"
    y_root = "/Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original/white_noise_box_data_no_edge_sync_vad"
    test_angle = "angle_90"
    
    # 載入數據
    x_dir = os.path.join(x_root, test_angle)
    y_dir = os.path.join(y_root, test_angle)
    
    x_files = [f for f in os.listdir(x_dir) if f.endswith('.npy')]
    y_files = [f for f in os.listdir(y_dir) if f.endswith('.npy')]
    
    x_audio = np.load(os.path.join(x_dir, x_files[0]))
    y_audio = np.load(os.path.join(y_dir, y_files[0]))
    
    # 處理數據
    audio_processor = AudioProcessor()
    
    freqs_x, _, _, magnitude_x = audio_processor.compute_stft_spectrogram(
        x_audio.astype(np.float32), fs=config.sample_rate,
        nperseg=config.n_fft, noverlap=config.n_fft - config.hop_length
    )
    
    freqs_y, _, _, magnitude_y = audio_processor.compute_stft_spectrogram(
        y_audio.astype(np.float32), fs=config.sample_rate,
        nperseg=config.n_fft, noverlap=config.n_fft - config.hop_length
    )
    
    X_tensor = torch.from_numpy(magnitude_x).float()
    Y_tensor = torch.from_numpy(magnitude_y).float()
    
    X_spec, _ = audio_processor.apply_frequency_filter(
        X_tensor, freqs_x, config.freq_min, config.freq_max
    )
    Y_spec, _ = audio_processor.apply_frequency_filter(
        Y_tensor, freqs_y, config.freq_min, config.freq_max
    )
    
    X_data = X_spec.detach().cpu().numpy()
    Y_data = Y_spec.detach().cpu().numpy()
    
    print(f"數據載入完成: X_data={X_data.shape}, Y_data={Y_data.shape}")
    
    # 獲取傳遞函數和角度對應
    processor = DataProcessor(config)
    H, angles, angle_folders, metadata = processor.estimate_transfer_functions(Path(x_root), Path(y_root))
    
    print(f"傳遞函數: H={H.shape}")
    print(f"角度數組: {angles}")
    print(f"角度資料夾: {[f.name for f in angle_folders]}")
    
    # 詳細檢查角度對應關係
    print(f"\n角度對應檢查:")
    for i, (angle, folder) in enumerate(zip(angles, angle_folders)):
        print(f"  索引 {i}: {angle:.0f}° ← {folder.name}")
        if folder.name == test_angle:
            print(f"    ★ 這是我們載入的 y_data 對應的角度！")
    
    # 找到對應的方向索引
    test_angle_deg = int(test_angle.replace('angle_', ''))
    direction_idx = None
    for i, angle in enumerate(angles):
        if abs(angle - test_angle_deg) < 1e-6:
            direction_idx = i
            break
    
    if direction_idx is None:
        print(f"\n⚠️  找不到角度 {test_angle_deg}°！")
        direction_idx = 0
    else:
        print(f"\n✓ 角度 {test_angle_deg}° → 方向索引 {direction_idx}")
        
    # 雙重確認：檢查資料夾名稱是否也匹配
    if direction_idx < len(angle_folders):
        corresponding_folder = angle_folders[direction_idx].name
        if corresponding_folder == test_angle:
            print(f"✓ 資料夾名稱確認: {corresponding_folder} 匹配!")
        else:
            print(f"⚠️  資料夾不匹配: 期望 {test_angle}, 但方向 {direction_idx} 對應 {corresponding_folder}")
    
    print(f"\n實際測試:")
    print(f"  載入的 y_data 來自: {y_dir}")
    print(f"  使用的 H[:, {direction_idx}] 對應: {angles[direction_idx]:.0f}° ({angle_folders[direction_idx].name})")
    
    # 從 X_data 學習源特徵
    W_from_X, U_from_X = learn_W_from_X(X_data, n_components=15)
    
    # 執行基本一致性測試
    result = basic_consistency_test(Y_data, H, W_from_X, U_from_X, direction_idx, config)
    
    print(f"\n✅ 測試完成！")
    print(f"結果：相關性 {result['correlation']:.4f}")
    
    return result


if __name__ == "__main__":
    result = main()