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


def basic_consistency_test_raw(Y_data, H_raw, W_from_X, U_from_X, config):
    """基本一致性測試：使用未正規化的 H_raw 驗證數學框架"""
    print("\n" + "="*80)
    print("基本一致性測試 (使用真實未正規化 H_raw)")
    print("="*80)
    
    # 建構 A 矩陣：A = diag(H_raw) @ W_from_X
    # H_raw 是單一方向的傳遞函數，直接與 W_from_X 組合
    print(f"設定檢查：")
    print(f"  H_raw shape: {H_raw.shape} (單一方向的真實傳遞函數)")
    print(f"  W_from_X shape: {W_from_X.shape}")
    print(f"  U_from_X shape: {U_from_X.shape}")
    
    # 轉換為 torch tensor
    H_raw_tensor = torch.from_numpy(H_raw).float().view(-1, 1)  # [F, 1]
    W_from_X_tensor = W_from_X  # [F, K]
    
    # 建構混合矩陣 A = diag(H_raw) @ W_from_X
    # diag(H_raw) @ W 等於 H_raw[:, None] * W  (廣播)
    A = H_raw_tensor * W_from_X_tensor  # [F, K]
    
    print(f"  A (mixing matrix) shape: {A.shape}")
    print(f"  A = diag(H_raw) @ W_from_X")
    print(f"  A 統計: mean={A.mean():.6f}, std={A.std():.6f}")
    
    # 直接使用 U_from_X 作為係數矩陣（無需擴展）
    U_tensor = U_from_X  # [K, N]
    
    print(f"  使用原始 U_from_X (無需擴展到多方向)")
    print(f"  U_from_X 統計: mean={U_tensor.mean():.6f}, range={U_tensor.min():.3f}-{U_tensor.max():.3f}")
    
    # 重建測試：Y_hat = A @ U_from_X
    print(f"\n重建測試：Y_hat = diag(H_raw) @ W_from_X @ U_from_X")
    Y_hat_tensor = A @ U_tensor  # [F, N]
    Y_hat = Y_hat_tensor.detach().cpu().numpy()
    
    print(f"  重建形狀: Y_hat={Y_hat.shape}, Y_data={Y_data.shape}")
    
    # 計算品質指標
    epsilon = 1e-12
    mse = float(np.mean((Y_data - Y_hat) ** 2))
    scale_ratio = float(np.mean(Y_data) / max(np.mean(Y_hat), epsilon))
    corr = float(np.corrcoef(Y_data.flatten(), Y_hat.flatten())[0, 1])
    
    print(f"\n基本一致性測試結果（真實未正規化 H_raw）：")
    print(f"  MSE: {mse:.2e}")
    print(f"  規模比例: {scale_ratio:.4f} (1.0 = 完美)")
    print(f"  相關性: {corr:.4f}")
    
    # 統計對比
    print(f"\n數據統計對比：")
    print(f"  Y_data: mean={np.mean(Y_data):.2e}, std={np.std(Y_data):.2e}")
    print(f"  Y_hat:  mean={np.mean(Y_hat):.2e}, std={np.std(Y_hat):.2e}")
    print(f"  絕對尺度保留: {np.mean(Y_hat)/np.mean(Y_data):.4f}")
    
    if corr > 0.8:
        print("✓ 高相關性 - 數學框架一致性良好！")
    elif corr > 0.5:
        print("⚠ 中等相關性 - 框架部分有效")
    else:
        print("✗ 低相關性 - 仍有其他問題")
    
    return {
        'mse': mse,
        'scale_ratio': scale_ratio,
        'correlation': corr,
        'Y_hat': Y_hat,
        'H_raw_stats': {
            'mean': float(np.mean(H_raw)),
            'std': float(np.std(H_raw)),
            'min': float(np.min(H_raw)),
            'max': float(np.max(H_raw))
        }
    }


def basic_consistency_test(Y_data, H, W_from_X, U_from_X, direction_idx, config):
    """基本一致性測試：使用擴展的 U_from_X 驗證數學框架（舊版本，保留以備比較）"""
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
        max_iter=50, beta=0, tolerance=1e-6,
        apply_contrast_enhancement=False    # 保持 H 的原始絕對尺度
    )
    
    # 數據路徑
    x_root = "/Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original/white_noise_original_data_no_edge_sync_vad"
    y_root = "/Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original/white_noise_box_data_no_edge_sync_vad"
    test_angle = "angle_90"
    
    # 載入數據
    x_dir = os.path.join(x_root, test_angle)
    y_dir = os.path.join(y_root, test_angle)
    
    # 使用 sorted() 確保與 H 計算時的檔案選擇一致
    x_files = sorted([f for f in os.listdir(x_dir) if f.endswith('.npy')])
    y_files = sorted([f for f in os.listdir(y_dir) if f.endswith('.npy')])
    
    x_audio = np.load(os.path.join(x_dir, x_files[0]))
    y_audio = np.load(os.path.join(y_dir, y_files[0]))
    
    # 處理數據 - 保留複數 STFT 用於計算未正規化的 H_raw
    from scipy import signal
    
    # 計算複數 STFT（保留相位資訊）
    stft_params = {
        'fs': config.sample_rate,
        'nperseg': config.n_fft,
        'noverlap': config.n_fft - config.hop_length,
        'window': 'hann'  # 與 STFT processor 一致
    }
    
    freqs_x, times_x, X_stft = signal.stft(x_audio.astype(np.float32), **stft_params)
    freqs_y, times_y, Y_stft = signal.stft(y_audio.astype(np.float32), **stft_params)
    
    # 驗證一致性
    assert np.allclose(freqs_x, freqs_y), "頻率陣列必須匹配"
    assert X_stft.shape == Y_stft.shape, f"STFT 形狀必須匹配: {X_stft.shape} vs {Y_stft.shape}"
    
    print(f"STFT 形狀: X_stft={X_stft.shape}, Y_stft={Y_stft.shape}")
    
    # 計算未正規化的傳遞函數 H_raw = |Y_stft| / |X_stft|
    epsilon = 1e-12
    H_stft_complex = Y_stft / (X_stft + epsilon)
    
    # 時間平均取幅度（完全保留絕對尺度）
    H_raw_full = np.mean(np.abs(H_stft_complex), axis=1)  # [freq]
    
    print(f"H_raw_full 形狀: {H_raw_full.shape}")
    print(f"H_raw_full 統計: mean={np.mean(H_raw_full):.4f}, min={np.min(H_raw_full):.4f}, max={np.max(H_raw_full):.4f}")
    
    # 應用相同的頻率濾波
    freq_mask = (freqs_x >= config.freq_min) & (freqs_x <= config.freq_max)
    H_raw = H_raw_full[freq_mask]  # 未正規化的傳遞函數
    freqs_filtered = freqs_x[freq_mask]
    
    # 取得對應的 magnitude spectra 並濾波
    magnitude_x = np.abs(X_stft)
    magnitude_y = np.abs(Y_stft)
    
    X_data = magnitude_x[freq_mask, :]
    Y_data = magnitude_y[freq_mask, :]
    
    print(f"數據載入完成: X_data={X_data.shape}, Y_data={Y_data.shape}, H_raw={H_raw.shape}")
    
    # 直接使用計算出的 H_raw，無需獲取多方向的傳遞函數
    # H_raw 是當前 (x_audio, y_audio) 對的真實未正規化傳遞函數
    
    print(f"\n真實傳遞函數 H_raw:")
    print(f"  H_raw 統計: mean={np.mean(H_raw):.4f}, std={np.std(H_raw):.4f}, min={np.min(H_raw):.4f}, max={np.max(H_raw):.4f}")
    print(f"  這是從載入的 (x_audio, y_audio) 直接計算的未正規化傳遞函數")
    print(f"  保留了絕對尺度: Y_stft / X_stft 的真實比例")
    
    # 將 H_raw 擴展為與其他方向一致的矩陣格式（僅用於測試框架相容性）
    test_angle_deg = int(test_angle.replace('angle_', ''))
    print(f"\n目標角度: {test_angle_deg}°")
    print(f"  載入的 y_data 來自: {y_dir}")
    print(f"  使用就地計算的 H_raw (完全對應的傳遞函數)")
    
    # 從 X_data 學習源特徵
    W_from_X, U_from_X = learn_W_from_X(X_data, n_components=15)
    
    # 執行基本一致性測試（使用真實未正規化的 H_raw）
    result = basic_consistency_test_raw(Y_data, H_raw, W_from_X, U_from_X, config)
    
    print(f"\n✅ 測試完成！")
    print(f"結果：相關性 {result['correlation']:.4f}")
    
    return result


if __name__ == "__main__":
    result = main()