#!/usr/bin/env python3
"""
驗證 NMF 概念性問題：
1. 當前 Y, X_data 是什麼？
2. H 是如何建立的？ 
3. Activation matrix X 對應什麼？
4. 理論上的重建品質是多少？
5. 與當前結果的差異分析
"""

import numpy as np
import torch
import os
import sys
import datetime
from pathlib import Path

# Add project path
sys.path.append('/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/development-workspace')

from nmf_localizer.core.data_processor import DataProcessor
from nmf_localizer.core.localizer import NMFSoundLocalizer
from nmf_localizer.config import NMFConfig
from nmf_localizer.utils.audio_utils import AudioProcessor


def analyze_data_concepts():
    """分析當前數據的概念定義"""
    print("="*80)
    print("步驟 1: 分析當前數據概念")
    print("="*80)
    
    # 使用與 debug_is_divergence.py 相同的配置
    config = NMFConfig(
        sample_rate=16000, n_fft=2048, hop_length=512,
        freq_min=500.0, freq_max=3000.0, n_files_per_angle=1,
        max_iter=10, beta=0, lambda_group=0.1, gamma_sparse=0.01, tolerance=1e-6
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
    
    print(f"原始音頻數據:")
    print(f"  x_audio (proxy): shape={x_audio.shape}, mean={np.mean(x_audio):.2e}")
    print(f"  y_audio (LDV): shape={y_audio.shape}, mean={np.mean(y_audio):.2e}")
    
    # 處理為頻譜圖
    audio_processor = AudioProcessor()
    
    freqs_x, _, _, magnitude_x = audio_processor.compute_stft_spectrogram(
        x_audio.astype(np.float32), fs=config.sample_rate,
        nperseg=config.n_fft, noverlap=config.n_fft - config.hop_length
    )
    
    freqs_y, _, _, magnitude_y = audio_processor.compute_stft_spectrogram(
        y_audio.astype(np.float32), fs=config.sample_rate,
        nperseg=config.n_fft, noverlap=config.n_fft - config.hop_length
    )
    
    print(f"\nSTFT 頻譜圖:")
    print(f"  X_data (proxy spectrogram): shape={magnitude_x.shape}, mean={np.mean(magnitude_x):.2e}")
    print(f"  Y_data (LDV spectrogram): shape={magnitude_y.shape}, mean={np.mean(magnitude_y):.2e}")
    
    # 應用頻率濾波
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
    
    print(f"\n頻率濾波後的頻譜圖 (這是我們的真實數據):")
    print(f"  X_data: shape={X_data.shape}, mean={np.mean(X_data):.2e}")
    print(f"  Y_data: shape={Y_data.shape}, mean={np.mean(Y_data):.2e}")
    
    return X_data, Y_data, config, x_root, y_root


def analyze_transfer_function_estimation(X_data, Y_data, config, x_root, y_root):
    """分析傳遞函數 H 的估計過程"""
    print("\n" + "="*80)
    print("步驟 2: 分析傳遞函數 H 的估計")
    print("="*80)
    
    # 使用 DataProcessor 估計傳遞函數
    processor = DataProcessor(config)
    H, angles, folders, metadata = processor.estimate_transfer_functions(
        Path(x_root), Path(y_root)
    )
    
    print(f"傳遞函數估計結果:")
    print(f"  H shape: {H.shape} (頻率 × 方向)")
    print(f"  H stats: mean={H.mean():.4f}, max={H.max():.4f}, min={H.min():.4f}")
    print(f"  方向數: {len(angles)} 個角度")
    print(f"  角度: {angles.tolist()}")
    
    # 檢查 H 的物理意義
    print(f"\n傳遞函數 H 的物理意義:")
    print(f"  H[f, d] = 從角度 d 到接收器在頻率 f 的傳遞特性")
    print(f"  這是通過分析 X_data 和 Y_data 的統計關係估計的")
    print(f"  NOT 通過 Y = H @ X 的線性關係!")
    
    # 檢查條件數
    H_numpy = H.detach().cpu().numpy()
    condition_number = np.linalg.cond(H_numpy)
    print(f"\n傳遞函數品質:")
    print(f"  條件數: {condition_number:.2e}")
    print(f"  動態範圍: {H.max().item()/H.min().item():.2f}")
    
    return H, angles


def analyze_nmf_components(Y_data, H, config):
    """分析 NMF 組件的構建"""
    print("\n" + "="*80)
    print("步驟 3: 分析 NMF 組件構建")
    print("="*80)
    
    # 創建源字典 W (這是隨機生成的，不來自真實數據)
    n_components = 15
    F = Y_data.shape[0]
    
    W = torch.randn(F, n_components) * 0.1
    W = torch.abs(W) + 0.01
    W = W / W.sum(dim=0, keepdim=True)
    
    print(f"源信號字典 W:")
    print(f"  W shape: {W.shape} (頻率 × 源數)")
    print(f"  W stats: mean={W.mean():.4f}, max={W.max():.4f}, min={W.min():.4f}")
    print(f"  物理意義: W[:, k] = 第 k 個源信號的頻譜特徵")
    print(f"  重要: W 是隨機生成的，不代表真實源信號!")
    
    # 構建混合矩陣 A
    localizer = NMFSoundLocalizer(config)
    localizer.load_source_dictionary(W)
    localizer.load_transfer_functions(H)
    
    A = localizer.A.detach().cpu().numpy()
    print(f"\n混合矩陣 A = [diag(H₁)W, ..., diag(H_D)W]:")
    print(f"  A shape: {A.shape} (頻率 × 源×方向)")
    print(f"  A stats: mean={A.mean():.4f}, max={A.max():.4f}, min={A.min():.4f}")
    print(f"  物理意義: A[:, kd] = 源 k 從方向 d 到接收器的頻譜傳遞")
    print(f"  條件數: {np.linalg.cond(A):.2e}")
    
    return W, A, localizer


def theoretical_reconstruction_test(Y_data, A):
    """理論重建測試：如果存在完美的 X，重建品質如何？"""
    print("\n" + "="*80)
    print("步驟 4: 理論重建品質測試")  
    print("="*80)
    
    # 使用最小二乘法求解理論最優 X
    print("計算理論最優激活矩陣 X_optimal...")
    
    # 解 A @ X = Y_data，求 X
    # 使用 Moore-Penrose 偽逆
    A_tensor = torch.from_numpy(A).float()
    Y_tensor = torch.from_numpy(Y_data).float()
    
    # X_optimal = A⁺ @ Y
    A_pinv = torch.linalg.pinv(A_tensor)
    X_optimal = A_pinv @ Y_tensor
    
    # 計算理論重建
    Y_hat_optimal = A_tensor @ X_optimal
    
    X_optimal_np = X_optimal.detach().cpu().numpy()
    Y_hat_optimal_np = Y_hat_optimal.detach().cpu().numpy()
    
    print(f"理論最優解:")
    print(f"  X_optimal shape: {X_optimal_np.shape}")
    print(f"  X_optimal stats: mean={np.mean(X_optimal_np):.2e}, min={np.min(X_optimal_np):.2e}")
    print(f"  X_optimal 負值比例: {np.sum(X_optimal_np < 0) / X_optimal_np.size:.3f}")
    
    # 計算重建品質
    mse = np.mean((Y_data - Y_hat_optimal_np) ** 2)
    scale_ratio = np.mean(Y_data) / np.mean(Y_hat_optimal_np)
    correlation = np.corrcoef(Y_data.flatten(), Y_hat_optimal_np.flatten())[0, 1]
    
    print(f"\n理論最佳重建品質:")
    print(f"  MSE: {mse:.2e}")
    print(f"  規模比例: {scale_ratio:.4f}")
    print(f"  相關性: {correlation:.4f}")
    print(f"  Y_data mean: {np.mean(Y_data):.2e}")
    print(f"  Y_hat_optimal mean: {np.mean(Y_hat_optimal_np):.2e}")
    
    return X_optimal_np, Y_hat_optimal_np


def compare_with_nmf_result(Y_data, localizer, X_optimal, Y_hat_optimal):
    """比較 NMF 結果與理論最優解"""
    print("\n" + "="*80)
    print("步驟 5: NMF 結果 vs 理論最優解比較")
    print("="*80)
    
    # 運行 NMF
    print("運行 NMF 優化...")
    Y_tensor = torch.from_numpy(Y_data).float()
    X_nmf_tensor, nmf_result = localizer.factorize(Y_tensor)
    
    # 計算 NMF 重建
    Y_hat_nmf_tensor = localizer.A @ X_nmf_tensor
    
    X_nmf = X_nmf_tensor.detach().cpu().numpy()
    Y_hat_nmf = Y_hat_nmf_tensor.detach().cpu().numpy()
    
    print(f"NMF 結果:")
    print(f"  收斂: {nmf_result['converged']} ({nmf_result['n_iter']} 迭代)")
    print(f"  最終損失: {nmf_result['final_loss']:.2e}")
    print(f"  X_nmf shape: {X_nmf.shape}")
    print(f"  X_nmf 稀疏性: {1.0 - np.count_nonzero(X_nmf > 1e-10) / X_nmf.size:.3f}")
    
    # 比較激活矩陣
    x_optimal_norm = np.linalg.norm(X_optimal)
    x_nmf_norm = np.linalg.norm(X_nmf)
    x_diff_norm = np.linalg.norm(X_optimal - X_nmf)
    
    print(f"\n激活矩陣比較:")
    print(f"  ||X_optimal||: {x_optimal_norm:.2e}")
    print(f"  ||X_nmf||: {x_nmf_norm:.2e}")
    print(f"  ||X_optimal - X_nmf||: {x_diff_norm:.2e}")
    print(f"  相對誤差: {x_diff_norm / x_optimal_norm:.4f}")
    
    # 比較重建品質
    nmf_mse = np.mean((Y_data - Y_hat_nmf) ** 2)
    nmf_scale_ratio = np.mean(Y_data) / np.mean(Y_hat_nmf)
    nmf_correlation = np.corrcoef(Y_data.flatten(), Y_hat_nmf.flatten())[0, 1]
    
    optimal_mse = np.mean((Y_data - Y_hat_optimal) ** 2)
    optimal_scale_ratio = np.mean(Y_data) / np.mean(Y_hat_optimal)
    optimal_correlation = np.corrcoef(Y_data.flatten(), Y_hat_optimal.flatten())[0, 1]
    
    print(f"\n重建品質比較:")
    print(f"{'指標':<15} {'理論最優':<15} {'NMF結果':<15} {'比值':<10}")
    print("-" * 60)
    print(f"{'MSE':<15} {optimal_mse:<15.2e} {nmf_mse:<15.2e} {nmf_mse/optimal_mse:<10.2f}")
    print(f"{'規模比例':<15} {optimal_scale_ratio:<15.4f} {nmf_scale_ratio:<15.4f} {nmf_scale_ratio/optimal_scale_ratio:<10.4f}")
    print(f"{'相關性':<15} {optimal_correlation:<15.4f} {nmf_correlation:<15.4f} {nmf_correlation/optimal_correlation:<10.4f}")
    
    # IS 散度比較
    epsilon = 1e-12
    Y_safe = np.maximum(Y_data, epsilon)
    
    # 理論最優的 IS 散度
    Y_hat_opt_safe = np.maximum(Y_hat_optimal, epsilon)
    ratio_opt = Y_safe / Y_hat_opt_safe
    is_div_optimal = np.sum(ratio_opt - np.log(ratio_opt) - 1)
    
    # NMF 的 IS 散度  
    Y_hat_nmf_safe = np.maximum(Y_hat_nmf, epsilon)
    ratio_nmf = Y_safe / Y_hat_nmf_safe
    is_div_nmf = np.sum(ratio_nmf - np.log(ratio_nmf) - 1)
    
    print(f"\nIS 散度比較:")
    print(f"  理論最優: {is_div_optimal:.2e}")
    print(f"  NMF結果: {is_div_nmf:.2e}")
    print(f"  比值: {is_div_nmf / is_div_optimal:.2f}")
    
    return {
        'X_nmf': X_nmf,
        'Y_hat_nmf': Y_hat_nmf,
        'nmf_result': nmf_result,
        'comparison': {
            'mse_ratio': nmf_mse / optimal_mse,
            'scale_ratio_ratio': nmf_scale_ratio / optimal_scale_ratio,
            'correlation_ratio': nmf_correlation / optimal_correlation,
            'is_divergence_ratio': is_div_nmf / is_div_optimal,
            'x_relative_error': x_diff_norm / x_optimal_norm
        }
    }


def main():
    """主驗證流程"""
    print("NMF 概念性驗證分析")
    print("="*80)
    print("目標：理解當前 NMF 設置的物理意義和重建限制")
    print()
    
    # 步驟 1: 分析數據概念
    X_data, Y_data, config, x_root, y_root = analyze_data_concepts()
    
    # 步驟 2: 分析傳遞函數估計
    H, angles = analyze_transfer_function_estimation(X_data, Y_data, config, x_root, y_root)
    
    # 步驟 3: 分析 NMF 組件構建  
    W, A, localizer = analyze_nmf_components(Y_data, H, config)
    
    # 步驟 4: 理論重建測試
    X_optimal, Y_hat_optimal = theoretical_reconstruction_test(Y_data, A)
    
    # 步驟 5: 比較 NMF 與理論最優解
    comparison_result = compare_with_nmf_result(Y_data, localizer, X_optimal, Y_hat_optimal)
    
    # 總結
    print("\n" + "="*80)
    print("總結與解釋")
    print("="*80)
    
    print("關鍵發現:")
    print("1. X_data, Y_data = 原始音頻的頻譜表示")
    print("2. H = 從 X_data, Y_data 統計估計的傳遞函數 (不是線性關係)")
    print("3. W = 隨機生成的源字典 (不代表真實源)")
    print("4. A = diag(H) @ W 構建的混合矩陣")  
    print("5. NMF 求解: Y_data ≈ A @ X，其中 X 是激活矩陣")
    
    comp = comparison_result['comparison']
    print(f"\nNMF vs 理論最優性能:")
    print(f"  MSE 比值: {comp['mse_ratio']:.2f} (越小越好)")
    print(f"  規模比值: {comp['scale_ratio_ratio']:.4f} (越接近1越好)")
    print(f"  相關性比值: {comp['correlation_ratio']:.4f} (越接近1越好)")
    print(f"  IS散度比值: {comp['is_divergence_ratio']:.2f} (越小越好)")
    print(f"  X相對誤差: {comp['x_relative_error']:.4f} (越小越好)")
    
    if comp['scale_ratio_ratio'] > 0.8 and comp['correlation_ratio'] > 0.8:
        print("\n✅ NMF 表現接近理論最優，設置合理")
    elif comp['mse_ratio'] < 2.0:
        print("\n⚠️  NMF 表現可接受，但有改進空間")
    else:
        print("\n❌ NMF 表現明顯低於理論最優，設置可能有問題")
    
    print(f"\n保存結果到: nmf_conceptual_validation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")


if __name__ == "__main__":
    main()