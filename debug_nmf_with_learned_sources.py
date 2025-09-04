#!/usr/bin/env python3
"""
修正版 NMF：從真實數據中學習源特徵 W，而不是使用隨機 W
測試物理上合理的 NMF 設置是否能提供更好的重建
"""

import numpy as np
import torch
import os
import sys
import datetime
from pathlib import Path
from sklearn.decomposition import NMF as sklearn_NMF

# Add project path
sys.path.append('/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/development-workspace')

from nmf_localizer.core.data_processor import DataProcessor
from nmf_localizer.core.localizer import NMFSoundLocalizer
from nmf_localizer.config import NMFConfig
from nmf_localizer.utils.audio_utils import AudioProcessor


def learn_source_features_from_data(X_data, Y_data, n_components=15):
    """從真實數據中學習源特徵 W，並回傳係數 U_from_X 等資訊

    命名統一：
    - W_from_X: 字典/基底 (F×K) = sklearn components_.T
    - U_from_X: 係數/activations (K×N) = sklearn fit_transform(X.T).T
    - 驗證關係：X_data ≈ W_from_X @ U_from_X
    """
    print("從真實數據學習源特徵...")
    
    # 方法1: 從 X_data 中學習源特徵 (proxy 信號包含源信息)
    print("方法1: 從 X_data (proxy) 學習源特徵")
    
    # 使用 sklearn NMF 分解 X_data ≈ W_x @ H_x.T
    nmf_x = sklearn_NMF(n_components=n_components, init='nndsvd', max_iter=1000, random_state=42)
    H_x_time = nmf_x.fit_transform(X_data.T)  # (N, K)
    W_x = nmf_x.components_.T               # (F, K)
    U_from_X = torch.from_numpy(H_x_time.T).float()  # (K, N)
    
    print(f"  X_data NMF: W_x (dictionary) shape={W_x.shape}, reconstruction_error={nmf_x.reconstruction_err_:.2e}")
    print(f"  X_data NMF: U_from_X (coefficients) shape={U_from_X.shape}")
    
    # 方法2: 從 Y_data 中學習源特徵
    print("方法2: 從 Y_data (LDV) 學習源特徵")
    
    nmf_y = sklearn_NMF(n_components=n_components, init='nndsvd', max_iter=1000, random_state=42)
    H_y_time = nmf_y.fit_transform(Y_data.T)  # (N, K)
    W_y = nmf_y.components_.T               # (F, K)
    
    print(f"  Y_data NMF: W_y shape={W_y.shape}, reconstruction_error={nmf_y.reconstruction_err_:.2e}")
    
    # 方法3: 從 X_data 和 Y_data 的聯合分析學習
    print("方法3: 從聯合數據學習源特徵")
    
    # 使用 X_data 的主要頻譜模式，但根據 Y_data 調整權重
    combined_data = np.concatenate([X_data, Y_data], axis=1)  # (F, 2N)
    nmf_combined = sklearn_NMF(n_components=n_components, init='nndsvd', max_iter=1000, random_state=42)
    H_combined_time = nmf_combined.fit_transform(combined_data.T)  # (2N, K)
    W_combined = nmf_combined.components_.T                      # (F, K)
    
    print(f"  聯合 NMF: W_combined shape={W_combined.shape}, reconstruction_error={nmf_combined.reconstruction_err_:.2e}")
    
    return {
        'W_from_X': torch.from_numpy(W_x).float(),
        'W_from_Y': torch.from_numpy(W_y).float(), 
        'W_combined': torch.from_numpy(W_combined).float(),
        'coefficients': {
            # 僅提供參考用，不作為字典 W 用於後續 localizer
            'U_from_X': U_from_X,  # (K, N)
            # 可視需要再加：'U_from_Y': torch.from_numpy(H_y_time.T).float(),
            # 以及 'U_combined': torch.from_numpy(H_combined_time.T).float(),
        },
        'reconstruction_errors': {
            'X_data': nmf_x.reconstruction_err_,
            'Y_data': nmf_y.reconstruction_err_,
            'combined': nmf_combined.reconstruction_err_
        }
    }


def test_learned_sources_nmf(Y_data, H, learned_sources, config):
    """測試使用學習到的源特徵進行 NMF"""
    print("\n" + "="*80)
    print("測試學習到的源特徵")
    print("="*80)
    
    results = {}
    
    for method_name, W_learned in learned_sources.items():
        # 跳過非字典型資料（例如係數、重建誤差等）
        if method_name in ('reconstruction_errors', 'coefficients'):
            continue
        # 僅處理形狀為 (F, K) 的字典矩陣
        if not isinstance(W_learned, torch.Tensor) or W_learned.dim() != 2 or W_learned.shape[0] != H.shape[0]:
            continue
            
        print(f"\n測試 {method_name}...")
        
        # 創建 NMF localizer
        localizer = NMFSoundLocalizer(config)
        localizer.load_source_dictionary(W_learned)
        localizer.load_transfer_functions(H)
        
        # 運行 NMF
        Y_tensor = torch.from_numpy(Y_data).float()
        X_nmf_tensor, nmf_result = localizer.factorize(Y_tensor)
        
        # 計算重建
        Y_hat_nmf_tensor = localizer.A @ X_nmf_tensor
        Y_hat_nmf = Y_hat_nmf_tensor.detach().cpu().numpy()
        X_nmf = X_nmf_tensor.detach().cpu().numpy()
        
        # 計算品質指標
        mse = np.mean((Y_data - Y_hat_nmf) ** 2)
        scale_ratio = np.mean(Y_data) / np.mean(Y_hat_nmf)
        correlation = np.corrcoef(Y_data.flatten(), Y_hat_nmf.flatten())[0, 1]
        sparsity = 1.0 - np.count_nonzero(X_nmf > 1e-10) / X_nmf.size
        
        # IS 散度
        epsilon = 1e-12
        Y_safe = np.maximum(Y_data, epsilon)
        Y_hat_safe = np.maximum(Y_hat_nmf, epsilon)
        ratio = Y_safe / Y_hat_safe
        is_divergence = np.sum(ratio - np.log(ratio) - 1)
        
        results[method_name] = {
            'converged': nmf_result['converged'],
            'n_iter': nmf_result['n_iter'],
            'final_loss': nmf_result['final_loss'],
            'mse': mse,
            'scale_ratio': scale_ratio,
            'correlation': correlation,
            'sparsity': sparsity,
            'is_divergence': is_divergence,
            'Y_hat': Y_hat_nmf,
            'X_nmf': X_nmf
        }
        
        print(f"  收斂: {nmf_result['converged']} ({nmf_result['n_iter']} 迭代)")
        print(f"  MSE: {mse:.2e}")
        print(f"  規模比例: {scale_ratio:.4f}")
        print(f"  相關性: {correlation:.4f}")
        print(f"  稀疏性: {sparsity:.3f}")
        print(f"  IS 散度: {is_divergence:.2e}")
    
    return results


def single_direction_scan_using_U(Y_data, H, W_from_X, U_from_X):
    """使用從 X 學得的係數 U_from_X 進行單方向掃描以估計最佳方向。

    公式：對每個方向 d，B_d = diag(H[:, d]) @ W_from_X，Y_hat_d = B_d @ U_from_X。
    回傳最佳方向與各向度的指標。
    """
    print("\n" + "="*80)
    print("單方向掃描（使用 U_from_X 直接重建 Y 並估計方向）")
    print("="*80)

    # Ensure tensors
    if not isinstance(W_from_X, torch.Tensor):
        W_from_X = torch.from_numpy(W_from_X).float()
    if not isinstance(U_from_X, torch.Tensor):
        U_from_X = torch.from_numpy(U_from_X).float()

    Y_np = Y_data  # (F, N) numpy
    F, D = H.shape

    metrics = []
    best = None

    epsilon = 1e-12
    Y_safe = np.maximum(Y_np, epsilon)

    for d in range(D):
        H_d_diag = torch.diag(H[:, d])
        B_d = H_d_diag @ W_from_X  # (F, K)
        Y_hat_d = (B_d @ U_from_X).detach().cpu().numpy()  # (F, N)

        # Quality metrics
        mse = float(np.mean((Y_np - Y_hat_d) ** 2))
        scale_ratio = float(np.mean(Y_np) / max(np.mean(Y_hat_d), epsilon))
        corr = float(np.corrcoef(Y_np.flatten(), Y_hat_d.flatten())[0, 1])
        Y_hat_safe = np.maximum(Y_hat_d, epsilon)
        ratio = Y_safe / Y_hat_safe
        is_div = float(np.sum(ratio - np.log(ratio) - 1))

        metrics.append({
            'direction_index': d,
            'mse': mse,
            'scale_ratio': scale_ratio,
            'correlation': corr,
            'is_divergence': is_div,
        })

        if best is None or mse < best['mse']:
            best = {
                'direction_index': d,
                'mse': mse,
                'scale_ratio': scale_ratio,
                'correlation': corr,
                'is_divergence': is_div,
                'Y_hat': Y_hat_d,
            }

    # Print summary
    print(f"掃描完成：共 {D} 個方向。")
    print(f"最佳方向 d* = {best['direction_index']}")
    print(f"  MSE: {best['mse']:.2e}")
    print(f"  規模比例: {best['scale_ratio']:.4f}")
    print(f"  相關性: {best['correlation']:.4f}")
    print(f"  IS 散度: {best['is_divergence']:.2e}")

    return best, metrics


def compare_all_methods(Y_data, results, optimal_baseline):
    """比較所有方法與理論最優基線"""
    print("\n" + "="*80)
    print("所有方法性能比較")
    print("="*80)
    
    print(f"{'方法':<20} {'MSE比值':<10} {'規模比值':<10} {'相關性比值':<12} {'IS散度比值':<12} {'稀疏性':<8}")
    print("-" * 85)
    
    # 理論最優基線
    opt_mse = optimal_baseline['mse']
    opt_scale = optimal_baseline['scale_ratio']  
    opt_corr = optimal_baseline['correlation']
    opt_is_div = optimal_baseline['is_divergence']
    
    print(f"{'理論最優 (基線)':<20} {'1.00':<10} {'1.00':<10} {'1.00':<12} {'1.00':<12} {'N/A':<8}")
    
    best_method = None
    best_score = float('inf')
    
    for method, result in results.items():
        mse_ratio = result['mse'] / opt_mse
        scale_ratio_ratio = abs(result['scale_ratio'] - 1.0) / abs(opt_scale - 1.0) if abs(opt_scale - 1.0) > 1e-10 else 1.0
        corr_ratio = result['correlation'] / opt_corr if opt_corr > 1e-10 else 1.0
        is_div_ratio = result['is_divergence'] / opt_is_div if opt_is_div > 1e-10 else result['is_divergence'] / 1e6
        
        # 計算綜合評分 (越小越好)
        score = mse_ratio + abs(scale_ratio_ratio - 1.0) + (1 - corr_ratio) + is_div_ratio / 1e6
        
        if score < best_score:
            best_score = score
            best_method = method
        
        print(f"{method:<20} {mse_ratio:<10.2f} {scale_ratio_ratio:<10.4f} {corr_ratio:<12.4f} {is_div_ratio:<12.2e} {result['sparsity']:<8.3f}")
    
    print(f"\n🏆 最佳方法: {best_method} (綜合評分: {best_score:.4f})")
    return best_method, results[best_method]


def save_visualization_data(Y_data, results, optimal_baseline, method_name):
    """Save Y and Y_hat data for visualization"""
    import json
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get specified method Y_hat
    method_Y_hat = results[method_name]['Y_hat']
    
    # Sample data for visualization (reduce size)
    freq_step = max(1, Y_data.shape[0] // 100)  # Max 100 frequency points
    time_step = max(1, Y_data.shape[1] // 200)  # Max 200 time points
    
    Y_sampled = Y_data[::freq_step, ::time_step]
    Y_hat_sampled = method_Y_hat[::freq_step, ::time_step]
    
    # Calculate IS divergence matrix
    epsilon = 1e-12
    Y_safe = np.maximum(Y_data, epsilon)
    Y_hat_safe = np.maximum(method_Y_hat, epsilon)
    ratio = Y_safe / Y_hat_safe
    is_div_matrix = ratio - np.log(ratio) - 1
    is_div_sampled = is_div_matrix[::freq_step, ::time_step]
    
    vis_data = {
        'metadata': {
            'timestamp': timestamp,
            'best_method': method_name,
            'Y_shape': Y_data.shape,
            'Y_hat_shape': method_Y_hat.shape,
            'sampling_info': {
                'freq_step': freq_step,
                'time_step': time_step,
                'sampled_shape': Y_sampled.shape
            }
        },
        'reconstruction_quality': {
            'scale_ratio': float(results[method_name]['scale_ratio']),
            'correlation': float(results[method_name]['correlation']),
            'mse': float(results[method_name]['mse']),
            'is_divergence': float(results[method_name]['is_divergence']),
            'Y_stats': {
                'mean': float(np.mean(Y_data)),
                'std': float(np.std(Y_data)),
                'min': float(np.min(Y_data)),
                'max': float(np.max(Y_data))
            },
            'Y_hat_stats': {
                'mean': float(np.mean(method_Y_hat)),
                'std': float(np.std(method_Y_hat)),
                'min': float(np.min(method_Y_hat)),
                'max': float(np.max(method_Y_hat))
            }
        },
        'comparison_data': {
            'Y_values': [float(x) for x in Y_sampled.flatten()],
            'Y_hat_values': [float(x) for x in Y_hat_sampled.flatten()],
            'is_divergence_values': [float(x) for x in is_div_sampled.flatten()],
            'data_points': int(Y_sampled.size)
        },
        'curve_comparison': {
            'diagonal_Y': [float(x) for x in np.diag(Y_sampled)],
            'diagonal_Y_hat': [float(x) for x in np.diag(Y_hat_sampled)],
            'diagonal_is_div': [float(x) for x in np.diag(is_div_sampled)],
            'diagonal_length': int(len(np.diag(Y_sampled))),
            'time_slice_Y': [float(x) for x in Y_sampled[:, Y_sampled.shape[1]//2]],
            'time_slice_Y_hat': [float(x) for x in Y_hat_sampled[:, Y_hat_sampled.shape[1]//2]],
            'freq_slice_Y': [float(x) for x in Y_sampled[Y_sampled.shape[0]//2, :]],
            'freq_slice_Y_hat': [float(x) for x in Y_hat_sampled[Y_hat_sampled.shape[0]//2, :]]
        }
    }
    
    output_file = f"learned_sources_{method_name}_reconstruction_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(vis_data, f, indent=2)
    
    print(f"\n💾 Visualization data saved: {output_file}")
    return output_file


def main():
    """主函數"""
    print("修正版 NMF: 使用從真實數據學習的源特徵")
    print("="*80)
    
    # 載入數據 (重用驗證腳本的邏輯)
    config = NMFConfig(
        sample_rate=16000, n_fft=2048, hop_length=512,
        freq_min=500.0, freq_max=3000.0, n_files_per_angle=1,
        max_iter=50, beta=0, lambda_group=0.1, gamma_sparse=0.01, tolerance=1e-6
    )
    
    x_root = "/Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original/white_noise_original_data_no_edge_sync_vad"
    y_root = "/Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original/white_noise_box_data_no_edge_sync_vad"
    test_angle = "angle_90"
    
    # 載入和處理數據
    x_dir = os.path.join(x_root, test_angle)
    y_dir = os.path.join(y_root, test_angle)
    
    x_files = [f for f in os.listdir(x_dir) if f.endswith('.npy')]
    y_files = [f for f in os.listdir(y_dir) if f.endswith('.npy')]
    
    x_audio = np.load(os.path.join(x_dir, x_files[0]))
    y_audio = np.load(os.path.join(y_dir, y_files[0]))
    
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
    
    # 獲取傳遞函數
    processor = DataProcessor(config)
    H, _, _, _ = processor.estimate_transfer_functions(Path(x_root), Path(y_root))
    
    print(f"傳遞函數: H={H.shape}")
    
    # 計算理論最優基線 (重用驗證腳本邏輯)
    n_components = 15
    F = Y_data.shape[0]
    
    W_random = torch.randn(F, n_components) * 0.1
    W_random = torch.abs(W_random) + 0.01
    W_random = W_random / W_random.sum(dim=0, keepdim=True)
    
    localizer_baseline = NMFSoundLocalizer(config)
    localizer_baseline.load_source_dictionary(W_random)
    localizer_baseline.load_transfer_functions(H)
    
    A_baseline = localizer_baseline.A.detach().cpu().numpy()
    A_tensor = torch.from_numpy(A_baseline).float()
    Y_tensor = torch.from_numpy(Y_data).float()
    A_pinv = torch.linalg.pinv(A_tensor)
    X_optimal = A_pinv @ Y_tensor
    Y_hat_optimal = A_tensor @ X_optimal
    
    Y_hat_optimal_np = Y_hat_optimal.detach().cpu().numpy()
    
    optimal_mse = np.mean((Y_data - Y_hat_optimal_np) ** 2)
    optimal_scale = np.mean(Y_data) / np.mean(Y_hat_optimal_np)
    optimal_corr = np.corrcoef(Y_data.flatten(), Y_hat_optimal_np.flatten())[0, 1]
    
    epsilon = 1e-12
    Y_safe = np.maximum(Y_data, epsilon)
    Y_hat_opt_safe = np.maximum(Y_hat_optimal_np, epsilon)
    ratio_opt = Y_safe / Y_hat_opt_safe
    optimal_is_div = np.sum(ratio_opt - np.log(ratio_opt) - 1)
    
    optimal_baseline = {
        'mse': optimal_mse,
        'scale_ratio': optimal_scale,
        'correlation': optimal_corr,
        'is_divergence': optimal_is_div
    }
    
    print(f"理論最優基線: MSE={optimal_mse:.2e}, Scale={optimal_scale:.4f}, Corr={optimal_corr:.4f}")
    
    # 從數據學習源特徵
    learned_sources = learn_source_features_from_data(X_data, Y_data, n_components)

    # 命名統一檢查：X_data ≈ W_from_X @ U_from_X
    if 'coefficients' in learned_sources and 'U_from_X' in learned_sources['coefficients']:
        W_from_X = learned_sources['W_from_X']            # (F, K) torch
        U_from_X = learned_sources['coefficients']['U_from_X']  # (K, N) torch
        X_recon = (W_from_X @ U_from_X).detach().cpu().numpy()
        nmf_x_recon_mse = np.mean((X_data - X_recon) ** 2)
        print(f"命名檢查：X_data ≈ W_from_X @ U_from_X → MSE={nmf_x_recon_mse:.2e}")

        # 單方向掃描並估計最佳方向（使用 U_from_X 直接重建 Y）
        best_scan, scan_metrics = single_direction_scan_using_U(Y_data, H, W_from_X, U_from_X)
    else:
        best_scan, scan_metrics = None, None
    
    # 測試學習到的源特徵
    results = test_learned_sources_nmf(Y_data, H, learned_sources, config)
    
    # 比較所有方法
    best_method, best_result = compare_all_methods(Y_data, results, optimal_baseline)

    # 保存視覺化數據 - 特別保存 W_from_X 方法用於比較
    vis_file = save_visualization_data(Y_data, results, optimal_baseline, "W_from_X")

    # 若有單方向掃描結果，也保存其視覺化數據（命名為 scan_U_best_d）
    if best_scan is not None:
        # 構造最小結果結構以複用可視化函式
        results_for_scan = {
            'scan_U_best_d': {
                'mse': best_scan['mse'],
                'scale_ratio': best_scan['scale_ratio'],
                'correlation': best_scan['correlation'],
                'is_divergence': best_scan['is_divergence'],
                'sparsity': float('nan'),
                'Y_hat': best_scan['Y_hat'],
                'X_nmf': None,
                'converged': True,
                'n_iter': 0,
                'final_loss': float('nan'),
            }
        }
        vis_file_scan = save_visualization_data(Y_data, results_for_scan, optimal_baseline, 'scan_U_best_d')
        print(f"附加：單方向掃描視覺化檔案：{vis_file_scan}")
    
    print(f"\n✅ 分析完成!")
    print(f"🏆 最佳方法: {best_method}")
    print(f"最佳結果相對於理論最優:")
    print(f"  相關性: {best_result['correlation'] / optimal_corr:.3f}")
    print(f"  規模匹配: {best_result['scale_ratio']:.4f} vs {optimal_scale:.4f}")
    
    return vis_file
    

if __name__ == "__main__":
    vis_file = main()
    print(f"\n🎯 Use visualize_nmf_reconstruction.py to plot: {vis_file}")
