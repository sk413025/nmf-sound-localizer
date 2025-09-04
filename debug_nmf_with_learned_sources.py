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
    """從真實數據中學習源特徵 W"""
    print("從真實數據學習源特徵...")
    
    # 方法1: 從 X_data 中學習源特徵 (proxy 信號包含源信息)
    print("方法1: 從 X_data (proxy) 學習源特徵")
    
    # 使用 sklearn NMF 分解 X_data = W_x @ H_x
    nmf_x = sklearn_NMF(n_components=n_components, init='nndsvd', max_iter=1000, random_state=42)
    H_x = nmf_x.fit_transform(X_data.T)  # (N, K)
    W_x = nmf_x.components_.T  # (F, K)
    
    print(f"  X_data NMF: W_x shape={W_x.shape}, reconstruction_error={nmf_x.reconstruction_err_:.2e}")
    
    # 方法2: 從 Y_data 中學習源特徵
    print("方法2: 從 Y_data (LDV) 學習源特徵")
    
    nmf_y = sklearn_NMF(n_components=n_components, init='nndsvd', max_iter=1000, random_state=42)
    H_y = nmf_y.fit_transform(Y_data.T)  # (N, K)
    W_y = nmf_y.components_.T  # (F, K)
    
    print(f"  Y_data NMF: W_y shape={W_y.shape}, reconstruction_error={nmf_y.reconstruction_err_:.2e}")
    
    # 方法3: 從 X_data 和 Y_data 的聯合分析學習
    print("方法3: 從聯合數據學習源特徵")
    
    # 使用 X_data 的主要頻譜模式，但根據 Y_data 調整權重
    combined_data = np.concatenate([X_data, Y_data], axis=1)  # (F, 2N)
    nmf_combined = sklearn_NMF(n_components=n_components, init='nndsvd', max_iter=1000, random_state=42)
    H_combined = nmf_combined.fit_transform(combined_data.T)  # (2N, K)
    W_combined = nmf_combined.components_.T  # (F, K)
    
    print(f"  聯合 NMF: W_combined shape={W_combined.shape}, reconstruction_error={nmf_combined.reconstruction_err_:.2e}")
    
    return {
        'W_from_X': torch.from_numpy(W_x).float(),
        'W_from_Y': torch.from_numpy(W_y).float(), 
        'W_combined': torch.from_numpy(W_combined).float(),
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
        if method_name == 'reconstruction_errors':
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


def save_visualization_data(Y_data, results, optimal_baseline, best_method):
    """Save Y and Y_hat data for visualization"""
    import json
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get best result Y_hat
    best_Y_hat = results[best_method]['Y_hat']
    
    # Sample data for visualization (reduce size)
    freq_step = max(1, Y_data.shape[0] // 100)  # Max 100 frequency points
    time_step = max(1, Y_data.shape[1] // 200)  # Max 200 time points
    
    Y_sampled = Y_data[::freq_step, ::time_step]
    Y_hat_sampled = best_Y_hat[::freq_step, ::time_step]
    
    # Calculate IS divergence matrix
    epsilon = 1e-12
    Y_safe = np.maximum(Y_data, epsilon)
    Y_hat_safe = np.maximum(best_Y_hat, epsilon)
    ratio = Y_safe / Y_hat_safe
    is_div_matrix = ratio - np.log(ratio) - 1
    is_div_sampled = is_div_matrix[::freq_step, ::time_step]
    
    vis_data = {
        'metadata': {
            'timestamp': timestamp,
            'best_method': best_method,
            'Y_shape': Y_data.shape,
            'Y_hat_shape': best_Y_hat.shape,
            'sampling_info': {
                'freq_step': freq_step,
                'time_step': time_step,
                'sampled_shape': Y_sampled.shape
            }
        },
        'reconstruction_quality': {
            'scale_ratio': float(results[best_method]['scale_ratio']),
            'correlation': float(results[best_method]['correlation']),
            'mse': float(results[best_method]['mse']),
            'is_divergence': float(results[best_method]['is_divergence']),
            'Y_stats': {
                'mean': float(np.mean(Y_data)),
                'std': float(np.std(Y_data)),
                'min': float(np.min(Y_data)),
                'max': float(np.max(Y_data))
            },
            'Y_hat_stats': {
                'mean': float(np.mean(best_Y_hat)),
                'std': float(np.std(best_Y_hat)),
                'min': float(np.min(best_Y_hat)),
                'max': float(np.max(best_Y_hat))
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
    
    output_file = f"learned_sources_reconstruction_{timestamp}.json"
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
    
    # 測試學習到的源特徵
    results = test_learned_sources_nmf(Y_data, H, learned_sources, config)
    
    # 比較所有方法
    best_method, best_result = compare_all_methods(Y_data, results, optimal_baseline)
    
    # 保存視覺化數據
    vis_file = save_visualization_data(Y_data, results, optimal_baseline, best_method)
    
    print(f"\n✅ 分析完成!")
    print(f"🏆 最佳方法: {best_method}")
    print(f"最佳結果相對於理論最優:")
    print(f"  相關性: {best_result['correlation'] / optimal_corr:.3f}")
    print(f"  規模匹配: {best_result['scale_ratio']:.4f} vs {optimal_scale:.4f}")
    
    return vis_file
    

if __name__ == "__main__":
    vis_file = main()
    print(f"\n🎯 Use visualize_nmf_reconstruction.py to plot: {vis_file}")