#!/usr/bin/env python3
"""
視覺化 NMF 重建結果：Y vs Y_hat 對比分析
讀取 debug_is_divergence.py 生成的 JSON 數據進行繪圖分析
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse

def load_reconstruction_data(json_file):
    """載入重建數據 JSON 檔案"""
    print(f"載入視覺化數據：{json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"數據時間戳：{data['metadata']['timestamp']}")
    print(f"參數設置：λ_group={data['metadata']['lambda_group']}, γ_sparse={data['metadata']['gamma_sparse']}")
    print(f"數據點數量：{data['comparison_data']['data_points']}")
    print(f"規模比例：{data['reconstruction_quality']['scale_ratio']:.4f}")
    print(f"X 稀疏性：{data['reconstruction_quality']['x_sparsity']:.3f}")
    
    return data

def create_y_vs_yhat_comparison(data, save_plots=True):
    """創建 Y vs Y_hat 對比曲線圖"""
    
    # 設置中文字體
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 創建多子圖布局
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'NMF 重建品質分析 (λ_group={data["metadata"]["lambda_group"]}, γ_sparse={data["metadata"]["gamma_sparse"]})', 
                 fontsize=16, fontweight='bold')
    
    # 獲取數據
    Y_values = np.array(data['comparison_data']['Y_values'])
    Y_hat_values = np.array(data['comparison_data']['Y_hat_values'])
    is_div_values = np.array(data['comparison_data']['is_divergence_values'])
    
    diagonal_Y = np.array(data['curve_comparison']['diagonal_Y'])
    diagonal_Y_hat = np.array(data['curve_comparison']['diagonal_Y_hat'])
    diagonal_is_div = np.array(data['curve_comparison']['diagonal_is_div'])
    
    # 子圖 1: Y vs Y_hat 散點圖
    ax1 = axes[0, 0]
    # 取樣點以避免過密
    sample_indices = np.random.choice(len(Y_values), min(5000, len(Y_values)), replace=False)
    Y_sample = Y_values[sample_indices]
    Y_hat_sample = Y_hat_values[sample_indices]
    
    scatter = ax1.scatter(Y_sample, Y_hat_sample, alpha=0.6, s=1, c='blue')
    # 添加理想重建線 (y=x)
    max_val = max(np.max(Y_sample), np.max(Y_hat_sample))
    ax1.plot([0, max_val], [0, max_val], 'r--', alpha=0.8, linewidth=2, label='理想重建 (Y=Y_hat)')
    ax1.set_xlabel('Y (原始值)')
    ax1.set_ylabel('Y_hat (重建值)')
    ax1.set_title('Y vs Y_hat 散點圖')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子圖 2: 對角線切片曲線對比
    ax2 = axes[0, 1]
    x_axis = np.arange(len(diagonal_Y))
    ax2.plot(x_axis, diagonal_Y, 'b-', label='Y (原始)', linewidth=2, alpha=0.8)
    ax2.plot(x_axis, diagonal_Y_hat, 'r-', label='Y_hat (重建)', linewidth=2, alpha=0.8)
    ax2.set_xlabel('頻率-時間對角線索引')
    ax2.set_ylabel('振幅值')
    ax2.set_title('對角線切片：Y vs Y_hat')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子圖 3: 數值統計比較
    ax3 = axes[1, 0]
    stats_labels = ['平均值', '標準差', '最小值', '最大值']
    y_stats = [data['reconstruction_quality']['Y_stats']['mean'],
               data['reconstruction_quality']['Y_stats']['std'],
               data['reconstruction_quality']['Y_stats']['min'],
               data['reconstruction_quality']['Y_stats']['max']]
    y_hat_stats = [data['reconstruction_quality']['Y_hat_stats']['mean'],
                   data['reconstruction_quality']['Y_hat_stats']['std'],
                   data['reconstruction_quality']['Y_hat_stats']['min'],
                   data['reconstruction_quality']['Y_hat_stats']['max']]
    
    x_pos = np.arange(len(stats_labels))
    width = 0.35
    
    ax3.bar(x_pos - width/2, y_stats, width, label='Y (原始)', alpha=0.8, color='blue')
    ax3.bar(x_pos + width/2, y_hat_stats, width, label='Y_hat (重建)', alpha=0.8, color='red')
    ax3.set_xlabel('統計指標')
    ax3.set_ylabel('數值')
    ax3.set_title('統計數值對比')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(stats_labels)
    ax3.legend()
    ax3.set_yscale('log')  # 使用對數尺度
    ax3.grid(True, alpha=0.3)
    
    # 子圖 4: IS Divergence 熱力圖 (對角線切片)
    ax4 = axes[1, 1]
    ax4.plot(x_axis, diagonal_is_div, 'g-', linewidth=2, alpha=0.8)
    ax4.set_xlabel('頻率-時間對角線索引')
    ax4.set_ylabel('IS Divergence')
    ax4.set_title('IS Divergence 分布 (對角線切片)')
    ax4.grid(True, alpha=0.3)
    
    # 添加文字框顯示關鍵指標
    textstr = f'''關鍵指標:
規模比例: {data["reconstruction_quality"]["scale_ratio"]:.4f}
X 稀疏性: {data["reconstruction_quality"]["x_sparsity"]:.1%}
IS Divergence 總計: {data["reconstruction_quality"]["is_divergence_stats"]["total"]:.2e}
IS Divergence 平均: {data["reconstruction_quality"]["is_divergence_stats"]["mean"]:.2f}'''
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    fig.text(0.02, 0.02, textstr, fontsize=10, verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    
    if save_plots:
        timestamp = data['metadata']['timestamp']
        filename = f'nmf_reconstruction_analysis_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"圖表已保存：{filename}")
    
    # plt.show()  # 移除互動式顯示
    
    return fig

def create_iteration_progression_plot(data, save_plots=True):
    """創建迭代過程分析圖"""
    
    if not data['iteration_progression']:
        print("沒有迭代過程數據可供視覺化")
        return None
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('NMF 優化迭代過程分析', fontsize=16, fontweight='bold')
    
    # 提取迭代數據
    iterations = [item['iteration'] for item in data['iteration_progression']]
    losses = [item['loss'] for item in data['iteration_progression']]
    x_sparsities = [item['X_sparsity'] for item in data['iteration_progression']]
    x_means = [item['X_mean'] for item in data['iteration_progression']]
    y_hat_means = [item['Y_hat_mean'] for item in data['iteration_progression']]
    
    # 子圖 1: 損失函數收斂
    axes[0, 0].plot(iterations, losses, 'b-', marker='o', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('迭代次數')
    axes[0, 0].set_ylabel('損失函數值')
    axes[0, 0].set_title('損失函數收斂過程')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 子圖 2: X 稀疏性變化
    axes[0, 1].plot(iterations, x_sparsities, 'r-', marker='s', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('迭代次數')
    axes[0, 1].set_ylabel('X 稀疏性')
    axes[0, 1].set_title('X 因子稀疏性變化')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子圖 3: X 因子平均值增長
    axes[1, 0].plot(iterations, x_means, 'g-', marker='^', linewidth=2, markersize=4)
    axes[1, 0].set_xlabel('迭代次數')
    axes[1, 0].set_ylabel('X 因子平均值')
    axes[1, 0].set_title('X 因子平均值增長')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # 子圖 4: Y_hat 重建值增長
    axes[1, 1].plot(iterations, y_hat_means, 'm-', marker='d', linewidth=2, markersize=4)
    axes[1, 1].set_xlabel('迭代次數')
    axes[1, 1].set_ylabel('Y_hat 平均值')
    axes[1, 1].set_title('重建值 Y_hat 增長')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_plots:
        timestamp = data['metadata']['timestamp']
        filename = f'nmf_iteration_progression_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"迭代過程圖表已保存：{filename}")
    
    # plt.show()  # 移除互動式顯示
    
    return fig

def print_detailed_analysis(data):
    """打印詳細的數值分析結果"""
    
    print("\n" + "="*60)
    print("NMF 重建品質詳細分析")
    print("="*60)
    
    # 基本信息
    print(f"時間戳: {data['metadata']['timestamp']}")
    print(f"參數設置: λ_group={data['metadata']['lambda_group']}, γ_sparse={data['metadata']['gamma_sparse']}")
    print(f"原始數據形狀: {data['metadata']['Y_shape']}")
    print(f"重建數據形狀: {data['metadata']['Y_hat_shape']}")
    
    # 重建品質指標
    print(f"\n📊 重建品質指標:")
    print(f"規模比例 (Scale Ratio): {data['reconstruction_quality']['scale_ratio']:.6f}")
    print(f"X 稀疏性: {data['reconstruction_quality']['x_sparsity']:.1%}")
    
    # Y 和 Y_hat 統計對比
    print(f"\n📈 Y (原始) vs Y_hat (重建) 統計對比:")
    y_stats = data['reconstruction_quality']['Y_stats']
    y_hat_stats = data['reconstruction_quality']['Y_hat_stats']
    
    print(f"平均值: {y_stats['mean']:.2e} vs {y_hat_stats['mean']:.2e} (比例: {y_hat_stats['mean']/y_stats['mean']:.4f})")
    print(f"標準差: {y_stats['std']:.2e} vs {y_hat_stats['std']:.2e} (比例: {y_hat_stats['std']/y_stats['std']:.4f})")
    print(f"最大值: {y_stats['max']:.2e} vs {y_hat_stats['max']:.2e} (比例: {y_hat_stats['max']/y_stats['max']:.4f})")
    print(f"最小值: {y_stats['min']:.2e} vs {y_hat_stats['min']:.2e}")
    
    # IS Divergence 分析
    print(f"\n🔥 IS Divergence 分析:")
    is_div_stats = data['reconstruction_quality']['is_divergence_stats']
    print(f"總計: {is_div_stats['total']:.2e}")
    print(f"平均: {is_div_stats['mean']:.2f}")
    print(f"標準差: {is_div_stats['std']:.2f}")
    print(f"最大值: {is_div_stats['max']:.2f}")
    print(f"最小值: {is_div_stats['min']:.2f}")
    
    # 數據抽樣信息
    print(f"\n📋 視覺化採樣信息:")
    sampling_info = data['metadata']['sampling_info']
    print(f"頻率採樣步長: {sampling_info['freq_step']}")
    print(f"時間採樣步長: {sampling_info['time_step']}")
    print(f"採樣後形狀: {sampling_info['sampled_shape']}")
    print(f"比較數據點數: {data['comparison_data']['data_points']}")
    print(f"對角線數據點數: {data['curve_comparison']['diagonal_length']}")
    
    # 迭代過程摘要
    if data['iteration_progression']:
        print(f"\n⚙️ 優化迭代過程摘要:")
        print(f"追蹤的迭代點數: {len(data['iteration_progression'])}")
        first_iter = data['iteration_progression'][0]
        last_iter = data['iteration_progression'][-1]
        print(f"初始損失: {first_iter['loss']:.2e}")
        print(f"最終損失: {last_iter['loss']:.2e}")
        print(f"損失降低: {(first_iter['loss'] - last_iter['loss'])/first_iter['loss']*100:.1f}%")
        print(f"初始 X 稀疏性: {first_iter['X_sparsity']:.3f}")
        print(f"最終 X 稀疏性: {last_iter['X_sparsity']:.3f}")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='視覺化 NMF 重建結果')
    parser.add_argument('--data_file', type=str, 
                      help='JSON 數據檔案路徑 (預設使用最新的檔案)')
    parser.add_argument('--no_save', action='store_true',
                      help='不儲存圖表檔案')
    
    args = parser.parse_args()
    
    # 尋找數據檔案
    if args.data_file:
        data_file = args.data_file
    else:
        # 尋找最新的 JSON 檔案
        json_files = list(Path('.').glob('nmf_reconstruction_data_*.json'))
        if not json_files:
            print("找不到 nmf_reconstruction_data_*.json 檔案")
            print("請先運行 debug_is_divergence.py 生成數據")
            return
        
        data_file = max(json_files, key=lambda p: p.stat().st_mtime)
        print(f"使用最新的數據檔案: {data_file}")
    
    # 載入和分析數據
    try:
        data = load_reconstruction_data(data_file)
        
        # 打印詳細分析
        print_detailed_analysis(data)
        
        # 創建視覺化圖表
        print(f"\n🎨 正在生成視覺化圖表...")
        fig1 = create_y_vs_yhat_comparison(data, save_plots=not args.no_save)
        fig2 = create_iteration_progression_plot(data, save_plots=not args.no_save)
        
        print(f"\n✅ 視覺化完成！")
        
    except Exception as e:
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()