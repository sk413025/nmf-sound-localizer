#!/usr/bin/env python3
"""
临时脚本：处理 LDV-data 的所有 37 个角度
基于原始 white_noise_to_nmf_converter_no_edge.py，但扩展了 angle_mapping
"""

import sys
import os

# 添加原始脚本路径
sys.path.insert(0, '/Users/sbplab/jiawei/datasets')

# 导入原始转换器
from white_noise_to_nmf_converter_no_edge import WhiteNoiseNMFConverter
import argparse

# 创建扩展的转换器类
class LDVDataConverter(WhiteNoiseNMFConverter):
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)
        
        # 扩展 angle_mapping 支持所有 37 个角度 (0-180 每 5°)
        self.angle_mapping = {}
        for angle in range(0, 181, 5):
            angle_str = f"{angle:03d}"  # 格式化为 "000", "005", "010", ...
            self.angle_mapping[angle_str] = angle
        
        print(f"✅ 已扩展角度映射: 支持 {len(self.angle_mapping)} 个角度")

def main():
    parser = argparse.ArgumentParser(description='LDV-data 全角度转换器')
    parser.add_argument('--dataset_path', required=True,
                       help='LDV-data 数据集路径')
    parser.add_argument('--material', choices=['box', 'IrregularBox'], required=True,
                       help='要处理的材料类型')
    parser.add_argument('--output_dir', default='./nmf_data',
                       help='输出目录')
    
    args = parser.parse_args()
    
    try:
        # 使用扩展的转换器
        converter = LDVDataConverter(args.dataset_path)
        
        # 载入数据集信息
        converter.load_dataset_info()
        
        # 提取标准间隔
        converter.extract_intervals_from_standard()
        
        # 创建 NMF 数据集
        output_path = converter.create_nmf_dataset(args.material, args.output_dir)
        
        print("\n✅ 转换完成!")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
