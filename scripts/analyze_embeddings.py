#!/usr/bin/env python
"""Detailed analysis of token and position embeddings in the Transformer pipeline."""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List

# Import project modules
from doa_rl.model.transformer import TransformerPolicy
from doa_rl.text import Vocab
from doa_rl.features import PatchTokenizer


def analyze_embedding_layers():
    """Analyze the embedding layers initialization and structure."""

    print("=" * 80)
    print("EMBEDDING LAYERS: 初始化與結構分析")
    print("=" * 80)

    # Model parameters
    vocab_size = 129  # 126 patches + 3 special tokens
    max_len = 512     # Maximum sequence length
    d_model = 256     # Embedding dimension

    print(f"\n1. EMBEDDING 層的創建 (在 __init__ 中):")
    print(f"   當 TransformerPolicy 被實例化時，創建兩個 Embedding 層：")
    print()

    # 展示實際的初始化代碼
    print("   ```python")
    print("   class TransformerPolicy(nn.Module):")
    print("       def __init__(self, vocab_size, n_dirs, d_model=256, ...):")
    print("           super().__init__()")
    print(f"           # Token Embedding 層")
    print(f"           self.tok_embed = nn.Embedding(vocab_size={vocab_size}, d_model={d_model})")
    print(f"           # Position Embedding 層")
    print(f"           self.pos_embed = nn.Embedding(max_len={max_len}, d_model={d_model})")
    print("   ```")

    # 實際創建 embedding 層來分析
    tok_embed = nn.Embedding(vocab_size, d_model)
    pos_embed = nn.Embedding(max_len, d_model)

    print(f"\n2. EMBEDDING 層的內部結構:")
    print(f"\n   Token Embedding (self.tok_embed):")
    print(f"   - 類型: nn.Embedding (PyTorch 的可學習查找表)")
    print(f"   - Weight 矩陣形狀: {tok_embed.weight.shape} = (vocab_size, d_model)")
    print(f"   - 參數數量: {tok_embed.weight.numel():,} 個可學習參數")
    print(f"   - 初始化: 默認使用 N(0, 1) 正態分布隨機初始化")
    print(f"   - 每個 token ID 對應一個 {d_model} 維的向量")

    print(f"\n   Position Embedding (self.pos_embed):")
    print(f"   - 類型: nn.Embedding (可學習的位置編碼)")
    print(f"   - Weight 矩陣形狀: {pos_embed.weight.shape} = (max_len, d_model)")
    print(f"   - 參數數量: {pos_embed.weight.numel():,} 個可學習參數")
    print(f"   - 初始化: 默認使用 N(0, 1) 正態分布隨機初始化")
    print(f"   - 每個位置 (0-{max_len-1}) 對應一個 {d_model} 維的向量")

    print(f"\n3. EMBEDDING 矩陣的物理意義:")
    print(f"\n   Token Embedding 矩陣 (129 × 256):")
    print(f"   行 0: [PAD] token 的 256 維表示")
    print(f"   行 1: [CLS] token 的 256 維表示")
    print(f"   行 2: [SEP] token 的 256 維表示")
    print(f"   行 3: <P_0_0_0> (最低頻、最早時間、無能量) 的表示")
    print(f"   行 4: <P_0_0_1> (最低頻、最早時間、低能量) 的表示")
    print(f"   ...")
    print(f"   行 128: <P_6_17_15> (最高頻、最晚時間、最高能量) 的表示")

    print(f"\n   Position Embedding 矩陣 (512 × 256):")
    print(f"   行 0: 序列位置 0 的 256 維編碼")
    print(f"   行 1: 序列位置 1 的 256 維編碼")
    print(f"   ...")
    print(f"   行 126: 序列位置 126 的 256 維編碼")
    print(f"   (我們只用前 127 個位置)")

    return tok_embed, pos_embed


def trace_embedding_computation():
    """Trace how embeddings are computed in the forward pass."""

    print("\n" + "=" * 80)
    print("EMBEDDING 計算過程: Forward Pass 追蹤")
    print("=" * 80)

    # Create example input
    vocab_size = 129
    model = TransformerPolicy(vocab_size=vocab_size, n_dirs=24, d_model=256)

    # Example token sequence
    example_tokens = ["[CLS]", "<P_0_0_4>", "<P_0_1_4>", "<P_1_0_5>", "<P_2_3_8>"]
    example_ids = [1, 10, 11, 25, 48]  # Simulated token IDs

    print(f"\n1. 輸入準備:")
    print(f"   Token 序列: {example_tokens[:3]} ...")
    print(f"   Token IDs: {example_ids[:3]} ...")

    # Create tensors
    input_ids = torch.tensor([example_ids + [0] * (127 - len(example_ids))])  # Pad to 127
    attention_mask = torch.ones(1, 127)
    attention_mask[0, len(example_ids):] = 0  # Mask padding

    print(f"   Input shape: {input_ids.shape} = (batch_size=1, seq_len=127)")

    print(f"\n2. FORWARD PASS 中的 EMBEDDING 計算:")

    print(f"\n   Step 1: 查找 Token Embeddings")
    print(f"   ```python")
    print(f"   # input_ids: [1, 10, 11, 25, 48, 0, 0, ...] (長度 127)")
    print(f"   tok_embeddings = self.tok_embed(input_ids)")
    print(f"   # 對每個 ID，從 embedding 矩陣中查找對應的行")
    print(f"   # tok_embeddings[0,0,:] = tok_embed.weight[1,:] (CLS token)")
    print(f"   # tok_embeddings[0,1,:] = tok_embed.weight[10,:] (P_0_0_4)")
    print(f"   # Shape: (1, 127, 256)")
    print(f"   ```")

    with torch.no_grad():
        tok_embeddings = model.tok_embed(input_ids)
        print(f"   實際輸出形狀: {tok_embeddings.shape}")
        print(f"   第一個 token 的 embedding 前 5 維: {tok_embeddings[0, 0, :5].numpy()}")

    print(f"\n   Step 2: 生成位置索引")
    print(f"   ```python")
    print(f"   L = 127  # 序列長度")
    print(f"   pos_idx = torch.arange(L)  # [0, 1, 2, ..., 126]")
    print(f"   pos = pos_idx.unsqueeze(0).expand(1, L)  # (1, 127)")
    print(f"   ```")

    L = 127
    pos_idx = torch.arange(L)
    pos = pos_idx.unsqueeze(0).expand(1, L)
    print(f"   位置索引: {pos[0, :10].tolist()} ...")

    print(f"\n   Step 3: 查找 Position Embeddings")
    print(f"   ```python")
    print(f"   pos_embeddings = self.pos_embed(pos)")
    print(f"   # 對每個位置，從 position embedding 矩陣中查找對應的行")
    print(f"   # pos_embeddings[0,0,:] = pos_embed.weight[0,:] (位置 0)")
    print(f"   # pos_embeddings[0,1,:] = pos_embed.weight[1,:] (位置 1)")
    print(f"   # Shape: (1, 127, 256)")
    print(f"   ```")

    with torch.no_grad():
        pos_embeddings = model.pos_embed(pos)
        print(f"   實際輸出形狀: {pos_embeddings.shape}")
        print(f"   位置 0 的 embedding 前 5 維: {pos_embeddings[0, 0, :5].numpy()}")

    print(f"\n   Step 4: 組合 Token 和 Position Embeddings")
    print(f"   ```python")
    print(f"   x = tok_embeddings + pos_embeddings  # Element-wise addition")
    print(f"   # 每個位置的最終 embedding = token 語義 + 位置信息")
    print(f"   # Shape: (1, 127, 256)")
    print(f"   ```")

    with torch.no_grad():
        x = tok_embeddings + pos_embeddings
        print(f"   組合後的形狀: {x.shape}")
        print(f"   第一個位置的組合 embedding 前 5 維: {x[0, 0, :5].numpy()}")

    return model, input_ids, x


def explain_embedding_learning():
    """Explain how embeddings are learned during training."""

    print("\n" + "=" * 80)
    print("EMBEDDING 的學習過程")
    print("=" * 80)

    print("""
1. 初始化階段 (模型創建時):

   Token Embeddings 初始化:
   - 隨機初始化 129 個 256 維向量
   - 每個 token 開始時有隨機的表示
   - 相似的 tokens (如 P_0_0_4 和 P_0_0_5) 初始時可能完全不同

   Position Embeddings 初始化:
   - 隨機初始化 512 個 256 維向量
   - 每個位置開始時有隨機的編碼
   - 相鄰位置 (如位置 0 和 1) 初始時可能完全不同

2. 訓練過程中的更新:

   反向傳播通過以下路徑更新 embeddings:

   Loss ← Logits ← Pooled ← Transformer ← Embeddings
                                            ↑
                                    梯度回傳到這裡

   Token Embedding 學習什麼:
   - P_3_8_12 (高能量中頻) → 學習表示 "中頻強信號"
   - P_0_0_0 (無能量低頻) → 學習表示 "安靜區域"
   - 相似能量模式的 tokens 會學到相似的 embeddings

   Position Embedding 學習什麼:
   - 位置 0 → 學習表示 "序列開始"
   - 位置 63 → 學習表示 "序列中間"
   - 相鄰位置會學到能夠表達時間連續性的 embeddings

3. 學習目標:

   Embeddings 被優化來最大化方向預測的準確性:

   - 如果 P_3_*_12 經常出現在 0° 方向
     → P_3_*_12 的 embeddings 會包含 "指向 0°" 的信息

   - 如果位置 0-20 的 patches 對預測很重要
     → 這些位置的 embeddings 會被賦予更高的重要性
    """)


def analyze_embedding_transformer_relationship():
    """Analyze the relationship between embeddings and Transformer."""

    print("\n" + "=" * 80)
    print("EMBEDDING 與 TRANSFORMER 的關係")
    print("=" * 80)

    print("""
1. EMBEDDING 是 TRANSFORMER 的輸入接口:

   原始 Token IDs → Embedding 層 → 256維向量 → Transformer
   (離散符號)      (查找表)      (連續表示)   (可以處理)

   為什麼需要 Embedding？
   - Transformer 只能處理連續向量，不能直接處理離散 token IDs
   - Embedding 將離散空間映射到連續空間
   - 256 維提供足夠的表達能力

2. TOKEN EMBEDDING 的作用:

   語義信息編碼:
   - <P_0_0_0> → 編碼 "低頻低能量" 的語義
   - <P_3_8_12> → 編碼 "中頻高能量" 的語義
   - [CLS] → 編碼 "序列開始" 的特殊含義

   在 Transformer 中:
   - Self-attention 計算 Q, K, V 時使用這些語義信息
   - 相似語義的 tokens 會有更高的 attention 分數

3. POSITION EMBEDDING 的作用:

   位置信息編碼:
   - 告訴 Transformer 每個 patch 在序列中的位置
   - 使模型能夠區分 P_0_0 在位置 1 vs 位置 10

   為什麼重要？
   - Transformer 本身沒有位置概念（並行處理所有位置）
   - Position embedding 注入順序信息
   - 使模型能學習 "早期 patches" vs "後期 patches" 的模式

4. EMBEDDING 加法的含義:

   final_embedding = token_emb + pos_emb

   這種加法組合:
   - 保留 token 的語義信息
   - 同時編碼位置信息
   - 允許 attention 同時考慮 "是什麼" 和 "在哪裡"

5. 在 ATTENTION 計算中的角色:

   Attention(Q, K, V) 中:
   - Q = Linear(embeddings) → 查詢向量
   - K = Linear(embeddings) → 鍵向量
   - V = Linear(embeddings) → 值向量

   Attention 分數 = Q · K^T / √d
   - 如果兩個 embeddings 相似 → 高 attention 分數
   - Token 語義相似 + 位置相關 → 更高分數

6. EMBEDDING 維度 (256) 的意義:

   為什麼是 256 維？
   - 足夠的容量編碼 129 個不同的 tokens
   - 與 Transformer 的 d_model 一致
   - 8 個 attention heads × 32 維/head = 256
   - 平衡表達能力和計算效率
    """)


def visualize_embedding_space():
    """Visualize the embedding space conceptually."""

    print("\n" + "=" * 80)
    print("EMBEDDING 空間的概念可視化")
    print("=" * 80)

    print("""
Token Embedding 空間 (256維，這裡簡化為 2D):

         高能量 patches
              ↑
    P_6_*_15  •  • P_5_*_14
              •  •
    P_4_*_12  •  • P_3_*_11

    ← 低頻 ─────────────── 高頻 →

    P_2_*_5   •  • P_1_*_4
              •  •
    P_0_*_1   •  • P_0_*_0
              ↓
         低能量 patches

相似的 patches 在空間中靠近，不同的 patches 遠離。

Position Embedding 空間:

    序列末尾
        ↑
    126 •
        •
     63 • • • (中間位置聚集)
        •
      1 •
      0 •
        ↓
    序列開始

連續的位置在空間中相對靠近，表達時間連續性。

組合後的 Embedding 空間:

每個點 = Token 語義 + 位置信息
例如: P_3_8_12 在位置 45
     = embedding("中頻高能量") + embedding("序列中間")
     = 一個獨特的 256 維向量

這個組合向量進入 Transformer，參與 attention 計算和特徵學習。
    """)


def create_detailed_diagram():
    """Create a detailed diagram of the embedding process."""

    print("\n" + "=" * 80)
    print("EMBEDDING 完整流程圖")
    print("=" * 80)

    print("""
    輸入 Token 序列
    ["[CLS]", "<P_0_0_4>", "<P_0_1_4>", ..., "<P_6_17_8>"]
           ↓
    ┌──────────────────────────────────────────────┐
    │              詞彙表查找                      │
    │  {"[CLS]": 1, "<P_0_0_4>": 10, ...}        │
    └──────────────────────────────────────────────┘
           ↓
    Token IDs: [1, 10, 11, ..., 125]
           ↓
    ┌──────────────────────────────────────────────┐
    │          TOKEN EMBEDDING 查找                │
    │                                              │
    │  self.tok_embed = nn.Embedding(129, 256)    │
    │                                              │
    │  tok_embed.weight:                           │
    │  ┌─────────────────┐                        │
    │  │ 129 × 256 矩陣  │ ← 可學習參數          │
    │  └─────────────────┘                        │
    │                                              │
    │  input_ids[i] → weight[input_ids[i], :]     │
    └──────────────────────────────────────────────┘
           ↓
    Token Embeddings: (1, 127, 256)
           ↓
           ⊕ (Element-wise Addition)
           ↓
    ┌──────────────────────────────────────────────┐
    │         POSITION EMBEDDING 查找              │
    │                                              │
    │  self.pos_embed = nn.Embedding(512, 256)    │
    │                                              │
    │  pos_embed.weight:                           │
    │  ┌─────────────────┐                        │
    │  │ 512 × 256 矩陣  │ ← 可學習參數          │
    │  └─────────────────┘                        │
    │                                              │
    │  position[i] → weight[position[i], :]       │
    └──────────────────────────────────────────────┘
           ↓
    Position Embeddings: (1, 127, 256)
           ↓
    ┌──────────────────────────────────────────────┐
    │           組合後的 Embeddings                │
    │                                              │
    │   X = Token_Emb + Pos_Emb                   │
    │   Shape: (1, 127, 256)                      │
    │                                              │
    │   每個 256 維向量包含:                       │
    │   - Token 的語義信息                        │
    │   - 在序列中的位置信息                      │
    └──────────────────────────────────────────────┘
           ↓
    ┌──────────────────────────────────────────────┐
    │            TRANSFORMER ENCODER               │
    │                                              │
    │  Multi-Head Self-Attention:                 │
    │  - 使用 embeddings 計算 Q, K, V             │
    │  - 學習 patches 之間的關係                  │
    │                                              │
    │  Feed-Forward Network:                      │
    │  - 進一步轉換特徵                          │
    │                                              │
    │  × 2 層                                      │
    └──────────────────────────────────────────────┘
           ↓
    Contextualized Representations: (1, 127, 256)
           ↓
    [Pooling → Classification → Direction]

    訓練時的梯度回傳:
    Loss → ... → Transformer → Embeddings → tok_embed.weight
                                         → pos_embed.weight
    通過梯度下降更新這兩個 embedding 矩陣
    """)


def main():
    """Run complete embedding analysis."""

    # Analyze embedding initialization
    tok_embed, pos_embed = analyze_embedding_layers()

    # Trace computation
    model, input_ids, embeddings = trace_embedding_computation()

    # Explain learning
    explain_embedding_learning()

    # Analyze relationships
    analyze_embedding_transformer_relationship()

    # Visualize
    visualize_embedding_space()

    # Create diagram
    create_detailed_diagram()

    print("\n" + "=" * 80)
    print("總結: Embeddings 是離散 tokens 和連續 Transformer 之間的橋樑")
    print("=" * 80)
    print("""
關鍵要點:
1. Token Embedding: 學習每個 patch pattern 的語義表示
2. Position Embedding: 編碼序列中的位置信息
3. 兩者相加: 同時保留 "是什麼" 和 "在哪裡" 的信息
4. 可學習參數: 通過訓練自動優化這些表示
5. Transformer 輸入: 這些 embeddings 是 Transformer 的實際輸入
    """)


if __name__ == "__main__":
    main()