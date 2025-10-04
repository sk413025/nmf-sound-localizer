#!/usr/bin/env python
"""Detailed analysis of Transformer output and direction prediction mechanism."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from doa_rl.model.transformer import TransformerPolicy
from doa_rl.text import Vocab, pad_sequences
from doa_rl.features import PatchTokenizer


def analyze_transformer_architecture():
    """Analyze whether this is seq2seq or classification architecture."""

    print("=" * 80)
    print("TRANSFORMER 架構類型分析")
    print("=" * 80)

    print("""
這個 TransformerPolicy 的架構類型:

❌ 不是 Sequence-to-Sequence (Seq2Seq):
   - Seq2Seq: 輸入序列 → 輸出序列 (如翻譯: "Hello" → "你好")
   - Seq2Seq 特徵:
     * 有 Encoder 和 Decoder
     * 輸出長度可變
     * 每個位置輸出一個 token

✅ 是 Sequence-to-Classification:
   - 輸入: 127 個 patch tokens (序列)
   - 輸出: 1 個方向類別 (分類)
   - 架構特徵:
     * 只有 Encoder (沒有 Decoder)
     * 使用 Mean Pooling 聚合序列信息
     * 最後用 Linear 層做分類

架構流程:
┌─────────────────────────────────────────┐
│  Input: Sequence of 127 tokens         │
│  [CLS, P_0_0, P_0_1, ..., P_6_17]      │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Transformer Encoder (2 layers)        │
│  Self-Attention + Feed-Forward         │
│  輸出: (1, 127, 256) - 每個 token 的表示│
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Mean Pooling                          │
│  將 127 個向量平均成 1 個向量          │
│  輸出: (1, 256)                        │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Linear Classification Head            │
│  256 → 24 (方向數)                     │
│  輸出: (1, 24) logits                  │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Argmax                                │
│  輸出: 單一方向 index (0-23)           │
└─────────────────────────────────────────┘
    """)


def trace_transformer_output():
    """Trace the complete output flow through Transformer."""

    print("\n" + "=" * 80)
    print("TRANSFORMER 輸出維度追蹤")
    print("=" * 80)

    # Create model
    model = TransformerPolicy(vocab_size=129, n_dirs=24, d_model=256)

    # Create dummy input
    input_ids = torch.randint(0, 129, (1, 127))
    attention_mask = torch.ones(1, 127)

    print("\n1. 輸入維度:")
    print(f"   Input IDs: {input_ids.shape} = (batch=1, seq_len=127)")
    print(f"   Attention Mask: {attention_mask.shape}")

    # Manual forward pass to track dimensions
    with torch.no_grad():
        # Embedding
        B, L = input_ids.shape
        pos_idx = torch.arange(L)
        pos = pos_idx.unsqueeze(0).expand(B, L)

        tok_emb = model.tok_embed(input_ids)
        pos_emb = model.pos_embed(pos)
        x = tok_emb + pos_emb

        print(f"\n2. Embedding 後:")
        print(f"   Combined embeddings: {x.shape} = (1, 127, 256)")
        print(f"   每個 token 都有 256 維表示")

        # Transformer Encoder
        key_padding_mask = attention_mask.eq(0)
        h = model.encoder(x, src_key_padding_mask=key_padding_mask)

        print(f"\n3. Transformer Encoder 後:")
        print(f"   Contextualized representations: {h.shape} = (1, 127, 256)")
        print(f"   仍然是 127 個 tokens，但每個都包含了全局信息")

        # Layer Norm
        h = model.norm(h)
        print(f"\n4. Layer Normalization 後:")
        print(f"   Normalized features: {h.shape} = (1, 127, 256)")

        # Mean Pooling
        denom = attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (h * attention_mask.unsqueeze(-1)).sum(dim=1) / denom

        print(f"\n5. Mean Pooling 後:")
        print(f"   Pooled representation: {pooled.shape} = (1, 256)")
        print(f"   127 個 tokens → 1 個全局向量")

        # Classification Head
        logits = model.head(pooled)

        print(f"\n6. Linear Classification Head 後:")
        print(f"   Direction logits: {logits.shape} = (1, 24)")
        print(f"   24 個方向的未歸一化分數")

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        print(f"\n7. Softmax 後:")
        print(f"   Direction probabilities: {probs.shape} = (1, 24)")
        print(f"   總和 = {probs.sum().item():.4f}")

        # Final prediction
        prediction = torch.argmax(logits, dim=-1)

        print(f"\n8. 最終預測:")
        print(f"   Predicted direction index: {prediction.item()}")
        print(f"   對應角度: {prediction.item() * 15}° (假設 15° 間隔)")

    return model, logits, probs, prediction


def explain_direction_tokens():
    """Explain whether directions have tokens."""

    print("\n" + "=" * 80)
    print("方向是否有 TOKENS？")
    print("=" * 80)

    print("""
❌ 方向沒有 tokens！

輸入 tokens vs 輸出方向:

輸入側 (有 tokens):
- [CLS]: token ID = 1
- <P_0_0_4>: token ID = 10
- <P_0_1_5>: token ID = 15
- ...
- 總共 129 種可能的 input tokens

輸出側 (沒有 tokens):
- 方向 0°: index = 0 (不是 token，是類別標籤)
- 方向 15°: index = 1
- 方向 30°: index = 2
- ...
- 方向 345°: index = 23
- 總共 24 個類別（不是 tokens）

為什麼方向不需要 tokens？
1. 這是分類任務，不是生成任務
2. 輸出是固定的 24 個類別之一
3. 使用 Linear 層直接輸出 24 個 logits
4. 不需要詞彙表或 embedding

如果是 Seq2Seq 架構，才會有輸出 tokens:
- 例如翻譯: 輸出 "你", "好" 都是 tokens
- 需要輸出詞彙表和 decoder
- 但我們的任務不是這樣
    """)


def analyze_decision_mechanism():
    """Analyze how Transformer decides the direction."""

    print("\n" + "=" * 80)
    print("TRANSFORMER 如何決定方向？")
    print("=" * 80)

    print("""
決策機制詳解:

1. 信息聚合 (通過 Self-Attention):
   ```
   每個 patch token 通過 attention 獲取其他 patches 的信息

   例如 P_3_8 (中頻中間時刻) 會關注:
   - P_3_7, P_3_9 (時間鄰近)
   - P_2_8, P_4_8 (頻率鄰近)
   - P_1_5 (可能的諧波關係)

   Attention 權重學習哪些 patches 組合對方向重要
   ```

2. 特徵提取 (通過 Transformer Layers):
   ```
   Layer 1: 學習局部 patch 關係
   Layer 2: 學習全局 patch 模式

   每層包含:
   - Multi-Head Attention: 8 個 heads 學習不同關係
   - Feed-Forward: 進一步轉換特徵
   ```

3. 全局表示 (通過 Mean Pooling):
   ```
   127 個 contextualized tokens
           ↓ 平均
   1 個 256 維全局向量

   這個向量編碼了所有 patches 的聚合信息
   ```

4. 方向分類 (通過 Linear Head):
   ```python
   # Linear 層的權重矩陣 W: (256, 24)
   # 每一列對應一個方向的 "檢測器"

   logits = pooled @ W.T + bias
   # logits[0] = 檢測 0° 的分數
   # logits[1] = 檢測 15° 的分數
   # ...
   # logits[23] = 檢測 345° 的分數
   ```

5. 最終決策:
   ```python
   # 選擇分數最高的方向
   prediction = argmax(logits)

   # 或者獲取概率分布
   probabilities = softmax(logits)
   # 可以看到每個方向的信心度
   ```
    """)


def visualize_output_flow():
    """Visualize the complete output flow."""

    print("\n" + "=" * 80)
    print("完整輸出流程圖")
    print("=" * 80)

    print("""
    126 Patches + [CLS]
         ↓
    ┌───────────────────────────────────┐
    │     Input IDs (1, 127)           │
    └───────────────────────────────────┘
         ↓
    ┌───────────────────────────────────┐
    │    Embeddings (1, 127, 256)      │
    │  每個 token 的 256 維初始表示     │
    └───────────────────────────────────┘
         ↓
    ┌───────────────────────────────────┐
    │   Transformer Layer 1             │
    │  - Self-Attention: 學習 patch 關係│
    │  - FFN: 特徵轉換                 │
    │  輸出: (1, 127, 256)             │
    └───────────────────────────────────┘
         ↓
    ┌───────────────────────────────────┐
    │   Transformer Layer 2             │
    │  - Self-Attention: 全局模式      │
    │  - FFN: 深層特徵                 │
    │  輸出: (1, 127, 256)             │
    └───────────────────────────────────┘
         ↓
    ┌───────────────────────────────────┐
    │   Layer Normalization             │
    │  輸出: (1, 127, 256)             │
    └───────────────────────────────────┘
         ↓
    ┌───────────────────────────────────┐
    │      Mean Pooling                 │
    │  127 vectors → 1 vector          │
    │  輸出: (1, 256)                  │
    └───────────────────────────────────┘
         ↓
    ┌───────────────────────────────────┐
    │   Linear Classification Head      │
    │     W: (256, 24), b: (24)        │
    │   輸出: (1, 24) logits           │
    └───────────────────────────────────┘
         ↓
    ┌───────────────────────────────────┐
    │         Softmax                   │
    │   輸出: (1, 24) probabilities    │
    └───────────────────────────────────┘
         ↓
    ┌───────────────────────────────────┐
    │         Argmax                    │
    │   輸出: scalar (0-23)            │
    └───────────────────────────────────┘
         ↓
    Direction: 0°, 15°, 30°, ..., or 345°
    """)


def show_example_computation():
    """Show a concrete example of direction computation."""

    print("\n" + "=" * 80)
    print("實際計算示例")
    print("=" * 80)

    # Create model
    model = TransformerPolicy(vocab_size=129, n_dirs=24, d_model=256)

    # Dummy input
    input_ids = torch.randint(0, 129, (1, 127))
    attention_mask = torch.ones(1, 127)

    # Forward pass
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)

        print("實際輸出示例:\n")
        print(f"Logits 形狀: {logits.shape}")
        print(f"前 5 個 logits: {logits[0, :5].numpy()}")
        print()

        print(f"Probabilities 形狀: {probs.shape}")
        print(f"前 5 個概率:")
        for i in range(5):
            angle = i * 15
            prob = probs[0, i].item()
            print(f"  方向 {angle:3d}°: {prob:.4f} ({prob*100:.2f}%)")

        print(f"\n概率總和: {probs.sum().item():.6f} (應該 = 1.0)")

        # Prediction
        pred_idx = torch.argmax(logits, dim=-1).item()
        pred_angle = pred_idx * 15
        pred_prob = probs[0, pred_idx].item()

        print(f"\n最終預測:")
        print(f"  Index: {pred_idx}")
        print(f"  角度: {pred_angle}°")
        print(f"  信心度: {pred_prob:.4f} ({pred_prob*100:.2f}%)")

        # Show decision boundary
        print(f"\n決策邊界:")
        sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)
        for i in range(3):
            idx = sorted_indices[i].item()
            angle = idx * 15
            prob = sorted_probs[i].item()
            print(f"  Top {i+1}: {angle:3d}° ({prob*100:.2f}%)")


def explain_training_supervision():
    """Explain how the model is trained."""

    print("\n" + "=" * 80)
    print("訓練監督信號")
    print("=" * 80)

    print("""
訓練過程:

1. 標籤格式:
   ```python
   # 每個訓練樣本
   input: 127 個 patch tokens
   target: 單一方向標籤 (0-23)

   # 不是 token，是類別 index
   target = 5  # 表示 75° (5 * 15°)
   ```

2. Loss 計算:
   ```python
   # Cross-Entropy Loss
   logits = model(input_ids, attention_mask)  # (1, 24)
   loss = F.cross_entropy(logits, target)

   # 等價於
   log_probs = F.log_softmax(logits, dim=-1)
   loss = -log_probs[target]  # 負對數似然
   ```

3. 梯度更新:
   ```
   Loss 對每個方向的梯度:
   ∂L/∂logits[i] = probs[i] - 1(i==target)

   如果預測錯誤:
   - 正確方向的 logit 會增加
   - 錯誤方向的 logit 會減少
   - 通過反向傳播更新所有參數
   ```

4. 學習目標:
   - 使正確方向的 logit 最大
   - 學習 patch 模式與方向的對應關係
   - 優化 attention 權重來關注重要 patches
    """)


def main():
    """Run complete analysis."""

    # Architecture type
    analyze_transformer_architecture()

    # Output dimensions
    model, logits, probs, prediction = trace_transformer_output()

    # Direction tokens
    explain_direction_tokens()

    # Decision mechanism
    analyze_decision_mechanism()

    # Output flow
    visualize_output_flow()

    # Example computation
    show_example_computation()

    # Training
    explain_training_supervision()

    print("\n" + "=" * 80)
    print("總結")
    print("=" * 80)
    print("""
關鍵要點:

1. 架構類型: Sequence-to-Classification (不是 Seq2Seq)
   - 輸入: 序列 (127 tokens)
   - 輸出: 單一類別 (24 個方向之一)

2. 輸出維度:
   - Transformer 輸出: (1, 127, 256)
   - Pooling 後: (1, 256)
   - 最終 logits: (1, 24)

3. 方向預測:
   - 沒有方向 tokens
   - 使用 Linear 層直接分類
   - Argmax 選擇最高分數的方向

4. 決策機制:
   - Self-Attention 學習 patch 關係
   - Mean Pooling 聚合信息
   - Linear Head 做 24 分類
    """)


if __name__ == "__main__":
    main()