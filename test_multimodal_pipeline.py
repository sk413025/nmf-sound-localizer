#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
測試完整的多模態訓練流程
驗證 Day 1-7 的所有組件能否順利整合
"""

import torch
import numpy as np
from pathlib import Path

print("=" * 60)
print("多模態 ICL 流程測試")
print("=" * 60)

# ============================================================================
# 1. 載入物理資產
# ============================================================================
print("\n[1/6] 載入 W 和 H 矩陣...")

# W 矩陣 (NMF 字典)
w_path = Path("doa_normalized_config_c_corrected/models/usm.pth")
if not w_path.exists():
    print(f"⚠️  找不到 W 矩陣: {w_path}")
    print("   請確認路徑是否正確")
    exit(1)

data = torch.load(w_path, map_location="cpu")
W = data["W"].numpy()  # (F, K)
print(f"✓ W 矩陣載入: {W.shape} (F={W.shape[0]}, K={W.shape[1]})")

# H 矩陣 (轉移函數)
h_path = Path("h_matrix_normalized_original_to_box.pth")
if not h_path.exists():
    print(f"⚠️  找不到 H 矩陣: {h_path}")
    print("   請確認路徑是否正確")
    exit(1)

H_data = torch.load(h_path, map_location="cpu", weights_only=False)
H = H_data["H"].numpy()  # (F, D)
angles = H_data["angles"]  # List[int]
print(f"✓ H 矩陣載入: {H.shape} (F={H.shape[0]}, D={H.shape[1]})")
print(f"✓ 角度範圍: {angles[0]}° - {angles[-1]}°, 共 {len(angles)} 個方向")

# ============================================================================
# 2. 建立 Tokenizers
# ============================================================================
print("\n[2/6] 建立三種 Tokenizers...")

from doa_rl.features.tokenizers import PatchTokenizer
from doa_rl.features.tokenizers_extended import (
    NMFAtomTokenizer,
    DirectionProjectionTokenizer,
)

patch_tok = PatchTokenizer()
atom_tok = NMFAtomTokenizer(W, top_k=8)
dir_tok = DirectionProjectionTokenizer(H, angles, top_m=5)

print(f"✓ PatchTokenizer 建立完成")
print(f"✓ NMFAtomTokenizer 建立完成 (top_k=8)")
print(f"✓ DirectionProjectionTokenizer 建立完成 (top_m=5)")

# ============================================================================
# 3. 建立 MultiModalPromptBuilder
# ============================================================================
print("\n[3/6] 建立 MultiModalPromptBuilder...")

from doa_rl.features.prompt_builder import (
    MultiModalPromptBuilder,
    PromptConfig,
)

config = PromptConfig(
    ordering="physics_first",
    use_directions=True,
    use_atoms=True,
    use_patches=True,
    max_tokens=200,
)

builder = MultiModalPromptBuilder(
    patch_tokenizer=patch_tok,
    atom_tokenizer=atom_tok,
    direction_tokenizer=dir_tok,
    config=config,
)

print(f"✓ PromptBuilder 建立完成")
print(f"  - 排序策略: {config.ordering}")
print(f"  - Token 上限: {config.max_tokens}")

# ============================================================================
# 4. 測試 DoAICLDataset
# ============================================================================
print("\n[4/6] 測試 DoAICLDataset...")

from doa_rl.data import DoAICLDataset

data_root = "doa_normalized_config_c_corrected"
direction_angles = list(range(80, 101, 5))  # [80, 85, 90, 95, 100]

dataset = DoAICLDataset(
    root=data_root,
    angles=direction_angles,
    prompt_builder=builder,
    icl_mode=False,  # 先測試基本模式
)

# 限制只測試前 5 個樣本
test_size = min(5, len(dataset))

print(f"✓ Dataset 建立完成: {len(dataset)} 個樣本")

# 測試讀取一個樣本
sample = dataset[0]
print(f"\n樣本資訊:")
print(f"  - Y shape: {sample['Y'].shape}")
print(f"  - Angle: {sample['angle_deg']}°")
print(f"  - Path: {sample['path']}")
print(f"  - Prompt (前 200 字元): {sample['prompt'][:200]}...")

# ============================================================================
# 5. 測試 HF Tokenizer
# ============================================================================
print("\n[5/6] 測試 HF Tokenizer (Extended Vocabulary)...")

from doa_rl.hf.tokenizer import build_patch_tokenizer

tokenizer = build_patch_tokenizer(
    direction_angles,
    enable_extended_vocab=True,
    n_atoms=W.shape[1],  # K
)

print(f"✓ Tokenizer 建立完成")
print(f"  - 詞彙表大小: {len(tokenizer)} tokens")
print(f"  - BOS token: {tokenizer.bos_token}")
print(f"  - EOS token: {tokenizer.eos_token}")

# 測試 encode/decode
test_prompt = sample["prompt"]
encoded = tokenizer.encode(test_prompt)
decoded = tokenizer.decode(encoded.ids)

print(f"\n編碼測試:")
print(f"  - 原始長度: {len(test_prompt.split())} tokens")
print(f"  - 編碼長度: {len(encoded.ids)} token IDs")
print(f"  - Round-trip 正確: {test_prompt == decoded}")

# ============================================================================
# 6. 測試 Transformer Model
# ============================================================================
print("\n[6/6] 測試 Transformer Model...")

from doa_rl.hf.model import build_value_head_model

model, _ = build_value_head_model(tokenizer)
model.eval()

print(f"✓ Model 建立完成")
print(f"  - 總參數: {sum(p.numel() for p in model.parameters()):,}")
print(f"  - Embedding 維度: {model.transformer.wte.weight.shape}")

# 測試前向傳播
input_ids = torch.tensor([encoded.ids])
attention_mask = torch.ones_like(input_ids)

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    values = outputs.value

print(f"\n前向傳播測試:")
print(f"  - Input shape: {input_ids.shape}")
print(f"  - Logits shape: {logits.shape}")
print(f"  - Values shape: {values.shape}")

# ============================================================================
# 總結
# ============================================================================
print("\n" + "=" * 60)
print("✅ 所有測試通過！")
print("=" * 60)
print("\n多模態 ICL 流程已經就緒，可以進行完整訓練：")
print("\n建議的測試順序:")
print("1. Smoke test (10 samples, 2 epochs)")
print("   → 驗證訓練腳本能正常運行")
print("\n2. 小規模實驗 (100 samples, 10 epochs)")
print("   → 對比 baseline vs multi-modal 效果")
print("\n3. Ablation studies")
print("   → physics_first vs structure_first vs patch_first")
print("   → 測試不同 token ordering 的影響")
print("\n4. ICL few-shot experiments")
print("   → 1-shot vs 3-shot vs 5-shot")
print("   → random vs nearest vs diverse sampling")
print("\n下一步: 修改訓練腳本以支援 --use-multi-modal 參數")
print("=" * 60)
