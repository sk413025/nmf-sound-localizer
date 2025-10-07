#!/usr/bin/env python3
"""Debug script to understand PEFT embedding layer structure."""

import sys
sys.path.insert(0, '/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/development-workspace')

from peft import get_peft_model, LoraConfig
from doa_rl.hf import build_patch_tokenizer, build_value_head_model

# Discover angles
import pathlib
angles = sorted([
    int(p.name.split("_")[1])
    for p in pathlib.Path("tmp/trl_test_data").glob("angle_*")
    if p.is_dir() and p.name.split("_")[1].isdigit()
])
print(f"Found angles: {angles}\n")

# Build model
tokenizer = build_patch_tokenizer(direction_angles=angles)
rm_model, _ = build_value_head_model(tokenizer)

print("=" * 60)
print("BEFORE LoRA")
print("=" * 60)
emb = rm_model.pretrained_model.get_input_embeddings()
print(f"Embedding layer type: {type(emb)}")
print(f"Embedding weight shape: {emb.weight.shape}")
print(f"Requires grad: {emb.weight.requires_grad}")
print()

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['c_attn', 'c_proj'],
    task_type='CAUSAL_LM'
)
rm_model.pretrained_model = get_peft_model(rm_model.pretrained_model, lora_config)

print("=" * 60)
print("AFTER LoRA (before unfreezing)")
print("=" * 60)
emb = rm_model.pretrained_model.get_input_embeddings()
print(f"Embedding layer type: {type(emb)}")
print(f"Has 'base_layer': {hasattr(emb, 'base_layer')}")
print(f"Has 'weight': {hasattr(emb, 'weight')}")
if hasattr(emb, 'base_layer'):
    print(f"Base layer type: {type(emb.base_layer)}")
    print(f"Base layer weight shape: {emb.base_layer.weight.shape}")
    print(f"Base layer requires_grad: {emb.base_layer.weight.requires_grad}")
if hasattr(emb, 'weight'):
    print(f"Weight shape: {emb.weight.shape}")
    print(f"Requires grad: {emb.weight.requires_grad}")
print()

# Attempt to unfreeze
if hasattr(emb, 'base_layer'):
    emb.base_layer.weight.requires_grad = True
else:
    emb.weight.requires_grad = True

print("=" * 60)
print("AFTER unfreezing")
print("=" * 60)
if hasattr(emb, 'base_layer'):
    print(f"Base layer requires_grad: {emb.base_layer.weight.requires_grad}")
if hasattr(emb, 'weight'):
    print(f"Weight requires_grad: {emb.weight.requires_grad}")
print()

# Check all parameters
print("=" * 60)
print("First 20 named parameters:")
print("=" * 60)
for i, (n, p) in enumerate(rm_model.named_parameters()):
    if i < 20:
        print(f"  {n}: shape={p.shape}, requires_grad={p.requires_grad}")
print()

print("=" * 60)
print("All parameters with 'embed' in name:")
print("=" * 60)
for n, p in rm_model.named_parameters():
    if 'embed' in n.lower():
        print(f"  {n}:")
        print(f"    - shape: {p.shape}")
        print(f"    - requires_grad: {p.requires_grad}")
        print(f"    - numel: {p.numel():,}")
print()

# Count trainable
def count_params(model, pattern=None):
    if pattern:
        return sum(
            p.numel() for n, p in model.named_parameters()
            if p.requires_grad and pattern.lower() in n.lower()
        )
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=" * 60)
print("Parameter counts:")
print("=" * 60)
print(f"LoRA:       {count_params(rm_model, 'lora'):,}")
print(f"Embeddings: {count_params(rm_model, 'embed'):,}")
print(f"V-head:     {sum(p.numel() for p in rm_model.v_head.parameters()):,}")
print(f"Total trainable: {sum(p.numel() for p in rm_model.parameters() if p.requires_grad):,}")
print(f"Total params: {sum(p.numel() for p in rm_model.parameters()):,}")
