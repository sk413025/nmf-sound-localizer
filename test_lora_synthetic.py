#!/usr/bin/env python3
"""Test LoRA training with synthetic data to verify the mechanism works."""

import sys
sys.path.insert(0, '/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/development-workspace')

import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig
from doa_rl.hf import build_patch_tokenizer, build_value_head_model
import pathlib

# Discover angles
angles = sorted([
    int(p.name.split("_")[1])
    for p in pathlib.Path("tmp/trl_test_data").glob("angle_*")
    if p.is_dir() and p.name.split("_")[1].isdigit()
])

# Build model
tokenizer = build_patch_tokenizer(direction_angles=angles)
rm_model, _ = build_value_head_model(tokenizer)

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['c_attn', 'c_proj'],
    task_type='CAUSAL_LM'
)
rm_model.pretrained_model = get_peft_model(rm_model.pretrained_model, lora_config)

# Unfreeze embeddings
embedding_layer = rm_model.pretrained_model.get_input_embeddings()
embedding_layer.weight.requires_grad = True

# Create synthetic training data with CLEAR pattern
# Pattern: reward = length of sequence
texts = [
    "P001 P002 <D_000>",
    "P001 P002 P003 <D_000> <D_015>",
    "P001 P002 P003 P004 <D_000> <D_015> <D_030>",
    "P001 <D_000>",
    "P001 P002 P003 P004 P005 <D_015>",
    "P001 P002 P003 <D_030>",
]
# Rewards are simply the number of tokens (easy pattern to learn)
rewards = [float(len(t.split())) for t in texts]

print("Synthetic training data:")
for t, r in zip(texts, rewards):
    print(f"  '{t}' → {r}")
print(f"Reward range: [{min(rewards)}, {max(rewards)}]")
print(f"Reward std: {torch.tensor(rewards).std().item():.2f}\n")

# Tokenize
enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
input_ids = enc["input_ids"]
attn = enc["attention_mask"]
tgt = torch.tensor(rewards, dtype=torch.float32)

# Optimizer - include ALL trainable parameters
optimizer = torch.optim.Adam([
    {"params": [p for n, p in rm_model.named_parameters() if p.requires_grad and "lora" in n.lower()], "lr": 1e-4},
    {"params": [p for n, p in rm_model.named_parameters() if p.requires_grad and ("wte" in n.lower() or "wpe" in n.lower())], "lr": 1e-4},
    {"params": rm_model.v_head.parameters(), "lr": 1e-3},
])

# Check optimizer has parameters
total_opt_params = sum(len(group['params']) for group in optimizer.param_groups)
print(f"Optimizer parameter groups:")
for i, group in enumerate(optimizer.param_groups):
    print(f"  Group {i}: {len(group['params'])} parameters, LR={group['lr']}")
print(f"Total parameters in optimizer: {total_opt_params}\n")

loss_fn = nn.MSELoss()

print("Training with synthetic data:")
print("=" * 60)

for epoch in range(50):
    optimizer.zero_grad()
    
    # Forward
    out = rm_model.pretrained_model(
        input_ids=input_ids,
        attention_mask=attn,
        output_hidden_states=True,
        return_dict=True
    )
    last_hidden = out.hidden_states[-1]
    value = rm_model.v_head(last_hidden).squeeze(-1)
    
    # Take last token
    lengths = attn.sum(dim=1) - 1
    pred = value[torch.arange(value.size(0)), lengths]
    
    # Loss
    loss = loss_fn(pred, tgt)
    
    # Backward
    loss.backward()
    torch.nn.utils.clip_grad_norm_(rm_model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Metrics
    with torch.no_grad():
        mae = torch.mean(torch.abs(pred - tgt))
        if len(pred) > 1:
            correlation = torch.corrcoef(torch.stack([pred, tgt]))[0, 1]
            if torch.isnan(correlation):
                correlation = torch.tensor(0.0)
        else:
            correlation = torch.tensor(0.0)
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch:3d}: loss={loss.item():10.4f}, mae={mae.item():6.2f}, corr={correlation.item():6.3f}")

print("=" * 60)
print("\nFinal predictions vs targets:")
with torch.no_grad():
    out = rm_model.pretrained_model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, return_dict=True)
    last_hidden = out.hidden_states[-1]
    value = rm_model.v_head(last_hidden).squeeze(-1)
    lengths = attn.sum(dim=1) - 1
    pred = value[torch.arange(value.size(0)), lengths]
    
    for i, (t, r, p) in enumerate(zip(texts, rewards, pred)):
        print(f"  {i}: target={r:5.1f}, pred={p.item():5.1f}, error={abs(r-p.item()):5.1f}")

print("\n✅ If loss decreased significantly, LoRA training works!")
print("❌ If loss stayed at ~10²⁸, there's a fundamental bug.")
