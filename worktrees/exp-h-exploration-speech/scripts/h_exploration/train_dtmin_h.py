import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HTrajectoryDataset(Dataset):
    def __init__(self, data_path: str):
        self.data = torch.load(data_path, weights_only=False)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # Item: {steps: [...], angle, freq_bin, ...}
        traj = self.data[idx]
        steps = traj["steps"]
        
        # Prepare Sequences
        obs_seq = []
        action_seq = []
        rtg_seq = []
        
        # Calculate Returns (Cumulative Reward/Cost?)
        # For standard DT, RTG is sum of future rewards.
        # Here OMP minimizes MSE. Improvement is reward.
        # Let's use MSE as the "Return" we want to achieve (0).
        # So RTG at step t is the Final MSE achievable? Or current MSE?
        # Let's use Final MSE as target (Conditional BC).
        final_mse = steps[-1]["mse"]
        
        for step in steps:
            obs = torch.tensor(step["observation"], dtype=torch.float32)
            action = torch.tensor(step["action"], dtype=torch.long)
            # RTG: Condition on the outcome
            rtg = torch.tensor([final_mse], dtype=torch.float32)
            
            obs_seq.append(obs)
            action_seq.append(action)
            rtg_seq.append(rtg)
            
        return {
            "obs_seq": torch.stack(obs_seq),    # (K, C)
            "action_seq": torch.stack(action_seq), # (K,)
            "rtg_seq": torch.stack(rtg_seq),       # (K, 1)
            "length": len(steps)
        }

def collate_fn(batch):
    # Pad to max length
    max_len = max(b["length"] for b in batch)
    
    obs_batch = []
    act_batch = []
    rtg_batch = []
    mask_batch = []
    
    for b in batch:
        l = b["length"]
        obs = b["obs_seq"]
        act = b["action_seq"]
        rtg = b["rtg_seq"]
        
        # Pad
        pad_len = max_len - l
        if pad_len > 0:
            obs_pad = torch.zeros(pad_len, obs.shape[1])
            act_pad = torch.zeros(pad_len, dtype=torch.long) # 0 index? careful
            rtg_pad = torch.zeros(pad_len, 1)
            
            obs = torch.cat([obs, obs_pad], dim=0)
            act = torch.cat([act, act_pad], dim=0)
            rtg = torch.cat([rtg, rtg_pad], dim=0)
            
        mask = torch.arange(max_len) < l
        
        obs_batch.append(obs)
        act_batch.append(act)
        rtg_batch.append(rtg)
        mask_batch.append(mask)
        
    return {
        "obs": torch.stack(obs_batch),
        "action": torch.stack(act_batch),
        "rtg": torch.stack(rtg_batch),
        "mask": torch.stack(mask_batch)
    }

class H_DT(nn.Module):
    def __init__(self, input_dim, num_actions, d_model=64):
        super().__init__()
        self.obs_emb = nn.Linear(input_dim, d_model)
        self.rtg_emb = nn.Linear(1, d_model)
        # self.act_emb = nn.Embedding(num_actions, d_model) # If we use past actions
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=2
        )
        
        self.head = nn.Linear(d_model, num_actions)
        
    def forward(self, obs, rtg, mask=None):
        # obs: (B, T, Input)
        # rtg: (B, T, 1)
        x = self.obs_emb(obs) + self.rtg_emb(rtg)
        
        # Add Positional Encoding? 
        # For OMP steps K is small (4-6), maybe not strictly needed or simple additive
        B, T, D = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        # pos_emb = ... embedded
        
        out = self.transformer(x, src_key_padding_mask=~mask if mask is not None else None)
        logits = self.head(out)
        return logits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    dataset = HTrajectoryDataset(args.data_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    # Determine dims from first item
    sample = dataset[0]
    input_dim = sample["obs_seq"].shape[1]
    # Max actions? Assumed 6 for now or derive from data
    num_actions = 16 # Safe upper bound for channels?
    
    model = H_DT(input_dim, num_actions)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            obs = batch["obs"]
            rtg = batch["rtg"]
            target = batch["action"]
            mask = batch["mask"]
            
            logits = model(obs, rtg, mask)
            
            # mask logits and targets
            # flat
            logit_flat = logits[mask]
            target_flat = target[mask]
            
            loss = crit(logit_flat, target_flat)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        logger.info(f"Epoch {epoch}: Loss {total_loss/len(loader)}")
        
    # Save
    torch.save(model.state_dict(), f"{args.out_dir}/model.pt")
    logger.info("Saved model.")

if __name__ == "__main__":
    main()
