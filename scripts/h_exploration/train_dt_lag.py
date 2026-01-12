import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LagTrajectoryDataset(Dataset):
    def __init__(self, pt_path):
        raw_data = torch.load(pt_path, weights_only=False)
        
        # Data is list of dicts: { "corrs": (F, K, M), "actions": (F, K) }
        # We flat it out to (N_samples * K, M)
        # This treats every step as an independent sample (Markov assumption)
        
        self.all_corrs = []
        self.all_actions = []
        
        logger.info("Loading and flattening data...")
        for block in raw_data:
            # block["corrs"] is shape (F, K, M)
            K = block["corrs"].shape[1]
            c = block["corrs"].float() # (F, K, M)
            a = block["actions"].long() # (F, K)
            
            # Reshape to (F*K, M) and (F*K)
            c = c.view(-1, 16)
            a = a.view(-1)
            
            self.all_corrs.append(c)
            self.all_actions.append(a)
            
        self.corrs = torch.cat(self.all_corrs, dim=0) # (N_total*K, M)
        self.actions = torch.cat(self.all_actions, dim=0) # (N_total*K)
        
        logger.info(f"Loaded {len(self.corrs)} steps.")
        
    def __len__(self):
        return len(self.corrs)
    
    def __getitem__(self, idx):
        return self.corrs[idx], self.actions[idx]

class StepwiseDT(nn.Module):
    def __init__(self, M_lags=16, d_model=256):
        super().__init__()
        # Simple MLP: Input Correlation (M) -> Action Logits (M)
        self.net = nn.Sequential(
            nn.Linear(M_lags, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, M_lags)
        )
        
    def forward(self, x):
        return self.net(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/dtmin_lag")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    # Load Data
    dataset = LagTrajectoryDataset(args.data_path)
    
    # Split
    N = len(dataset)
    train_len = int(0.9 * N)
    val_len = N - train_len
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len])
    
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024)
    
    # Model
    model = StepwiseDT(M_lags=16, d_model=256)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    logger.info("Starting Training...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device) 
            batch_y = batch_y.to(device)
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.numel()
            
        train_acc = correct / total
        
        # Val
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.numel()
                
        val_acc = val_correct / val_total
        logger.info(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
    torch.save(model.state_dict(), Path(args.out_dir) / "dt_lag_stepwise.pth")

if __name__ == "__main__":
    main()
