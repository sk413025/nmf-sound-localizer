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

class LagSequenceDataset(Dataset):
    def __init__(self, pt_path):
        raw_data = torch.load(pt_path, weights_only=False)
        self.seq_corrs = []
        self.seq_actions = []
        
        logger.info(f"Loading sequence data from {pt_path}...")
        for block in raw_data:
            # block["corrs"]: (F, K, M)
            # block["actions"]: (F, K)
            c = block["corrs"].float()
            a = block["actions"].long()
            
            # We treat each Frequency bin as an independent sequence of length K
            # c shape: (F, K, M) -> add to list
            self.seq_corrs.append(c)
            self.seq_actions.append(a)
            
        self.corrs = torch.cat(self.seq_corrs, dim=0)   # (N_total, K, M)
        self.actions = torch.cat(self.seq_actions, dim=0) # (N_total, K)
        
        self.N, self.K, self.M = self.corrs.shape
        logger.info(f"Loaded {self.N} sequences of length {self.K}. Feature dim: {self.M}")
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        return self.corrs[idx], self.actions[idx]

class SeqDT(nn.Module):
    def __init__(self, M_lags=16, d_model=128, hidden_dim=256, n_layers=2):
        super().__init__()
        
        # Feature Tokenizer: Map inputs to d_model
        # Input is (Batch, K, M)
        self.embedding = nn.Sequential(
            nn.Linear(M_lags, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # Sequence Modeler: GRU (Simpler than Transformer for short sequences K=3~5)
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, M_lags)
        )
        
    def forward(self, x):
        # x: (B, K, M)
        B, K, M = x.shape
        
        # Embed
        emb = self.embedding(x) # (B, K, d_model)
        
        # Run RNN
        # out: (B, K, hidden_dim)
        out, _ = self.rnn(emb)
        
        # Predict logits for each step
        logits = self.head(out) # (B, K, M)
        
        return logits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/dtmin_lag_seq")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Data
    dataset = LagSequenceDataset(args.data_path)
    
    # Split
    N = len(dataset)
    train_len = int(0.9 * N)
    val_len = N - train_len
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    
    # Model
    model = SeqDT(M_lags=dataset.M, d_model=128, hidden_dim=256).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    logger.info("Starting Sequence Training...")
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device) 
            batch_y = batch_y.to(device) # (B, K)
            
            # Forward
            logits = model(batch_x) # (B, K, M)
            
            # Flatten for Loss
            # logits: (B*K, M), target: (B*K)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), batch_y.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=2) # (B, K)
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
                preds = torch.argmax(logits, dim=2)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.numel()
                
        val_acc = val_correct / val_total
        logger.info(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Path(args.out_dir) / "dt_lag_seq_best.pth")
            
    logger.info(f"Best Val Acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
