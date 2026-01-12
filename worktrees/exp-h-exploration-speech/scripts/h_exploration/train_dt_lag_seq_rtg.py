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
    def __init__(self, pt_path, freq_idx_range=None):
        raw_data = torch.load(pt_path, weights_only=False)
        self.seq_corrs = []
        self.seq_actions = []
        self.seq_rtg = [] # Return-to-Go
        
        logger.info(f"Loading sequence data with RTG from {pt_path}...")
        for block in raw_data:
            # block["corrs"]: (F, K, M)
            c = block["corrs"].float()
            a = block["actions"].long()
            red = block.get("reductions", None)
            final_target = block.get("scores", None)
            
            if red is None or final_target is None:
                continue
                
            # Filter Frequencies if requested
            if freq_idx_range is not None:
                start, end = freq_idx_range
                # Ensure range is valid
                if start >= c.shape[0]: 
                    logger.warning(f"Freq Filter start {start} >= F {c.shape[0]}")
                    continue
                end = min(end, c.shape[0])
                c = c[start:end]
                a = a[start:end]
                red = red[start:end]
                final_target = final_target[start:end]
                if c.shape[0] == 0: continue

            red = red.float()
            final_target = final_target.float()
            
            F, K, M = c.shape
            rtgs = torch.zeros(F, K)
            
            for k in range(K):
                if k == 0:
                    prev_red = 0.0
                else:
                    prev_red = red[:, k-1]
                rtgs[:, k] = final_target - prev_red
                
            self.seq_corrs.append(c)
            self.seq_actions.append(a)
            self.seq_rtg.append(rtgs)

            
        if len(self.seq_corrs) == 0:
            raise ValueError("No valid data with 'reductions' found. Regenerate data.")
            
        self.corrs = torch.cat(self.seq_corrs, dim=0)   # (N_total, K, M)
        self.actions = torch.cat(self.seq_actions, dim=0) # (N_total, K)
        self.rtgs = torch.cat(self.seq_rtg, dim=0)    # (N_total, K)
        
        self.N, self.K, self.M = self.corrs.shape
        logger.info(f"Loaded {self.N} sequences. Feature dim: {self.M}")
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        return self.corrs[idx], self.actions[idx], self.rtgs[idx]

class SeqDT_RTG(nn.Module):
    def __init__(self, M_lags=16, d_model=128, hidden_dim=256, n_layers=2):
        super().__init__()
        
        # RTG Embedding: Simple linear projection of scaler
        self.rtg_embed = nn.Linear(1, d_model)
        
        # State Embedding
        self.state_embed = nn.Linear(M_lags, d_model)
        
        # Input Normalization (Critical for small correlation values vs RTG~1.0)
        self.corr_norm = nn.LayerNorm(M_lags)
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Sequence Modeler: GRU
        self.rnn = nn.GRU(
            input_size=d_model, # We add them? Or concat? 
            # Usual DT sums embeddings: Embed(s) + Embed(RTG) + Embed(t)
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
        
    def forward(self, x, rtg):
        # x: (B, K, M)
        # rtg: (B, K)
        B, K, M = x.shape
        
        # Apply Input Normalization
        x = self.corr_norm(x)
        
        s_emb = self.state_embed(x) # (B, K, d)
        r_emb = self.rtg_embed(rtg.unsqueeze(-1)) # (B, K, 1) -> (B, K, d)
        
        # Combine: Additive is standard for transformers/DT
        emb = self.layer_norm(s_emb + r_emb)
        
        # RNN
        out, _ = self.rnn(emb)
        
        logits = self.head(out)
        return logits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/dtmin_lag_rtg")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freq_range", type=str, default=None, help="Start,End bin indices e.g. 50,60")
    args = parser.parse_args()
    
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Parse freq range
    f_range = None
    if args.freq_range:
        parts = args.freq_range.split(",")
        f_range = (int(parts[0]), int(parts[1]))
        logger.info(f"Training on filtered Frequency Range: {f_range}")

    try:
        dataset = LagSequenceDataset(args.data_path, freq_idx_range=f_range)
    except ValueError as e:
        logger.error(e)
        return
    
    N = len(dataset)
    train_len = int(0.9 * N)
    val_len = N - train_len
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    
    model = SeqDT_RTG(M_lags=dataset.M, d_model=128, hidden_dim=256).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    logger.info("Starting RTG Training...")
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y, batch_rtg in train_loader:
            batch_x = batch_x.to(device) 
            batch_y = batch_y.to(device)
            batch_rtg = batch_rtg.to(device)
            
            logits = model(batch_x, batch_rtg)
            
            loss = criterion(logits.reshape(-1, logits.shape[-1]), batch_y.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=2)
            correct += (preds == batch_y).sum().item()
            total += batch_y.numel()
            
        train_acc = correct / total
        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_rtg in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_rtg = batch_rtg.to(device)
                
                logits = model(batch_x, batch_rtg)
                preds = torch.argmax(logits, dim=2)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.numel()
                
        val_acc = val_correct / val_total
        logger.info(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Path(args.out_dir) / "dt_lag_rtg_best.pth")
            
    logger.info(f"Best Val Acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
