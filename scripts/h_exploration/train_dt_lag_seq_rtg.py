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
        self.seq_freqs = [] # Frequency Indices

        logger.info(f"Loading sequence data with RTG from {pt_path}...")
        for block in raw_data:
            # block["corrs"]: (F, K, M)
            c = block["corrs"].float()
            a = block["actions"].long()
            red = block.get("reductions", None)
            final_target = block.get("scores", None)
            
            if red is None or final_target is None:
                continue
                
            F_orig = c.shape[0]
            
            # Determine Frequency Indices involved
            if freq_idx_range is not None:
                start, end = freq_idx_range
                if start >= F_orig: 
                    logger.warning(f"Freq Filter start {start} >= F {F_orig}")
                    continue
                end = min(end, F_orig)
                
                # Slice Data
                c = c[start:end]
                a = a[start:end]
                red = red[start:end]
                final_target = final_target[start:end]
                
                # Create Indices
                f_ids = torch.arange(start, end, dtype=torch.long)
            else:
                f_ids = torch.arange(F_orig, dtype=torch.long)

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
            
            # Replicate f_ids across K is NOT needed if we just input scalar freq to Model
            # But wait, self.seq_corrs stores (F, K, M).
            # self.seq_freqs.append(f_ids) -> (F,)
            # When we concat, we get (N_total, K, M) and (N_total,)
            self.seq_freqs.append(f_ids)

            
        if len(self.seq_corrs) == 0:
            raise ValueError("No valid data with 'reductions' found. Regenerate data.")
            
        self.corrs = torch.cat(self.seq_corrs, dim=0)   # (N_total, K, M)
        self.actions = torch.cat(self.seq_actions, dim=0) # (N_total, K)
        self.rtgs = torch.cat(self.seq_rtg, dim=0)    # (N_total, K)
        self.freqs = torch.cat(self.seq_freqs, dim=0) # (N_total, )
        
        self.N, self.K, self.M = self.corrs.shape
        logger.info(f"Loaded {self.N} sequences. Feature dim: {self.M}")
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        # Return: corr, action, rtg, freq_idx
        return self.corrs[idx], self.actions[idx], self.rtgs[idx], self.freqs[idx]

class SeqDT_FreqAware(nn.Module):
    def __init__(self, M_lags=16, d_model=128, hidden_dim=256, n_layers=2, max_freq=1025):
        super().__init__()
        
        # RTG Embedding
        self.rtg_embed = nn.Linear(1, d_model)
        
        # State Embedding
        self.state_embed = nn.Linear(M_lags, d_model)

        # Frequency Embedding (New)
        self.freq_embed = nn.Embedding(max_freq, d_model)
        
        # Input Normalization
        self.corr_norm = nn.LayerNorm(M_lags)
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Sequence Modeler: GRU
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
        
    def forward(self, x, rtg, freq_idx):
        # x: (B, K, M)
        # rtg: (B, K)
        # freq_idx: (B,)
        B, K, M = x.shape
        
        # Apply Input Normalization
        x = self.corr_norm(x)
        
        s_emb = self.state_embed(x) # (B, K, d)
        r_emb = self.rtg_embed(rtg.unsqueeze(-1)) # (B, K, d)
        
        # Freq Embedding: (B, d) -> (B, 1, d) -> Expand to (B, K, d)
        f_emb = self.freq_embed(freq_idx).unsqueeze(1).expand(-1, K, -1)
        
        # Combine: Add Physics(s) + Goal(r) + Context(f)
        emb = self.layer_norm(s_emb + r_emb + f_emb)
        
        # RNN
        out, _ = self.rnn(emb)
        
        logits = self.head(out)
        return logits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/dtmin_freq_aware")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freq_range", type=str, default=None, help="Start,End bin indices e.g. 50,60")
    args = parser.parse_args()
    
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Dataset
    f_range = None
    if args.freq_range:
        parts = args.freq_range.split(",")
        f_range = (int(parts[0]), int(parts[1]))
    
    dataset = LagSequenceDataset(args.data_path, freq_idx_range=f_range)
    
    # Train/Val Split
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # Model
    model = SeqDT_FreqAware(M_lags=dataset.M).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    best_loss = float("inf")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (bx, ba, br, bf) in enumerate(train_loader):
            bx, ba, br, bf = bx.to(device), ba.to(device), br.to(device), bf.to(device)
            
            optimizer.zero_grad()
            logits = model(bx, br, bf) # (B, K, M_lags)
            
            # Flatten for CE
            # logits: (B*K, M_lags)
            # actions: (B*K)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), ba.reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # Val
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
             for bx, ba, br, bf in val_loader:
                bx, ba, br, bf = bx.to(device), ba.to(device), br.to(device), bf.to(device)
                logits = model(bx, br, bf)
                
                loss = criterion(logits.reshape(-1, logits.shape[-1]), ba.reshape(-1))
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == ba).sum().item()
                total += ba.numel()
        
        avg_val_loss = val_loss / len(val_loader)
        acc = correct / total
        
        logger.info(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Loss={avg_val_loss:.4f}, Acc={acc:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), f"{args.out_dir}/dt_freq_aware_best.pth")
            
if __name__ == "__main__":
    main()
