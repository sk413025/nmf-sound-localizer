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
        
        # NOTE: Do NOT stack here easily because K varies!
        # Instead, we flatten into a list of single-frequency samples
        # Each sample is (K, M)
        
        self.flat_corrs = []
        self.flat_actions = []
        self.flat_rtgs = []
        self.flat_freqs = []
        
        for i in range(len(self.seq_corrs)):
             # seq_corrs[i]: (F_i, K_i, M)
             # seq_actions[i]: (F_i, K_i)
             # seq_rtgs[i]: (F_i, K_i)
             # seq_freqs[i]: (F_i,)
             
             c_block = self.seq_corrs[i]
             a_block = self.seq_actions[i]
             r_block = self.seq_rtg[i]
             f_block = self.seq_freqs[i]
             
             num_freqs = c_block.shape[0]
             
             for f in range(num_freqs):
                 self.flat_corrs.append(c_block[f])   # (K_i, M)
                 self.flat_actions.append(a_block[f]) # (K_i,)
                 self.flat_rtgs.append(r_block[f])    # (K_i,)
                 self.flat_freqs.append(f_block[f])   # Scalar
        
        self.N = len(self.flat_corrs)
        if self.N > 0:
            self.M = self.flat_corrs[0].shape[-1]
        else:
            self.M = 0
            
        logger.info(f"Loaded {self.N} variable-length sequences. Feature dim: {self.M}")
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        # Return: corr, action, rtg, freq_idx
        # corr: (K, M)
        # action: (K,)
        # rtg: (K,)
        # freq: scalar
        return self.flat_corrs[idx], self.flat_actions[idx], self.flat_rtgs[idx], self.flat_freqs[idx]

def pad_collate_fn(batch):
    """
    Collate variable length sequences:
    x: (B, K_max, M)
    a: (B, K_max)
    r: (B, K_max)
    f: (B,)
    mask: (B, K_max) - 1 for valid, 0 for pad
    """
    batch_corrs = [b[0] for b in batch]
    batch_actions = [b[1] for b in batch]
    batch_rtgs = [b[2] for b in batch]
    batch_freqs = torch.stack([b[3] for b in batch])
    
    # Pad sequences
    # pad_sequence expects list of (L, *) -> (B, L, *) if batch_first=True
    
    # Corrs: Pad with 0
    x_padded = torch.nn.utils.rnn.pad_sequence(batch_corrs, batch_first=True, padding_value=0.0)
    
    # Actions: Pad with -100 (standard ignore index for CE)
    a_padded = torch.nn.utils.rnn.pad_sequence(batch_actions, batch_first=True, padding_value=-100)
    
    # RTGs: Pad with 0
    r_padded = torch.nn.utils.rnn.pad_sequence(batch_rtgs, batch_first=True, padding_value=0.0)
    
    # Generate mask
    lengths = torch.tensor([len(x) for x in batch_corrs])
    max_len = x_padded.shape[1]
    
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    mask = mask.to(x_padded.device) # Will move later
    
    return x_padded, a_padded, r_padded, batch_freqs, mask


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
        
    def forward(self, x, rtg, freq_idx, mask=None):
        # x: (B, K, M)
        # rtg: (B, K)
        # freq_idx: (B,)
        # mask: (B, K)
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
        out, _ = self.rnn(emb) # (B, K, hidden)
        
        logits = self.head(out) # (B, K, M_actions)
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
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_fn)
    
    # Model
    model = SeqDT_FreqAware(M_lags=dataset.M).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # CE Loss with ignore_index to handle padding
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    best_loss = float("inf")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (bx, ba, br, bf, mask) in enumerate(train_loader):
            bx, ba, br, bf = bx.to(device), ba.to(device), br.to(device), bf.to(device)
            # mask is not strictly needed for GRU if padding is right-side and we ignore_index in loss
            
            optimizer.zero_grad()
            logits = model(bx, br, bf, mask) # (B, K, M_lags)
            
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
             for bx, ba, br, bf, mask in val_loader:
                bx, ba, br, bf = bx.to(device), ba.to(device), br.to(device), bf.to(device)
                logits = model(bx, br, bf)
                
                loss = criterion(logits.reshape(-1, logits.shape[-1]), ba.reshape(-1))
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=-1)
                
                # Accuracy masking: Only count valid positions
                valid_mask = (ba != -100)
                
                correct += (preds[valid_mask] == ba[valid_mask]).sum().item()
                total += valid_mask.sum().item()

        
        avg_val_loss = val_loss / len(val_loader)
        acc = correct / total
        
        logger.info(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Loss={avg_val_loss:.4f}, Acc={acc:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), f"{args.out_dir}/dt_freq_aware_best.pth")
            
if __name__ == "__main__":
    main()
