import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
import logging
import random
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def grad_norm(module: nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        total += float(p.grad.detach().norm().item())
    return total

class LagSequenceDataset(Dataset):
    def __init__(
        self,
        pt_path,
        freq_idx_range=None,
        rtg_dim: int = 2,
        rtg_mode: str = "target1",
        target_return: float = 1.0,
        lambda_c_values: str | None = None,
        use_stop_action: bool = False,
        rtg1_mode: str = "seq_len",
        rtg1_max_k: int | None = None,
    ):
        raw_data = torch.load(pt_path, weights_only=False)
        self.seq_corrs = []
        self.seq_actions = []
        self.seq_rtg = [] # Return-to-Go
        self.seq_freqs = [] # Frequency Indices
        self.flat_corrs = []
        self.flat_actions = []
        self.flat_rtgs = []
        self.flat_freqs = []

        if rtg_dim not in (1, 2):
            raise ValueError("rtg_dim must be 1 or 2")
        if rtg_mode not in ("teacher_final", "target1", "lambda_cost"):
            raise ValueError("rtg_mode must be one of: teacher_final, target1, lambda_cost")
        self.rtg_dim = rtg_dim
        self.rtg_mode = rtg_mode
        self.target_return = float(target_return)
        self.use_stop_action = bool(use_stop_action)
        self.lambda_c_values = lambda_c_values
        self.rtg1_mode = rtg1_mode
        self.rtg1_max_k = rtg1_max_k
        self._logc_min = None
        self._logc_max = None
        if self.rtg_mode == "lambda_cost":
            if not self.lambda_c_values:
                raise ValueError("lambda_cost requires --lambda_c_values")
            if self.rtg1_mode not in ("seq_len", "max_k"):
                raise ValueError("rtg1_mode must be one of: seq_len, max_k")
            if self.rtg1_mode == "max_k" and self.rtg1_max_k is None:
                raise ValueError("rtg1_mode=max_k requires --rtg1_max_k")
            c_vals = [float(x) for x in str(self.lambda_c_values).split(",") if x.strip() != ""]
            if not c_vals:
                raise ValueError("lambda_cost requires non-empty --lambda_c_values")
            logc = [np.log10(c) for c in c_vals]
            self._logc_min = min(logc)
            self._logc_max = max(logc)

        logger.info(f"Loading sequence data with RTG from {pt_path}...")
        for block in raw_data:
            # block["corrs"]: (F, K, M)
            c = block["corrs"].float()
            a = block["actions"].long()
            red = block.get("reductions", None)
            final_target = block.get("scores", None)
            valid_len = block.get("valid_len", None)
            lambda_c = block.get("lambda_c", None)
            
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

            if c.shape[0] == 0:
                continue

            red = red.float()
            final_target = final_target.float()
            
            if self.rtg_mode == "lambda_cost":
                if valid_len is None or lambda_c is None:
                    raise ValueError("lambda_cost requires 'valid_len' and 'lambda_c' in trajectories.")
                valid_len = valid_len.long()
                lambda_c = lambda_c.float()

                F, K, M = c.shape
                stop_id = M
                for f in range(F):
                    L = int(valid_len[f].item())
                    if L <= 0:
                        continue

                    if self.use_stop_action and L < K:
                        corr_seq = c[f, : L + 1]
                        act_seq = torch.cat([a[f, :L], torch.tensor([stop_id], dtype=a.dtype)], dim=0)
                    else:
                        corr_seq = c[f, :L]
                        act_seq = a[f, :L]

                    c_val = float(lambda_c[f].item())
                    if self._logc_max == self._logc_min:
                        rtg0 = 0.0
                    else:
                        rtg0 = (np.log10(c_val) - self._logc_min) / (self._logc_max - self._logc_min)
                    rtg0 = float(np.clip(rtg0, 0.0, 1.0))

                    seq_len = corr_seq.shape[0]
                    if self.rtg_dim == 2:
                        if self.rtg1_mode == "max_k":
                            denom = max(int(self.rtg1_max_k), 1)
                            remaining = (
                                torch.arange(seq_len, dtype=torch.float32).mul(-1).add(denom) / float(denom)
                            ).clamp(min=0.0)
                        else:
                            remaining = (
                                torch.arange(seq_len, dtype=torch.float32).mul(-1).add(seq_len) / max(seq_len, 1)
                            )
                        rtg_seq = torch.stack([torch.full((seq_len,), rtg0), remaining], dim=1)
                    else:
                        rtg_seq = torch.full((seq_len,), rtg0)

                    self.flat_corrs.append(corr_seq)
                    self.flat_actions.append(act_seq)
                    self.flat_rtgs.append(rtg_seq)
                    self.flat_freqs.append(f_ids[f])
                continue

            F, K, M = c.shape
            if self.rtg_dim == 2:
                rtgs = torch.zeros(F, K, 2)
            else:
                rtgs = torch.zeros(F, K)

            remaining_steps = (
                (torch.arange(K, dtype=torch.float32).mul(-1).add(K) / max(K, 1))
                .view(1, K)
                .expand(F, -1)
            )

            for k in range(K):
                if k == 0:
                    prev_red = 0.0
                else:
                    prev_red = red[:, k - 1]

                if self.rtg_mode == "teacher_final":
                    gap = final_target - prev_red
                else:
                    # Align with eval: rtg0 == remaining energy ratio == (1 - reduction_so_far)
                    gap = self.target_return - prev_red

                if self.rtg_dim == 2:
                    rtgs[:, k, 0] = gap
                    rtgs[:, k, 1] = remaining_steps[:, k]
                else:
                    rtgs[:, k] = gap

            self.seq_corrs.append(c)
            self.seq_actions.append(a)
            self.seq_rtg.append(rtgs)
            self.seq_freqs.append(f_ids)

            
        if self.rtg_mode != "lambda_cost":
            if len(self.seq_corrs) == 0:
                raise ValueError("No valid data with 'reductions' found. Regenerate data.")
        
        # NOTE: Do NOT stack here easily because K varies!
        # Instead, we flatten into a list of single-frequency samples
        # Each sample is (K, M)
        
        if self.rtg_mode != "lambda_cost":
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
        if self.N == 0:
            raise ValueError("No valid sequences found. Regenerate data or adjust filters.")
        if self.N > 0:
            self.M = self.flat_corrs[0].shape[-1]
        else:
            self.M = 0
        self.action_dim = self.M + (1 if self.use_stop_action else 0)
            
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
    def __init__(
        self,
        M_lags=16,
        d_model=128,
        hidden_dim=256,
        n_layers=2,
        max_freq=1025,
        rtg_dim: int = 2,
        action_dim: int | None = None,
    ):
        super().__init__()
        
        if rtg_dim not in (1, 2):
            raise ValueError("rtg_dim must be 1 or 2")
        self.rtg_dim = rtg_dim

        # RTG Embedding
        self.rtg_embed = nn.Linear(rtg_dim, d_model)
        
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
        out_dim = M_lags if action_dim is None else int(action_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )
        
    def forward(self, x, rtg, freq_idx, mask=None):
        # x: (B, K, M)
        # rtg: (B, K) or (B, K, rtg_dim)
        # freq_idx: (B,)
        # mask: (B, K)
        B, K, M = x.shape
        
        # Apply Input Normalization
        x = self.corr_norm(x)
        
        s_emb = self.state_embed(x) # (B, K, d)
        if rtg.dim() == 2:
            rtg_in = rtg.unsqueeze(-1)
        else:
            rtg_in = rtg
        r_emb = self.rtg_embed(rtg_in) # (B, K, d)
        
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
    parser.add_argument("--rtg_dim", type=int, default=2, help="RTG dimension: 1 or 2")
    parser.add_argument(
        "--rtg_mode",
        type=str,
        default="target1",
        choices=["teacher_final", "target1", "lambda_cost"],
        help="RTG0 definition: teacher_final uses per-sample OMP final score; target1 uses fixed target_return; lambda_cost uses normalized log10(lambda_c).",
    )
    parser.add_argument(
        "--lambda_c_values",
        type=str,
        default="",
        help="Comma-separated lambda scaling values used for normalization when rtg_mode=lambda_cost.",
    )
    parser.add_argument(
        "--target_return",
        type=float,
        default=1.0,
        help="Target return used when rtg_mode=target1. Typically 1.0 (full reduction).",
    )
    parser.add_argument(
        "--rtg1_mode",
        type=str,
        default="seq_len",
        choices=["seq_len", "max_k"],
        help="RTG1 definition for lambda_cost: seq_len uses per-sample length; max_k uses a fixed global budget.",
    )
    parser.add_argument(
        "--rtg1_max_k",
        type=int,
        default=None,
        help="Global K_max used when rtg1_mode=max_k (required in that mode).",
    )
    parser.add_argument("--use_stop_action", action="store_true", help="Enable STOP token (action_dim=M+1).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for split and training")
    args = parser.parse_args()
    
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    set_seed(int(args.seed))

    # Dataset
    f_range = None
    if args.freq_range:
        parts = args.freq_range.split(",")
        f_range = (int(parts[0]), int(parts[1]))
    
    dataset = LagSequenceDataset(
        args.data_path,
        freq_idx_range=f_range,
        rtg_dim=int(args.rtg_dim),
        rtg_mode=str(args.rtg_mode),
        target_return=float(args.target_return),
        lambda_c_values=str(args.lambda_c_values),
        use_stop_action=bool(args.use_stop_action),
        rtg1_mode=str(args.rtg1_mode),
        rtg1_max_k=(int(args.rtg1_max_k) if args.rtg1_max_k is not None else None),
    )
    
    # Train/Val Split
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_fn)
    
    # Model
    action_dim = dataset.M + (1 if bool(args.use_stop_action) else 0)
    model = SeqDT_FreqAware(M_lags=dataset.M, rtg_dim=int(args.rtg_dim), action_dim=action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # CE Loss with ignore_index to handle padding
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    best_loss = float("inf")
    grad_history = {
        "rtg_embed": [],
        "state_embed": [],
        "freq_embed": [],
    }
    
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
            grad_history["rtg_embed"].append(grad_norm(model.rtg_embed))
            grad_history["state_embed"].append(grad_norm(model.state_embed))
            grad_history["freq_embed"].append(grad_norm(model.freq_embed))
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

    # Diagnostics: RTG ablation on one val batch
    model.eval()
    ablation = {"base_loss": None, "shuffle_rtg_loss": None, "zero_rtg_loss": None}
    try:
        bx, ba, br, bf, mask = next(iter(val_loader))
        bx, ba, br, bf = bx.to(device), ba.to(device), br.to(device), bf.to(device)
        with torch.no_grad():
            logits = model(bx, br, bf, mask)
            base_loss = criterion(logits.reshape(-1, logits.shape[-1]), ba.reshape(-1)).item()
            ablation["base_loss"] = float(base_loss)

            # Shuffle RTG across batch
            idx = torch.randperm(br.shape[0])
            br_shuf = br[idx]
            logits_shuf = model(bx, br_shuf, bf, mask)
            shuf_loss = criterion(logits_shuf.reshape(-1, logits_shuf.shape[-1]), ba.reshape(-1)).item()
            ablation["shuffle_rtg_loss"] = float(shuf_loss)

            # Zero RTG
            br_zero = torch.zeros_like(br)
            logits_zero = model(bx, br_zero, bf, mask)
            zero_loss = criterion(logits_zero.reshape(-1, logits_zero.shape[-1]), ba.reshape(-1)).item()
            ablation["zero_rtg_loss"] = float(zero_loss)
    except Exception as e:
        logger.warning(f"RTG ablation diagnostics failed: {e}")

    # Write diagnostics JSON
    diag_dir = Path(args.out_dir).parent / "train"
    diag_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = {
        "grad_norm_mean": {
            "rtg_embed": float(np.mean(grad_history["rtg_embed"])) if grad_history["rtg_embed"] else None,
            "state_embed": float(np.mean(grad_history["state_embed"])) if grad_history["state_embed"] else None,
            "freq_embed": float(np.mean(grad_history["freq_embed"])) if grad_history["freq_embed"] else None,
        },
        "rtg_ablation": ablation,
        "rtg_mode": args.rtg_mode,
        "rtg_dim": int(args.rtg_dim),
        "use_stop_action": bool(args.use_stop_action),
        "rtg1_mode": str(args.rtg1_mode),
        "rtg1_max_k": (int(args.rtg1_max_k) if args.rtg1_max_k is not None else None),
        "lambda_c_values": args.lambda_c_values,
    }
    (diag_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
            
if __name__ == "__main__":
    main()
