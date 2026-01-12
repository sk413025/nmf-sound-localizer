import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scripts.h_exploration.dataset_lag import DoALagDataset, create_dataloader
from scripts.h_exploration.train_dt_lag_seq import SeqDT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_model_inference(model, X_history, y_target, K_max=3, device="cpu"):
    """
    Runs the physics loop but uses Model for decision making.
    """
    model.eval()
    Tw, F = y_target.shape
    M = 16

    # Prepare Dictionary
    Dict_tensor = torch.zeros(F, Tw, M, dtype=X_history.dtype, device=device)
    for k in range(M):
        start = 16 - k
        end = 16 + Tw - k
        slice_k = X_history[start : end, :]
        Dict_tensor[:, :, k] = slice_k.T 
        
    Targets = y_target.T.unsqueeze(2) # (F, Tw, 1)
    Residuals = Targets.clone()
    
    Active_Sets = torch.zeros(F, K_max, dtype=torch.long, device=device) - 1
    # We maintain a "Sequence History" for the model
    # Model expects (Batch, K_so_far, M)
    # But since we run autoregressively step by step, 
    # at step k, we need to pass a sequence of length k+1?
    # No, our trained SeqDT takes (B, K_max, M) usually or we can pass (B, Current_Seq_Len, M).
    # The GRU can handle variable lengths if trained so, OR we just pad.
    # But our `train_dt_lag_seq.py` trained on fixed length sequences of K=3 (or whatever was generated).
    # Let's assume the model can take partial sequences if we just run it forward.
    # Actually, simpler: We construct the full input tensor step by step.
    
    # Storage for inputs to model
    verify_corrs = [] # List of (F, M)
    
    # Initial Calculation
    # Safe fallback for MPS complex norm
    initial_norms = torch.norm(Targets.abs(), dim=1).squeeze()
    
    # State for RNN (if we were stateless we'd pass full history every time)
    # Our SeqDT re-runs the whole sequence every forward pass (it's not stateful-cached).
    # So we build the history buffer.
    
    Corrs_History = torch.zeros(F, K_max, M, device=device)
    
    for k in range(K_max):
        # 1. Physics: Calc Correlations
        Norms = torch.norm(Dict_tensor.abs(), dim=1, keepdim=True) + 1e-8
        Dict_Norm = Dict_tensor / Norms
        
        Corr_Vals = torch.bmm(Dict_Norm.conj().transpose(1, 2), Residuals)
        Abs_Corrs = torch.abs(Corr_Vals).squeeze(2) # (F, M)
        
        # Mask previous selections
        if k > 0:
            for b in range(F):
                prev = Active_Sets[b, :k]
                valid_prev = prev[prev >= 0]
                Abs_Corrs[b, valid_prev] = -1.0 # Mask out
        
        # Store in history
        Corrs_History[:, k, :] = Abs_Corrs
        
        # 2. Model Inference
        # Input to model: the history so far? 
        # Train protocol: Model saw full sequence of K items.
        # Eval protocol: At step k, we have k+1 items.
        # We can feed `Corrs_History[:, :k+1, :]`
        
        input_seq = Corrs_History[:, :k+1, :] # (F, k+1, M)
        
        with torch.no_grad():
            logits_seq = model(input_seq) # (F, k+1, M)
            # We only care about the last output
            last_logits = logits_seq[:, -1, :] # (F, M)
            
        # 3. Action Selection
        # Masking in logits? The model should learn to output low logits for masked inputs (if -1 input).
        # But to be safe, we can enforce mask again.
        if k > 0:
            for b in range(F):
                prev = Active_Sets[b, :k]
                valid_prev = prev[prev >= 0]
                last_logits[b, valid_prev] = -1e9
                
        Actions = torch.argmax(last_logits, dim=1) # (F,)
        Active_Sets[:, k] = Actions
        
        # 4. Physics: Projection & Update
        for b in range(F):
            indices = Active_Sets[b, :k+1]
            A_active = Dict_tensor[b, :, indices] 
            y_b = Targets[b, :, 0]
            
            # Robust LS
            try:
                h = torch.linalg.lstsq(A_active, y_b).solution.flatten()
                recon = A_active @ h
                Residuals[b, :, 0] = y_b - recon
            except:
                pass # Singular matrix?
                
    final_norms = torch.norm(Residuals.abs(), dim=1).squeeze()
    reduction = (initial_norms - final_norms) / (initial_norms + 1e-6)
    return reduction

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--mic_root", type=str, default="/Users/sbplab/LDV-data-processed/speech260_original_16k_no_edge_sync_vad_normalized")
    parser.add_argument("--ldv_root", type=str, default="/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized")
    parser.add_argument("--max_items", type=int, default=50) # Eval on 50 items
    args = parser.parse_args()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load Model
    # Need to know M from somewhere, assumed 16
    model = SeqDT(M_lags=16, d_model=128, hidden_dim=256)
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    model.to(device)
    model.eval()
    
    dataset = DoALagDataset(args.mic_root, args.ldv_root, angle=90.0)
    indices = list(range(args.max_items))
    dataset = torch.utils.data.Subset(dataset, indices)
    loader = create_dataloader(dataset, batch_size=1)
    
    all_reductions = []
    
    logger.info("Running Model Evaluation Loop...")
    
    Tw = 16
    MaxLag = 16
    stride = 32
    
    for i, batch in tqdm(enumerate(loader)):
        mic_stft = batch["mic_stft"][0].to(device)
        ldv_stft = batch["ldv_stft"][0].to(device)
        
        T_full, F_bins = mic_stft.shape
        
        # Batching: We will run all T windows for this clip in one batch?
        # F is 257. Each window has 257 independent problems.
        # We process one window at a time (Batch=257).
        
        for t in range(MaxLag, T_full - Tw, stride):
            X_chunk = mic_stft[t-MaxLag : t+Tw]
            Y_chunk = ldv_stft[t : t+Tw]
            
            if X_chunk.shape[0] < MaxLag+Tw or Y_chunk.shape[0] < Tw:
                continue
                
            red = run_model_inference(model, X_chunk, Y_chunk, K_max=3, device=device)
            all_reductions.append(red.detach().cpu())
            
    all_reductions = torch.cat(all_reductions)
    avg_red = all_reductions.mean().item()
    logger.info(f"Model Average Energy Reduction: {avg_red*100:.2f}%")
    
    # Compare with Baseline (Static Lag? or just report this number)
    # The user knows 35% is baseline, 74% is Oracle.
    # We hope for > 50%.

if __name__ == "__main__":
    main()
