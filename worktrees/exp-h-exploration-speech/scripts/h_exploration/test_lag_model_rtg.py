import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scripts.h_exploration.dataset_lag import DoALagDataset, create_dataloader
from scripts.h_exploration.train_dt_lag_seq_rtg import SeqDT_RTG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_rtg_inference(model, X_history, y_target, K_max=3, device="cpu", target_return=1.0):
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
        
    Targets = y_target.T.unsqueeze(2)
    Residuals = Targets.clone()
    
    Active_Sets = torch.zeros(F, K_max, dtype=torch.long, device=device) - 1
    
    # RTG State
    # RTG[t] = Desired_Total - Reduction_So_Far
    # Initially: RTG[0] = Desired_Total
    current_rtg = torch.ones(F, 1, device=device) * target_return
    
    initial_norms = torch.norm(Targets.abs(), dim=1).squeeze()
    
    Corrs_History = torch.zeros(F, K_max, M, device=device)
    RTG_History = torch.zeros(F, K_max, device=device)
    
    for k in range(K_max):
        # 1. Physics
        Norms = torch.norm(Dict_tensor.abs(), dim=1, keepdim=True) + 1e-8
        Dict_Norm = Dict_tensor / Norms
        Corr_Vals = torch.bmm(Dict_Norm.conj().transpose(1, 2), Residuals)
        Abs_Corrs = torch.abs(Corr_Vals).squeeze(2) 
        
        if k > 0:
            for b in range(F):
                prev = Active_Sets[b, :k]
                valid_prev = prev[prev >= 0]
                Abs_Corrs[b, valid_prev] = -1.0
        
        Corrs_History[:, k, :] = Abs_Corrs
        
        # Update RTG History for model input
        # RTG_History[:, k] = current_rtg[:, 0]
        # Wait, if we use a sequence model, we need to correct the RTG history?
        # In DT, RTG input at step t is the rtg *at that step*.
        RTG_History[:, k] = current_rtg.squeeze()
        
        input_seq = Corrs_History[:, :k+1, :]
        rtg_input = RTG_History[:, :k+1]
        
        # 2. Model
        with torch.no_grad():
            logits_seq = model(input_seq, rtg_input)
            last_logits = logits_seq[:, -1, :]
            
        if k > 0:
            for b in range(F):
                prev = Active_Sets[b, :k]
                valid_prev = prev[prev >= 0]
                last_logits[b, valid_prev] = -1e9
                
        Actions = torch.argmax(last_logits, dim=1)
        Active_Sets[:, k] = Actions
        
        # 3. Physics Update
        for b in range(F):
            indices = Active_Sets[b, :k+1]
            A_active = Dict_tensor[b, :, indices] 
            y_b = Targets[b, :, 0]
            try:
                # CPU fallback for lstsq if needed
                if device.type == 'mps':
                     h = torch.linalg.lstsq(A_active.cpu(), y_b.cpu()).solution.flatten().to(device)
                else: 
                     h = torch.linalg.lstsq(A_active, y_b).solution.flatten()
                
                recon = A_active @ h
                Residuals[b, :, 0] = y_b - recon
            except:
                pass 
                
        # 4. Update RTG for next step
        # RTG_{t+1} = RTG_t - reward_t
        # Reward_t = Reduction_after - Reduction_before
        # Reduction = (Init - Curr) / Init
        # So RTG_{t+1} = (Target - Red_{t+1})
        current_norms_val = torch.norm(Residuals.abs(), dim=1).squeeze()
        current_reduction = (initial_norms - current_norms_val) / (initial_norms + 1e-6)
        
        # New RTG = Target - Current Reduction
        # e.g. Target=1.0. Current=0.2. New RTG=0.8.
        current_rtg = (target_return - current_reduction).unsqueeze(1).clamp(min=0.0)
        
    final_norms = torch.norm(Residuals.abs(), dim=1).squeeze()
    reduction = (initial_norms - final_norms) / (initial_norms + 1e-6)
    return reduction

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--mic_root", type=str, default="/Users/sbplab/LDV-data-processed/speech260_original_16k_no_edge_sync_vad_normalized")
    parser.add_argument("--ldv_root", type=str, default="/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized")
    parser.add_argument("--max_items", type=int, default=50) 
    parser.add_argument("--target_rtg", type=float, default=1.0)
    args = parser.parse_args()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model = SeqDT_RTG(M_lags=16, d_model=128, hidden_dim=256)
    try:
        model.load_state_dict(torch.load(args.model_path, weights_only=True))
    except: 
        logger.warning("Strict loading failed, trying sloppy")
        model.load_state_dict(torch.load(args.model_path, weights_only=True, map_location=device))
        
    model.to(device)
    model.eval()
    
    dataset = DoALagDataset(args.mic_root, args.ldv_root, angle=90.0)
    indices = list(range(args.max_items))
    dataset = torch.utils.data.Subset(dataset, indices)
    loader = create_dataloader(dataset, batch_size=1)
    
    all_reductions = []
    logger.info(f"Running RTG Inference (Target={args.target_rtg})...")
    
    Tw = 16
    MaxLag = 16
    stride = 32
    
    for i, batch in tqdm(enumerate(loader)):
        mic_stft = batch["mic_stft"][0].to(device)
        ldv_stft = batch["ldv_stft"][0].to(device)
        T_full = mic_stft.shape[0]
        
        for t in range(MaxLag, T_full - Tw, stride):
            X_chunk = mic_stft[t-MaxLag : t+Tw]
            Y_chunk = ldv_stft[t : t+Tw]
            if X_chunk.shape[0] < MaxLag+Tw or Y_chunk.shape[0] < Tw: continue
            
            red = run_rtg_inference(model, X_chunk, Y_chunk, K_max=3, device=device, target_return=args.target_rtg)
            all_reductions.append(red.detach().cpu())
            
    all_reductions = torch.cat(all_reductions)
    avg_red = all_reductions.mean().item()
    logger.info(f"RTG Model Average Energy Reduction: {avg_red*100:.2f}%")

if __name__ == "__main__":
    main()
