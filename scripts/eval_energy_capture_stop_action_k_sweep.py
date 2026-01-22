import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from scripts.h_exploration.dataset_lag import DoALagDataset


class SeqDT_FreqAware(nn.Module):
    def __init__(self, M_lags=16, d_model=128, hidden_dim=256, n_layers=2, max_freq=1025, rtg_dim: int = 2):
        super().__init__()
        if rtg_dim not in (1, 2):
            raise ValueError("rtg_dim must be 1 or 2")
        self.rtg_dim = rtg_dim

        self.rtg_embed = nn.Linear(rtg_dim, d_model)
        self.state_embed = nn.Linear(M_lags, d_model)
        self.freq_embed = nn.Embedding(max_freq, d_model)
        self.corr_norm = nn.LayerNorm(M_lags)
        self.layer_norm = nn.LayerNorm(d_model)
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, M_lags + 1),  # STOP action at index M_lags
        )

    def forward(self, x, rtg, freq_idx, mask=None):
        # x: (B, K, M)
        B, K, M = x.shape
        x_emb = self.state_embed(x)
        if rtg.dim() == 2:
            rtg_in = rtg.unsqueeze(-1)
        else:
            rtg_in = rtg
        r_emb = self.rtg_embed(rtg_in)
        f_emb = self.freq_embed(freq_idx).unsqueeze(1).expand(-1, K, -1)

        h = x_emb + r_emb + f_emb
        h = self.layer_norm(h)
        out, _ = self.rnn(h)
        logits = self.head(out)
        return logits


def manual_complex_norm(tensor, dim=None, keepdim=False):
    if dim is None:
        return torch.sqrt(tensor.real.pow(2) + tensor.imag.pow(2) + 1e-10)
    return torch.sqrt(tensor.real.pow(2).sum(dim=dim, keepdim=keepdim) + tensor.imag.pow(2).sum(dim=dim, keepdim=keepdim) + 1e-10)


def gather_atoms(Dict, Ids_till_k):
    F, Tw, M = Dict.shape
    K_curr = Ids_till_k.shape[1]
    Ids_exp = Ids_till_k.unsqueeze(1).expand(-1, Tw, -1)
    return torch.gather(Dict, 2, Ids_exp)


def eval_clip_stop_action(model, dataset, idx, device, max_lag=50, max_k=16, Tw=32, gain=100.0, rtg_dim: int = 2):
    Lag_Min = -max_lag
    Lag_Max = max_lag
    Lags = torch.arange(Lag_Min, Lag_Max + 1, device=device)
    M_lags = len(Lags)
    stop_id = M_lags

    item = dataset[idx]
    mic_stft = item["mic_stft"].to(device)
    ldv_stft = item["ldv_stft"].to(device)

    if gain != 1.0:
        mic_stft = mic_stft * float(gain)
        ldv_stft = ldv_stft * float(gain)

    T_total, F = ldv_stft.shape

    start_limit = Lag_Max
    end_limit = T_total - Tw + Lag_Min
    if end_limit <= start_limit:
        return torch.zeros(0, max_k), torch.zeros(0, max_k)

    valid_starts = range(start_limit, end_limit, Tw)
    if len(valid_starts) == 0:
        return torch.zeros(0, max_k), torch.zeros(0, max_k)

    dt_energies = []
    omp_energies = []
    freq_ids = torch.arange(F, device=device)
    model.eval()

    for start_t in valid_starts:
        end_t = start_t + Tw
        y_block = ldv_stft[start_t:end_t, :].T  # (F, Tw)

        D = torch.zeros(F, Tw, M_lags, dtype=mic_stft.dtype, device=device)
        for m, k_lag in enumerate(Lags):
            s = int(start_t - k_lag)
            e = int(s + Tw)
            D[:, :, m] = mic_stft[s:e, :].T

        norms = manual_complex_norm(D, dim=1).unsqueeze(1) + 1e-8
        D_norm = D / norms

        Y = y_block.unsqueeze(2)  # (F, Tw, 1)
        initial_energy = manual_complex_norm(Y.squeeze(), dim=1) ** 2
        initial_energy_cpu = initial_energy.cpu()

        res_dt = Y.clone()
        res_omp = Y.clone()

        mask_dt = torch.zeros(F, M_lags, dtype=torch.bool, device=device)
        mask_omp = torch.zeros(F, M_lags, dtype=torch.bool, device=device)
        stop_mask = torch.zeros(F, dtype=torch.bool, device=device)

        indices_dt = torch.zeros(F, max_k, dtype=torch.long, device=device)
        indices_omp = torch.zeros(F, max_k, dtype=torch.long, device=device)

        active_indices = [[] for _ in range(F)]
        recon_dt_cpu = torch.zeros(F, Tw, dtype=Y.dtype, device="cpu")
        res_dt_cpu = Y.cpu().squeeze(2)

        accum_dt = []
        accum_omp = []

        D_cpu = D.cpu()
        Y_cpu = Y.cpu()

        for k in range(max_k):
            corrs_dt = torch.bmm(D_norm.conj().transpose(1, 2), res_dt).squeeze(2)
            abs_corrs_dt = torch.abs(corrs_dt)
            corrs_omp = torch.bmm(D_norm.conj().transpose(1, 2), res_omp).squeeze(2)
            abs_corrs_omp = torch.abs(corrs_omp)

            abs_corrs_dt[mask_dt] = -1.0
            abs_corrs_omp[mask_omp] = -1.0

            act_omp = torch.argmax(abs_corrs_omp, dim=1)

            cur_E_dt = manual_complex_norm(res_dt.squeeze(), dim=1) ** 2
            vals = cur_E_dt / (initial_energy + 1e-6)
            if int(rtg_dim) == 2:
                remaining = float(max_k - k) / max(float(max_k), 1.0)
                rtg_in = torch.stack([vals, torch.full_like(vals, remaining)], dim=-1)
            else:
                rtg_in = vals

            logits = model(abs_corrs_dt.unsqueeze(1), rtg_in.unsqueeze(1), freq_ids).squeeze(1)
            logits[:, :M_lags][mask_dt] = -float("inf")

            act_dt = torch.argmax(logits, dim=1)
            act_dt = torch.where(stop_mask, torch.full_like(act_dt, stop_id), act_dt)

            for f in range(F):
                a = int(act_dt[f].item())
                if stop_mask[f]:
                    pass
                elif a == stop_id:
                    stop_mask[f] = True
                else:
                    mask_dt[f, a] = True
                    active_indices[f].append(a)
                indices_dt[f, k] = act_dt[f]

                mask_omp[f, act_omp[f]] = True
                indices_omp[f, k] = act_omp[f]

            for f in range(F):
                if stop_mask[f]:
                    continue
                idxs = active_indices[f]
                if not idxs:
                    continue
                active_dt = D_cpu[f, :, idxs]
                y_f = Y_cpu[f]
                try:
                    sol_dt = torch.linalg.lstsq(active_dt, y_f).solution
                except Exception:
                    sol_dt = torch.linalg.pinv(active_dt) @ y_f
                recon_f = (active_dt @ sol_dt).squeeze(1)
                recon_dt_cpu[f] = recon_f
                res_dt_cpu[f] = y_f.squeeze(1) - recon_f

            res_dt = res_dt_cpu.to(device).unsqueeze(2)

            try:
                active_omp = gather_atoms(D_cpu, indices_omp[:, : k + 1].cpu())
                sol_omp = torch.linalg.lstsq(active_omp, Y_cpu).solution
            except Exception:
                sol_omp = torch.zeros(F, active_omp.shape[2], 1, dtype=torch.complex64)
                for f in range(F):
                    sol_omp[f] = torch.linalg.pinv(active_omp[f]) @ Y_cpu[f]
            recon_omp = active_omp @ sol_omp
            res_omp = (Y_cpu - recon_omp).to(device)

            E_recon_dt = manual_complex_norm(recon_dt_cpu, dim=1) ** 2
            E_recon_omp = manual_complex_norm(recon_omp.squeeze(), dim=1) ** 2

            ratio_dt = E_recon_dt / (initial_energy_cpu + 1e-6)
            ratio_omp = E_recon_omp / (initial_energy_cpu + 1e-6)

            accum_dt.append(ratio_dt)
            accum_omp.append(ratio_omp)

        dt_energies.append(torch.stack(accum_dt, dim=1))
        omp_energies.append(torch.stack(accum_omp, dim=1))

    if len(dt_energies) == 0:
        return torch.zeros(0, max_k), torch.zeros(0, max_k)

    return torch.cat(dt_energies, dim=0), torch.cat(omp_energies, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mic_root", type=str, required=True)
    parser.add_argument("--ldv_root", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/eval_capture_stop_action")
    parser.add_argument("--hop_length", type=int, default=160)
    parser.add_argument("--max_lag", type=int, default=50)
    parser.add_argument("--max_k", type=int, default=16)
    parser.add_argument("--tw", type=int, default=32)
    parser.add_argument("--gain", type=float, default=100.0)
    parser.add_argument("--rtg_dim", type=int, default=2)
    parser.add_argument("--num_clips", type=int, default=10)
    parser.add_argument("--angle", type=float, default=90.0)
    parser.add_argument("--all_angles", action="store_true")
    parser.add_argument("--use_stop_action", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps"],
        help="Device selection: auto picks mps if available else cpu.",
    )
    args = parser.parse_args()

    if not args.use_stop_action:
        raise SystemExit("use_stop_action is required for stop-action evaluator.")

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps, but MPS is not available.")
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    dataset = DoALagDataset(
        args.mic_root,
        args.ldv_root,
        angle=None if args.all_angles else float(args.angle),
        hop_length=int(args.hop_length),
        freq_min=0,
        freq_max=8000,
    )
    print(f"Dataset Size: {len(dataset)}")

    M_lags = args.max_lag * 2 + 1
    print(f"Initializing Model with M_lags={M_lags} (stop-action)")

    model = SeqDT_FreqAware(M_lags=M_lags, d_model=128, hidden_dim=256, rtg_dim=int(args.rtg_dim)).to(device)

    state = torch.load(args.ckpt_path, map_location=device)
    head_out = state["head.2.weight"].shape[0]
    if head_out != M_lags + 1:
        raise SystemExit(f"Checkpoint head_out mismatch. Expected {M_lags + 1}, got {head_out}.")
    model.load_state_dict(state)
    print("Model Loaded.")

    Total_DT = []
    Total_OMP = []

    N_clips = min(int(args.num_clips), len(dataset))
    for i in tqdm(range(N_clips)):
        dt_flat, omp_flat = eval_clip_stop_action(
            model,
            dataset,
            i,
            device,
            max_lag=args.max_lag,
            max_k=args.max_k,
            Tw=args.tw,
            gain=args.gain,
            rtg_dim=int(args.rtg_dim),
        )
        if dt_flat.shape[0] == 0:
            continue
        item = dataset[i]
        F_dim = item["mic_stft"].shape[1]
        num_windows = dt_flat.shape[0] // F_dim
        if num_windows == 0:
            continue
        dt_3d = dt_flat.reshape(num_windows, F_dim, -1)
        omp_3d = omp_flat.reshape(num_windows, F_dim, -1)

        dt_mean = dt_3d.mean(dim=0).cpu()
        omp_mean = omp_3d.mean(dim=0).cpu()
        Total_DT.append(dt_mean)
        Total_OMP.append(omp_mean)

    if len(Total_DT) == 0:
        print("No valid data collected.")
        return

    Grand_DT = torch.stack(Total_DT, dim=0).mean(dim=0)
    Grand_OMP = torch.stack(Total_OMP, dim=0).mean(dim=0)

    torch.save({"DT": Grand_DT, "OMP": Grand_OMP}, os.path.join(args.out_dir, "eval_stats.pt"))
    print(f"Wrote {Path(args.out_dir) / 'eval_stats.pt'}")


if __name__ == "__main__":
    main()
