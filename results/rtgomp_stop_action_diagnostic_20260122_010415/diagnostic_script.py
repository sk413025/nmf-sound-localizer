import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from scripts.h_exploration.dataset_lag import DoALagDataset

RUN_DIR = os.environ["RUN_DIR"]
MIC_ROOT = os.environ["MIC_ROOT"]
LDV_ROOT = os.environ["LDV_ROOT"]

CKPT = "results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth"
HOP_LENGTH = 160
MAX_LAG = 50
MAX_K = 16
TW = 32
GAIN = 100.0
RTG_DIM = 2
NUM_CLIPS = 3

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
            nn.Linear(d_model, M_lags + 1),
        )

    def forward(self, x, rtg, freq_idx, mask=None):
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


def eval_clip_stop_steps(model, dataset, idx, device):
    Lag_Min = -MAX_LAG
    Lag_Max = MAX_LAG
    Lags = torch.arange(Lag_Min, Lag_Max + 1, device=device)
    M_lags = len(Lags)
    stop_id = M_lags

    item = dataset[idx]
    mic_stft = item["mic_stft"].to(device)
    ldv_stft = item["ldv_stft"].to(device)

    if GAIN != 1.0:
        mic_stft = mic_stft * float(GAIN)
        ldv_stft = ldv_stft * float(GAIN)

    T_total, F = ldv_stft.shape
    start_limit = Lag_Max
    end_limit = T_total - TW + Lag_Min
    if end_limit <= start_limit:
        return []

    valid_starts = range(start_limit, end_limit, TW)
    if len(valid_starts) == 0:
        return []

    freq_ids = torch.arange(F, device=device)
    model.eval()

    stop_steps_all = []

    for start_t in valid_starts:
        end_t = start_t + TW
        y_block = ldv_stft[start_t:end_t, :].T

        D = torch.zeros(F, TW, M_lags, dtype=mic_stft.dtype, device=device)
        for m, k_lag in enumerate(Lags):
            s = int(start_t - k_lag)
            e = int(s + TW)
            D[:, :, m] = mic_stft[s:e, :].T

        norms = manual_complex_norm(D, dim=1).unsqueeze(1) + 1e-8
        D_norm = D / norms

        Y = y_block.unsqueeze(2)
        initial_energy = manual_complex_norm(Y.squeeze(), dim=1) ** 2

        res_dt = Y.clone()
        mask_dt = torch.zeros(F, M_lags, dtype=torch.bool, device=device)
        stop_mask = torch.zeros(F, dtype=torch.bool, device=device)
        active_indices = [[] for _ in range(F)]

        D_cpu = D.cpu()
        Y_cpu = Y.cpu()
        res_dt_cpu = Y_cpu.squeeze(2).clone()

        stop_steps = np.full(F, MAX_K + 1, dtype=np.int64)

        for k in range(MAX_K):
            corrs_dt = torch.bmm(D_norm.conj().transpose(1, 2), res_dt).squeeze(2)
            abs_corrs_dt = torch.abs(corrs_dt)
            abs_corrs_dt[mask_dt] = -1.0

            cur_E_dt = manual_complex_norm(res_dt.squeeze(), dim=1) ** 2
            vals = cur_E_dt / (initial_energy + 1e-6)
            if int(RTG_DIM) == 2:
                remaining = float(MAX_K - k) / max(float(MAX_K), 1.0)
                rtg_in = torch.stack([vals, torch.full_like(vals, remaining)], dim=-1)
            else:
                rtg_in = vals

            logits = model(abs_corrs_dt.unsqueeze(1), rtg_in.unsqueeze(1), freq_ids).squeeze(1)
            logits[:, :M_lags][mask_dt] = -float("inf")
            act_dt = torch.argmax(logits, dim=1)
            act_dt = torch.where(stop_mask, torch.full_like(act_dt, stop_id), act_dt)

            for f in range(F):
                if stop_mask[f]:
                    continue
                a = int(act_dt[f].item())
                if a == stop_id:
                    stop_mask[f] = True
                    stop_steps[f] = k + 1
                else:
                    mask_dt[f, a] = True
                    active_indices[f].append(a)

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
                res_dt_cpu[f] = y_f.squeeze(1) - recon_f

            res_dt = res_dt_cpu.to(device).unsqueeze(2)

        stop_steps_all.extend(stop_steps.tolist())

    return stop_steps_all


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = DoALagDataset(
        MIC_ROOT,
        LDV_ROOT,
        angle=None,
        hop_length=int(HOP_LENGTH),
        freq_min=0,
        freq_max=8000,
    )
    print(f"Dataset Size: {len(dataset)}")

    M_lags = MAX_LAG * 2 + 1
    model = SeqDT_FreqAware(M_lags=M_lags, d_model=128, hidden_dim=256, rtg_dim=int(RTG_DIM)).to(device)
    state = torch.load(CKPT, map_location=device)
    head_out = state["head.2.weight"].shape[0]
    if head_out != M_lags + 1:
        raise SystemExit(f"Checkpoint head_out mismatch. Expected {M_lags + 1}, got {head_out}.")
    model.load_state_dict(state)
    print("Model Loaded.")

    n = min(NUM_CLIPS, len(dataset))
    all_stop_steps = []
    for i in tqdm(range(n)):
        all_stop_steps.extend(eval_clip_stop_steps(model, dataset, i, device))

    arr = np.array(all_stop_steps, dtype=np.int64)
    total = int(arr.size)
    no_stop = int(np.sum(arr > MAX_K))

    ks = [1, 2, 4, 8, 16]
    stop_rate = {int(k): float(np.mean(arr <= k)) for k in ks}

    hist = {str(k): int(np.sum(arr == k)) for k in range(1, MAX_K + 1)}
    hist["no_stop"] = no_stop

    stats = {
        "total_freq_windows": total,
        "max_k": MAX_K,
        "stop_rate_by_k": stop_rate,
        "no_stop_rate": float(no_stop / total) if total else None,
        "stop_step_mean": float(arr.mean()) if total else None,
        "stop_step_median": float(np.median(arr)) if total else None,
        "stop_step_p90": float(np.quantile(arr, 0.9)) if total else None,
        "histogram": hist,
    }

    out = {
        "config": {
            "ckpt": CKPT,
            "mic_root": MIC_ROOT,
            "ldv_root": LDV_ROOT,
            "hop_length": HOP_LENGTH,
            "max_lag": MAX_LAG,
            "tw": TW,
            "max_k": MAX_K,
            "gain": GAIN,
            "rtg_dim": RTG_DIM,
            "num_clips": n,
        },
        "stats": stats,
    }

    out_path = Path(RUN_DIR) / "summary" / "stop_diagnostics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    print("STOP diagnostic summary:")
    print(f"total_freq_windows={total}")
    print(f"no_stop_rate={stats['no_stop_rate']:.4f}")
    for k in ks:
        print(f"stop_rate<=K={k}: {stop_rate[k]:.4f}")
    print(f"stop_step_mean={stats['stop_step_mean']:.2f}")
    print(f"stop_step_median={stats['stop_step_median']:.2f}")
    print(f"stop_step_p90={stats['stop_step_p90']:.2f}")


if __name__ == "__main__":
    main()
