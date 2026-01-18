import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from scripts.eval_energy_capture_generic import SeqDT_FreqAware, manual_complex_norm, gather_atoms
from scripts.h_exploration.dataset_lag import DoALagDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@torch.no_grad()
def eval_clip_rtg1_override(
    model,
    dataset,
    idx: int,
    device: torch.device,
    *,
    max_lag: int,
    max_k: int,
    tw: int,
    gain: float,
    rtg1_override: float,
):
    Lag_Min = -max_lag
    Lag_Max = max_lag
    Lags = torch.arange(Lag_Min, Lag_Max + 1, device=device)
    M_lags = len(Lags)

    item = dataset[idx]
    mic_stft = item["mic_stft"].to(device)
    ldv_stft = item["ldv_stft"].to(device)

    if gain != 1.0:
        mic_stft = mic_stft * float(gain)
        ldv_stft = ldv_stft * float(gain)

    T_total, F = ldv_stft.shape

    start_limit = Lag_Max
    end_limit = T_total - tw + Lag_Min
    if end_limit <= start_limit:
        return None

    valid_starts = range(start_limit, end_limit, tw)
    if len(list(valid_starts)) == 0:
        return None

    model.eval()

    final_ratios = []
    freq_ids = torch.arange(F, device=device)

    for start_t in valid_starts:
        end_t = start_t + tw
        y_block = ldv_stft[start_t:end_t, :].T  # (F, Tw)

        D = torch.zeros(F, tw, M_lags, dtype=mic_stft.dtype, device=device)
        for m, k_lag in enumerate(Lags):
            s = int(start_t - k_lag)
            e = int(s + tw)
            D[:, :, m] = mic_stft[s:e, :].T

        norms = manual_complex_norm(D, dim=1).unsqueeze(1) + 1e-8
        D_norm = D / norms

        Y = y_block.unsqueeze(2)
        initial_energy = manual_complex_norm(Y.squeeze(), dim=1) ** 2

        res_dt = Y.clone()
        mask_dt = torch.zeros(F, M_lags, dtype=torch.bool, device=device)
        indices_dt = torch.zeros(F, max_k, dtype=torch.long, device=device)

        for k in range(max_k):
            corrs_dt = torch.bmm(D_norm.conj().transpose(1, 2), res_dt).squeeze(2)
            abs_corrs_dt = torch.abs(corrs_dt)
            abs_corrs_dt[mask_dt] = -1.0

            cur_E_dt = manual_complex_norm(res_dt.squeeze(), dim=1) ** 2
            rtg0 = (cur_E_dt / (initial_energy + 1e-6)).clamp(0.0, 1.0)
            rtg_in = torch.stack([rtg0, torch.full_like(rtg0, float(rtg1_override))], dim=-1)

            logits = model(abs_corrs_dt.unsqueeze(1), rtg_in.unsqueeze(1), freq_ids).squeeze(1)
            logits[mask_dt] = -float("inf")

            act_dt = torch.argmax(logits, dim=1)
            mask_dt.scatter_(1, act_dt.unsqueeze(1), True)
            indices_dt[:, k] = act_dt

            active_dt = gather_atoms(D.cpu(), indices_dt[:, : k + 1].cpu())
            Y_cpu = Y.cpu()
            sol_dt = torch.linalg.lstsq(active_dt, Y_cpu).solution
            recon_dt = active_dt @ sol_dt
            res_dt = (Y_cpu - recon_dt).to(device)

        E_recon_dt = manual_complex_norm(recon_dt.squeeze(), dim=1) ** 2
        ratio_dt = (E_recon_dt / (initial_energy.cpu() + 1e-6)).clamp(0.0, 1.0)
        final_ratios.append(ratio_dt)

    all_ratios = torch.cat(final_ratios, dim=0)
    return float(all_ratios.mean().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mic_root", type=str, required=True)
    p.add_argument("--ldv_root", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--angle", type=float, default=90.0)
    p.add_argument("--all_angles", action="store_true")
    p.add_argument("--num_clips", type=int, default=3)
    p.add_argument("--hop_length", type=int, default=160, help="STFT hop length (samples)")
    p.add_argument("--max_lag", type=int, default=50)
    p.add_argument("--max_k", type=int, default=16)
    p.add_argument("--tw", type=int, default=32)
    p.add_argument("--gain", type=float, default=100.0)
    p.add_argument(
        "--rtg1_values",
        type=str,
        default="0.0,0.5,1.0",
        help="Comma-separated constants to override RTG1 (remaining_steps) during eval",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset = DoALagDataset(
        args.mic_root,
        args.ldv_root,
        angle=None if args.all_angles else float(args.angle),
        hop_length=int(args.hop_length),
        freq_min=0,
        freq_max=8000,
    )
    n = min(int(args.num_clips), len(dataset))

    M_lags = int(args.max_lag) * 2 + 1
    model = SeqDT_FreqAware(M_lags=M_lags, d_model=128, hidden_dim=256, n_layers=2, rtg_dim=2).to(device)
    state = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state)

    rtg1_vals = [float(x) for x in str(args.rtg1_values).split(",") if x.strip() != ""]

    grid = []
    for v in rtg1_vals:
        per_clip = []
        for i in tqdm(range(n), desc=f"rtg1={v}"):
            score = eval_clip_rtg1_override(
                model,
                dataset,
                i,
                device,
                max_lag=int(args.max_lag),
                max_k=int(args.max_k),
                tw=int(args.tw),
                gain=float(args.gain),
                rtg1_override=float(v),
            )
            if score is not None:
                per_clip.append(score)
        grid.append({"rtg1": float(v), "dt_final_mean": float(np.mean(per_clip)) if per_clip else None})

    payload = {
        "config": {
            "mic_root": args.mic_root,
            "ldv_root": args.ldv_root,
            "ckpt_path": args.ckpt_path,
            "angle": float(args.angle),
            "num_clips": int(n),
            "max_lag": int(args.max_lag),
            "max_k": int(args.max_k),
            "tw": int(args.tw),
            "gain": float(args.gain),
            "rtg1_values": rtg1_vals,
        },
        "grid": grid,
    }

    out_path = out_dir / "rtg1_override_grid.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
