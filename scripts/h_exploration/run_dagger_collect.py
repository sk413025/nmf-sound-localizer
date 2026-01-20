import argparse
import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch

from scripts.h_exploration.dataset_lag import DoALagDataset
from scripts.h_exploration.run_lambda_override_grid_eval import SeqDT_FreqAware, gather_atoms, manual_complex_norm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_logc(c: float, logc_min: float, logc_max: float) -> float:
    if logc_max == logc_min:
        return 0.0
    return float(np.clip((np.log10(c) - logc_min) / (logc_max - logc_min), 0.0, 1.0))


@torch.no_grad()
def collect_dagger_blocks(
    *,
    model: torch.nn.Module,
    dataset: DoALagDataset,
    clip_indices: List[int],
    device: torch.device,
    lambda_c_values: List[float],
    max_lag: int,
    max_k: int,
    tw: int,
    gain: float,
    rtg_dim: int,
    use_stop_action: bool,
    rtg1_mode: str,
    rtg1_max_k: int,
    min_k: int,
) -> List[dict]:
    Lag_Min = -max_lag
    Lag_Max = max_lag
    Lags = torch.arange(Lag_Min, Lag_Max + 1, device=device)
    M_lags = len(Lags)

    logc_vals = [np.log10(c) for c in lambda_c_values]
    logc_min, logc_max = min(logc_vals), max(logc_vals)

    all_blocks: List[dict] = []

    model.eval()
    stride = 32

    for idx in clip_indices:
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
            continue

        freq_ids = torch.arange(F, device=device)

        for start_t in range(start_limit, end_limit, stride):
            end_t = start_t + tw
            y_block = ldv_stft[start_t:end_t, :].T  # (F, Tw)

            D = torch.zeros(F, tw, M_lags, dtype=mic_stft.dtype, device=device)
            for m, k_lag in enumerate(Lags):
                s = int(start_t - k_lag)
                e = int(s + tw)
                if e > mic_stft.shape[0]:
                    raise ValueError("Mic STFT too short for requested lag window.")
                D[:, :, m] = mic_stft[s:e, :].T

            norms = manual_complex_norm(D, dim=1).unsqueeze(1) + 1e-8
            D_norm = D / norms

            Y = y_block.unsqueeze(2)  # (F, Tw, 1)
            initial_energy = manual_complex_norm(Y.squeeze(), dim=1) ** 2  # (F,)

            for lambda_c in lambda_c_values:
                rtg0 = normalize_logc(lambda_c, logc_min, logc_max)

                res_student = Y.clone()
                res_label = Y.clone()

                student_mask = torch.zeros(F, M_lags, dtype=torch.bool, device=device)
                teacher_mask = torch.zeros(F, M_lags, dtype=torch.bool, device=device)
                teacher_stop = torch.zeros(F, dtype=torch.bool, device=device)
                valid_len = torch.full((F,), max_k, dtype=torch.long, device=device)

                corrs_steps = []
                actions_steps = []
                reductions_steps = []
                eres_steps = []
                delta_steps = []

                for k in range(max_k):
                    corrs_state = torch.bmm(D_norm.conj().transpose(1, 2), res_student).squeeze(2)
                    abs_corrs = torch.abs(corrs_state)
                    abs_corrs[student_mask] = -1.0

                    corrs_steps.append(abs_corrs.detach().cpu())

                    # Teacher label: greedy action on student state.
                    best_teacher = torch.argmax(abs_corrs, dim=1)
                    best_teacher[teacher_stop] = 0
                    actions_steps.append(best_teacher.detach().cpu())

                    # Compute teacher deltaE on label residual (separate from student residual).
                    E_before = manual_complex_norm(res_label.squeeze(), dim=1) ** 2

                    D_cpu = D.cpu()
                    Y_cpu = Y.cpu()
                    for f in range(F):
                        if teacher_stop[f]:
                            continue
                        if teacher_mask[f, best_teacher[f]]:
                            continue
                        teacher_mask[f, best_teacher[f]] = True
                        active_ids = torch.nonzero(teacher_mask[f], as_tuple=False).squeeze(1).cpu()
                        A_active = gather_atoms(D_cpu[f : f + 1], active_ids.unsqueeze(0))
                        y_f = Y_cpu[f : f + 1]
                        sol_f = torch.linalg.lstsq(A_active, y_f).solution
                        recon_f = A_active @ sol_f
                        res_label[f : f + 1] = (y_f - recon_f).to(device)

                    E_after = manual_complex_norm(res_label.squeeze(), dim=1) ** 2
                    delta = (E_before - E_after).clamp(min=0.0)
                    lambda_abs = initial_energy * float(lambda_c)

                    if k + 1 >= int(min_k):
                        newly_stop = (~teacher_stop) & (delta <= lambda_abs)
                        teacher_stop[newly_stop] = True
                        valid_len[newly_stop] = k + 1

                    eres_steps.append(E_after.detach().cpu())
                    delta_steps.append(delta.detach().cpu())
                    reductions = (initial_energy - E_after) / (initial_energy + 1e-6)
                    reductions_steps.append(reductions.detach().cpu())

                    # Student action (no STOP during data collection).
                    if rtg_dim == 2:
                        if rtg1_mode == "max_k":
                            remaining = float(max_k - k) / max(float(rtg1_max_k), 1.0)
                        else:
                            remaining = float(max_k - k) / max(float(max_k), 1.0)
                        rtg_in = torch.stack(
                            [torch.full((F,), rtg0, device=device), torch.full((F,), remaining, device=device)], dim=-1
                        )
                    else:
                        rtg_in = torch.full((F,), rtg0, device=device)

                    logits = model(abs_corrs.unsqueeze(1), rtg_in.unsqueeze(1), freq_ids).squeeze(1)
                    if use_stop_action:
                        logits[:, M_lags] = -float("inf")
                    act_student = torch.argmax(logits, dim=1)

                    for f in range(F):
                        if student_mask[f, act_student[f]]:
                            continue
                        student_mask[f, act_student[f]] = True
                        active_ids = torch.nonzero(student_mask[f], as_tuple=False).squeeze(1).cpu()
                        A_active = gather_atoms(D_cpu[f : f + 1], active_ids.unsqueeze(0))
                        y_f = Y_cpu[f : f + 1]
                        sol_f = torch.linalg.lstsq(A_active, y_f).solution
                        recon_f = A_active @ sol_f
                        res_student[f : f + 1] = (y_f - recon_f).to(device)

                corrs = torch.stack(corrs_steps, dim=1)
                actions = torch.stack(actions_steps, dim=1)
                reductions = torch.stack(reductions_steps, dim=1)
                E_res = torch.stack(eres_steps, dim=1)
                deltaE = torch.stack(delta_steps, dim=1)

                all_blocks.append(
                    {
                        "corrs": corrs.half(),
                        "actions": actions.to(torch.int16),
                        "reductions": reductions.half(),
                        "scores": reductions[:, -1].half(),
                        "lambda_c": torch.full_like(initial_energy, float(lambda_c)),
                        "lambda_abs": lambda_abs.float(),
                        "E0": initial_energy.float(),
                        "E_res": E_res.float(),
                        "deltaE": deltaE.float(),
                        "valid_len": valid_len.to(torch.int16),
                    }
                )

    return all_blocks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mic_root", type=str, required=True)
    p.add_argument("--ldv_root", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--angle", type=float, default=None)
    p.add_argument("--num_clips", type=int, default=None)
    p.add_argument("--hop_length", type=int, default=160)
    p.add_argument("--freq_min", type=float, default=0.0)
    p.add_argument("--freq_max", type=float, default=8000.0)
    p.add_argument("--max_lag", type=int, default=50)
    p.add_argument("--max_k", type=int, default=16)
    p.add_argument("--tw", type=int, default=32)
    p.add_argument("--gain", type=float, default=100.0)
    p.add_argument("--rtg_dim", type=int, default=2)
    p.add_argument("--rtg1_mode", type=str, default="max_k", choices=["seq_len", "max_k"])
    p.add_argument("--rtg1_max_k", type=int, default=16)
    p.add_argument("--use_stop_action", action="store_true")
    p.add_argument("--min_k", type=int, default=1)
    p.add_argument(
        "--lambda_c_values",
        type=str,
        required=True,
        help="Comma-separated lambda values (e.g., 1e-4,3e-4,1e-3).",
    )
    p.set_defaults(use_stop_action=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset = DoALagDataset(
        args.mic_root,
        args.ldv_root,
        angle=args.angle,
        hop_length=int(args.hop_length),
        freq_min=float(args.freq_min),
        freq_max=float(args.freq_max),
    )

    if args.num_clips is None:
        num_clips = 3
    else:
        num_clips = int(args.num_clips)
    clip_indices = list(range(min(num_clips, len(dataset))))

    lambda_c_values = [float(x) for x in str(args.lambda_c_values).split(",") if x.strip() != ""]
    if not lambda_c_values:
        raise ValueError("--lambda_c_values must contain at least one value.")

    M_lags = int(args.max_lag) * 2 + 1
    action_dim = M_lags + (1 if args.use_stop_action else 0)
    model = SeqDT_FreqAware(M_lags=M_lags, rtg_dim=int(args.rtg_dim), action_dim=action_dim).to(device)
    state = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state)

    blocks = collect_dagger_blocks(
        model=model,
        dataset=dataset,
        clip_indices=clip_indices,
        device=device,
        lambda_c_values=lambda_c_values,
        max_lag=int(args.max_lag),
        max_k=int(args.max_k),
        tw=int(args.tw),
        gain=float(args.gain),
        rtg_dim=int(args.rtg_dim),
        use_stop_action=bool(args.use_stop_action),
        rtg1_mode=str(args.rtg1_mode),
        rtg1_max_k=int(args.rtg1_max_k),
        min_k=int(args.min_k),
    )

    out_path = out_dir / "dagger_trajectories.pt"
    torch.save(blocks, out_path)
    logger.info(f"Saved {len(blocks)} DAgger blocks to {out_path}")

    config = {
        "mic_root": args.mic_root,
        "ldv_root": args.ldv_root,
        "ckpt_path": args.ckpt_path,
        "num_clips": num_clips,
        "hop_length": int(args.hop_length),
        "freq_min": float(args.freq_min),
        "freq_max": float(args.freq_max),
        "max_lag": int(args.max_lag),
        "max_k": int(args.max_k),
        "tw": int(args.tw),
        "gain": float(args.gain),
        "lambda_c_values": lambda_c_values,
        "rtg_dim": int(args.rtg_dim),
        "rtg1_mode": str(args.rtg1_mode),
        "rtg1_max_k": int(args.rtg1_max_k),
        "use_stop_action": bool(args.use_stop_action),
        "min_k": int(args.min_k),
    }
    (out_dir / "dagger_config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
