import argparse
import json
import logging
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from scripts.h_exploration.dataset_lag import DoALagDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        out_dim = M_lags if action_dim is None else int(action_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x, rtg, freq_idx):
        # x: (B, K, M)
        B, K, M = x.shape
        x = self.corr_norm(x)
        s_emb = self.state_embed(x)
        if rtg.dim() == 2:
            rtg_in = rtg.unsqueeze(-1)
        else:
            rtg_in = rtg
        r_emb = self.rtg_embed(rtg_in)
        f_emb = self.freq_embed(freq_idx).unsqueeze(1).expand(-1, K, -1)
        h = self.layer_norm(s_emb + r_emb + f_emb)
        out, _ = self.rnn(h)
        logits = self.head(out)
        return logits


def manual_complex_norm(tensor, dim=None, keepdim=False):
    if dim is None:
        return torch.sqrt(tensor.real.pow(2) + tensor.imag.pow(2) + 1e-10)
    return torch.sqrt(
        tensor.real.pow(2).sum(dim=dim, keepdim=keepdim) + tensor.imag.pow(2).sum(dim=dim, keepdim=keepdim) + 1e-10
    )


def gather_atoms(Dict, Ids_till_k):
    F, Tw, M = Dict.shape
    K_curr = Ids_till_k.shape[1]
    Ids_exp = Ids_till_k.unsqueeze(1).expand(-1, Tw, -1)
    return torch.gather(Dict, 2, Ids_exp)


def normalize_logc(c: float, logc_min: float, logc_max: float) -> float:
    if logc_max == logc_min:
        return 0.0
    return float(np.clip((np.log10(c) - logc_min) / (logc_max - logc_min), 0.0, 1.0))


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


@torch.no_grad()
def simulate_for_lambda(
    model: nn.Module,
    dataset: DoALagDataset,
    clip_indices: list[int],
    device: torch.device,
    *,
    lambda_c: float,
    logc_min: float,
    logc_max: float,
    max_lag: int,
    max_k: int,
    tw: int,
    gain: float,
    rtg_dim: int,
    use_stop_action: bool,
    eps_energy: float,
    rollout_mode: str,
    teacher_min_k: int,
) -> tuple[list[list[int]], list[list[np.ndarray]], list[int], list[float]]:
    Lag_Min = -max_lag
    Lag_Max = max_lag
    Lags = torch.arange(Lag_Min, Lag_Max + 1, device=device)
    M_lags = len(Lags)
    action_dim = M_lags + (1 if use_stop_action else 0)
    stop_id = M_lags if use_stop_action else None

    rtg0 = normalize_logc(lambda_c, logc_min, logc_max)

    all_actions: list[list[int]] = []
    all_logits: list[list[np.ndarray]] = []
    steps_used: list[int] = []
    final_capture: list[float] = []
    teacher_steps: list[int] = []
    student_stop_before_teacher: list[int] = []
    student_stop_at_teacher: list[int] = []

    model.eval()

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

        valid_starts = range(start_limit, end_limit, tw)
        if len(list(valid_starts)) == 0:
            continue

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

            Y = y_block.unsqueeze(2)  # (F, Tw, 1)
            initial_energy = manual_complex_norm(Y.squeeze(), dim=1) ** 2  # (F,)

            res_dt = Y.clone()
            mask_dt = torch.zeros(F, action_dim, dtype=torch.bool, device=device)
            indices_dt = torch.full((F, max_k), -1, dtype=torch.long, device=device)

            # Teacher-forced state path uses a separate residual and mask.
            res_teacher = Y.clone()
            mask_teacher = torch.zeros(F, M_lags, dtype=torch.bool, device=device)
            indices_teacher = torch.full((F, max_k), -1, dtype=torch.long, device=device)
            teacher_stop_mask = torch.zeros(F, dtype=torch.bool, device=device)
            teacher_stop_step = torch.full((F,), -1, dtype=torch.long, device=device)

            stop_mask = torch.zeros(F, dtype=torch.bool, device=device)
            per_freq_actions = [[] for _ in range(F)]
            per_freq_logits = [[] for _ in range(F)]
            per_freq_steps = torch.full((F,), -1, dtype=torch.long, device=device)

            for k in range(max_k):
                if rollout_mode == "teacher_forced":
                    corrs_state = torch.bmm(D_norm.conj().transpose(1, 2), res_teacher).squeeze(2)
                else:
                    corrs_state = torch.bmm(D_norm.conj().transpose(1, 2), res_dt).squeeze(2)

                abs_corrs_dt = torch.abs(corrs_state)
                abs_corrs_dt[mask_dt[:, :M_lags]] = -1.0

                if rtg_dim == 2:
                    remaining = float(max_k - k) / max(float(max_k), 1.0)
                    rtg_in = torch.stack(
                        [torch.full((F,), rtg0, device=device), torch.full((F,), remaining, device=device)], dim=-1
                    )
                else:
                    rtg_in = torch.full((F,), rtg0, device=device)

                logits = model(abs_corrs_dt.unsqueeze(1), rtg_in.unsqueeze(1), freq_ids).squeeze(1)
                logits[mask_dt] = -float("inf")
                act_dt = torch.argmax(logits, dim=1)

                for f in range(F):
                    if stop_mask[f]:
                        continue
                    act = int(act_dt[f].item())
                    per_freq_actions[f].append(act)
                    per_freq_logits[f].append(logits[f].detach().cpu().numpy())
                    if use_stop_action and act == stop_id:
                        stop_mask[f] = True
                        per_freq_steps[f] = k + 1
                        continue
                    mask_dt[f, act] = True
                    indices_dt[f, k] = act

                if rollout_mode == "teacher_forced":
                    # Teacher greedy action + residual update; penalty-OMP stop is diagnostic only.
                    abs_corrs_teacher = torch.abs(corrs_state)
                    abs_corrs_teacher[mask_teacher] = -1.0
                    best_teacher = torch.argmax(abs_corrs_teacher, dim=1)
                    best_teacher[stop_mask] = 0
                    indices_teacher[:, k] = best_teacher
                    active_teacher = ~stop_mask
                    if active_teacher.any():
                        mask_teacher[active_teacher, best_teacher[active_teacher]] = True

                    # Energy before update (teacher residual)
                    E_before = manual_complex_norm(res_teacher.squeeze(), dim=1) ** 2

                    D_cpu = D.cpu()
                    Y_cpu = Y.cpu()
                    for f in range(F):
                        if stop_mask[f]:
                            continue
                        active_ids = indices_teacher[f, : k + 1].cpu()
                        if (active_ids < 0).any():
                            continue
                        A_active = gather_atoms(D_cpu[f : f + 1], active_ids.unsqueeze(0))
                        y_f = Y_cpu[f : f + 1]
                        sol_f = torch.linalg.lstsq(A_active, y_f).solution
                        recon_f = A_active @ sol_f
                        res_teacher[f : f + 1] = (y_f - recon_f).to(device)

                    E_after = manual_complex_norm(res_teacher.squeeze(), dim=1) ** 2
                    delta = (E_before - E_after).clamp(min=0.0)
                    if k + 1 >= int(teacher_min_k):
                        lambda_abs = initial_energy * float(lambda_c)
                        newly_stop = active_teacher & (delta <= lambda_abs)
                        teacher_stop_mask[newly_stop] = True
                        teacher_stop_step[newly_stop] = k + 1

                    if bool(stop_mask.all()):
                        break
                else:
                    if bool(stop_mask.all()):
                        break

                    # Update residuals for active freqs only (student actions)
                    D_cpu = D.cpu()
                    Y_cpu = Y.cpu()
                    for f in range(F):
                        if stop_mask[f]:
                            continue
                        active_ids = indices_dt[f, : k + 1].cpu()
                        if (active_ids < 0).any():
                            continue
                        A_active = gather_atoms(D_cpu[f : f + 1], active_ids.unsqueeze(0))
                        y_f = Y_cpu[f : f + 1]
                        sol_f = torch.linalg.lstsq(A_active, y_f).solution
                        recon_f = A_active @ sol_f
                        res_dt[f : f + 1] = (y_f - recon_f).to(device)

            # Final energies after stopping
            if rollout_mode == "teacher_forced":
                E_res = manual_complex_norm(res_teacher.squeeze(), dim=1) ** 2
            else:
                E_res = manual_complex_norm(res_dt.squeeze(), dim=1) ** 2
            capture = 1.0 - (E_res / torch.clamp(initial_energy, min=eps_energy))

            for f in range(F):
                if per_freq_steps[f] < 0:
                    per_freq_steps[f] = max_k
                all_actions.append(per_freq_actions[f])
                all_logits.append(per_freq_logits[f])
                steps_used.append(int(per_freq_steps[f].item()))
                final_capture.append(float(capture[f].item()))

                if rollout_mode == "teacher_forced":
                    t_step = int(teacher_stop_step[f].item()) if teacher_stop_step[f] >= 0 else max_k
                    teacher_steps.append(t_step)
                    s_step = int(per_freq_steps[f].item())
                    student_stop_before_teacher.append(int(s_step < t_step))
                    student_stop_at_teacher.append(int(s_step == t_step))

    return all_actions, all_logits, steps_used, final_capture, teacher_steps, student_stop_before_teacher, student_stop_at_teacher


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mic_root", type=str, required=True)
    p.add_argument("--ldv_root", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--subset_manifest", type=str, default=None)
    p.add_argument("--angle", type=float, default=None)
    p.add_argument("--num_clips", type=int, default=None)
    p.add_argument("--hop_length", type=int, default=160, help="STFT hop length (samples)")
    p.add_argument("--freq_min", type=float, default=0.0)
    p.add_argument("--freq_max", type=float, default=8000.0)
    p.add_argument("--max_lag", type=int, default=50)
    p.add_argument("--max_k", type=int, default=16)
    p.add_argument("--tw", type=int, default=32)
    p.add_argument("--gain", type=float, default=100.0)
    p.add_argument("--rtg_dim", type=int, default=2)
    p.add_argument("--eps_energy", type=float, default=1e-12)
    p.add_argument("--use_stop_action", action="store_true")
    p.add_argument(
        "--rollout_mode",
        type=str,
        default="free",
        choices=["free", "teacher_forced"],
        help="free: student updates residuals; teacher_forced: teacher updates residuals, student only decides STOP.",
    )
    p.add_argument(
        "--teacher_min_k",
        type=int,
        default=1,
        help="Minimum steps before teacher STOP in teacher_forced mode (penalty-OMP rule).",
    )
    p.add_argument(
        "--lambda_c_values",
        type=str,
        required=True,
        help="Comma-separated lambda scaling values (e.g., 1e-4,3e-4,1e-3).",
    )
    p.set_defaults(use_stop_action=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    subset_manifest = None
    if args.subset_manifest:
        subset_manifest = json.loads(Path(args.subset_manifest).read_text())

    dataset = DoALagDataset(
        args.mic_root,
        args.ldv_root,
        angle=args.angle,
        hop_length=int(args.hop_length),
        freq_min=float(args.freq_min),
        freq_max=float(args.freq_max),
    )

    if args.num_clips is None:
        if subset_manifest and isinstance(subset_manifest.get("num_pairs"), int):
            num_clips = int(subset_manifest["num_pairs"])
        else:
            num_clips = 3
    else:
        num_clips = int(args.num_clips)

    clip_indices = list(range(min(num_clips, len(dataset))))

    lambda_c_values = [float(x) for x in str(args.lambda_c_values).split(",") if x.strip() != ""]
    if not lambda_c_values:
        raise ValueError("--lambda_c_values must contain at least one value.")

    logc_vals = [np.log10(c) for c in lambda_c_values]
    logc_min, logc_max = min(logc_vals), max(logc_vals)

    M_lags = int(args.max_lag) * 2 + 1
    action_dim = M_lags + (1 if args.use_stop_action else 0)

    model = SeqDT_FreqAware(M_lags=M_lags, rtg_dim=int(args.rtg_dim), action_dim=action_dim).to(device)
    state = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state)

    ref_c = lambda_c_values[0]
    ref_actions, ref_logits, ref_steps, ref_capture, ref_teacher_steps, ref_before_rate, ref_at_rate = simulate_for_lambda(
        model,
        dataset,
        clip_indices,
        device,
        lambda_c=ref_c,
        logc_min=logc_min,
        logc_max=logc_max,
        max_lag=int(args.max_lag),
        max_k=int(args.max_k),
        tw=int(args.tw),
        gain=float(args.gain),
        rtg_dim=int(args.rtg_dim),
        use_stop_action=bool(args.use_stop_action),
        eps_energy=float(args.eps_energy),
        rollout_mode=str(args.rollout_mode),
        teacher_min_k=int(args.teacher_min_k),
    )

    grid = []
    for lambda_c in tqdm(lambda_c_values, desc="lambda_c sweep"):
        actions, logits, steps_used, final_capture, teacher_steps, student_before, student_at = simulate_for_lambda(
            model,
            dataset,
            clip_indices,
            device,
            lambda_c=lambda_c,
            logc_min=logc_min,
            logc_max=logc_max,
            max_lag=int(args.max_lag),
            max_k=int(args.max_k),
            tw=int(args.tw),
            gain=float(args.gain),
            rtg_dim=int(args.rtg_dim),
            use_stop_action=bool(args.use_stop_action),
            eps_energy=float(args.eps_energy),
            rollout_mode=str(args.rollout_mode),
            teacher_min_k=int(args.teacher_min_k),
        )

        action_change_rates: list[float] = []
        logits_kls: list[float] = []

        for i in range(min(len(actions), len(ref_actions))):
            a_ref = ref_actions[i]
            a_cur = actions[i]
            len_common = min(len(a_ref), len(a_cur))
            if len_common == 0:
                continue
            diff = sum(int(a_ref[j] != a_cur[j]) for j in range(len_common)) / float(len_common)
            action_change_rates.append(diff)

            kl_steps = []
            for j in range(len_common):
                p = softmax_np(np.asarray(ref_logits[i][j]))
                q = softmax_np(np.asarray(logits[i][j]))
                kl_steps.append(kl_divergence(p, q))
            if kl_steps:
                logits_kls.append(float(np.mean(kl_steps)))

        summary = {
            "steps_used_mean": float(np.mean(steps_used)) if steps_used else None,
            "final_capture_mean": float(np.mean(final_capture)) if final_capture else None,
            "action_change_rate_vs_ref": float(np.mean(action_change_rates)) if action_change_rates else 0.0,
            "logits_kl_mean_vs_ref": float(np.mean(logits_kls)) if logits_kls else 0.0,
        }
        if str(args.rollout_mode) == "teacher_forced":
            summary.update(
                {
                    "teacher_steps_mean": float(np.mean(teacher_steps)) if teacher_steps else None,
                    "student_stop_before_teacher_rate": float(np.mean(student_before)) if student_before else None,
                    "student_stop_at_teacher_rate": float(np.mean(student_at)) if student_at else None,
                }
            )

        grid.append(
            {
                "lambda_c": float(lambda_c),
                "rtg0": normalize_logc(lambda_c, logc_min, logc_max),
                "summary": summary,
            }
        )

    payload: dict[str, Any] = {
        "config": {
            "mic_root": args.mic_root,
            "ldv_root": args.ldv_root,
            "ckpt_path": args.ckpt_path,
            "subset_manifest_path": args.subset_manifest,
            "angle": None if args.angle is None else float(args.angle),
            "num_clips": int(num_clips),
            "max_lag": int(args.max_lag),
            "max_k": int(args.max_k),
            "tw": int(args.tw),
            "gain": float(args.gain),
            "lambda_c_values": lambda_c_values,
            "rtg0_semantics": "lambda_cost_logc_norm",
            "rtg1_semantics": "remaining_steps_fraction",
            "rtg_dim": int(args.rtg_dim),
            "use_stop_action": bool(args.use_stop_action),
            "rollout_mode": str(args.rollout_mode),
            "teacher_min_k": int(args.teacher_min_k),
            "eps_energy": float(args.eps_energy),
        },
        "grid": grid,
    }

    out_path = out_dir / "lambda_grid.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
