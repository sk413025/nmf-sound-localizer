import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from scripts.h_exploration.dataset_lag import DoALagDataset

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


class SeqDT_FreqAware(nn.Module):
    def __init__(
        self,
        M_lags: int = 16,
        d_model: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 2,
        max_freq: int = 1025,
        rtg_dim: int = 2,
        action_dim: Optional[int] = None,
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
        tensor.real.pow(2).sum(dim=dim, keepdim=keepdim)
        + tensor.imag.pow(2).sum(dim=dim, keepdim=keepdim)
        + 1e-10
    )


def gather_atoms(dict_atoms, ids_till_k):
    # dict_atoms: (F, Tw, M); ids_till_k: (F, k)
    _, tw, _ = dict_atoms.shape
    ids_exp = ids_till_k.unsqueeze(1).expand(-1, tw, -1)
    return torch.gather(dict_atoms, 2, ids_exp)


def normalize_logc(c: float, logc_min: float, logc_max: float) -> float:
    if logc_max == logc_min:
        return 0.0
    return float(np.clip((np.log10(c) - logc_min) / (logc_max - logc_min), 0.0, 1.0))


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_subset_manifest(
    dataset: DoALagDataset,
    mic_root: str,
    ldv_root: str,
    num_pairs: int,
    out_path: Path,
) -> Dict[str, object]:
    if num_pairs > len(dataset):
        raise SystemExit(f"Dataset too small: len(dataset)={len(dataset)} < num_pairs={num_pairs}")

    file_hashes: List[Dict[str, str]] = []
    for idx in range(num_pairs):
        mic_path, ldv_path = dataset.clips[idx]
        mic_path = Path(mic_path)
        ldv_path = Path(ldv_path)
        if not mic_path.exists():
            raise SystemExit(f"Missing mic file: {mic_path}")
        if not ldv_path.exists():
            raise SystemExit(f"Missing ldv file: {ldv_path}")
        file_hashes.append({"path": str(mic_path), "md5": md5_file(mic_path)})
        file_hashes.append({"path": str(ldv_path), "md5": md5_file(ldv_path)})

    lines = sorted([f'{r["md5"]}  {r["path"]}' for r in file_hashes])
    fingerprint_md5 = hashlib.md5("\n".join(lines).encode("utf-8")).hexdigest()

    manifest = {
        "mic_root": mic_root,
        "ldv_root": ldv_root,
        "num_pairs": int(num_pairs),
        "num_files": int(len(file_hashes)),
        "selection": f"first {num_pairs} clip pairs in dataset order (all angles)",
        "file_hashes": file_hashes,
        "fingerprint_md5": fingerprint_md5,
    }
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def validate_manifest(dataset: DoALagDataset, manifest: Dict[str, object]) -> Tuple[int, int]:
    missing_files = 0
    md5_mismatches = 0
    file_hashes = manifest["file_hashes"]
    expected_paths = [entry["path"] for entry in file_hashes]

    actual_paths = []
    for idx in range(int(manifest["num_pairs"])):
        mic_path, ldv_path = dataset.clips[idx]
        actual_paths.extend([str(mic_path), str(ldv_path)])

    if expected_paths != actual_paths:
        raise SystemExit("Subset mismatch: manifest order does not match dataset order.")

    for entry in file_hashes:
        path = Path(entry["path"])
        if not path.exists():
            missing_files += 1
            continue
        md5_now = md5_file(path)
        if md5_now != entry["md5"]:
            md5_mismatches += 1
    return missing_files, md5_mismatches


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = float("inf")
        self.max = float("-inf")

    def update(self, value: float) -> None:
        self.count += 1
        self.sum += value
        self.sumsq += value * value
        if value < self.min:
            self.min = value
        if value > self.max:
            self.max = value

    def mean(self) -> Optional[float]:
        if self.count == 0:
            return None
        return self.sum / self.count

    def std(self) -> Optional[float]:
        if self.count == 0:
            return None
        mean = self.sum / self.count
        var = max(self.sumsq / self.count - mean * mean, 0.0)
        return float(np.sqrt(var))


class P2Quantile:
    def __init__(self, q: float):
        self.q = q
        self.n = 0
        self.initial: List[float] = []
        self.qv: List[float] = []
        self.npos: List[float] = []
        self.np: List[float] = []
        self.dn: List[float] = []

    def update(self, x: float) -> None:
        if not np.isfinite(x):
            return
        if self.n < 5:
            self.initial.append(x)
            self.n += 1
            if self.n == 5:
                self.initial.sort()
                self.qv = [float(v) for v in self.initial]
                self.npos = [1, 2, 3, 4, 5]
                self.np = [1, 1 + 2 * self.q, 1 + 4 * self.q, 3 + 2 * self.q, 5]
                self.dn = [0.0, self.q / 2.0, self.q, (1 + self.q) / 2.0, 1.0]
            return

        self.n += 1
        if x < self.qv[0]:
            self.qv[0] = x
            k = 0
        elif x < self.qv[1]:
            k = 0
        elif x < self.qv[2]:
            k = 1
        elif x < self.qv[3]:
            k = 2
        elif x <= self.qv[4]:
            k = 3
        else:
            self.qv[4] = x
            k = 3

        for i in range(k + 1, 5):
            self.npos[i] += 1
        for i in range(5):
            self.np[i] += self.dn[i]

        for i in range(1, 4):
            d = self.np[i] - self.npos[i]
            if (d >= 1 and self.npos[i + 1] - self.npos[i] > 1) or (d <= -1 and self.npos[i] - self.npos[i - 1] > 1):
                s = 1 if d >= 0 else -1
                qv_new = self.qv[i] + (s / (self.npos[i + 1] - self.npos[i - 1])) * (
                    (self.npos[i] - self.npos[i - 1] + s) * (self.qv[i + 1] - self.qv[i]) / (self.npos[i + 1] - self.npos[i])
                    + (self.npos[i + 1] - self.npos[i] - s) * (self.qv[i] - self.qv[i - 1]) / (self.npos[i] - self.npos[i - 1])
                )
                if self.qv[i - 1] < qv_new < self.qv[i + 1]:
                    self.qv[i] = qv_new
                else:
                    self.qv[i] = self.qv[i] + s * (self.qv[i + s] - self.qv[i]) / (self.npos[i + s] - self.npos[i])
                self.npos[i] += s

    def result(self) -> Optional[float]:
        if self.n == 0:
            return None
        if self.n <= 5:
            return float(np.median(self.initial))
        return float(self.qv[2])


def rankdata(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=float)
    sorted_vals = arr[order]
    n = len(arr)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def spearmanr(values_x: List[float], values_y: List[float]) -> Optional[float]:
    if len(values_x) == 0 or len(values_x) != len(values_y):
        return None
    rx = rankdata(values_x)
    ry = rankdata(values_y)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return None
    corr = np.corrcoef(rx, ry)[0, 1]
    return float(corr)


def compute_omp_capture_by_k(
    D: torch.Tensor,
    D_norm: torch.Tensor,
    Y: torch.Tensor,
    initial_energy: torch.Tensor,
    *,
    max_k: int,
    eps_energy: float,
    tol: float,
) -> Tuple[torch.Tensor, int]:
    F, _, M_lags = D.shape
    res = Y.clone()
    chosen = torch.zeros(F, 0, dtype=torch.long)
    captures = torch.zeros(F, max_k)
    prev_res_energy = None
    violations = 0

    for k in range(max_k):
        corrs = torch.bmm(D_norm.conj().transpose(1, 2), res).squeeze(2)
        abs_corrs = torch.abs(corrs)
        if chosen.numel() > 0:
            abs_corrs.scatter_(1, chosen, -1.0)
        act = torch.argmax(abs_corrs, dim=1).unsqueeze(1)
        chosen = torch.cat([chosen, act], dim=1)

        active = gather_atoms(D, chosen)
        sol = torch.linalg.lstsq(active, Y).solution
        recon = active @ sol
        res = Y - recon
        res_energy = manual_complex_norm(res.squeeze(), dim=1) ** 2

        if prev_res_energy is not None:
            if torch.any(res_energy > prev_res_energy + tol):
                violations += int(torch.sum(res_energy > prev_res_energy + tol).item())
        prev_res_energy = res_energy

        capture = 1.0 - (res_energy / torch.clamp(initial_energy, min=eps_energy))
        captures[:, k] = capture

    return captures, violations


def compute_random_capture_by_k(
    D: torch.Tensor,
    Y: torch.Tensor,
    initial_energy: torch.Tensor,
    *,
    max_k: int,
    random_trials: int,
    rng: np.random.Generator,
    eps_energy: float,
) -> torch.Tensor:
    F, _, M_lags = D.shape
    captures = torch.zeros(F, max_k)

    for _ in range(random_trials):
        rand_ids = rng.integers(0, M_lags, size=(F, max_k))
        rand_ids = torch.from_numpy(rand_ids).long()
        for k in range(max_k):
            active = gather_atoms(D, rand_ids[:, : k + 1])
            sol = torch.linalg.lstsq(active, Y).solution
            recon = active @ sol
            res_energy = manual_complex_norm((Y - recon).squeeze(), dim=1) ** 2
            capture = 1.0 - (res_energy / torch.clamp(initial_energy, min=eps_energy))
            captures[:, k] += capture

    captures /= float(random_trials)
    return captures


def compute_dt_forced_k_capture(
    model: nn.Module,
    D: torch.Tensor,
    D_norm: torch.Tensor,
    Y: torch.Tensor,
    initial_energy: torch.Tensor,
    *,
    max_k: int,
    rtg0: float,
    rtg_dim: int,
    stop_id: int,
    eps_energy: float,
    freq_ids: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    F, _, M_lags = D.shape
    device = D.device
    D_cpu = D.cpu()
    Y_cpu = Y.cpu()
    res_cpu = Y_cpu.clone()
    res = res_cpu.to(device)
    mask = torch.zeros(F, M_lags, dtype=torch.bool, device=device)
    indices = torch.full((F, max_k), -1, dtype=torch.long, device=device)
    captures = torch.zeros(F, max_k, device=device)
    duplicates = 0

    for k in range(max_k):
        corrs = torch.bmm(D_norm.conj().transpose(1, 2), res).squeeze(2)
        abs_corrs = torch.abs(corrs)
        abs_corrs[mask] = -1.0

        if rtg_dim == 2:
            remaining = float(max_k - k) / max(float(max_k), 1.0)
            rtg_in = torch.stack(
                [torch.full((F,), rtg0, device=device), torch.full((F,), remaining, device=device)],
                dim=-1,
            )
        else:
            rtg_in = torch.full((F,), rtg0, device=device)

        logits = model(abs_corrs.unsqueeze(1), rtg_in.unsqueeze(1), freq_ids).squeeze(1)
        logits[:, stop_id] = -float("inf")
        logits[:, :M_lags][mask] = -float("inf")
        act = torch.argmax(logits, dim=1)

        row_idx = torch.arange(F, device=device)
        dup_mask = mask[row_idx, act]
        if dup_mask.any():
            duplicates += int(torch.sum(dup_mask).item())
        mask[row_idx, act] = True
        indices[:, k] = act

        if (indices[:, : k + 1] < 0).any():
            raise RuntimeError("DT forced-K has invalid indices; duplicate masking failed.")
        active = gather_atoms(D_cpu, indices[:, : k + 1].cpu())
        sol = torch.linalg.lstsq(active, Y_cpu).solution
        recon = active @ sol
        res_cpu = Y_cpu - recon
        res = res_cpu.to(device)
        res_energy = manual_complex_norm(res.squeeze(), dim=1) ** 2
        captures[:, k] = 1.0 - (res_energy / torch.clamp(initial_energy, min=eps_energy))

    return captures, duplicates


def compute_dt_free_rollout(
    model: nn.Module,
    D: torch.Tensor,
    D_norm: torch.Tensor,
    Y: torch.Tensor,
    initial_energy: torch.Tensor,
    *,
    lambda_c: float,
    logc_min: float,
    logc_max: float,
    max_k: int,
    rtg_dim: int,
    stop_id: int,
    eps_energy: float,
    freq_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    F, _, M_lags = D.shape
    device = D.device
    action_dim = M_lags + 1
    D_cpu = D.cpu()
    Y_cpu = Y.cpu()
    res_cpu = Y_cpu.clone()
    res = res_cpu.to(device)
    mask = torch.zeros(F, action_dim, dtype=torch.bool, device=device)
    indices = torch.full((F, max_k), -1, dtype=torch.long, device=device)
    stop_mask = torch.zeros(F, dtype=torch.bool, device=device)
    steps_used = torch.full((F,), -1, dtype=torch.long, device=device)

    rtg0 = normalize_logc(lambda_c, logc_min, logc_max)

    for k in range(max_k):
        corrs = torch.bmm(D_norm.conj().transpose(1, 2), res).squeeze(2)
        abs_corrs = torch.abs(corrs)
        abs_corrs[mask[:, :M_lags]] = -1.0

        if rtg_dim == 2:
            remaining = float(max_k - k) / max(float(max_k), 1.0)
            rtg_in = torch.stack(
                [torch.full((F,), rtg0, device=device), torch.full((F,), remaining, device=device)],
                dim=-1,
            )
        else:
            rtg_in = torch.full((F,), rtg0, device=device)

        logits = model(abs_corrs.unsqueeze(1), rtg_in.unsqueeze(1), freq_ids).squeeze(1)
        logits[mask] = -float("inf")
        act = torch.argmax(logits, dim=1)

        for f in range(F):
            if stop_mask[f]:
                continue
            a = int(act[f].item())
            if a == stop_id:
                stop_mask[f] = True
                steps_used[f] = k + 1
                continue
            mask[f, a] = True
            indices[f, k] = a

        if bool(stop_mask.all()):
            break

        active = ~stop_mask
        if bool(active.any()):
            active_cpu = active.cpu()
            active_ids = indices[active, : k + 1].cpu()
            if (active_ids < 0).any():
                raise RuntimeError("DT free rollout has invalid indices for active frequencies.")
            active_dict = gather_atoms(D_cpu[active_cpu], active_ids)
            active_y = Y_cpu[active_cpu]
            sol = torch.linalg.lstsq(active_dict, active_y).solution
            recon = active_dict @ sol
            res_cpu[active_cpu] = active_y - recon
            res = res_cpu.to(device)

    for f in range(F):
        if steps_used[f] < 0:
            steps_used[f] = max_k

    res_energy = manual_complex_norm(res.squeeze(), dim=1) ** 2
    capture = 1.0 - (res_energy / torch.clamp(initial_energy, min=eps_energy))
    return steps_used, capture


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mic_root", type=str, required=True)
    p.add_argument("--ldv_root", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--subset_manifest", type=str, default=None)
    p.add_argument("--mode", type=str, choices=["scale_check_subset", "full_dataset", "smoke"], required=True)
    p.add_argument("--num_pairs", type=int, default=None)
    p.add_argument("--hop_length", type=int, default=160)
    p.add_argument("--fs", type=int, default=16000)
    p.add_argument("--n_fft", type=int, default=2048)
    p.add_argument("--freq_min", type=float, default=300.0)
    p.add_argument("--freq_max", type=float, default=3000.0)
    p.add_argument("--max_lag", type=int, default=50)
    p.add_argument("--max_k", type=int, default=16)
    p.add_argument("--tw", type=int, default=32)
    p.add_argument("--gain", type=float, default=100.0)
    p.add_argument("--rtg_dim", type=int, default=2)
    p.add_argument("--eps_energy", type=float, default=1e-12)
    p.add_argument("--lambda_c_values", type=str, required=True)
    p.add_argument("--random_trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--write_per_sample", type=int, default=0)
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "auto"],
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary").mkdir(parents=True, exist_ok=True)

    mic_root = Path(args.mic_root)
    ldv_root = Path(args.ldv_root)
    if not mic_root.exists():
        raise SystemExit(f"Missing mic_root: {mic_root}")
    if not ldv_root.exists():
        raise SystemExit(f"Missing ldv_root: {ldv_root}")

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps, but MPS is not available.")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    dataset = DoALagDataset(
        str(mic_root),
        str(ldv_root),
        angle=None,
        hop_length=int(args.hop_length),
    )

    if args.mode == "scale_check_subset":
        expected_pairs = 48
        if args.num_pairs is not None and int(args.num_pairs) != expected_pairs:
            raise SystemExit(f"scale_check_subset requires num_pairs={expected_pairs}")
        num_pairs = expected_pairs
    elif args.mode == "full_dataset":
        expected_pairs = len(dataset)
        if args.num_pairs is not None and int(args.num_pairs) != expected_pairs:
            raise SystemExit("full_dataset requires num_pairs=len(dataset)")
        num_pairs = expected_pairs
    else:
        num_pairs = 3 if args.num_pairs is None else int(args.num_pairs)
        if num_pairs <= 0:
            raise SystemExit("smoke mode requires num_pairs > 0")
        if num_pairs > len(dataset):
            raise SystemExit(f"smoke num_pairs={num_pairs} exceeds dataset length {len(dataset)}")

    manifest_path = Path(args.subset_manifest) if args.subset_manifest else (out_dir / "subset_manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = generate_subset_manifest(
            dataset,
            str(mic_root),
            str(ldv_root),
            num_pairs,
            manifest_path,
        )
        logger.info("Wrote subset manifest: %s", manifest_path)

    missing_files, md5_mismatches = validate_manifest(dataset, manifest)
    if missing_files > 0:
        raise SystemExit(f"Missing files in manifest: {missing_files}")

    if md5_mismatches > 0:
        raise SystemExit(f"MD5 mismatches in manifest: {md5_mismatches}")

    lambda_c_values = [float(x) for x in str(args.lambda_c_values).split(",") if x.strip() != ""]
    if not lambda_c_values:
        raise SystemExit("--lambda_c_values must be non-empty.")
    logc_vals = [np.log10(c) for c in lambda_c_values]
    logc_min, logc_max = min(logc_vals), max(logc_vals)

    M_lags = int(args.max_lag) * 2 + 1
    action_dim = M_lags + 1
    state = torch.load(args.ckpt_path, map_location="cpu")
    state_mlags = int(state["state_embed.weight"].shape[1])
    head_out = int(state["head.2.weight"].shape[0])
    if state_mlags != M_lags:
        raise SystemExit(f"Checkpoint state_embed M_lags mismatch. Expected {M_lags}, got {state_mlags}.")
    if head_out != action_dim:
        raise SystemExit(f"Checkpoint head_out mismatch. Expected {action_dim}, got {head_out}.")

    model = SeqDT_FreqAware(M_lags=M_lags, rtg_dim=int(args.rtg_dim), action_dim=action_dim).to(device)
    model.load_state_dict(state)
    model.eval()

    tol = 1e-7
    forced_k_values = [1, 2, 4, 8, 16]
    forced_k_values = [k for k in forced_k_values if k <= int(args.max_k)]

    per_sample_path = out_dir / "per_sample.jsonl"
    per_sample_fp = per_sample_path.open("w", encoding="utf-8") if int(args.write_per_sample) == 1 else None

    stats_by_lambda = {}
    steps_stats = {}
    medians_by_lambda = {}
    for lambda_c in lambda_c_values:
        stats_by_lambda[lambda_c] = {
            "dt": RunningStats(),
            "omp": RunningStats(),
            "random": RunningStats(),
        }
        steps_stats[lambda_c] = RunningStats()
        medians_by_lambda[lambda_c] = {
            "dt": P2Quantile(0.5),
            "omp": P2Quantile(0.5),
            "random": P2Quantile(0.5),
        }

    forced_stats = {}
    for k in forced_k_values:
        forced_stats[k] = {
            "dt": RunningStats(),
            "omp": RunningStats(),
            "random": RunningStats(),
        }

    num_samples_total = 0
    num_samples_used = 0
    num_nan_or_inf = 0
    num_capture_out_of_range = 0
    num_omp_monotonicity_violations = 0
    num_dt_duplicate_actions_forced_k = 0

    rng = np.random.default_rng(int(args.seed))
    freq_ids = None
    freq_idx_cpu = None
    log_every = 1

    for clip_idx in range(num_pairs):
        item = dataset[clip_idx]
        mic_stft = item["mic_stft"]
        ldv_stft = item["ldv_stft"]

        if float(args.gain) != 1.0:
            mic_stft = mic_stft * float(args.gain)
            ldv_stft = ldv_stft * float(args.gain)

        T_total, F = ldv_stft.shape
        if freq_ids is None:
            expected_bins = int(args.n_fft) // 2 + 1
            if F != expected_bins:
                raise SystemExit(f"STFT F mismatch: expected {expected_bins}, got {F}")
            freqs = np.linspace(0.0, float(args.fs) / 2.0, expected_bins)
            idx = np.where((freqs >= float(args.freq_min)) & (freqs <= float(args.freq_max)))[0]
            if len(idx) == 0:
                raise SystemExit("Frequency band selection is empty; check freq_min/freq_max.")
            freq_idx_cpu = torch.from_numpy(idx.astype(np.int64))
            freq_ids_full = torch.arange(F, device=device)
            freq_ids = freq_ids_full[freq_idx_cpu.to(device)]

        Lag_Min = -int(args.max_lag)
        Lag_Max = int(args.max_lag)
        Lags = torch.arange(Lag_Min, Lag_Max + 1)

        start_limit = Lag_Max
        end_limit = T_total - int(args.tw) + Lag_Min
        if end_limit <= start_limit:
            continue

        valid_starts = list(range(start_limit, end_limit, int(args.tw)))
        if len(valid_starts) == 0:
            continue

        for w_idx, start_t in enumerate(valid_starts):
            end_t = start_t + int(args.tw)
            y_block = ldv_stft[start_t:end_t, :].T

            D = torch.zeros(F, int(args.tw), M_lags, dtype=mic_stft.dtype)
            for m, k_lag in enumerate(Lags):
                s = int(start_t - k_lag)
                e = int(s + int(args.tw))
                D[:, :, m] = mic_stft[s:e, :].T

            if freq_idx_cpu is not None:
                D = D[freq_idx_cpu]
                y_block = y_block[freq_idx_cpu]
            norms = manual_complex_norm(D, dim=1).unsqueeze(1) + 1e-8
            D_norm = D / norms
            Y = y_block.unsqueeze(2)
            initial_energy = manual_complex_norm(Y.squeeze(), dim=1) ** 2

            D_device = D.to(device)
            D_norm_device = D_norm.to(device)
            Y_device = Y.to(device)
            initial_energy_device = initial_energy.to(device)

            omp_capture_by_k, violations = compute_omp_capture_by_k(
                D,
                D_norm,
                Y,
                initial_energy,
                max_k=int(args.max_k),
                eps_energy=float(args.eps_energy),
                tol=tol * max(float(initial_energy.max().item()), 1.0),
            )
            num_omp_monotonicity_violations += int(violations)

            rand_seed = int(args.seed) + clip_idx * 100000 + w_idx
            rng = np.random.default_rng(rand_seed)
            random_capture_by_k = compute_random_capture_by_k(
                D,
                Y,
                initial_energy,
                max_k=int(args.max_k),
                random_trials=int(args.random_trials),
                rng=rng,
                eps_energy=float(args.eps_energy),
            )

            rtg0_forced = normalize_logc(lambda_c_values[0], logc_min, logc_max)
            dt_forced_capture_by_k, dupes = compute_dt_forced_k_capture(
                model,
                D_device,
                D_norm_device,
                Y_device,
                initial_energy_device,
                max_k=int(args.max_k),
                rtg0=rtg0_forced,
                rtg_dim=int(args.rtg_dim),
                stop_id=M_lags,
                eps_energy=float(args.eps_energy),
                freq_ids=freq_ids,
            )
            num_dt_duplicate_actions_forced_k += int(dupes)

            F_band = int(Y.shape[0])
            for k in forced_k_values:
                idx_k = k - 1
                dt_vals = dt_forced_capture_by_k[:, idx_k]
                omp_vals = omp_capture_by_k[:, idx_k]
                rnd_vals = random_capture_by_k[:, idx_k]
                for f in range(F_band):
                    dt_val = float(dt_vals[f].item())
                    omp_val = float(omp_vals[f].item())
                    rnd_val = float(rnd_vals[f].item())
                    if not np.isfinite(dt_val) or not np.isfinite(omp_val) or not np.isfinite(rnd_val):
                        num_nan_or_inf += 1
                        continue
                    if dt_val < -1e-6 or dt_val > 1.0 + 1e-6:
                        num_capture_out_of_range += 1
                    if omp_val < -1e-6 or omp_val > 1.0 + 1e-6:
                        num_capture_out_of_range += 1
                    if rnd_val < -1e-6 or rnd_val > 1.0 + 1e-6:
                        num_capture_out_of_range += 1
                    forced_stats[k]["dt"].update(dt_val)
                    forced_stats[k]["omp"].update(omp_val)
                    forced_stats[k]["random"].update(rnd_val)

            for lambda_c in lambda_c_values:
                steps_used, dt_capture = compute_dt_free_rollout(
                    model,
                    D_device,
                    D_norm_device,
                    Y_device,
                    initial_energy_device,
                    lambda_c=lambda_c,
                    logc_min=logc_min,
                    logc_max=logc_max,
                    max_k=int(args.max_k),
                    rtg_dim=int(args.rtg_dim),
                    stop_id=M_lags,
                    eps_energy=float(args.eps_energy),
                    freq_ids=freq_ids,
                )
                for f in range(F_band):
                    steps = int(steps_used[f].item())
                    dt_val = float(dt_capture[f].item())
                    omp_val = float(omp_capture_by_k[f, steps - 1].item())
                    rnd_val = float(random_capture_by_k[f, steps - 1].item())
                    if not np.isfinite(dt_val) or not np.isfinite(omp_val) or not np.isfinite(rnd_val):
                        num_nan_or_inf += 1
                        continue
                    if dt_val < -1e-6 or dt_val > 1.0 + 1e-6:
                        num_capture_out_of_range += 1
                    if omp_val < -1e-6 or omp_val > 1.0 + 1e-6:
                        num_capture_out_of_range += 1
                    if rnd_val < -1e-6 or rnd_val > 1.0 + 1e-6:
                        num_capture_out_of_range += 1

                    stats_by_lambda[lambda_c]["dt"].update(dt_val)
                    stats_by_lambda[lambda_c]["omp"].update(omp_val)
                    stats_by_lambda[lambda_c]["random"].update(rnd_val)
                    medians_by_lambda[lambda_c]["dt"].update(dt_val)
                    medians_by_lambda[lambda_c]["omp"].update(omp_val)
                    medians_by_lambda[lambda_c]["random"].update(rnd_val)
                    steps_stats[lambda_c].update(float(steps))

                    if per_sample_fp is not None:
                        row = {
                            "clip_idx": int(clip_idx),
                            "window_idx": int(w_idx),
                            "freq_idx": int(f),
                            "lambda_c": float(lambda_c),
                            "steps_used": int(steps),
                            "dt_capture": dt_val,
                            "omp_capture": omp_val,
                            "random_capture": rnd_val,
                        }
                        per_sample_fp.write(json.dumps(row) + "\n")

            num_samples_total += int(F_band)

        if (clip_idx + 1) % log_every == 0:
            logger.info("Processed clip %d / %d", clip_idx + 1, num_pairs)

    if per_sample_fp is not None:
        per_sample_fp.close()

    compute_rows = []
    steps_means = []
    for lambda_c in lambda_c_values:
        dt_mean = stats_by_lambda[lambda_c]["dt"].mean()
        omp_mean = stats_by_lambda[lambda_c]["omp"].mean()
        rnd_mean = stats_by_lambda[lambda_c]["random"].mean()
        dt_med = medians_by_lambda[lambda_c]["dt"].result()
        omp_med = medians_by_lambda[lambda_c]["omp"].result()
        rnd_med = medians_by_lambda[lambda_c]["random"].result()
        steps_mean = steps_stats[lambda_c].mean()
        steps_means.append(steps_mean)
        compute_rows.append(
            {
                "lambda_c": float(lambda_c),
                "dt_capture_mean": dt_mean,
                "omp_capture_mean": omp_mean,
                "random_capture_mean": rnd_mean,
                "dt_capture_median": dt_med,
                "omp_capture_median": omp_med,
                "random_capture_median": rnd_med,
                "dt_over_omp_mean": None if dt_mean is None or omp_mean is None else float(dt_mean / (omp_mean + 1e-12)),
                "dt_minus_random_mean": None if dt_mean is None or rnd_mean is None else float(dt_mean - rnd_mean),
                "steps_used_mean": steps_mean,
                "steps_used_std": steps_stats[lambda_c].std(),
            }
        )

    forced_rows = []
    for k in forced_k_values:
        dt_mean = forced_stats[k]["dt"].mean()
        omp_mean = forced_stats[k]["omp"].mean()
        rnd_mean = forced_stats[k]["random"].mean()
        forced_rows.append(
            {
                "k": int(k),
                "dt_capture_mean": dt_mean,
                "omp_capture_mean": omp_mean,
                "random_capture_mean": rnd_mean,
                "dt_over_omp_mean": None if dt_mean is None or omp_mean is None else float(dt_mean / (omp_mean + 1e-12)),
                "dt_minus_random_mean": None if dt_mean is None or rnd_mean is None else float(dt_mean - rnd_mean),
            }
        )

    steps_for_corr = [row["steps_used_mean"] for row in compute_rows]
    if any(s is None for s in steps_for_corr):
        spearman = None
    else:
        spearman = spearmanr(lambda_c_values, steps_for_corr)
    steps_range = None
    if steps_means and all(s is not None for s in steps_means):
        steps_range = float(max(steps_means) - min(steps_means))

    num_samples_used = num_samples_total
    integrity = {
        "num_samples_total": int(num_samples_total),
        "num_samples_used": int(num_samples_used),
        "num_missing_files": int(missing_files),
        "num_md5_mismatches": int(md5_mismatches),
        "num_nan_or_inf": int(num_nan_or_inf),
        "num_capture_out_of_range": int(num_capture_out_of_range),
        "num_omp_monotonicity_violations": int(num_omp_monotonicity_violations),
        "num_dt_duplicate_actions_forced_k": int(num_dt_duplicate_actions_forced_k),
    }

    compute_summary = {
        "config": {
            "mic_root": str(mic_root),
            "ldv_root": str(ldv_root),
            "ckpt_path": str(args.ckpt_path),
            "subset_manifest": str(manifest_path),
            "mode": args.mode,
            "num_pairs": int(num_pairs),
            "hop_length": int(args.hop_length),
            "fs": int(args.fs),
            "n_fft": int(args.n_fft),
            "freq_min": float(args.freq_min),
            "freq_max": float(args.freq_max),
            "max_lag": int(args.max_lag),
            "max_k": int(args.max_k),
            "tw": int(args.tw),
            "gain": float(args.gain),
            "rtg_dim": int(args.rtg_dim),
            "eps_energy": float(args.eps_energy),
            "lambda_c_values": lambda_c_values,
            "random_trials": int(args.random_trials),
            "random_sampling": "with_replacement",
            "forced_k_rtg0_lambda_c": float(lambda_c_values[0]),
            "device": str(device),
            "write_per_sample": bool(int(args.write_per_sample)),
        },
        "integrity": integrity,
        "rows": compute_rows,
    }

    forced_summary = {
        "config": compute_summary["config"],
        "integrity": integrity,
        "rows": forced_rows,
    }

    controllability_summary = {
        "lambda_c_values": lambda_c_values,
        "steps_used_mean": [row["steps_used_mean"] for row in compute_rows],
        "spearman_lambda_steps": spearman,
        "steps_range": steps_range,
    }

    (out_dir / "summary" / "compute_matched_summary.json").write_text(
        json.dumps(compute_summary, indent=2), encoding="utf-8"
    )
    (out_dir / "summary" / "forced_k_summary.json").write_text(
        json.dumps(forced_summary, indent=2), encoding="utf-8"
    )
    (out_dir / "summary" / "rtg_controllability_summary.json").write_text(
        json.dumps(controllability_summary, indent=2), encoding="utf-8"
    )

    logger.info("Wrote summaries to %s", out_dir / "summary")


if __name__ == "__main__":
    main()
