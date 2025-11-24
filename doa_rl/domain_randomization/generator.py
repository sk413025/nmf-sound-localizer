from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from scipy.signal import resample_poly
from sklearn.cluster import KMeans

from doa_rl.assets import load_H, load_W
from nmf_localizer.utils.audio_utils import AudioProcessor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AngleRange:
    name: str
    start_deg: int
    end_deg: int
    step: int = 5

    def angles(self) -> List[int]:
        if self.end_deg < self.start_deg:
            raise ValueError(f"AngleRange {self.name}: end_deg must be >= start_deg")
        angles = list(range(self.start_deg, self.end_deg + 1, self.step))
        if not angles:
            raise ValueError(f"AngleRange {self.name}: empty angle list")
        return angles


@dataclass
class GenerationConfig:
    data_root: Path
    w_path: Path
    h_path: Path
    output_root: Path
    clips_per_angle: int = 3
    k: int = 3
    m: int = 8
    reduction_seed: int = 200
    projection_seed: int = 200
    orig_sample_rate: int = 48000
    target_sample_rate: int = 16000
    n_fft: int = 2048
    freq_min: float = 300.0
    freq_max: float = 3000.0
    d_model: int = 128
    normalize_w: bool = True
    normalize_d: bool = True
    early_stop_eps: float = 0.0  # 0 disables; stop when resid_sq < eps (relative to 1.0 after norm)
    min_steps: int = 1  # enforce at least this many OMP steps
    early_stop_resid_ratio: float = 0.0  # 0 disables; stop when resid <= ratio * initial_resid
    reduction_mode: str = "kmeans"  # kmeans (default) or svd


def _normalize_columns(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    norms = x.norm(dim=0, keepdim=True).clamp_min(eps)
    return x / norms


def _compute_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class AngleRangeShardGenerator:
    def __init__(self, config: GenerationConfig):
        self.config = config
        random.seed(config.reduction_seed)
        np.random.seed(config.reduction_seed)
        torch.manual_seed(config.reduction_seed)

        self.W_full = load_W(str(config.w_path)).float()
        self.H_full, angles_tensor = load_H(str(config.h_path))
        self.H_full = self.H_full.float()
        self.angles = [int(a) for a in angles_tensor.tolist()]
        self._validate_assets()

        # DTMin dictionary configuration
        M_full = self.W_full.shape[1]
        logger.info("="*80)
        logger.info("DTMin Dictionary Configuration")
        logger.info("="*80)
        logger.info("W_full shape: %s (F=%d, M_full=%d)", self.W_full.shape, self.W_full.shape[0], M_full)
        logger.info("H_full shape: %s (F=%d, E=%d angles)", self.H_full.shape, self.H_full.shape[0], len(self.angles))
        
        # If requested M matches full dictionary, keep full W to preserve fidelity.
        if config.m >= M_full:
            logger.info("Using FULL W dictionary (m=%d >= M_full=%d) - preserves fidelity for DTMin", config.m, M_full)
            self.W_reduced = self.W_full.clone()
        else:
            mode_desc = (
                f"k-means (seed={config.reduction_seed})" if config.reduction_mode == "kmeans" else "SVD"
            )
            logger.info("Compressing W: M_full=%d → m=%d via %s", M_full, config.m, mode_desc)
            logger.warning("WARNING: Compression may degrade teacher quality for speech/complex signals!")
            if config.reduction_mode == "svd":
                self.W_reduced = self._reduce_atoms_svd(self.W_full, config.m)
            else:
                self.W_reduced = self._reduce_atoms_kmeans(self.W_full, config.m, config.reduction_seed)
        
        if config.normalize_w:
            logger.info("Applying column normalization to W")
            self.W_reduced = _normalize_columns(self.W_reduced)
        
        logger.info("Building dictionary D = H ⊙ W (Hadamard product)")
        self.D = self._build_dictionary(self.W_reduced, self.H_full, normalize=config.normalize_d)
        logger.info("Dictionary D shape: %s (F=%d, P=%d)", self.D.shape, self.D.shape[0], self.D.shape[1])
        logger.info("normalize_d: %s", config.normalize_d)
        logger.info("="*80)
        self.expected_F = self.W_reduced.shape[0]

        torch.manual_seed(config.projection_seed)
        self.P_R = torch.nn.Linear(self.expected_F, config.d_model, bias=False)
        self.proj_rtg = torch.nn.Linear(2, config.d_model, bias=False)
        self.proj_step = torch.nn.Linear(2, config.d_model, bias=False)
        self.type_R = torch.nn.Parameter(torch.randn(1, config.d_model) * 0.02)
        for layer in (self.P_R, self.proj_rtg, self.proj_step):
            torch.nn.init.xavier_uniform_(layer.weight)

        self.angle_to_index = {ang: idx for idx, ang in enumerate(self.angles)}

    def _validate_assets(self) -> None:
        if self.W_full.shape[0] != self.H_full.shape[0]:
            raise ValueError(
                f"W/H frequency mismatch: W.F={self.W_full.shape[0]} H.F={self.H_full.shape[0]}"
            )
        if len(self.angles) != self.H_full.shape[1]:
            raise ValueError(
                f"Angles length {len(self.angles)} does not match H shape {self.H_full.shape}"
            )
        if len(set(self.angles)) != len(self.angles):
            raise ValueError("Duplicate angles detected in H matrix")

    @staticmethod
    def _reduce_atoms_kmeans(W: torch.Tensor, M: int, seed: int) -> torch.Tensor:
        W_np = W.T.numpy()
        km = KMeans(n_clusters=M, random_state=seed, n_init=10)
        km.fit(W_np)
        centers = torch.from_numpy(km.cluster_centers_.T).float()
        return centers

    @staticmethod
    def _reduce_atoms_svd(W: torch.Tensor, M: int) -> torch.Tensor:
        # SVD-based reduction: keep top-M components; columns are U * S
        U, S, _ = torch.linalg.svd(W, full_matrices=False)
        return U[:, :M] * S[:M]

    def _build_dictionary(self, W_reduced: torch.Tensor, H: torch.Tensor, normalize: bool) -> torch.Tensor:
        F, M = W_reduced.shape
        _, D = H.shape
        D_blocks = []
        for d in range(D):
            D_blocks.append(W_reduced * H[:, d:d + 1])
        D_all = torch.cat(D_blocks, dim=1)
        if normalize:
            D_all = _normalize_columns(D_all)
        if D_all.shape[0] != F or D_all.shape[1] != D * M:
            raise ValueError(f"Dictionary shape mismatch: expected ({F}, {D*M}), got {tuple(D_all.shape)}")
        return D_all

    def _select_files(self, angles: Sequence[int], max_samples: int | None) -> Tuple[List[Path], List[int]]:
        files: List[Path] = []
        file_angles: List[int] = []
        for angle in angles:
            if angle not in self.angle_to_index:
                raise ValueError(f"Requested angle {angle} not available in H (angles: {self.angles})")
            angle_dir = self.config.data_root / f"angle_{int(angle)}"
            if not angle_dir.exists():
                raise FileNotFoundError(f"Missing angle directory: {angle_dir}")
            npy_files = sorted(angle_dir.glob("*.npy"))
            if len(npy_files) < self.config.clips_per_angle:
                raise ValueError(
                    f"Angle {angle} has {len(npy_files)} clips (<{self.config.clips_per_angle}); "
                    "No fallback allowed."
                )
            selected = npy_files[: self.config.clips_per_angle]
            for f in selected:
                files.append(f)
                file_angles.append(angle)
                if max_samples is not None and len(files) >= max_samples:
                    return files, file_angles
        return files, file_angles

    def _resample(self, wav: np.ndarray) -> np.ndarray:
        if self.config.orig_sample_rate == self.config.target_sample_rate:
            return wav
        return resample_poly(wav, self.config.target_sample_rate, self.config.orig_sample_rate)

    def _compute_spectrum(self, wav: np.ndarray) -> torch.Tensor:
        freqs, _, _, magnitude = AudioProcessor.compute_stft_spectrogram(
            wav,
            fs=self.config.target_sample_rate,
            nperseg=self.config.n_fft,
            window="hann",
        )
        mask = (freqs >= self.config.freq_min) & (freqs <= self.config.freq_max)
        mag_band = magnitude[mask, :]
        if mag_band.shape[0] != self.expected_F:
            raise ValueError(
                f"STFT F dimension mismatch: expected {self.expected_F}, got {mag_band.shape[0]}"
            )
        y = torch.from_numpy(mag_band.mean(axis=1)).float()
        if torch.isnan(y).any():
            raise ValueError("NaN detected in STFT magnitude")
        return y

    def _run_omp(self, y: torch.Tensor, angle_idx: int) -> Tuple[List[Dict], np.ndarray, int]:
        K = self.config.k
        M = self.config.m
        D = self.D
        r = y / (y.norm() + 1e-12)
        selected: List[int] = []
        residuals: List[np.ndarray] = []
        steps: List[Dict] = []
        prev_resid = (r @ r).item()
        init_resid = prev_resid
        min_steps = max(1, self.config.min_steps)
        for t in range(K):
            g = torch.matmul(D.T, r)
            j = int(torch.argmax(torch.abs(g)).item())
            selected.append(j)
            D_sel = D[:, selected]
            x = torch.linalg.lstsq(D_sel, y).solution
            y_hat = D_sel @ x
            r_raw = y - y_hat
            resid = (r_raw @ r_raw).item()
            r = r_raw / (r_raw.norm() + 1e-12)
            residuals.append(r.cpu().numpy())
            steps.append(
                {
                    "step": t,
                    "dict_index": j,
                    "expert": j // M,
                    "atom": j % M,
                    "resid_sq": resid,
                    "delta_resid_sq": resid - prev_resid,
                    "is_correct_expert": (j // M) == angle_idx,
                }
            )
            prev_resid = resid
            if t + 1 >= min_steps:
                stop_eps = self.config.early_stop_eps > 0.0 and resid < self.config.early_stop_eps
                stop_ratio = (
                    self.config.early_stop_resid_ratio > 0.0
                    and resid <= init_resid * self.config.early_stop_resid_ratio
                )
                if stop_eps or stop_ratio:
                    break
        return steps, np.stack(residuals), len(steps)

    def _convert_to_embeddings(self, residuals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        K = residuals.shape[0]
        rtg_seq = np.zeros((K, 2), dtype=np.float32)
        step_seq = np.zeros((K, 2), dtype=np.float32)
        for t in range(K):
            rtg_seq[t, 0] = residuals[t].mean()
            rtg_seq[t, 1] = 0.0
            step_seq[t, 0] = t / max(self.config.k, 1)
            step_seq[t, 1] = (self.config.k - t) / max(self.config.k, 1)

        r_tensor = torch.from_numpy(residuals).float()
        rtg_tensor = torch.from_numpy(rtg_seq).float()
        step_tensor = torch.from_numpy(step_seq).float()

        with torch.no_grad():
            h_seq = self.P_R(r_tensor) + self.proj_rtg(rtg_tensor) + self.proj_step(step_tensor) + self.type_R
            h_seq = torch.nn.functional.layer_norm(h_seq, [h_seq.shape[-1]])
        return (
            h_seq.cpu().numpy().astype(np.float32),
            rtg_seq.astype(np.float32),
            step_seq.astype(np.float32),
        )

    def _compute_fingerprint(self, files: Sequence[Path]) -> Dict[str, object]:
        entries = []
        for f in sorted(files):
            entries.append({"path": str(f), "md5": _compute_md5(f)})
        agg_input = "".join(e["md5"] + e["path"] for e in entries).encode()
        aggregate = hashlib.md5(agg_input).hexdigest()
        return {"files": entries, "aggregate_md5": aggregate}

    def generate_shard(self, angle_range: AngleRange, max_samples: int | None = None) -> Path:
        angles = angle_range.angles()
        files, file_angles = self._select_files(angles, max_samples)
        if not files:
            raise ValueError(f"No audio files selected for range {angle_range.name}")

        outputs = []
        expert_gt_list = []
        atom_gt_list = []
        angle_gt_list = []
        angle_deg_list = []
        rtg_list = []
        step_list = []
        traj_records = []
        teacher_records = []
        actual_lengths: List[int] = []
        per_angle_counts: Dict[float, Counter] = {}

        for path, angle_deg in zip(files, file_angles):
            wav = np.load(path)
            if wav.ndim != 1:
                raise ValueError(f"Expected mono waveform at {path}, got shape {wav.shape}")
            y = self._compute_spectrum(wav)
            angle_idx = self.angle_to_index[angle_deg]
            steps, residuals, actual_len = self._run_omp(y, angle_idx)
            actual_lengths.append(actual_len)

            # Pad residuals to length K for downstream batching.
            if actual_len < self.config.k:
                pad = np.zeros((self.config.k - actual_len, residuals.shape[1]), dtype=residuals.dtype)
                residuals_padded = np.concatenate([residuals, pad], axis=0)
            else:
                residuals_padded = residuals

            h_seq, rtg_seq, step_seq = self._convert_to_embeddings(residuals_padded)
            experts = [s["expert"] for s in steps]
            atoms = [s["atom"] for s in steps]

            expert_padded = np.full(self.config.k, fill_value=-100, dtype=np.int64)
            atom_padded = np.full(self.config.k, fill_value=-100, dtype=np.int64)
            expert_padded[:actual_len] = np.array(experts, dtype=np.int64)
            atom_padded[:actual_len] = np.array(atoms, dtype=np.int64)

            expert_gt_list.append(expert_padded)
            atom_gt_list.append(atom_padded)
            outputs.append(h_seq[: self.config.k])
            rtg_list.append(rtg_seq[: self.config.k])
            step_list.append(step_seq[: self.config.k])
            angle_gt_list.append(angle_idx)
            angle_deg_list.append(float(angle_deg))
            voted = Counter(experts).most_common(1)[0][0]
            first_hit = int(experts[0] == angle_idx)
            joint_hit = int(all(e == angle_idx for e in experts))
            voted_hit = int(voted == angle_idx)
            prefix_hits = [int(Counter(experts[:i]).most_common(1)[0][0] == angle_idx) for i in range(1, len(experts) + 1)]
            teacher_records.append(
                {
                    "path": str(path),
                    "angle_deg": float(angle_deg),
                    "angle_idx": angle_idx,
                    "first_step_correct": first_hit,
                    "joint_correct": joint_hit,
                    "voted_correct": voted_hit,
                    "any_prefix": int(any(prefix_hits)),
                    "avg_prefix": float(sum(prefix_hits) / len(prefix_hits)),
                }
            )
            traj_records.append(
                {"path": str(path), "angle_deg": float(angle_deg), "angle_idx": angle_idx, "steps": steps}
            )
            agg = per_angle_counts.setdefault(float(angle_deg), Counter())
            agg["count"] += 1
            agg["first"] += first_hit
            agg["voted"] += voted_hit
            agg["joint"] += joint_hit

        shard_dir = self.config.output_root / angle_range.name
        shard_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            shard_dir / "embeddings.npz",
            h_seq=np.stack(outputs),
            rtg_seq=np.stack(rtg_list),
            step_seq=np.stack(step_list),
            expert_gt=np.stack(expert_gt_list),
            atom_gt=np.stack(atom_gt_list),
            angle_gt=np.array(angle_gt_list, dtype=np.int64),
            angle_deg_gt=np.array(angle_deg_list, dtype=np.float32),
            actual_length=np.array(actual_lengths, dtype=np.int64),
        )

        with (shard_dir / "trajectories.jsonl").open("w") as f:
            for row in traj_records:
                f.write(json.dumps(row) + "\n")

        fingerprint = self._compute_fingerprint(files)

        first_step_acc = float(np.mean([r["first_step_correct"] for r in teacher_records]))
        joint_acc = float(np.mean([r["joint_correct"] for r in teacher_records]))
        voted_acc = float(np.mean([r["voted_correct"] for r in teacher_records]))
        any_prefix = float(np.mean([r["any_prefix"] for r in teacher_records]))
        avg_prefix = float(np.mean([r["avg_prefix"] for r in teacher_records]))
        step_hist = Counter(actual_lengths)
        summary = {
            "samples": len(outputs),
            "angles": angles,
            "clips_per_angle": self.config.clips_per_angle,
            "first_step_acc": first_step_acc,
            "joint_acc": joint_acc,
            "voted_acc": voted_acc,
            "any_prefix": any_prefix,
            "avg_prefix": avg_prefix,
            "actual_steps": {
                "min": min(actual_lengths),
                "max": max(actual_lengths),
                "hist": {int(k): int(v) for k, v in sorted(step_hist.items())},
            },
            "per_angle": {
                str(a): {
                    "count": int(c["count"]),
                    "first_step_acc": float(c["first"] / max(c["count"], 1)),
                    "voted_acc": float(c["voted"] / max(c["count"], 1)),
                    "joint_acc": float(c["joint"] / max(c["count"], 1)),
                }
                for a, c in sorted(per_angle_counts.items())
            },
            "fingerprint": fingerprint,
            "stft": {
                "fs": self.config.target_sample_rate,
                "n_fft": self.config.n_fft,
                "freq_min": self.config.freq_min,
                "freq_max": self.config.freq_max,
                "F_expected": self.expected_F,
            },
            "dictionary": {
                "W_path": str(self.config.w_path),
                "H_path": str(self.config.h_path),
                "M": self.config.m,
                "K": self.config.k,
                "reduction_seed": self.config.reduction_seed,
                "projection_seed": self.config.projection_seed,
                "normalize_w": self.config.normalize_w,
                "normalize_d": self.config.normalize_d,
            },
        }

        with (shard_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)

        torch.save(
            {"W_reduced": self.W_reduced, "D": self.D, "angles": self.angles},
            shard_dir / "dictionary.pth",
        )

        logger.info(
            "Generated shard %s: %d samples, first_step_acc=%.3f voted_acc=%.3f",
            angle_range.name,
            len(outputs),
            summary["first_step_acc"],
            summary["voted_acc"],
        )
        return shard_dir
