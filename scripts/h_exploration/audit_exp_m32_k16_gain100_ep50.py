import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def _run(cmd: List[str], cwd: Path | None = None) -> str:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="replace").strip()
    except Exception as e:
        return f"<failed: {e}>"


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass
class FileRecord:
    path: str
    size_bytes: int
    mtime_unix: float
    md5: str


@dataclass
class AuditSummary:
    timestamp: str
    exp_name: str
    out_dir: str
    python_executable: str
    platform: str
    git_head: str
    git_dirty: bool

    trajectories_path: str
    num_blocks: int
    total_flat_sequences: int
    per_block_num_freqs_min: int
    per_block_num_freqs_median: float
    per_block_num_freqs_max: int
    block_keys: List[str]
    duplicate_action_score_signatures: int
    max_signature_count: int

    mic_root: str
    ldv_root: str
    angle_dir: str
    num_clip_pairs: int
    dataset_fingerprint_md5: str

    notes: List[str]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="exp_m32_k16_full_gain100_ep50")
    parser.add_argument(
        "--repo_root",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
        help="Path to exp-interspeech-GRU1 repo root",
    )
    parser.add_argument("--mic_root", type=str, required=True)
    parser.add_argument("--ldv_root", type=str, required=True)
    parser.add_argument("--angle", type=int, default=90)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = repo_root / "results" / args.exp_name
    checks_dir = out_dir / "checks"
    checks_dir.mkdir(parents=True, exist_ok=True)

    traj_path = out_dir / "data" / "lag_trajectories.pt"
    if not traj_path.exists():
        raise FileNotFoundError(f"Missing trajectories: {traj_path}")

    # --- Load trajectories ---
    data = torch.load(traj_path, map_location="cpu")
    if not isinstance(data, list) or not data:
        raise ValueError("lag_trajectories.pt expected a non-empty list")

    block_keys = sorted(list(data[0].keys()))

    # Each block stores per-frequency sequences; training flattens across frequency.
    # This is important for leakage interpretation of train/val random_split.
    per_block_freqs = [int(b["corrs"].shape[0]) for b in data]
    total_flat = int(sum(per_block_freqs))
    per_block_sorted = sorted(per_block_freqs)
    if per_block_sorted:
        mid = len(per_block_sorted) // 2
        if len(per_block_sorted) % 2 == 1:
            median_freqs = float(per_block_sorted[mid])
        else:
            median_freqs = 0.5 * (per_block_sorted[mid - 1] + per_block_sorted[mid])
        min_freqs = int(per_block_sorted[0])
        max_freqs = int(per_block_sorted[-1])
    else:
        median_freqs = 0.0
        min_freqs = 0
        max_freqs = 0

    # Duplicate detection based on action+score signature (cheap but strict)
    signatures: Dict[str, int] = {}
    for b in data:
        a = b["actions"].numpy().tobytes()
        s = b["scores"].numpy().tobytes()
        sig = hashlib.sha1(a + s).hexdigest()
        signatures[sig] = signatures.get(sig, 0) + 1

    dup_count = sum(1 for c in signatures.values() if c > 1)
    max_count = max(signatures.values())

    # --- Enumerate dataset clip pairs ---
    mic_root = Path(args.mic_root)
    ldv_root = Path(args.ldv_root)
    angle_dir = f"angle_{args.angle}"
    mic_dir = mic_root / angle_dir

    if not mic_dir.exists():
        raise FileNotFoundError(f"Missing mic angle dir: {mic_dir}")

    pairs: List[Tuple[Path, Path]] = []
    for mic_path in sorted(mic_dir.glob("*.npy")):
        rel = mic_path.relative_to(mic_root)
        ldv_path = ldv_root / rel
        if ldv_path.exists():
            pairs.append((mic_path, ldv_path))

    # Save subset manifest (paths only)
    manifest = {
        "exp_name": args.exp_name,
        "mic_root": str(mic_root),
        "ldv_root": str(ldv_root),
        "angle": args.angle,
        "num_pairs": len(pairs),
        "pairs": [{"mic": str(m), "ldv": str(l)} for (m, l) in pairs],
    }
    (checks_dir / "subset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Compute file-level md5s (can take a bit; required for reproducibility)
    file_records: List[FileRecord] = []
    for m, l in pairs:
        for p in (m, l):
            st = p.stat()
            file_records.append(
                FileRecord(
                    path=str(p),
                    size_bytes=st.st_size,
                    mtime_unix=st.st_mtime,
                    md5=_md5_file(p),
                )
            )

    (checks_dir / "file_md5.json").write_text(
        json.dumps([asdict(r) for r in file_records], indent=2), encoding="utf-8"
    )

    # Dataset fingerprint = md5 of sorted (md5 path) lines
    lines = sorted([f"{r.md5}  {r.path}" for r in file_records])
    fp = hashlib.md5("\n".join(lines).encode("utf-8")).hexdigest()
    (checks_dir / "data_fingerprint.txt").write_text(fp + "\n", encoding="utf-8")

    # Code state
    git_head = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    git_dirty = _run(["git", "status", "--porcelain"], cwd=repo_root) != ""

    key_files = [
        repo_root / "run_pipeline_m32.sh",
        repo_root / "scripts" / "h_exploration" / "generate_lag_omp.py",
        repo_root / "scripts" / "h_exploration" / "train_dt_lag_seq_rtg.py",
        repo_root / "scripts" / "eval_energy_capture_generic.py",
    ]
    code_state = {
        "timestamp": datetime.now().isoformat(),
        "git_head": git_head,
        "git_dirty": git_dirty,
        "python": sys.executable,
        "platform": platform.platform(),
        "files": [
            {
                "path": str(p.relative_to(repo_root)),
                "sha256": _sha256_file(p) if p.exists() else None,
            }
            for p in key_files
        ],
    }
    (checks_dir / "code_state.json").write_text(json.dumps(code_state, indent=2), encoding="utf-8")

    notes: List[str] = []
    notes.append(
        "No exact duplicate blocks detected (signature = sha1(actions||scores)). This rules out trivial duplication."
    )
    notes.append(
        "Trajectory blocks do not store clip identifiers. This prevents rigorous clip-level leakage checks (e.g., ensuring train/val are split by clip)."
    )
    notes.append(
        "Training flattens each block across frequency (one sample per frequency). With random_split at this flat-sequence level, it is very likely that the same underlying time-window block contributes samples to both train and val (train/val contamination within-block)."
    )
    notes.append(
        "Evaluation uses the same dataset root as trajectory generation; therefore it is in-distribution and not a held-out generalization test."
    )
    notes.append(
        "RTG in eval is computed from residual energy over the observed LDV window Y; this is not leakage if Y is available at inference, but it is a non-causal/lookahead feature if deployment requires streaming decisions without access to the full Tw-block."
    )

    summary = AuditSummary(
        timestamp=datetime.now().isoformat(),
        exp_name=args.exp_name,
        out_dir=str(out_dir),
        python_executable=sys.executable,
        platform=platform.platform(),
        git_head=git_head,
        git_dirty=git_dirty,
        trajectories_path=str(traj_path),
        num_blocks=len(data),
        total_flat_sequences=total_flat,
        per_block_num_freqs_min=min_freqs,
        per_block_num_freqs_median=median_freqs,
        per_block_num_freqs_max=max_freqs,
        block_keys=block_keys,
        duplicate_action_score_signatures=dup_count,
        max_signature_count=max_count,
        mic_root=str(mic_root),
        ldv_root=str(ldv_root),
        angle_dir=angle_dir,
        num_clip_pairs=len(pairs),
        dataset_fingerprint_md5=fp,
        notes=notes,
    )

    (checks_dir / "audit_summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
