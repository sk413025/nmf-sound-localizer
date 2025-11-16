#!/usr/bin/env python3
"""
Post-hoc modal visualizations for the OMP Transformer Speech260 train/val split.

Goal: focus on the relation between the learned routing and the underlying
modal dictionary D (built from H and W), rather than on a teacher/student
story. All analysis is post-hoc and does not modify or retrain the model.

This script:
  1) Reloads the trained OMP Transformer and LDV dataset for a given run_dir.
  2) Runs a routing-only pass on the validation subset to collect:
       - routing scores over dictionary atoms,
       - modal baseline energy from |D^T y|.
  3) Produces six modal-focused figures under run_dir:
       (1) DOA×atom modal activation map
       (2) DOA×DOA similarity in modal space vs confusion
       (3) Modal sparsity/entropy per DOA
       (4) Modal “families”: atom frequency shapes + DOA profiles
       (5) Alignment between routing over angles and modal baseline
       (6) Error-angle analysis (55°, 100°, 175°) in modal space
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_run_state(run_dir: Path):
    """Load code_state.json, metrics.npz, diagnostics.jsonl for a run."""
    code_state_path = run_dir / "code_state.json"
    metrics_path = run_dir / "metrics.npz"
    diagnostics_path = run_dir / "diagnostics.jsonl"

    if not code_state_path.is_file():
        raise SystemExit(f"Missing code_state.json in run_dir: {code_state_path}")
    if not metrics_path.is_file():
        raise SystemExit(f"Missing metrics.npz in run_dir: {metrics_path}")
    if not diagnostics_path.is_file():
        raise SystemExit(f"Missing diagnostics.jsonl in run_dir: {diagnostics_path}")

    with code_state_path.open("r") as f:
        code_state = json.load(f)
    metrics = np.load(metrics_path, allow_pickle=True)

    return code_state, metrics, diagnostics_path


def build_model_and_data(run_dir: Path, code_state, device: torch.device):
    """
    Rebuild dictionary, dataset and model using the same configuration as training.

    Uses helper utilities from scripts.omp-transformer-ldv and
    scripts.eval_omp_transformer_split.
    """
    import importlib

    omp_mod = importlib.import_module("scripts.omp-transformer-ldv")
    eval_mod = importlib.import_module("scripts.eval_omp_transformer_split")

    args_dict = code_state.get("args", {})

    h_path = args_dict.get("h_path")
    w_path = args_dict.get("w_path")
    dataset_root = args_dict.get("dataset_root")
    if not (h_path and w_path and dataset_root):
        raise SystemExit("code_state.args is missing h_path, w_path, or dataset_root.")

    # Rebuild data pipeline (mirrors training/eval scripts)
    H_raw, W_raw, angles = omp_mod.load_raw_ldv_matrices(h_path, w_path, device="cpu")

    atom_reduce_mode = args_dict.get("atom_reduce_mode", "kmeans")
    n_atoms = int(args_dict.get("n_atoms", 8))
    atom_min_cos = float(args_dict.get("atom_min_cos", 0.98))

    if atom_reduce_mode == "kmeans":
        W_reduced, _, _ = omp_mod.reduce_atoms_kmeans(
            W_raw, n_clusters=n_atoms, random_state=42
        )
    else:
        W_reduced, _, _ = omp_mod.reduce_atoms(
            W_raw,
            mode=atom_reduce_mode,
            n_clusters=n_atoms,
            random_state=42,
            min_cos=atom_min_cos,
        )

    H_final = H_raw
    W_final = W_reduced
    D, idx2angle = omp_mod.build_dictionary(H_final, W_final, angles)

    Y_samples, labels, metadata = omp_mod.load_ldv_samples(
        dataset_root, H_final, W_final, angles, device="cpu"
    )

    # Deterministic train/validation split based on clip_id % 5
    _, val_mask, _ = omp_mod.split_train_val_by_clip_id(metadata, val_mod=5)
    Y_val = Y_samples[val_mask]
    labels_val = labels[val_mask]

    # Build model and load checkpoint from run_dir
    model = eval_mod.build_model_from_args(
        omp_mod, H_final, W_final, angles, args_dict, device=device
    )
    ckpt_path = run_dir / "model_best.pth"
    if not ckpt_path.is_file():
        raise SystemExit(f"Missing checkpoint in run_dir: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    return omp_mod, model.to(device), D.to(device), Y_val, labels_val, np.array(angles)


def dump_modal_routing(
    run_dir: Path,
    device: torch.device,
    max_val_samples: int | None = None,
):
    """
    Run a routing-only pass on the validation subset and dump modal routing
    signals to modal_routing_val.npz under run_dir.

    We collect:
      - scores_atoms: routing scores over atoms (E×M) from QK-based router
      - scores_expert: routing scores over experts (angles)
      - g_energy_expert: modal baseline energy from |D^T y| aggregated per expert
    """
    code_state, _, _ = load_run_state(run_dir)
    omp_mod, model, D, Y_val, labels_val, angles = build_model_and_data(
        run_dir, code_state, device
    )

    model.eval()

    E = model.E
    M = model.M
    N_val = Y_val.size(0)
    N_eff = min(N_val, int(max_val_samples)) if max_val_samples is not None else N_val

    scores_expert_all = np.zeros((N_eff, E), dtype=np.float32)
    scores_atoms_all = np.zeros((N_eff, E, M), dtype=np.float32)
    g_energy_expert_all = np.zeros((N_eff, E), dtype=np.float32)
    selected_counts = np.zeros(N_eff, dtype=np.int32)
    selected_mask = np.zeros((N_eff, E * M), dtype=bool)

    labels_np = labels_val.cpu().numpy()

    with torch.no_grad():
        for i in range(N_eff):
            y = Y_val[i].to(device)
            model.enable_diag = True
            x_hat, r_curve = model(y, D, train_mode=True)

            if getattr(model, "last_outputs", None) is None:
                raise RuntimeError("model.last_outputs is None after forward; cannot extract routing.")

            se = model.last_outputs.get("scores_expert", None)
            sa = model.last_outputs.get("scores_atoms", None)
            if se is None or sa is None:
                raise RuntimeError("scores_expert or scores_atoms missing from last_outputs.")

            scores_expert_all[i] = se.detach().cpu().numpy().astype(np.float32)
            scores_atoms_all[i] = sa.detach().cpu().numpy().astype(np.float32)

            # Hard-selection sparsity: support_indices 存的是這個樣本在所有步驟中選到的全域 atom index
            support = None
            if getattr(model, "last_outputs", None) is not None:
                support = model.last_outputs.get("support_indices", None)
            if support is not None:
                # 去重並保護 index 範圍
                uniq = []
                for idx in support:
                    idx_int = int(idx)
                    if 0 <= idx_int < E * M and idx_int not in uniq:
                        uniq.append(idx_int)
                selected_counts[i] = len(uniq)
                if uniq:
                    selected_mask[i, uniq] = True

            # Modal baseline from |D^T y|
            g_vec = (D.T @ y)  # (P,)
            g_energy = g_vec.abs().view(E, M).sum(dim=1)  # (E,)
            g_energy_expert_all[i] = g_energy.detach().cpu().numpy().astype(np.float32)

    # Save corresponding Y_val slice for spectral visualization
    Y_val_np = Y_val[:N_eff].cpu().numpy().astype(np.float32)

    out_npz = run_dir / "modal_routing_val.npz"
    np.savez(
        out_npz,
        labels=labels_np[:N_eff],
        angles_deg=angles,
        scores_expert=scores_expert_all,
        scores_atoms=scores_atoms_all,
        g_energy_expert=g_energy_expert_all,
        Y_val=Y_val_np,
        selected_counts=selected_counts,
        selected_mask=selected_mask,
    )
    print(f"[modal] Saved modal routing signals to: {out_npz}")


def aggregate_per_angle_modal(routing_npz: Path):
    """Aggregate per-angle modal routing statistics from modal_routing_val.npz."""
    data = np.load(routing_npz, allow_pickle=True)
    labels = data["labels"]
    angles_deg = data["angles_deg"]
    scores_expert = data["scores_expert"]  # (N,E)
    scores_atoms = data["scores_atoms"]  # (N,E,M)
    g_energy_expert = data["g_energy_expert"]  # (N,E)
    Y_val = data.get("Y_val", None)
    selected_counts = data.get("selected_counts", None)
    selected_mask = data.get("selected_mask", None)

    E = scores_expert.shape[1]
    M = scores_atoms.shape[2]
    P = E * M

    scores_atoms_flat = scores_atoms.reshape(scores_atoms.shape[0], P)

    per_angle_scores_atoms = np.zeros((E, P), dtype=np.float32)
    per_angle_scores_expert = np.zeros((E, E), dtype=np.float32)
    per_angle_g_expert = np.zeros((E, E), dtype=np.float32)

    for e in range(E):
        mask = (labels == e)
        if mask.sum() == 0:
            continue
        per_angle_scores_atoms[e] = scores_atoms_flat[mask].mean(axis=0)
        per_angle_scores_expert[e] = scores_expert[mask].mean(axis=0)
        per_angle_g_expert[e] = g_energy_expert[mask].mean(axis=0)

    return {
        "labels": labels,
        "angles_deg": angles_deg,
        "per_angle_scores_atoms": per_angle_scores_atoms,
        "per_angle_scores_expert": per_angle_scores_expert,
        "per_angle_g_expert": per_angle_g_expert,
        "scores_expert_all": scores_expert,
        "scores_atoms_all": scores_atoms,
        "selected_counts": selected_counts,
        "selected_mask": selected_mask,
        "Y_val": Y_val,
    }


def plot_doa_atom_activation(run_dir: Path, agg):
    """Experiment 1: DOA×atom modal activation map."""
    angles_deg = agg["angles_deg"]
    per_angle_scores_atoms = agg["per_angle_scores_atoms"]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(per_angle_scores_atoms, aspect="auto", cmap="viridis")
    ax.set_xlabel("Dictionary atom index")
    ax.set_ylabel("Angle (deg)")
    ax.set_yticks(np.arange(len(angles_deg)))
    ax.set_yticklabels([f"{int(a)}" for a in angles_deg])
    ax.set_title("Per-angle activation over modal dictionary atoms")
    plt.colorbar(im, ax=ax, label="Mean routing score")
    fig.tight_layout()
    out_path = run_dir / "modal_doa_atom_activation.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[exp1] Saved DOA×atom modal activation map: {out_path}")


def plot_doa_atom_clustered(run_dir: Path, agg, n_clusters: int = 8):
    """
    Alternative atom-level view: cluster atoms by their DOA profiles, then plot.

    Instead of按原始 index 0..P-1 排序，我們先把每個 atom 看成一個
    向量 v_p(θ) = per_angle_scores_atoms[:, p]，在 R^E 中做簡單 K-Means，
    再依 cluster 與「偏好 DOA」排序，畫出重新排序後的 heatmap。

    這樣：
      - 每個 cluster 是一群「對 DOA 有相似反應模式」的等效 modal fingerprints；
      - heatmap 的 x 軸會出現清楚的區段，而不是雜亂的 296 根柱子。
    """
    angles_deg = agg["angles_deg"]
    per_angle_scores_atoms = agg["per_angle_scores_atoms"]  # (E,P)
    E, P = per_angle_scores_atoms.shape

    X = per_angle_scores_atoms.T  # (P,E) 每一 row 是一個 atom 的 DOA profile

    # 簡單 K-Means（L2 距離），不依賴外部套件；P=296、E=37 很小。
    rng = np.random.RandomState(0)
    n_clusters = min(n_clusters, P)
    # 初始化：隨機挑 K 個 atoms 當中心
    init_idx = rng.choice(P, size=n_clusters, replace=False)
    centers = X[init_idx].copy()  # (K,E)

    for _ in range(50):
        # 指派 cluster
        # 距離 = squared L2
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # (P,K)
        labels = dists.argmin(axis=1)  # (P,)
        # 更新中心
        new_centers = np.zeros_like(centers)
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                new_centers[k] = X[mask].mean(axis=0)
            else:
                # 若有空 cluster，隨機重啟一個中心
                new_centers[k] = X[rng.randint(P)]
        if np.allclose(new_centers, centers, atol=1e-4):
            centers = new_centers
            break
        centers = new_centers

    # 對每個 cluster 計算「偏好 DOA」（加權平均角度），用來排序 cluster
    cluster_order = []
    for k in range(n_clusters):
        mask = labels == k
        if not mask.any():
            cluster_order.append((k, 999.0))
            continue
        v = centers[k]  # (E,)
        w = np.abs(v)
        mu = float((w * angles_deg).sum() / (w.sum() + 1e-8))
        cluster_order.append((k, mu))
    cluster_order.sort(key=lambda x: x[1])  # 依偏好角度排序

    # 依 cluster 排序 atoms，cluster 內再依偏好角度細排
    ordered_indices = []
    for k, _mu in cluster_order:
        mask = labels == k
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        # 依 atom 自己的偏好 DOA 排序
        sub_mu = []
        for p in idx:
            v = X[p]
            w = np.abs(v)
            mu_p = float((w * angles_deg).sum() / (w.sum() + 1e-8))
            sub_mu.append((p, mu_p))
        sub_mu.sort(key=lambda x: x[1])
        ordered_indices.extend([p for p, _ in sub_mu])

    # 重新排列 column：E×P
    reordered = per_angle_scores_atoms[:, ordered_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(reordered, aspect="auto", cmap="viridis")
    ax.set_xlabel("Atom index (clustered and sorted)")
    ax.set_ylabel("Angle (deg)")
    ax.set_yticks(np.arange(len(angles_deg))[::6])
    ax.set_yticklabels([f"{int(a)}" for a in angles_deg[::6]])
    ax.set_title(f"Per-angle activation over modal atoms (clustered, K={n_clusters})")
    plt.colorbar(im, ax=ax, label="Mean routing score")
    fig.tight_layout()
    out_path = run_dir / f"modal_doa_atom_activation_clustered_K{n_clusters}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[exp1] Saved clustered DOA×atom activation map: {out_path}")


def plot_doa_expert_activation(run_dir: Path, agg):
    """
    Compressed view of DOA×atom activation: aggregate atoms into expert blocks.

    Instead of 37×296 的 full matrix，這張圖只看：
        true DOA (row) × 字典 expert (column，對應 H 的角度)
    使用 per_angle_scores_expert (E×E)，直接顯示：
        「某個真實 DOA 下，路由在字典中每個 DOA-block 上的總分數」。
    這可以被讀成：routing 在 H(f,θ) 的角度軸上有多接近對角線。
    """
    angles_deg = agg["angles_deg"]
    per_angle_scores_expert = agg["per_angle_scores_expert"]  # (E,E)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(per_angle_scores_expert, aspect="auto", cmap="viridis")
    ax.set_xlabel("Dictionary expert angle (deg)")
    ax.set_ylabel("True DOA (deg)")
    xticks = np.arange(len(angles_deg))[::6]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(a)}" for a in angles_deg[xticks]])
    yticks = np.arange(len(angles_deg))[::6]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(a)}" for a in angles_deg[yticks]])
    ax.set_title("Per-angle activation over expert DOA blocks")
    plt.colorbar(im, ax=ax, label="Mean routing score")
    fig.tight_layout()
    out_path = run_dir / "modal_doa_expert_activation.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[exp1] Saved DOA×expert activation heatmap: {out_path}")


def plot_doa_similarity_and_confusion(run_dir: Path, agg, metrics):
    """Experiment 2: DOA×DOA similarity in modal space vs confusion."""
    angles_deg = agg["angles_deg"]
    per_angle_scores_atoms = agg["per_angle_scores_atoms"]  # (E,P)
    confusion = metrics["confusion_matrix"]

    X = per_angle_scores_atoms
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    Xn = X / norms
    sim = Xn @ Xn.T  # (E,E)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(sim, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axes[0].set_title("DOA×DOA similarity in modal dictionary space")
    axes[0].set_xlabel("Angle (deg)")
    axes[0].set_ylabel("Angle (deg)")
    ticks = np.arange(len(angles_deg))[::6]
    axes[0].set_xticks(ticks)
    axes[0].set_xticklabels([f"{int(a)}" for a in angles_deg[ticks]])
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels([f"{int(a)}" for a in angles_deg[ticks]])
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    conf = confusion.astype(np.float32)
    row_sums = conf.sum(axis=1, keepdims=True) + 1e-8
    conf_norm = conf / row_sums

    im1 = axes[1].imshow(conf_norm, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    axes[1].set_title("Validation confusion matrix (row-normalized)")
    axes[1].set_xlabel("Predicted angle (deg)")
    axes[1].set_ylabel("True angle (deg)")
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels([f"{int(a)}" for a in angles_deg[ticks]])
    axes[1].set_yticks(ticks)
    axes[1].set_yticklabels([f"{int(a)}" for a in angles_deg[ticks]])
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path = run_dir / "modal_similarity_vs_confusion.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[exp2] Saved DOA×DOA modal similarity vs confusion: {out_path}")


def plot_modal_sparsity_per_angle(run_dir: Path, agg):
    """Experiment 3: Modal sparsity and entropy per DOA."""
    angles_deg = agg["angles_deg"]
    per_angle_scores_atoms = agg["per_angle_scores_atoms"]  # (E,P)

    E, P = per_angle_scores_atoms.shape
    effective_atoms = np.zeros(E, dtype=np.float32)
    entropy = np.zeros(E, dtype=np.float32)
    top10_mass = np.zeros(E, dtype=np.float32)

    for e in range(E):
        s = per_angle_scores_atoms[e]
        s_abs = np.abs(s)
        if np.allclose(s_abs, 0):
            effective_atoms[e] = 0.0
            entropy[e] = 0.0
            top10_mass[e] = 0.0
            continue
        p = s_abs / (s_abs.sum() + 1e-8)
        H = -np.sum(p * np.log(p + 1e-12))
        entropy[e] = H
        effective_atoms[e] = float(np.exp(H))
        top10_mass[e] = float(np.sort(p)[-10:].sum())

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(angles_deg, effective_atoms, marker="o")
    axes[0].set_ylabel("Effective atoms (exp entropy)")
    axes[0].set_title("Effective modal atoms used per DOA")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(angles_deg, top10_mass, marker="x")
    axes[1].set_xlabel("Angle (deg)")
    axes[1].set_ylabel("Top-10 mass")
    axes[1].set_title("Fraction of routing mass in top 10 atoms")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = run_dir / "modal_sparsity_per_angle.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[exp3] Saved modal sparsity per angle: {out_path}")


def plot_atom_modal_families(run_dir: Path, agg, D: torch.Tensor, top_k: int = 8):
    """
    Experiment 4: Atom frequency shapes and DOA profiles (modal families).

    To avoid showing many nearly-identical atoms, we first cluster atoms in the
    DOA-profile space, then pick the top-1 most-used atom (by mean |score|)
    from each cluster as the representative modal fingerprint.
    """
    angles_deg = agg["angles_deg"]
    per_angle_scores_atoms = agg["per_angle_scores_atoms"]  # (E,P)
    D_np = D.cpu().numpy()  # (F,P) over STFT band [300,3000] Hz

    E, P = per_angle_scores_atoms.shape

    # 使用 mean(|score|) 作為重要度：代表此 atom 在各個 DOA 下被網路「強烈使用」的程度
    importance = np.abs(per_angle_scores_atoms).mean(axis=0)  # (P,)

    # Step 1: 在 DOA-profile 空間中對 atoms 分群
    X = per_angle_scores_atoms.T  # (P,E) 每 row 是一個 atom 的 DOA profile
    n_clusters = min(top_k, P)
    rng = np.random.RandomState(0)
    init_idx = rng.choice(P, size=n_clusters, replace=False)
    centers = X[init_idx].copy()  # (K,E)

    for _ in range(50):
        # 指派 cluster（L2 距離）
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # (P,K)
        labels = dists.argmin(axis=1)  # (P,)
        # 更新中心
        new_centers = np.zeros_like(centers)
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                new_centers[k] = X[mask].mean(axis=0)
            else:
                new_centers[k] = X[rng.randint(P)]
        if np.allclose(new_centers, centers, atol=1e-4):
            centers = new_centers
            break
        centers = new_centers

    # Step 2: 對每個 cluster 挑出 importance 最高的一個 atom 作為代表
    chosen_atoms: list[int] = []
    for k in range(n_clusters):
        mask = labels == k
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        best_local = idx[np.argmax(importance[idx])]
        chosen_atoms.append(int(best_local))

    # 若 cluster 數量 < top_k，就只畫實際可用的
    top_idx = chosen_atoms

    F, P = D_np.shape

    # 建立對應頻率軸，並只在 300–2000 Hz 區間內繪圖（更聚焦於主要情報頻帶）
    fs = 16000.0
    n_fft = 2048.0
    df = fs / n_fft  # 7.8125 Hz
    f_min = 300.0
    f_max = 3000.0
    k_start = int(np.ceil(f_min / df))
    # H/W 已經裁成 [300,3000]，所以第 i 列對應的 STFT bin index = k_start + i
    freqs = (k_start + np.arange(F)) * df
    # 只看 300–2000 Hz
    freq_mask = (freqs >= 300.0) & (freqs <= 2000.0)
    freqs_plot = freqs[freq_mask]
    D_plot = D_np[freq_mask, :]  # (F_plot, P)

    fig, axes = plt.subplots(top_k, 2, figsize=(10, 2.3 * top_k))

    for row, p in enumerate(top_idx):
        ax_freq = axes[row, 0]
        ax_doa = axes[row, 1]

        # 頻譜圖只顯示 300–2000 Hz
        ax_freq.plot(freqs_plot, D_plot[:, p])
        ax_freq.set_title(f"Atom {p} frequency shape (300–2000 Hz)")
        ax_freq.set_xlabel("Frequency (Hz)")
        ax_freq.set_ylabel("Amplitude")
        ax_freq.grid(True, alpha=0.3)

        E = per_angle_scores_atoms.shape[0]
        atom_profile = np.zeros(E, dtype=np.float32)
        for e in range(E):
            atom_profile[e] = per_angle_scores_atoms[e, p]
        ax_doa.plot(angles_deg, atom_profile, marker="o")
        ax_doa.set_title(f"Atom {p} activation vs DOA")
        ax_doa.set_xlabel("Angle (deg)")
        ax_doa.set_ylabel("Mean score")
        ax_doa.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = run_dir / "modal_atom_families.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[exp4] Saved modal atom families plot: {out_path}")


def plot_modal_alignment_with_baseline(run_dir: Path, agg):
    """Experiment 5: Alignment between routing and modal baseline per angle."""
    angles_deg = agg["angles_deg"]
    per_angle_scores_expert = agg["per_angle_scores_expert"]  # (E,E)
    per_angle_g_expert = agg["per_angle_g_expert"]  # (E,E)
    E = per_angle_scores_expert.shape[0]

    corrs = np.zeros(E, dtype=np.float32)
    for e in range(E):
        s = per_angle_scores_expert[e]
        g = per_angle_g_expert[e]
        if np.allclose(s, 0) or np.allclose(g, 0):
            corrs[e] = np.nan
            continue
        s_z = (s - s.mean()) / (s.std() + 1e-8)
        g_z = (g - g.mean()) / (g.std() + 1e-8)
        corrs[e] = float(np.mean(s_z * g_z))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(angles_deg, corrs, marker="o")
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("corr(routing, |D^T y|)")
    ax.set_title("Per-angle alignment between routing scores and modal baseline")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = run_dir / "modal_alignment_with_baseline.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[exp5] Saved modal alignment with baseline: {out_path}")


def plot_error_angle_modal_slices(run_dir: Path, agg, metrics):
    """Experiment 6: Modal view of error angles (55°, 100°, 175°)."""
    angles_deg = agg["angles_deg"]
    per_angle_scores_expert = agg["per_angle_scores_expert"]
    per_angle_g_expert = agg["per_angle_g_expert"]
    confusion = metrics["confusion_matrix"]
    E = confusion.shape[0]

    target_angles = [55, 100, 175]

    fig, axes = plt.subplots(len(target_angles), 2, figsize=(10, 4 * len(target_angles)))

    for row, target_deg in enumerate(target_angles):
        if target_deg not in list(angles_deg):
            continue
        e = int(np.where(angles_deg == target_deg)[0][0])

        conf_row = confusion[e].astype(np.float32)
        conf_row = conf_row / (conf_row.sum() + 1e-8)
        ax_conf = axes[row, 0]
        xs = np.arange(E)
        ax_conf.bar(xs, conf_row)
        ax_conf.set_title(f"Confusion row for true DOA {target_deg}°")
        ax_conf.set_xlabel("Predicted angle (deg)")
        ax_conf.set_ylabel("Probability")
        ax_conf.grid(True, alpha=0.3)
        ticks = np.arange(len(angles_deg))[::6]
        ax_conf.set_xticks(ticks)
        ax_conf.set_xticklabels([f"{int(a)}" for a in angles_deg[ticks]], rotation=45)

        ax_route = axes[row, 1]
        xs_expert = np.arange(E)
        ax_route.plot(angles_deg, per_angle_g_expert[e], label="Modal baseline |D^T y|", marker="o")
        ax_route.plot(angles_deg, per_angle_scores_expert[e], label="Routing scores", marker="x")
        ax_route.set_title(f"Expert-wise modal patterns for DOA {target_deg}°")
        ax_route.set_xlabel("Expert angle (deg)")
        ax_route.set_ylabel("Score")
        ax_route.grid(True, alpha=0.3)
        if row == 0:
            ax_route.legend()

    fig.tight_layout()
    out_path = run_dir / "modal_error_angle_slices.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[exp6] Saved modal error-angle slices: {out_path}")


def parse_diagnostics(diagnostics_path: Path):
    """Parse diagnostics.jsonl into per-epoch aggregates (for modal dynamics)."""
    epochs = []
    qk_g_corr = []
    qk_top1 = []
    g_top10_mean = []
    scores_expert_std = []
    w_a_entropy_mean = []

    with diagnostics_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "epoch" not in rec:
                continue
            e = rec["epoch"]
            if e not in epochs:
                epochs.append(e)
                qk_g_corr.append(rec.get("qk_g_corr_pearson_mean"))
                qk_top1.append(rec.get("qk_top1_match_rate"))
                g_stats = rec.get("g_stats", {})
                g_top10_mean.append(g_stats.get("g_top10_mean"))
                scores_stats = rec.get("scores_stats", {})
                scores_expert_std.append(scores_stats.get("scores_expert_std"))
                w_a_entropy_mean.append(rec.get("w_a_entropy_mean"))

    return {
        "epochs": np.array(epochs, dtype=np.int32),
        "qk_g_corr": np.array(qk_g_corr, dtype=np.float32),
        "qk_top1": np.array(qk_top1, dtype=np.float32),
        "g_top10_mean": np.array(g_top10_mean, dtype=np.float32),
        "scores_expert_std": np.array(scores_expert_std, dtype=np.float32),
        "w_a_entropy_mean": np.array(w_a_entropy_mean, dtype=np.float32),
    }


def plot_modal_dynamics(run_dir: Path, diag):
    """Optional: dynamics of modal alignment and sparsity over epochs."""
    epochs = diag["epochs"]

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(epochs, diag["qk_g_corr"], marker="o", label="corr(scores, |D^T y|)")
    axes[0].plot(epochs, diag["qk_top1"], marker="x", label="top-1 match rate")
    axes[0].set_ylabel("Alignment")
    axes[0].set_title("Alignment between routing and modal baseline over epochs")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, diag["g_top10_mean"], marker="o")
    axes[1].set_ylabel("g_top10_mean")
    axes[1].set_title("Top-atom |D^T y| energy over epochs")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, diag["scores_expert_std"], marker="o", label="scores_expert_std")
    axes[2].plot(epochs, diag["w_a_entropy_mean"], marker="x", label="w_a_entropy_mean")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Scale / entropy")
    axes[2].set_title("Routing scale and atom entropy over epochs")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    out_path = run_dir / "modal_dynamics_over_epochs.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[dyn] Saved modal dynamics over epochs: {out_path}")


def plot_atom_sparsity_histogram(run_dir: Path, agg):
    """
    Sparsity visualization 1: histogram of selected atom counts per clip.

    Uses selected_counts from modal_routing_val.npz, which is derived from
    model.last_outputs['support_indices'] (hard-Gumbel-selected atoms across steps).
    """
    selected_counts = agg.get("selected_counts", None)
    if selected_counts is None:
        print("[sparsity] selected_counts not found in routing dump; skip histogram.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(selected_counts, bins=np.arange(selected_counts.max() + 2) - 0.5, rwidth=0.8)
    ax.set_xlabel("Number of selected atoms per clip")
    ax.set_ylabel("Count (validation clips)")
    ax.set_title("Distribution of selected atoms (hard routing sparsity)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = run_dir / "modal_atom_sparsity_histogram.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[sparsity] Saved atom sparsity histogram: {out_path}")


def plot_example_clip_atom_selection(run_dir: Path, agg, max_examples: int = 4):
    """
    Sparsity visualization 2: per-clip selected atom indices for a few example DOAs.

    For a small set of angles (e.g., 0°, 55°, 100°, 175°), we draw a 1D bar/stem
    plot over atom indices, highlighting which atoms were selected (support_indices).
    """
    labels = agg["labels"]
    angles_deg = agg["angles_deg"]
    selected_mask = agg.get("selected_mask", None)
    if selected_mask is None:
        print("[sparsity] selected_mask not found in routing dump; skip per-clip plots.")
        return

    E = len(angles_deg)
    P = selected_mask.shape[1]

    target_angles = [0, 55, 100, 175]
    examples = []
    for ang in target_angles:
        if ang not in list(angles_deg):
            continue
        e = int(np.where(angles_deg == ang)[0][0])
        idx = np.where(labels == e)[0]
        if idx.size == 0:
            continue
        examples.append((ang, int(idx[0])))
        if len(examples) >= max_examples:
            break

    if not examples:
        print("[sparsity] No example clips found for target angles; skip per-clip plots.")
        return

    fig, axes = plt.subplots(len(examples), 1, figsize=(10, 2.5 * len(examples)), sharex=True)
    if len(examples) == 1:
        axes = [axes]

    xs = np.arange(P)
    for ax, (ang, idx) in zip(axes, examples):
        mask = selected_mask[idx]  # (P,)
        selected_idx = np.where(mask)[0]
        ax.stem(selected_idx, np.ones_like(selected_idx), linefmt="C0-", markerfmt="C0o", basefmt=" ")
        ax.set_ylabel("Selected")
        ax.set_title(f"Selected atoms for a single clip at DOA {ang}° (N={len(selected_idx)})")
        ax.grid(True, axis="x", alpha=0.3)

    axes[-1].set_xlabel("Dictionary atom index")
    fig.tight_layout()
    out_path = run_dir / "modal_atom_selection_examples.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[sparsity] Saved example clip atom selection plots: {out_path}")


def plot_example_clip_spectra(
    run_dir: Path,
    agg,
    target_angles: list[float] | None = None,
    predictions: np.ndarray | None = None,
    require_correct: bool | None = None,
    max_examples: int = 4,
    prefix: str = "",
):
    """
    Visualize raw spectra Y(f) for the same clips used in atom-family plots.

    For each selected clip:
      - Plot its magnitude spectrum Y(f) over 300–3000 Hz.
      - X-axis: frequency (Hz); title includes DOA and whether prediction is correct.
    """
    labels = agg["labels"]
    angles_deg = agg["angles_deg"]
    Y_val = agg.get("Y_val", None)
    if Y_val is None:
        print("[clip-spec] Y_val not found; skip clip spectra plots.")
        return

    N, F = Y_val.shape

    # Frequency axis: same as for H/W (300–3000 Hz band)
    fs = 16000.0
    n_fft = 2048.0
    df = fs / n_fft
    f_min = 300.0
    f_max = 3000.0
    k_start = int(np.ceil(f_min / df))
    freqs = (k_start + np.arange(F)) * df
    freq_mask = (freqs >= 300.0) & (freqs <= 3000.0)
    freqs_plot = freqs[freq_mask]

    # Default target angles: same as hard angles
    if target_angles is None:
        target_angles = [55.0, 100.0, 175.0]

    examples: list[tuple[float, int]] = []
    for ang in target_angles:
        if ang not in list(angles_deg):
            continue
        e = int(np.where(angles_deg == ang)[0][0])
        idx_all = np.where(labels == e)[0]
        if predictions is not None and require_correct is not None:
            if require_correct:
                idx_all = idx_all[predictions[idx_all] == e]
            else:
                idx_all = idx_all[predictions[idx_all] != e]
        if idx_all.size == 0:
            continue
        examples.append((ang, int(idx_all[0])))
        if len(examples) >= max_examples:
            break

    if not examples:
        print("[clip-spec] No example clips found; skip clip spectra plots.")
        return

    for ang, idx in examples:
        y = Y_val[idx]  # (F,)
        y_plot = y[freq_mask]
        correct_flag = None
        if predictions is not None:
            correct_flag = (int(predictions[idx]) == int(labels[idx]))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(freqs_plot, y_plot)
        status = ""
        if correct_flag is not None:
            status = " (correct)" if correct_flag else " (wrong)"
        ax.set_title(f"Clip {idx}, DOA {ang}° raw spectrum{status}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        tag = f"{prefix}{ang}"
        out_path = run_dir / f"modal_clip_{tag}_idx{idx}_spectrum.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[clip-spec] Saved clip spectrum plot: {out_path}")


def plot_example_clip_atom_families(
    run_dir: Path,
    agg,
    D: torch.Tensor,
    target_angles: list[float] | None = None,
    predictions: np.ndarray | None = None,
    require_correct: bool | None = None,
    max_examples: int = 4,
    prefix: str = "",
):
    """
    Clip-level modal view: for a few example clips, show the frequency shapes
    and global DOA profiles of the atoms selected by hard routing.

    For each chosen clip:
      - Left: D(:, p) frequency curve (300–2000 Hz) for each selected atom p
      - Right: per-angle routing profile w_{e,p} vs DOA, with the clip's DOA
               highlighted as a vertical line
    """
    labels = agg["labels"]
    angles_deg = agg["angles_deg"]
    per_angle_scores_atoms = agg["per_angle_scores_atoms"]  # (E,P)
    scores_expert_all = agg.get("scores_expert_all", None)   # (N,E)
    scores_atoms_all = agg.get("scores_atoms_all", None)     # (N,E,M)
    selected_mask = agg.get("selected_mask", None)
    if selected_mask is None:
        print("[clip-modal] selected_mask not found; skip clip-level atom families.")
        return

    E = len(angles_deg)
    P = selected_mask.shape[1]
    if scores_atoms_all is None:
        print("[clip-modal] scores_atoms_all not found; skip clip-level atom score plots.")
        return

    # Reuse the same STFT→Hz mapping as in plot_atom_modal_families
    D_np = D.cpu().numpy()  # (F,P)
    F, _ = D_np.shape
    fs = 16000.0
    n_fft = 2048.0
    df = fs / n_fft
    f_min = 300.0
    f_max = 3000.0
    k_start = int(np.ceil(f_min / df))
    freqs = (k_start + np.arange(F)) * df
    freq_mask = (freqs >= 300.0) & (freqs <= 2000.0)
    freqs_plot = freqs[freq_mask]
    D_plot = D_np[freq_mask, :]

    # Pick example clips at specified DOAs
    if target_angles is None:
        target_angles = [0, 55, 100, 175]
    examples: list[tuple[float, int]] = []
    for ang in target_angles:
        if ang not in list(angles_deg):
            continue
        e = int(np.where(angles_deg == ang)[0][0])
        idx_all = np.where(labels == e)[0]
        if predictions is not None and require_correct is not None:
            if require_correct:
                idx_all = idx_all[predictions[idx_all] == e]
            else:
                idx_all = idx_all[predictions[idx_all] != e]
        if idx_all.size == 0:
            continue
        examples.append((ang, int(idx_all[0])))
        if len(examples) >= max_examples:
            break

    if not examples:
        print("[clip-modal] No example clips found for target angles; skip clip-level atom families.")
        return

    for ang, idx in examples:
        mask = selected_mask[idx]  # (P,)
        sel = np.where(mask)[0]
        if sel.size == 0:
            continue
        n_sel = sel.size

        fig, axes = plt.subplots(n_sel, 2, figsize=(10, 2.5 * n_sel), sharex="col")
        if n_sel == 1:
            axes = np.array([axes])

        # For each clip, we want:
        #   左欄：每個選中 atom 的頻率 shape
        #   右欄：這個 clip 在所有 37 個 DOA 上，對「對應 atom family」的 scores_atoms
        #         （固定 atom index m_atom，沿著所有 expert e 掃過），
        #         並標出 true DOA 與 predicted DOA。
        s_expert = scores_expert_all[idx] if scores_expert_all is not None else None  # (E,) or None
        true_e = labels[idx]
        true_ang = float(angles_deg[true_e])
        pred_ang = None
        if predictions is not None:
            pred_e = int(predictions[idx])
            pred_ang = float(angles_deg[pred_e])

        for row, p in enumerate(sel):
            ax_freq = axes[row, 0]
            ax_score = axes[row, 1]

            # Frequency shape for this atom
            ax_freq.plot(freqs_plot, D_plot[:, p])
            ax_freq.set_title(f"Clip {idx}, DOA {ang}°: atom {p} freq (300–2000 Hz)")
            ax_freq.set_xlabel("Frequency (Hz)")
            ax_freq.set_ylabel("Amplitude")
            ax_freq.grid(True, alpha=0.3)

            # Clip-specific atom scores over all DOAs for this atom family:
            # 固定 atom index m_atom，對所有 expert e 取 scores_atoms_all[idx, e, m_atom]
            M = scores_atoms_all.shape[2]
            m_atom = p % M
            s_atom = scores_atoms_all[idx, :, m_atom]  # (E,)
            ax_score.plot(angles_deg, s_atom, marker="o")
            ax_score.axvline(true_ang, color="r", linestyle="--", alpha=0.6, label="true DOA" if row == 0 else None)
            if pred_ang is not None:
                ax_score.axvline(pred_ang, color="g", linestyle="--", alpha=0.6, label="pred DOA" if row == 0 else None)
            ax_score.set_title(f"Clip {idx}: atom-family scores vs DOA (m={m_atom})")
            ax_score.set_xlabel("Angle (deg)")
            ax_score.set_ylabel("Score")
            ax_score.grid(True, alpha=0.3)
            if row == 0 and pred_ang is not None:
                ax_score.legend()

        fig.tight_layout()
        tag = f"{prefix}{ang}"
        out_path = run_dir / f"modal_clip_{tag}_idx{idx}_atom_families.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[clip-modal] Saved clip-level atom families plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc modal visualizations for OMP Transformer Speech260 run."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to training run directory (must contain code_state.json, model_best.pth, metrics.npz, diagnostics.jsonl).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for model evaluation: cpu|mps|cuda (default: cpu for portability).",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        help="Optional cap on number of validation samples for modal routing dump (default: all).",
    )
    parser.add_argument(
        "--skip_dump",
        action="store_true",
        help="Skip recomputing modal_routing_val.npz if it already exists.",
    )

    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"run_dir does not exist: {run_dir}")

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    routing_npz = run_dir / "modal_routing_val.npz"
    if routing_npz.is_file() and args.skip_dump:
        print(f"[modal] Using existing modal routing dump: {routing_npz}")
    else:
        dump_modal_routing(run_dir, device=device, max_val_samples=args.max_val_samples)

    code_state, metrics, diagnostics_path = load_run_state(run_dir)
    agg = aggregate_per_angle_modal(routing_npz)
    # Metrics for per-angle accuracy and predictions
    per_angle_acc = metrics["per_angle_accuracy"]
    predictions = metrics["predictions"]

    # Rebuild dictionary for atom frequency shapes
    import importlib

    omp_mod = importlib.import_module("scripts.omp-transformer-ldv")
    args_dict = code_state.get("args", {})
    H_raw, W_raw, angles = omp_mod.load_raw_ldv_matrices(
        args_dict.get("h_path"), args_dict.get("w_path"), device="cpu"
    )
    atom_reduce_mode = args_dict.get("atom_reduce_mode", "kmeans")
    n_atoms = int(args_dict.get("n_atoms", 8))
    atom_min_cos = float(args_dict.get("atom_min_cos", 0.98))
    if atom_reduce_mode == "kmeans":
        W_reduced, _, _ = omp_mod.reduce_atoms_kmeans(
            W_raw, n_clusters=n_atoms, random_state=42
        )
    else:
        W_reduced, _, _ = omp_mod.reduce_atoms(
            W_raw,
            mode=atom_reduce_mode,
            n_clusters=n_atoms,
            random_state=42,
            min_cos=atom_min_cos,
        )
    H_final = H_raw
    W_final = W_reduced
    D, idx2angle = omp_mod.build_dictionary(H_final, W_final, angles)

    # Experiments 1–4: static modal structure
    # (1) 原始 DOA×atom matrix（保留作為原始資料快照）
    plot_doa_atom_activation(run_dir, agg)
    # (2) 先對 atoms 依 DOA profile 分群，再畫 clustered heatmap（更容易看出 atom family）
    plot_doa_atom_clustered(run_dir, agg, n_clusters=8)
    # (3) 壓縮成 37×37：true DOA × 字典 expert DOA-block，更容易看出是否接近對角。
    plot_doa_expert_activation(run_dir, agg)
    plot_doa_similarity_and_confusion(run_dir, agg, metrics)
    plot_modal_sparsity_per_angle(run_dir, agg)
    plot_atom_modal_families(run_dir, agg, D)

    # Experiments 5–6: alignment and error DOAs
    plot_modal_alignment_with_baseline(run_dir, agg)
    plot_error_angle_modal_slices(run_dir, agg, metrics)

    # Sparsity visualizations
    plot_atom_sparsity_histogram(run_dir, agg)
    plot_example_clip_atom_selection(run_dir, agg)
    # Clip-level modal families for hard angles (problematic DOAs) and easy angles
    hard_angles = [55.0, 100.0, 175.0]
    # Hard angles: visualize one correctly classified and one misclassified clip (if available)
    plot_example_clip_atom_families(
        run_dir,
        agg,
        D,
        target_angles=hard_angles,
        predictions=predictions,
        require_correct=True,
        prefix="hard_correct_",
    )
    plot_example_clip_atom_families(
        run_dir,
        agg,
        D,
        target_angles=hard_angles,
        predictions=predictions,
        require_correct=False,
        prefix="hard_wrong_",
    )
    # Easy angles: per-angle accuracy ~100%, 且排除已知困難角度
    angles_deg = agg["angles_deg"]
    hard_set = set(hard_angles)
    easy_angles = [float(a) for a, acc in zip(angles_deg, per_angle_acc) if acc >= 0.999 and float(a) not in hard_set][:3]
    if easy_angles:
        plot_example_clip_atom_families(
            run_dir,
            agg,
            D,
            target_angles=easy_angles,
            predictions=predictions,
            require_correct=True,
            prefix="easy_",
        )

    # Raw spectra for the same hard/easy clips (for side-by-side comparison)
    plot_example_clip_spectra(
        run_dir,
        agg,
        target_angles=hard_angles,
        predictions=predictions,
        require_correct=True,
        prefix="hard_correct_",
    )
    plot_example_clip_spectra(
        run_dir,
        agg,
        target_angles=hard_angles,
        predictions=predictions,
        require_correct=False,
        prefix="hard_wrong_",
    )
    if easy_angles:
        plot_example_clip_spectra(
            run_dir,
            agg,
            target_angles=easy_angles,
            predictions=predictions,
            require_correct=True,
            prefix="easy_",
        )

    # Optional dynamics figure from diagnostics
    diag = parse_diagnostics(diagnostics_path)
    plot_modal_dynamics(run_dir, diag)


if __name__ == "__main__":
    main()
