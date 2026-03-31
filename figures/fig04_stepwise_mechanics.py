"""Build the governed stepwise-mechanics artifact for Fig. 4.

The artifact is derived from the active primary run and stores only the
manuscript-facing quantities needed by the mechanism-first Fig. 4 generator:

- angle-conditioned stage-0 physics, QK, and routing summaries for panel b
- a single shared exemplar plus population summaries for the gated update
  ``Δx_t = w_t ⊙ g_t`` in panel c
- the shared exemplar's mode-resolved routing/update tensors for the
  aggregation-bridge panel d

It intentionally does not replace the upstream run artifacts recorded in
``figures/conf/experiments.yaml``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from figures.style import load_paths


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_SCRIPT = REPO_ROOT / "scripts" / "omp-transformer-ldv.py"
DEFAULT_REPRESENTATIVE_ANGLES_DEG = (70.0,)
DEFAULT_CANDIDATE_POOL_SIZE = 8
EPS = 1e-8
MODEL_MODULE: Any | None = None


def _load_model_module():
    global MODEL_MODULE
    if MODEL_MODULE is not None:
        return MODEL_MODULE

    spec = importlib.util.spec_from_file_location("omp_transformer_ldv", MODEL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load model definition from {MODEL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    MODEL_MODULE = module
    return module


def _load_model_class():
    return _load_model_module().FullTransformerRoutedSoftOMP


def _instantiate_model(run_dir: Path) -> tuple[Any, torch.Tensor]:
    args = json.loads((run_dir / "code_state.json").read_text(encoding="utf-8"))["args"]
    preprocessing = torch.load(
        run_dir / "preprocessing.pth",
        map_location="cpu",
        weights_only=False,
    )
    dictionary = preprocessing["D"].float()
    angles = preprocessing["angles"].float()

    model_cls = _load_model_class()
    model = model_cls(
        F=int(dictionary.shape[0]),
        E=int(angles.numel()),
        M=int(args["n_atoms"]),
        d=int(args["d_model"]),
        nhead=int(args["nhead"]),
        nlayers=int(args["nlayers"]),
        steps=int(args["steps"]),
        top_e=int(args["top_e"]),
        L=int(args["top_l"]),
        tau_e=0.5,
        tau_a=0.2,
        eta=0.5,
        routing=str(args["routing_activation_e"]),
        routing_mode=str(args["routing_mode"]),
        hybrid_alpha=float(args.get("hybrid_alpha", 0.5)),
        routing_e=str(args["routing_activation_e"]),
        routing_a=str(args["routing_activation_a"]),
        score_norm_mode=str(args.get("score_norm", "none")),
        score_reg_weight=float(args.get("score_reg_weight", 0.0)),
        expert_agg=str(args.get("expert_agg", "l2")),
    )
    for attr in (
        "no_type_bias",
        "encoder_identity",
        "single_gate_expert",
        "score_center_atoms",
        "score_center_expert",
        "d_can_attend_r",
        "use_hard_gumbel",
    ):
        if attr in args:
            setattr(model, attr, args[attr])

    checkpoint = torch.load(run_dir / "model_best.pth", map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, dictionary


def _normalize_positive_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    max_per_row = np.maximum(arr.max(axis=1, keepdims=True), EPS)
    return arr / max_per_row


def _mean_and_sem(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arr, dtype=np.float32)
    mean = arr.mean(axis=0, dtype=np.float64).astype(np.float32)
    if arr.shape[0] <= 1:
        sem = np.zeros_like(mean, dtype=np.float32)
    else:
        sem = (
            arr.std(axis=0, ddof=1, dtype=np.float64) / math.sqrt(arr.shape[0])
        ).astype(np.float32)
    return mean, sem


def _probability_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    total = np.maximum(arr.sum(axis=1, keepdims=True), EPS)
    return arr / total


def _pad_indices(values: list[int], length: int) -> np.ndarray:
    padded = np.full(length, -1, dtype=np.int16)
    if values:
        padded[: min(length, len(values))] = np.asarray(values[:length], dtype=np.int16)
    return padded


def _deterministic_picker(logits: torch.Tensor, tau: torch.Tensor | float, mode: str) -> torch.Tensor:
    if isinstance(tau, torch.Tensor):
        tau_tensor = tau.to(device=logits.device, dtype=logits.dtype).clamp_min(1e-8)
    else:
        tau_tensor = torch.tensor(float(tau), device=logits.device, dtype=logits.dtype).clamp_min(1e-8)

    if mode in {"gumbel", "softmax"}:
        return F.softmax(logits / tau_tensor, dim=-1)
    if mode == "entmax":
        return _load_model_module().entmax15(logits, dim=-1)
    if mode == "sparsemax":
        return _load_model_module().sparsemax(logits, dim=-1)
    raise ValueError(f"Unsupported routing activation for deterministic picker: {mode}")


def _deterministic_expert_gate(model: Any, logits: torch.Tensor) -> torch.Tensor:
    mode = str(getattr(model, "routing_e", "softmax"))
    tau = model.tau_e if isinstance(model.tau_e, torch.Tensor) else torch.tensor(float(model.tau_e))
    return _deterministic_picker(logits, tau=tau, mode=mode)


def _deterministic_atom_gate(model: Any, logits: torch.Tensor) -> torch.Tensor:
    mode = str(getattr(model, "routing_a", "softmax"))
    tau = model.tau_a if isinstance(model.tau_a, torch.Tensor) else torch.tensor(float(model.tau_a))
    return _deterministic_picker(logits, tau=tau, mode=mode)


def _compute_step_profiles(
    model: Any,
    residual: torch.Tensor,
    dictionary: torch.Tensor,
) -> dict[str, torch.Tensor]:
    tokens = model._build_tokens(residual, dictionary)
    mask = model._make_mask(tokens.size(0)).to(tokens.device)
    hidden = tokens if model.encoder_identity else model.encoder(tokens, mask=mask)
    h_r = hidden[0]
    h_d = hidden[1:]

    qk_atoms = (model.Wk(h_d) @ model.Wq(h_r)) / math.sqrt(float(model.d))
    qk_atoms = qk_atoms.reshape(model.E, model.M)
    if model.expert_agg == "l2":
        qk_expert = torch.sqrt((qk_atoms.abs() ** 2).sum(dim=1) + 1e-12)
    elif model.expert_agg == "max":
        qk_expert = qk_atoms.abs().max(dim=1).values
    else:
        qk_expert = torch.relu(qk_atoms).mean(dim=1)

    g_vec_all = dictionary.T @ residual
    g_atoms = g_vec_all.reshape(model.E, model.M)
    g_expert = torch.sqrt((g_atoms ** 2).sum(dim=1) + 1e-12)

    if model.routing_mode == "qk":
        scores_atoms = qk_atoms
        scores_expert = qk_expert
    elif model.routing_mode == "g":
        scores_atoms = g_atoms
        scores_expert = g_expert
    else:
        def _norm(tensor: torch.Tensor) -> torch.Tensor:
            return tensor / (tensor.norm() + 1e-12)

        scores_atoms = model.hybrid_alpha * _norm(g_atoms) + (1.0 - model.hybrid_alpha) * _norm(qk_atoms)
        scores_expert = model.hybrid_alpha * _norm(g_expert) + (1.0 - model.hybrid_alpha) * _norm(qk_expert)

    if model.score_norm_mode == "std":
        scores_expert = (scores_expert - scores_expert.mean()) / (scores_expert.std() + 1e-8)
        scores_atoms = (scores_atoms - scores_atoms.mean()) / (scores_atoms.std() + 1e-8)
    if model.score_center_expert:
        scores_expert = scores_expert - scores_expert.mean()
    if model.score_center_atoms:
        scores_atoms = scores_atoms - scores_atoms.mean(dim=1, keepdim=True)

    w_e = _deterministic_expert_gate(model, scores_expert)
    if getattr(model, "single_gate_expert", False):
        w_a = torch.ones((model.E, model.M), device=g_atoms.device, dtype=g_atoms.dtype)
    else:
        w_a = torch.stack(
            [_deterministic_atom_gate(model, scores_atoms[e].abs()) for e in range(model.E)],
            dim=0,
        )
    w_all = w_e[:, None] * w_a
    w_theta = w_all.sum(dim=1)
    delta_atoms = w_all * g_atoms
    delta_expert = torch.sqrt((delta_atoms ** 2).sum(dim=1) + 1e-12)
    return {
        "g_vec_all": g_vec_all,
        "g_atoms": g_atoms,
        "g_expert": g_expert,
        "qk_expert": qk_expert,
        "scores_atoms": scores_atoms,
        "scores_expert": scores_expert,
        "w_e": w_e,
        "w_a": w_a,
        "w_all": w_all,
        "w_theta": w_theta,
        "delta_atoms": delta_atoms,
        "delta_expert": delta_expert,
    }


@torch.no_grad()
def _extract_stage0_profiles(
    model: Any,
    observation: torch.Tensor,
    dictionary: torch.Tensor,
) -> dict[str, np.ndarray]:
    profiles = _compute_step_profiles(model, observation.float().clone(), dictionary.float())
    return {
        "g_expert": profiles["g_expert"].detach().cpu().numpy().astype(np.float32),
        "qk_expert": profiles["qk_expert"].detach().cpu().numpy().astype(np.float32),
        "scores_expert": profiles["scores_expert"].detach().cpu().numpy().astype(np.float32),
        "w_e": profiles["w_e"].detach().cpu().numpy().astype(np.float32),
        "w_theta": profiles["w_theta"].detach().cpu().numpy().astype(np.float32),
        "delta_expert": profiles["delta_expert"].detach().cpu().numpy().astype(np.float32),
    }


@torch.no_grad()
def _trace_hard_routing(
    model: Any,
    observation: torch.Tensor,
    dictionary: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Replay the active inference path and expose per-step internals."""
    observation = observation.float().clone()
    dictionary = dictionary.float()
    num_atoms = int(dictionary.shape[1])
    coeff = torch.zeros(num_atoms, dtype=dictionary.dtype)
    residual = observation.clone()
    initial_res_norm = float(torch.norm(residual).item())

    g_steps: list[np.ndarray] = []
    qk_steps: list[np.ndarray] = []
    score_steps: list[np.ndarray] = []
    gate_steps: list[np.ndarray] = []
    w_theta_steps: list[np.ndarray] = []
    delta_steps: list[np.ndarray] = []
    w_theta_mode_steps: list[np.ndarray] = []
    delta_theta_mode_steps: list[np.ndarray] = []
    res_norms: list[float] = []
    chosen_experts_steps: list[np.ndarray] = []
    chosen_indices_steps: list[np.ndarray] = []
    support_indices: list[int] = []
    eta = model.eta.detach().cpu() if isinstance(model.eta, torch.Tensor) else torch.tensor(float(model.eta))

    for step in range(int(model.steps)):
        profiles = _compute_step_profiles(model, residual, dictionary)
        g_vec_all = profiles["g_vec_all"]
        g_expert = profiles["g_expert"]
        qk_expert = profiles["qk_expert"]
        scores_atoms = profiles["scores_atoms"]
        scores_expert = profiles["scores_expert"]
        w_e = profiles["w_e"]
        w_theta = profiles["w_theta"]
        delta_expert = profiles["delta_expert"]
        w_all = profiles["w_all"]
        delta_atoms = profiles["delta_atoms"]

        use_hard = bool(getattr(model, "use_hard_gumbel", False))
        if use_hard and step > 0:
            scores_atoms_masked = scores_atoms.clone()
            scores_expert_masked = scores_expert.clone()
            for prev_idx in support_indices:
                prev_e = prev_idx // model.M
                prev_m = prev_idx % model.M
                scores_atoms_masked[prev_e, prev_m] = -1e10
            for expert_idx in range(model.E):
                if torch.all(scores_atoms_masked[expert_idx].abs() < -1e9):
                    scores_expert_masked[expert_idx] = -1e10
            chosen_experts = torch.topk(scores_expert_masked, k=min(model.top_e, model.E)).indices.tolist()
            chosen_indices: list[int] = []
            for expert_idx in chosen_experts:
                chosen_atoms = torch.topk(scores_atoms_masked[expert_idx], k=min(model.L, model.M)).indices.tolist()
                chosen_indices.extend(int(expert_idx) * model.M + int(atom_idx) for atom_idx in chosen_atoms)
        else:
            chosen_experts = torch.topk(scores_expert, k=min(model.top_e, model.E)).indices.tolist()
            chosen_indices = []
            for expert_idx in chosen_experts:
                chosen_atoms = torch.topk(scores_atoms[expert_idx].abs(), k=min(model.L, model.M)).indices.tolist()
                chosen_indices.extend(int(expert_idx) * model.M + int(atom_idx) for atom_idx in chosen_atoms)

        chosen_indices = list(dict.fromkeys(chosen_indices))
        support_indices.extend(chosen_indices)
        if chosen_indices:
            grad = dictionary.T @ residual
            coeff[chosen_indices] = coeff[chosen_indices] + eta * grad[chosen_indices]
        residual = observation - dictionary @ coeff

        g_steps.append(g_expert.detach().cpu().numpy().astype(np.float32))
        qk_steps.append(qk_expert.detach().cpu().numpy().astype(np.float32))
        score_steps.append(scores_expert.detach().cpu().numpy().astype(np.float32))
        gate_steps.append(w_e.detach().cpu().numpy().astype(np.float32))
        w_theta_steps.append(w_theta.detach().cpu().numpy().astype(np.float32))
        delta_steps.append(delta_expert.detach().cpu().numpy().astype(np.float32))
        w_theta_mode_steps.append(w_all.detach().cpu().numpy().astype(np.float32))
        delta_theta_mode_steps.append(
            (delta_atoms.abs() * eta.to(device=delta_atoms.device, dtype=delta_atoms.dtype))
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        res_norms.append(float(torch.norm(residual).item()))
        chosen_experts_steps.append(_pad_indices(chosen_experts, int(model.top_e)))
        chosen_indices_steps.append(_pad_indices(chosen_indices, int(model.top_e * model.L)))

    return {
        "initial_res_norm": np.asarray(initial_res_norm, dtype=np.float32),
        "g_expert_steps": np.stack(g_steps, axis=0).astype(np.float32),
        "qk_expert_steps": np.stack(qk_steps, axis=0).astype(np.float32),
        "scores_expert_steps": np.stack(score_steps, axis=0).astype(np.float32),
        "w_e_steps": np.stack(gate_steps, axis=0).astype(np.float32),
        "w_theta_steps": np.stack(w_theta_steps, axis=0).astype(np.float32),
        "delta_expert_steps": np.stack(delta_steps, axis=0).astype(np.float32),
        "w_theta_m_steps": np.stack(w_theta_mode_steps, axis=0).astype(np.float32),
        "delta_theta_m_steps": np.stack(delta_theta_mode_steps, axis=0).astype(np.float32),
        "res_norms": np.asarray(res_norms, dtype=np.float32),
        "chosen_experts_idx": np.stack(chosen_experts_steps, axis=0).astype(np.int16),
        "chosen_atom_indices": np.stack(chosen_indices_steps, axis=0).astype(np.int16),
    }


def build_stepwise_mechanics_artifact(
    data_root: Path,
    output_path: Path,
    *,
    representative_angles_deg: tuple[float, ...] = DEFAULT_REPRESENTATIVE_ANGLES_DEG,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
) -> Path:
    """Build the Fig. 4 stepwise-mechanics artifact under ``results/``."""
    paths_cfg = load_paths()
    run_dir = data_root / paths_cfg["primary_run"]
    modal = np.load(run_dir / "modal_routing_val.npz", allow_pickle=True)
    labels = np.asarray(modal["labels"], dtype=np.int64)
    angles_deg = np.asarray(modal["angles_deg"], dtype=np.float32)
    observations = torch.tensor(np.asarray(modal["Y_val"], dtype=np.float32))

    model, dictionary = _instantiate_model(run_dir)

    stage0_profiles_all = [
        _extract_stage0_profiles(model, observations[sample_idx], dictionary)
        for sample_idx in range(int(observations.shape[0]))
    ]
    g_rows_all = np.stack([item["g_expert"] for item in stage0_profiles_all], axis=0).astype(np.float32)
    qk_rows_all = np.stack([item["qk_expert"] for item in stage0_profiles_all], axis=0).astype(np.float32)
    scores_rows_all = np.stack([item["scores_expert"] for item in stage0_profiles_all], axis=0).astype(np.float32)
    w_theta_rows_all = np.stack([item["w_theta"] for item in stage0_profiles_all], axis=0).astype(np.float32)
    delta_rows_all = np.stack([item["delta_expert"] for item in stage0_profiles_all], axis=0).astype(np.float32)

    predictions = scores_rows_all.argmax(axis=1).astype(np.int64)
    correct_mask = predictions == labels
    top2 = np.sort(scores_rows_all, axis=1)[:, -2:]
    score_margins = (top2[:, 1] - top2[:, 0]).astype(np.float32)

    representative_label_indices: list[int] = []
    total_counts: list[int] = []
    correct_counts: list[int] = []
    angle_accuracy: list[float] = []
    stage0_g_mean: list[np.ndarray] = []
    stage0_g_sem: list[np.ndarray] = []
    stage0_qk_mean: list[np.ndarray] = []
    stage0_qk_sem: list[np.ndarray] = []
    stage0_w_theta_mean: list[np.ndarray] = []
    stage0_w_theta_sem: list[np.ndarray] = []
    stage0_delta_mean: list[np.ndarray] = []
    stage0_delta_sem: list[np.ndarray] = []
    sample_indices: list[int] = []

    for angle_deg in representative_angles_deg:
        matching = np.where(np.isclose(angles_deg, float(angle_deg)))[0]
        if matching.size != 1:
            raise ValueError(f"Representative angle {angle_deg} not found in active angle grid")
        label_idx = int(matching[0])
        angle_mask = labels == label_idx
        angle_correct = angle_mask & correct_mask
        if not np.any(angle_correct):
            raise RuntimeError(f"No correctly decoded validation clips found for {angle_deg:.0f} deg")

        representative_label_indices.append(label_idx)
        total_counts.append(int(angle_mask.sum()))
        correct_counts.append(int(angle_correct.sum()))
        angle_accuracy.append(float(correct_mask[angle_mask].mean()))

        g_rows = g_rows_all[angle_correct]
        qk_rows = qk_rows_all[angle_correct]
        w_theta_rows = w_theta_rows_all[angle_correct]
        delta_rows = delta_rows_all[angle_correct]
        g_mean, g_sem = _mean_and_sem(_normalize_positive_rows(g_rows))
        qk_mean, qk_sem = _mean_and_sem(_normalize_positive_rows(qk_rows))
        w_theta_mean, w_theta_sem = _mean_and_sem(_normalize_positive_rows(w_theta_rows))
        delta_mean, delta_sem = _mean_and_sem(_normalize_positive_rows(delta_rows))
        stage0_g_mean.append(g_mean)
        stage0_g_sem.append(g_sem)
        stage0_qk_mean.append(qk_mean)
        stage0_qk_sem.append(qk_sem)
        stage0_w_theta_mean.append(w_theta_mean)
        stage0_w_theta_sem.append(w_theta_sem)
        stage0_delta_mean.append(delta_mean)
        stage0_delta_sem.append(delta_sem)

        candidate_indices = np.flatnonzero(angle_correct)
        order = np.argsort(score_margins[candidate_indices])[::-1]
        shortlist = candidate_indices[order[: max(1, min(candidate_pool_size, candidate_indices.size))]]
        sample_indices.append(int(shortlist[0]))

    sample_traces: list[dict[str, np.ndarray]] = []
    sample_step0_margin: list[float] = []
    for rep_idx, label_idx in enumerate(representative_label_indices):
        angle_correct = (labels == label_idx) & correct_mask
        candidate_indices = np.flatnonzero(angle_correct)
        order = np.argsort(score_margins[candidate_indices])[::-1]
        shortlist = candidate_indices[order[: max(1, min(candidate_pool_size, candidate_indices.size))]]

        best_index = None
        best_trace = None
        best_key = None
        for candidate_idx in shortlist:
            trace = _trace_hard_routing(model, observations[candidate_idx], dictionary)
            final_ratio = 1.0 - float(trace["res_norms"][-1] / max(float(trace["initial_res_norm"]), EPS))
            key = (final_ratio, float(score_margins[candidate_idx]))
            if best_key is None or key > best_key:
                best_index = int(candidate_idx)
                best_trace = trace
                best_key = key
        if best_index is None or best_trace is None:
            raise RuntimeError(f"Failed to select representative trace for label index {label_idx}")

        sample_indices[rep_idx] = best_index
        sample_traces.append(best_trace)
        sample_step0_margin.append(float(score_margins[best_index]))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_initial_res_norm = np.asarray(
        [trace["initial_res_norm"] for trace in sample_traces],
        dtype=np.float32,
    )
    sample_res_norms = np.stack([trace["res_norms"] for trace in sample_traces], axis=0).astype(np.float32)
    sample_g_steps = np.stack([trace["g_expert_steps"] for trace in sample_traces], axis=0).astype(np.float32)
    sample_qk_steps = np.stack([trace["qk_expert_steps"] for trace in sample_traces], axis=0).astype(np.float32)
    sample_score_steps = np.stack([trace["scores_expert_steps"] for trace in sample_traces], axis=0).astype(np.float32)
    sample_gate_steps = np.stack([trace["w_e_steps"] for trace in sample_traces], axis=0).astype(np.float32)
    sample_w_theta_steps = np.stack([trace["w_theta_steps"] for trace in sample_traces], axis=0).astype(np.float32)
    sample_delta_steps = np.stack([trace["delta_expert_steps"] for trace in sample_traces], axis=0).astype(np.float32)
    sample_w_theta_m_steps = np.stack([trace["w_theta_m_steps"] for trace in sample_traces], axis=0).astype(np.float32)
    sample_delta_theta_m_steps = np.stack([trace["delta_theta_m_steps"] for trace in sample_traces], axis=0).astype(np.float32)
    sample_chosen_experts_idx = np.stack([trace["chosen_experts_idx"] for trace in sample_traces], axis=0).astype(np.int16)
    sample_chosen_atom_indices = np.stack([trace["chosen_atom_indices"] for trace in sample_traces], axis=0).astype(np.int16)
    sample_chosen_experts_angles = np.full_like(sample_chosen_experts_idx, -1, dtype=np.float32)
    valid_expert_mask = sample_chosen_experts_idx >= 0
    sample_chosen_experts_angles[valid_expert_mask] = angles_deg[sample_chosen_experts_idx[valid_expert_mask]]
    mode_axis = np.arange(1, sample_w_theta_m_steps.shape[-1] + 1, dtype=np.int16)

    aligned_radius_deg = np.arange(0.0, 65.0, 5.0, dtype=np.float32)
    g_prob_all = _probability_rows(g_rows_all)
    delta_prob_all = _probability_rows(delta_rows_all)
    aligned_g_rows: list[np.ndarray] = []
    aligned_delta_rows: list[np.ndarray] = []
    for sample_idx, label_idx in enumerate(labels.tolist()):
        target_angle = float(angles_deg[label_idx])
        distance = np.abs(angles_deg - target_angle)
        aligned_g_rows.append(
            np.asarray(
                [g_prob_all[sample_idx, distance <= radius].sum() for radius in aligned_radius_deg],
                dtype=np.float32,
            )
        )
        aligned_delta_rows.append(
            np.asarray(
                [delta_prob_all[sample_idx, distance <= radius].sum() for radius in aligned_radius_deg],
                dtype=np.float32,
            )
        )
    aligned_g_mean, aligned_g_sem = _mean_and_sem(np.stack(aligned_g_rows, axis=0).astype(np.float32))
    aligned_delta_mean, aligned_delta_sem = _mean_and_sem(np.stack(aligned_delta_rows, axis=0).astype(np.float32))

    np.savez_compressed(
        output_path,
        angles_deg=angles_deg.astype(np.float32),
        representative_angles_deg=np.asarray(representative_angles_deg, dtype=np.float32),
        representative_label_indices=np.asarray(representative_label_indices, dtype=np.int16),
        total_counts=np.asarray(total_counts, dtype=np.int16),
        correct_counts=np.asarray(correct_counts, dtype=np.int16),
        angle_accuracy=np.asarray(angle_accuracy, dtype=np.float32),
        stage0_g_norm_mean=np.stack(stage0_g_mean, axis=0).astype(np.float32),
        stage0_g_norm_sem=np.stack(stage0_g_sem, axis=0).astype(np.float32),
        stage0_qk_norm_mean=np.stack(stage0_qk_mean, axis=0).astype(np.float32),
        stage0_qk_norm_sem=np.stack(stage0_qk_sem, axis=0).astype(np.float32),
        stage0_w_theta_norm_mean=np.stack(stage0_w_theta_mean, axis=0).astype(np.float32),
        stage0_w_theta_norm_sem=np.stack(stage0_w_theta_sem, axis=0).astype(np.float32),
        stage0_delta_norm_mean=np.stack(stage0_delta_mean, axis=0).astype(np.float32),
        stage0_delta_norm_sem=np.stack(stage0_delta_sem, axis=0).astype(np.float32),
        stage0_gate_norm_mean=np.stack(stage0_w_theta_mean, axis=0).astype(np.float32),
        stage0_gate_norm_sem=np.stack(stage0_w_theta_sem, axis=0).astype(np.float32),
        aligned_radius_deg=aligned_radius_deg,
        aligned_cum_mass_g_mean=aligned_g_mean.astype(np.float32),
        aligned_cum_mass_g_sem=aligned_g_sem.astype(np.float32),
        aligned_cum_mass_delta_mean=aligned_delta_mean.astype(np.float32),
        aligned_cum_mass_delta_sem=aligned_delta_sem.astype(np.float32),
        aligned_clip_count=np.asarray(len(labels), dtype=np.int32),
        sample_indices=np.asarray(sample_indices, dtype=np.int32),
        sample_predictions=np.asarray(predictions[sample_indices], dtype=np.int16),
        sample_step0_margin=np.asarray(sample_step0_margin, dtype=np.float32),
        sample_initial_res_norm=sample_initial_res_norm,
        sample_res_norms=sample_res_norms,
        sample_g_expert_steps=sample_g_steps,
        sample_qk_expert_steps=sample_qk_steps,
        sample_scores_expert_steps=sample_score_steps,
        sample_w_e_steps=sample_gate_steps,
        sample_w_theta_steps=sample_w_theta_steps,
        sample_w_theta_m_steps=sample_w_theta_m_steps,
        sample_delta_expert_steps=sample_delta_steps,
        sample_delta_theta_m_steps=sample_delta_theta_m_steps,
        sample_chosen_experts_idx=sample_chosen_experts_idx,
        sample_chosen_experts_angles_deg=sample_chosen_experts_angles.astype(np.float32),
        sample_chosen_atom_indices=sample_chosen_atom_indices,
        mode_axis=mode_axis,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root used to resolve active figure data paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit output path for the derived NPZ artifact.",
    )
    parser.add_argument(
        "--representative-angles",
        type=float,
        nargs="+",
        default=list(DEFAULT_REPRESENTATIVE_ANGLES_DEG),
        help="Representative validation angles used for the mechanism panels.",
    )
    args = parser.parse_args()

    if args.output is None:
        paths_cfg = load_paths()
        output_path = args.data_root / paths_cfg["fig04_stepwise_mechanics"]
    else:
        output_path = args.output

    out = build_stepwise_mechanics_artifact(
        args.data_root,
        output_path,
        representative_angles_deg=tuple(args.representative_angles),
    )
    print(f"[fig04] wrote stepwise mechanics artifact -> {out}")


if __name__ == "__main__":
    main()
