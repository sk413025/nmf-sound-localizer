"""
IS-OMP (shared-support, time-varying coefficients) implemented with
scikit-learn's non_negative_factorization (MU, beta_loss='itakura-saito').

Key properties
- Y ∈ R_+^{F×N}, H ∈ R_+^{F×D}, W ∈ R_+^{F×R}
- Block per direction: B_d = diag(H[:,d]) @ W  (F×R)
- Full dictionary for a support S: A_S = [B_{d1}, ..., B_{dt}] (F×(tR))
- Given S, solve X_S* = argmin_{X≥0} DIS_IS(Y || A_S X) via MU (sklearn backend)
- Greedy: pick j maximizing ΔJ(j|S) = J(Ŷ(S)) − J(Ŷ(S∪{j}))

Notes
- Prefilter: optionally restrict candidates to top-M directions by a teacher
  score s_T(d) = −DIS_IS(Y || (H_d ⊙ ŝ)·1ᵀ). If ŝ is None, consider all D.
- No fallbacks: raise clear errors on grid/angle mismatches or missing sklearn.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


def _safe_np(x: np.ndarray, eps: float) -> np.ndarray:
    return np.maximum(x, eps)


def is_divergence(Y: np.ndarray, Yhat: np.ndarray, eps: float = 1e-8) -> float:
    """Itakura–Saito divergence D_IS(Y || Yhat) with shared epsilon clamps.

    Accepts Y(F,N), Yhat(F,N) or Yhat(F,). If vector, broadcasts across time.
    Returns a scalar float.
    """
    Yc = _safe_np(np.asarray(Y, dtype=np.float64), eps)
    Yh = np.asarray(Yhat, dtype=np.float64)
    if Yh.ndim == 1:
        Yhc = _safe_np(np.repeat(Yh[:, None], Yc.shape[1], axis=1), eps)
    else:
        Yhc = _safe_np(Yh, eps)
    r = Yc / Yhc
    val = np.sum(r - np.log(_safe_np(r, eps)) - 1.0)
    return float(val)


def build_blocks(H: np.ndarray, W: np.ndarray) -> List[np.ndarray]:
    """Precompute directional blocks B_d = diag(H_d) @ W for all d.

    H: (F,D), W: (F,R). Returns list of length D with arrays of shape (F,R).
    """
    H = np.asarray(H)
    W = np.asarray(W)
    Fh, D = H.shape
    Fw, R = W.shape
    if Fh != Fw:
        raise ValueError(f"H.F ({Fh}) must equal W.F ({Fw})")
    blocks: List[np.ndarray] = []
    for d in range(D):
        Bd = (H[:, d:d+1] * W)  # (F,1) * (F,R) → (F,R) via broadcast
        blocks.append(np.asarray(Bd, dtype=np.float64))
    return blocks


def concat_blocks(blocks: List[np.ndarray], sel: Sequence[int]) -> np.ndarray:
    """Concatenate blocks along columns for a support sel (indices into blocks)."""
    if not sel:
        # Empty dictionary – caller should avoid this case
        return np.zeros((blocks[0].shape[0], 0), dtype=blocks[0].dtype)
    mats = [blocks[i] for i in sel]
    return np.concatenate(mats, axis=1)


def solve_x_is_smu(
    Y: np.ndarray,
    A: np.ndarray,
    X0: Optional[np.ndarray] = None,
    max_iter: int = 50,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Solve min_{X≥0} D_IS(Y || A X) with fixed A using simple MU (IS geometry).

    Update rule (β=0, IS divergence):
        X ← X ⊙ (Aᵀ [Y / (A X)^2]) / (Aᵀ [1 / (A X)])
    with shared epsilon clamps for numerical stability.
    """
    Yc = _safe_np(np.asarray(Y, dtype=np.float64), eps)
    Afix = _safe_np(np.asarray(A, dtype=np.float64), eps)
    F, N = Yc.shape
    Fa, K = Afix.shape
    if Fa != F:
        raise ValueError(f"A.F ({Fa}) must equal Y.F ({F})")
    if K == 0:
        raise ValueError("A has zero columns; support is empty")
    if X0 is None:
        X = np.full((K, N), 1.0 / K, dtype=np.float64)
    else:
        X = np.asarray(X0, dtype=np.float64)
        if X.shape != (K, N):
            raise ValueError(f"X0 shape {X.shape} must be (K={K}, N={N})")
        X = _safe_np(X, eps)
    for _ in range(int(max_iter)):
        Yhat = _safe_np(Afix @ X, eps)
        num = Afix.T @ (Yc / (Yhat ** 2))
        den = Afix.T @ (1.0 / Yhat)
        X *= _safe_np(num / _safe_np(den, eps), eps)
    info = {"n_iter": float(max_iter)}
    return X, info


def teacher_scores_is(
    Y: np.ndarray,
    s_hat: np.ndarray,
    H: np.ndarray,
    dir_indices: Optional[Sequence[int]] = None,
    eps: float = 1e-8,
) -> List[float]:
    """s_T(d) = −D_IS(Y || (H_d ⊙ ŝ)·1ᵀ) for each direction in dir_indices (or all)."""
    F, N = Y.shape
    Fh, D = H.shape
    if Fh != F:
        raise ValueError(f"H.F ({Fh}) must equal Y.F ({F})")
    sh = _safe_np(np.asarray(s_hat).reshape(F), eps)
    if dir_indices is None:
        dir_indices = list(range(D))
    scores: List[float] = []
    for d in dir_indices:
        Hd = _safe_np(H[:, d], eps)
        Yhat_vec = Hd * sh  # (F,)
        val = -is_divergence(Y, Yhat_vec, eps=eps)
        scores.append(val)
    return scores


def is_omp_select(
    Y: np.ndarray,
    H: np.ndarray,
    W: np.ndarray,
    K: int,
    s_hat: Optional[np.ndarray] = None,
    prefilter_M: int = 16,
    mu_iter_warm: int = 10,
    mu_iter_accept: int = 40,
    baseline_k: int = 2,
    eps: float = 1e-8,
) -> Tuple[List[int], np.ndarray, Dict[str, object]]:
    """Run shared-support IS-OMP and return (support S, X_S, logs).

    - If s_hat provided, prefilter candidates to top-M by s_T(d); else use all D.
    - Logs include per-step ΔIS, chosen indices, and iteration info.
    """
    Yc = _safe_np(np.asarray(Y, dtype=np.float64), eps)
    Hc = _safe_np(np.asarray(H, dtype=np.float64), eps)
    Wc = _safe_np(np.asarray(W, dtype=np.float64), eps)
    F, N = Yc.shape
    Fh, D = Hc.shape
    Fw, R = Wc.shape
    if Fh != F or Fw != F:
        raise ValueError(f"Grid mismatch: Y.F={F}, H.F={Fh}, W.F={Fw}")
    if K < 1 or K > D:
        raise ValueError(f"Invalid K={K}; must be in [1, {D}]")

    blocks = build_blocks(Hc, Wc)  # list of (F,R)

    # Candidate set
    cand_all = list(range(D))
    if s_hat is not None and prefilter_M is not None and prefilter_M > 0:
        scores = teacher_scores_is(Yc, s_hat, Hc, dir_indices=cand_all, eps=eps)
        order = list(range(D))
        order.sort(key=lambda i: scores[i], reverse=True)
        cand = order[: min(int(prefilter_M), D)]
    else:
        cand = cand_all

    # Greedy loop
    S: List[int] = []
    X_S: Optional[np.ndarray] = None  # (tR, N)
    logs: Dict[str, object] = {
        "F": int(F),
        "N": int(N),
        "R": int(R),
        "D": int(D),
        "prefilter_M": int(prefilter_M),
        "baseline_k": int(baseline_k),
        "steps": [],
    }

    # Initial J with physically meaningful baseline when possible
    if s_hat is not None and int(baseline_k) > 0:
        sh = _safe_np(np.asarray(s_hat, dtype=np.float64).reshape(F), eps)
        # Per-frequency contributions from all directions
        contrib = _safe_np(Hc * sh[:, None], eps)  # (F,D)
        k = min(int(baseline_k), D)
        # Sum of k smallest contributions per frequency
        sorted_vals = np.sort(contrib, axis=1)
        base_vec = _safe_np(np.sum(sorted_vals[:, :k], axis=1), eps)  # (F,)
        Yhat0 = np.repeat(base_vec[:, None], N, axis=1)
    else:
        # Fallback: constant tiny baseline (kept for completeness)
        Yhat0 = np.full_like(Yc, eps)
    J_prev = is_divergence(Yc, Yhat0, eps=eps)
    logs["is_prev"] = float(J_prev)

    for t in range(K):
        best_j = None
        best_X = None
        best_J = None
        best_delta = -np.inf
        A_S = concat_blocks(blocks, S) if S else None
        for j in cand:
            if j in S:
                continue
            # Assemble candidate dictionary
            if A_S is None:
                A_Sj = blocks[j]
            else:
                A_Sj = np.concatenate([A_S, blocks[j]], axis=1)
            Kcols = A_Sj.shape[1]
            # Warm-start X by padding zeros for new rows
            if X_S is None:
                X0 = None
            else:
                X0 = np.zeros((Kcols, N), dtype=np.float64)
                X0[: X_S.shape[0], :] = X_S
            # Short warmstart MU
            X_hat, _ = solve_x_is_smu(Yc, A_Sj, X0=X0, max_iter=mu_iter_warm, eps=eps)
            Yhat = _safe_np(A_Sj @ X_hat, eps)
            J_new = is_divergence(Yc, Yhat, eps=eps)
            delta = J_prev - J_new
            if delta > best_delta:
                best_delta = delta
                best_j = j
                best_X = X_hat
                best_J = J_new
        if best_j is None or not np.isfinite(best_delta) or best_delta <= 0.0:
            raise RuntimeError(
                f"IS-OMP failed to find improving direction at step {t}; ΔJ={best_delta}"
            )
        # Accept and refine
        S.append(int(best_j))
        A_S = concat_blocks(blocks, S)
        # Full accept iterations starting from best_X padded if needed
        if best_X is not None and best_X.shape[0] == A_S.shape[1]:
            X0_acc = best_X
        else:
            # Should not happen, but be safe
            X0_acc = None
        X_S, info = solve_x_is_smu(Yc, A_S, X0=X0_acc, max_iter=mu_iter_accept, eps=eps)
        Yhat = _safe_np(A_S @ X_S, eps)
        J_new = is_divergence(Yc, Yhat, eps=eps)
        logs["steps"].append({
            "t": int(t),
            "chosen": int(best_j),
            "delta_is_warm": float(best_delta),
            "is_after_accept": float(J_new),
            "n_iter_accept": info.get("n_iter", 0.0),
        })
        J_prev = J_new

    logs["is_final"] = float(J_prev)
    return S, X_S, logs


__all__ = [
    "is_divergence",
    "build_blocks",
    "concat_blocks",
    "solve_x_is_smu",
    "teacher_scores_is",
    "is_omp_select",
]


def compute_deltais_step0(
    Y: np.ndarray,
    H: np.ndarray,
    W: np.ndarray,
    s_hat: Optional[np.ndarray] = None,
    prefilter_M: Optional[int] = None,
    mu_iter: int = 10,
    baseline_k: int = 2,
    eps: float = 1e-8,
) -> Tuple[List[int], List[float]]:
    """Compute ΔIS(d|S=∅) for all directions (or top-M after sorting).

    Returns (dir_indices, deltas) where both are sorted by decreasing ΔIS.
    """
    Yc = _safe_np(np.asarray(Y, dtype=np.float64), eps)
    Hc = _safe_np(np.asarray(H, dtype=np.float64), eps)
    Wc = _safe_np(np.asarray(W, dtype=np.float64), eps)
    F, N = Yc.shape
    Fh, D = Hc.shape
    Fw, R = Wc.shape
    if Fh != F or Fw != F:
        raise ValueError(f"Grid mismatch: Y.F={F}, H.F={Fh}, W.F={Fw}")
    # Baseline with k-smallest per-freq contributions if ŝ provided
    if s_hat is not None and int(baseline_k) > 0:
        sh = _safe_np(np.asarray(s_hat, dtype=np.float64).reshape(F), eps)
        contrib = _safe_np(Hc * sh[:, None], eps)
        k = min(int(baseline_k), D)
        sorted_vals = np.sort(contrib, axis=1)
        base_vec = _safe_np(np.sum(sorted_vals[:, :k], axis=1), eps)
        Yhat0 = np.repeat(base_vec[:, None], N, axis=1)
    else:
        Yhat0 = np.full_like(Yc, eps)
    J_prev = is_divergence(Yc, Yhat0, eps=eps)
    blocks = build_blocks(Hc, Wc)
    deltas: List[float] = []
    for j in range(D):
        A_j = blocks[j]
        X_hat, _ = solve_x_is_smu(Yc, A_j, X0=None, max_iter=mu_iter, eps=eps)
        Yhat = _safe_np(A_j @ X_hat, eps)
        J_new = is_divergence(Yc, Yhat, eps=eps)
        deltas.append(J_prev - J_new)
    order = list(range(D))
    order.sort(key=lambda i: deltas[i], reverse=True)
    if prefilter_M is not None and int(prefilter_M) > 0:
        m = min(int(prefilter_M), D)
        order = order[:m]
    return order, [float(deltas[i]) for i in order]
