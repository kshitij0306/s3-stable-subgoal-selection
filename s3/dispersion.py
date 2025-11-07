import math
from typing import Dict, Tuple

import numpy as np
import torch


VAR_FLOOR = 1e-3


def _safe_var(logS: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return variance and log-variance with a numerical floor."""
    var = torch.exp(logS).clamp_min(VAR_FLOOR)
    log_var = torch.log(var)
    return var, log_var


def mdn_moments(pi: torch.Tensor, mu: torch.Tensor, logS: torch.Tensor):
    """
    MDN mixture mean & covariance, batch-safe.
    Shapes in:
      pi   : (B, M) or (M,)
      mu   : (B, M, D) or (M, D)
      logS : (B, M, D) or (M, D)
    Shapes out:
      mu_mix : (B, D)
      cov    : (B, D, D)
    """
    # --- Force shapes ---
    if pi.dim() == 1:
        pi = pi.unsqueeze(0)        # (1, M)
    if mu.dim() == 2:
        mu = mu.unsqueeze(0)        # (1, M, D)
    if logS.dim() == 2:
        logS = logS.unsqueeze(0)    # (1, M, D)

    B, M = pi.shape
    _, M2, D = mu.shape
    assert M == M2, f"Mismatch: pi has {M}, mu has {M2}"

    w   = pi                        # (B, M)
    var, logS = _safe_var(logS)     # (B, M, D)

    # --- Mixture mean ---
    mu_mix = (w.unsqueeze(-1) * mu).sum(dim=1)               # (B, D)

    # --- Diagonal term from variances ---
    diag_term = torch.diag_embed((w.unsqueeze(-1) * var).sum(dim=1))  # (B, D, D)

    # --- Outer product term from component means ---
    sum_mu_outer = torch.einsum('bm,bmd,bme->bde', w, mu, mu)         # (B, D, D)

    # --- Covariance ---
    ExxT = diag_term + sum_mu_outer
    mu_outer = torch.einsum('bd,be->bde', mu_mix, mu_mix)             # (B, D, D)
    cov = ExxT - mu_outer

    # --- Jitter ---
    eps = 1e-6
    cov = cov + eps * torch.eye(D, device=cov.device).expand(B, D, D)
    return mu_mix, cov


def mdn_sample(pi: torch.Tensor, mu: torch.Tensor, logS: torch.Tensor, K: int) -> torch.Tensor:
    """Draw K samples from a diagonal-cov MDN (single batch element)."""
    w  = pi.squeeze(0)                  # (M,)
    Mu = mu.squeeze(0)                  # (M,D)
    var, _ = _safe_var(logS.squeeze(0))
    Sd = var.sqrt()                     # (M,D)

    cat = torch.distributions.Categorical(w)
    idx = cat.sample((K,))              # (K,)
    eps = torch.randn(K, Mu.size(1), device=Mu.device)
    samples = Mu[idx] + eps * Sd[idx]
    return samples                      # (K,D)


def _goal_abs(state_np: np.ndarray, subgoal_np: np.ndarray, d: int, absolute_goal: bool):
    """Compute absolute goal coordinates used for dispersion wrt g."""
    s_xy = state_np[:d]
    if absolute_goal:
        g_abs = subgoal_np[:d]
    else:
        g_abs = (state_np[:d] + subgoal_np[:d])
    return s_xy, g_abs


def _goal_abs_batch(states_np: np.ndarray, subgoals_np: np.ndarray, d: int, absolute_goal: bool):
    """Vectorised absolute goal helper."""
    s_xy = states_np[..., :d]
    if absolute_goal:
        g_abs = subgoals_np[..., :d]
    else:
        g_abs = states_np[..., :d] + subgoals_np[..., :d]
    return s_xy, g_abs


def dispersion_score_batch(
    pi: torch.Tensor,
    mu: torch.Tensor,
    logS: torch.Tensor,
    state_np: np.ndarray,
    subgoal_np: np.ndarray,
    d: int,
    args,
    device: torch.device,
) -> torch.Tensor:
    """Vectorised dispersion score for batched inputs (stats not returned)."""

    mu_mix, cov = mdn_moments(pi, mu, logS)
    B = mu_mix.shape[0]

    states = np.asarray(state_np)
    if states.ndim == 1:
        states = np.broadcast_to(states, (B, states.shape[0]))
    subgoals = np.asarray(subgoal_np)
    if subgoals.ndim == 1:
        subgoals = np.broadcast_to(subgoals, (B, subgoals.shape[0]))

    trace = cov.diagonal(dim1=1, dim2=2).sum(-1)

    s_xy, g_abs = _goal_abs_batch(states, subgoals, d, args.absolute_goal)
    g_t = torch.from_numpy(g_abs).float().to(device)
    mu_diff = mu_mix - g_t
    w2 = trace + torch.norm(mu_diff, dim=1).pow(2)

    score = trace

    if args.dispersion == "logdet":
        _, logabsdet = torch.linalg.slogdet(cov)
        score = logabsdet
    elif args.dispersion == "maxeig":
        evals = torch.linalg.eigvalsh(cov)
        score = evals.max(dim=1).values
    elif args.dispersion == "anisotropy":
        evals = torch.linalg.eigvalsh(cov)
        maxeig = evals.max(dim=1).values
        mineig = torch.clamp(evals.min(dim=1).values, min=1e-12)
        score = maxeig / mineig
    elif args.dispersion == "dir_perp":
        v = torch.from_numpy(g_abs - s_xy).float().to(device)
        v_norm = torch.norm(v, dim=1, keepdim=True) + 1e-12
        v_unit = v / v_norm
        dir_var = torch.einsum('bd,bde,be->b', v_unit, cov, v_unit)
        score = trace - dir_var
    elif args.dispersion == "w2":
        score = w2

    return score


def dispersion_score(
    pi: torch.Tensor,
    mu: torch.Tensor,
    logS: torch.Tensor,
    state_np: np.ndarray,
    subgoal_np: np.ndarray,
    d: int,
    args,
    device: torch.device,
    *,
    collect_stats: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the chosen dispersion score and a dict of TSD diagnostics.
    Returns:
      score: float  (the scalar used for PBRS bonus)
      stats: dict   (trace, logdet, maxeig, anisotropy, dir_var, perp_var, w2, chance_eps, cvar, mi_proxy)
    """
    mu_mix, cov = mdn_moments(pi, mu, logS)
    mu_mix = mu_mix[0]
    cov = cov[0]

    trace = float(torch.diagonal(cov).sum().item())
    stats: Dict[str, float] = {"trace": trace}

    s_xy, g_abs = _goal_abs(state_np, subgoal_np, d, args.absolute_goal)
    g_t = torch.from_numpy(g_abs).float().to(device)
    mu_diff = mu_mix - g_t
    w2 = float(trace + torch.norm(mu_diff, p=2).pow(2).item())
    stats["w2"] = w2

    score = trace

    need_logdet = collect_stats or args.dispersion == "logdet"
    if need_logdet:
        _, logabsdet = torch.linalg.slogdet(cov)
        logdet = float(logabsdet.item())
        stats["logdet"] = logdet
    else:
        logdet = float("nan")

    need_eigs = collect_stats or args.dispersion in {"maxeig", "anisotropy"}
    if need_eigs:
        evals = torch.linalg.eigvalsh(cov)
        maxeig = float(evals.max().item())
        mineig = float(torch.clamp(evals.min(), min=1e-12).item())
        stats["maxeig"] = maxeig
        stats["anisotropy"] = maxeig / max(mineig, 1e-12)
    else:
        maxeig = float("nan")

    need_dir = collect_stats or args.dispersion in {"dir_perp"}
    if need_dir:
        v = g_abs - s_xy
        v_norm = np.linalg.norm(v) + 1e-12
        v_unit = torch.from_numpy(v / v_norm).float().to(device)
        dir_var = float((v_unit @ cov @ v_unit).item())
        perp_var = trace - dir_var
        stats["dir_var"] = dir_var
        stats["perp_var"] = perp_var
    else:
        perp_var = trace

    if args.dispersion in ("chance", "cvar"):
        K = int(args.disp_samples)
        samples = mdn_sample(pi, mu, logS, K)
        dists = torch.linalg.norm(samples - g_t, dim=1)
        if args.dispersion == "chance":
            p_ok = float((dists <= args.disp_eps).float().mean().item())
            stats["chance_eps"] = p_ok
            score = 1.0 - p_ok
        else:
            alpha = float(args.disp_alpha)
            q = torch.quantile(dists, 1.0 - alpha)
            tail = dists[dists >= q]
            cvar = float(tail.mean().item() if tail.numel() > 0 else dists.mean().item())
            stats["cvar"] = cvar
            score = cvar

    if args.dispersion == "var":
        score = trace
    elif args.dispersion == "logdet":
        score = stats.get("logdet", logdet)
    elif args.dispersion == "maxeig":
        score = stats.get("maxeig", maxeig)
    elif args.dispersion == "anisotropy":
        score = stats.get("anisotropy", float("nan"))
    elif args.dispersion == "dir_perp":
        score = perp_var
    elif args.dispersion == "w2":
        score = w2

    dfloat = float(d)
    stats["mi_proxy"] = -0.5 * dfloat * math.log(max((2.0 * math.pi * math.e / dfloat) * trace, 1e-12))
    return float(score), stats
