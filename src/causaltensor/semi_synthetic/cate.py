"""
Semi-synthetic CATE evaluation from a fitted untreated surface.

Block assignment on a clean baseline ``M``. Unit effects are a fixed linear
function of observed covariates, centered on the full sample so their
population average is ``tau*``. Each trial's ATT is the average of those
effects over the units treated in that trial. Estimators are scored by the
unit-level effects implied by their counterfactual baselines.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from causaltensor.semi_synthetic.aa_test import DEFAULT_METHODS
from causaltensor.semi_synthetic.utils import (
    build_baseline_M,
    sample_treatment_parameters,
)
from causaltensor.utils.common import (
    get_fit_result_from_method,
    treated_states_and_starts_from_Z,
)
from causaltensor.utils.treatment_patterns import Z_block

logger = logging.getLogger(__name__)

# Weights on standardized hc, csh_i, openness, chosen before seeing estimates.
# gamma scales heterogeneity in units of |tau*|.
BETAS: Tuple[float, ...] = (1.0, 0.5, -0.5)
GAMMA: float = 0.5


def _standardize(raw: np.ndarray) -> np.ndarray:
    center = raw.mean(axis=0)
    scale = raw.std(axis=0, ddof=1)
    scale = np.where(scale > 0, scale, 1.0)
    return (raw - center) / scale


def _fixed_unit_effects(M: np.ndarray, h: np.ndarray, treatment_level: float, gamma: float):
    """tau_i depends only on X. Its average over all units equals tau*."""
    tau_star = float(np.mean(np.abs(M)) * treatment_level)
    h_c = h - np.mean(h)
    tau_i = tau_star + gamma * abs(tau_star) * h_c
    return tau_i, tau_star


def _unit_hat(O: np.ndarray, Mhat: np.ndarray, Z: np.ndarray) -> np.ndarray:
    treated = np.asarray(Z, dtype=float) > 0
    counts = treated.sum(axis=1)
    gap = np.where(treated, O - Mhat, 0.0).sum(axis=1)
    hat = np.full(O.shape[0], np.nan)
    ok = counts > 0
    hat[ok] = gap[ok] / counts[ok]
    return hat


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _gate_rows(trial, method, standardized, names, treated, tau_i, hat, thresholds):
    rows = []
    idx = np.where(treated)[0]
    for j, name in enumerate(names):
        x = standardized[idx, j]
        for group, sel in (("Low", x <= thresholds[j]), ("High", x > thresholds[j])):
            take = idx[sel]
            rows.append({
                "trial": trial,
                "method": method,
                "covariate": name,
                "group": group,
                "n_units": int(take.size),
                "tau_true": float(np.mean(tau_i[take])) if take.size else np.nan,
                "tau_hat": float(np.nanmean(hat[take])) if take.size else np.nan,
            })
    return rows


def run_cate(
    O: np.ndarray,
    Z: np.ndarray,
    covariates: np.ndarray,
    *,
    covariate_names: Sequence[str],
    betas: Sequence[float] = BETAS,
    gamma: float = GAMMA,
    treatment_level: float = 0.1,
    n_trials: int = 100,
    seed: int = 0,
    methods: Optional[Sequence[str]] = None,
    baseline_type: str = "control",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Block-assignment CATE experiment on one panel.

    ``covariates`` has one row per row of ``O``. A control baseline drops the
    ever-treated rows of ``O``, and the same rows are dropped from the
    covariates. Columns are standardized across the remaining units.

    Returns
    -------
    metrics : DataFrame
        One row per ``(trial, method)`` with ``root_pehe``, ``corr``, and the
        trial ATT.
    gates : DataFrame
        One row per ``(trial, method, covariate, group)`` with the true and
        estimated group average. Groups use full-sample median splits.
    """
    O = np.asarray(O, dtype=float)
    Z = np.asarray(Z, dtype=float)
    raw = np.asarray(covariates, dtype=float)
    names = list(covariate_names)
    if raw.ndim != 2 or raw.shape[0] != O.shape[0] or raw.shape[1] != len(names):
        raise ValueError(
            f"covariates must be shape (n, {len(names)}); got {raw.shape} for O {O.shape}."
        )
    if len(betas) != len(names):
        raise ValueError(f"betas has length {len(betas)}; covariates have {len(names)} columns.")

    treated_states, treat_starts = treated_states_and_starts_from_Z(Z)
    M, _, _ = build_baseline_M(O, treated_states, treat_starts, baseline_type)
    if baseline_type == "control":
        keep = np.ones(O.shape[0], dtype=bool)
        keep[treated_states] = False
        raw = raw[keep]
    standardized = _standardize(raw)
    h = standardized @ np.asarray(betas, dtype=float)
    tau_i, tau_star = _fixed_unit_effects(M, h, treatment_level, gamma)
    gate_thresholds = np.median(standardized, axis=0)
    n, T = M.shape
    method_names: List[str] = list(DEFAULT_METHODS if methods is None else methods)
    metric_rows = []
    gate_rows = []

    for trial in range(n_trials):
        rng = np.random.default_rng(np.random.SeedSequence([seed, trial]))
        m1, m2, _, _ = sample_treatment_parameters(n, T, rng)
        Z_syn = np.asarray(Z_block(M, m1=m1, m2=m2, rng=rng), dtype=float)
        O_syn = M + tau_i[:, None] * Z_syn
        treated = Z_syn.sum(axis=1) > 0
        treated_idx = np.where(treated)[0]
        truth = tau_i[treated_idx]
        att_true = float(np.mean(truth))
        for method_name in method_names:
            res, err = get_fit_result_from_method(method_name, O_syn, Z_syn)
            root_pehe = corr = np.nan
            hat = np.full(n, np.nan)
            if err is None and res is not None and res.baseline is not None:
                Mhat = np.asarray(res.baseline, dtype=float)
                if Mhat.shape == O_syn.shape:
                    hat = _unit_hat(O_syn, Mhat, Z_syn)
                    est = hat[treated_idx]
                    if np.all(np.isfinite(est)):
                        root_pehe = float(np.sqrt(np.mean((est - truth) ** 2)))
                        corr = _corr(est, truth)
            elif verbose and err is not None:
                logger.info("trial %s %s failed: %s", trial, method_name, err)
            metric_rows.append({
                "trial": trial,
                "method": method_name,
                "root_pehe": root_pehe,
                "corr": corr,
                "n_treated": int(treated_idx.size),
                "tau_star": tau_star,
                "att_true": att_true,
            })
            gate_rows.extend(
                _gate_rows(
                    trial, method_name, standardized, names, treated, tau_i, hat, gate_thresholds,
                )
            )
        if verbose:
            logger.info("Finished trial %s/%s", trial + 1, n_trials)

    return pd.DataFrame(metric_rows), pd.DataFrame(gate_rows)
