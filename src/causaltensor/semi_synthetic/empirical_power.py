r"""
Empirical rejection thresholds and Monte Carlo power (fixed baseline :math:`M`, random :math:`Z_{\mathrm{syn}}`).

**Size / power (short).** Under :math:`H_0`, reject with probability ≈ :math:`\alpha` (size).
**Power** is :math:`P(\text{reject} \mid H_1)` and increases with true effect size.

**This module**

1. Empirical threshold :math:`c`: :math:`(1-\alpha)` quantile of :math:`|\hat\tau|` using null draws
   from :func:`~causaltensor.semi_synthetic.aa_test.run_aa_test`.
2. Power grid: inject :math:`O = M + \delta\,\mathrm{mean}(|M|)\, Z_{\mathrm{syn}}`, reject if :math:`|\hat\tau|>c`.

For the **null histogram** of :math:`\hat\tau`, see :func:`~causaltensor.semi_synthetic.aa_test.plot_aa_null_figure`.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from causaltensor.semi_synthetic.aa_test import (
    VALID_PATTERNS,
    _normalise_methods,
    draw_synthetic_z,
)
from causaltensor.semi_synthetic.utils import build_baseline_M
from causaltensor.utils.common import get_tau_from_method, treated_states_and_starts_from_Z


def empirical_critical_abs_tau(
    null_df: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Per ``(method, pattern)``, set ``c`` so ~``alpha`` of null draws have ``|tau_hat| > c``.

    Uses the empirical ``(1 - alpha)`` quantile of ``abs(tau_hat)`` among finite values.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}.")

    rows = []
    for (method, pattern), g in null_df.groupby(["method", "pattern"], sort=True):
        x = g["tau_hat"].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 5:
            c = np.nan
        else:
            c = float(np.quantile(np.abs(x), 1.0 - alpha))
        rows.append(
            {
                "method": method,
                "pattern": pattern,
                "critical_abs_tau": c,
                "n_null": int(x.size),
            }
        )
    return pd.DataFrame(rows)


def run_empirical_power_grid(
    O: np.ndarray,
    Z: np.ndarray,
    null_df: pd.DataFrame,
    relative_effects: Sequence[float],
    *,
    baseline_type: str = "control",
    alpha: float = 0.05,
    n_trials_per_effect: int = 200,
    seed: int = 0,
    methods: Optional[List[str]] = None,
    patterns: Optional[List[str]] = None,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Monte Carlo power at each **relative** effect (fraction of ``mean(|M|)`` added on treated cells).

    Parameters
    ----------
    O, Z
        Observed panel and real treatment (only to build baseline ``M``).
    null_df
        Output of :func:`~causaltensor.semi_synthetic.aa_test.run_aa_test` on
        same ``O, Z`` / ``baseline_type`` — used only to calibrate ``c``.
    relative_effects
        Grid of multipliers ``delta`` in ``O = M + delta * mean(|M|) * Z_syn``.
    n_trials_per_effect
        Monte Carlo replications per ``(method, pattern, delta)``.
    alpha
        Target Type I rate for the empirical ``|tau|`` threshold.

    Returns
    -------
    thresholds_df
        Copy of empirical critical values (from ``null_df``).
    power_df
        Columns ``method, pattern, relative_effect, power, rejections, n_trials, std_err`` .
    """
    O = np.asarray(O, dtype=float)
    Z = np.asarray(Z, dtype=float)
    patterns = list(patterns) if patterns is not None else VALID_PATTERNS
    unknown = set(patterns) - set(VALID_PATTERNS)
    if unknown:
        raise ValueError(f"Unknown pattern(s): {unknown}. Valid: {VALID_PATTERNS}")
    methods_dict = _normalise_methods(methods, patterns)

    treated_states, treat_start_years = treated_states_and_starts_from_Z(Z)
    M, _, _ = build_baseline_M(O, treated_states, treat_start_years, baseline_type)
    scale = float(np.mean(np.abs(M))) if np.any(M) else 1.0

    null_slice = null_df
    if "baseline_type" in null_slice.columns:
        null_slice = null_slice[null_slice["baseline_type"] == baseline_type]

    thr_df = empirical_critical_abs_tau(null_slice, alpha=alpha)
    crit = thr_df.set_index(["method", "pattern"])["critical_abs_tau"]

    power_rows = []
    for p_idx, pattern_name in enumerate(patterns):
        for m_idx, (method_name, valid) in enumerate(methods_dict.items()):
            if pattern_name not in valid:
                continue
            key = (method_name, pattern_name)
            try:
                c = float(crit.loc[key])
            except KeyError:
                c = float("nan")
            if not 0 < n_trials_per_effect:
                raise ValueError("n_trials_per_effect must be positive.")

            for e_idx, delta in enumerate(relative_effects):
                rej = 0
                ok = 0
                inject = float(delta) * scale

                for t in range(n_trials_per_effect):
                    rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [int(seed), p_idx, m_idx, e_idx, t]
                        )
                    )
                    Z_syn = draw_synthetic_z(M, pattern_name, rng)
                    O_alt = M + inject * Z_syn
                    tau_hat = get_tau_from_method(method_name, O_alt, Z_syn)
                    if not np.isfinite(c) or np.isnan(tau_hat):
                        continue
                    ok += 1
                    if abs(float(tau_hat)) > c:
                        rej += 1

                p_hat = rej / ok if ok > 0 else float("nan")
                se = np.sqrt(p_hat * (1 - p_hat) / ok) if ok > 0 else float("nan")
                power_rows.append(
                    {
                        "method": method_name,
                        "pattern": pattern_name,
                        "relative_effect": float(delta),
                        "power": p_hat,
                        "rejections": rej,
                        "n_trials": ok,
                        "std_err": se,
                    }
                )
                if verbose:
                    print(
                        f"{method_name} {pattern_name} delta={delta}: "
                        f"power≈{p_hat:.3f} (n={ok})"
                    )

    return thr_df, pd.DataFrame(power_rows)


def plot_empirical_power_figure(
    power_df: pd.DataFrame,
    *,
    figsize: Tuple[float, float] = (12, 10),
):
    """
    Power vs ``relative_effect`` for each pattern (subplot), lines per method.

    Expects a frame like the second return of :func:`run_empirical_power_grid`.
    Requires ``matplotlib``.
    """
    import matplotlib.pyplot as plt

    patterns = list(dict.fromkeys(power_df["pattern"].tolist()))
    methods = list(dict.fromkeys(power_df["method"].tolist()))
    n_p = len(patterns)
    fig, axes = plt.subplots(
        n_p, 1, figsize=(figsize[0], max(2.5, 2.8 * n_p)), squeeze=False
    )
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(methods), 2)))

    for ax, pat in zip(axes.flat, patterns):
        sub = power_df[power_df["pattern"] == pat]
        for k, meth in enumerate(methods):
            s2 = sub[sub["method"] == meth].sort_values("relative_effect")
            if s2.empty:
                continue
            ax.plot(
                s2["relative_effect"],
                s2["power"],
                marker="o",
                ms=3,
                color=colors[k % len(colors)],
                label=meth,
            )
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.8, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(f"Empirical power — pattern = {pat}")
        ax.set_xlabel(r"relative effect $\delta$ (inject $\delta \cdot \mathrm{mean}(|M|)$ on treated cells)")
        ax.set_ylabel("power")
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        r"Reject $H_0:\tau=0$ when $|\hat\tau|>c$; "
        r"$c$ = empirical $(1-\alpha)$ quantile of $|\hat\tau|$ under null",
        fontsize=11,
    )
    fig.tight_layout()
    return fig, axes
