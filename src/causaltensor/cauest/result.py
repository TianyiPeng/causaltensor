import numpy as np


def _fmt_fe(arr, label):
    """One-line summary of a fixed-effect vector (min / mean / max)."""
    v = np.asarray(arr).ravel()
    return f"{label:<24s}: min={v.min():.4g}  mean={v.mean():.4g}  max={v.max():.4g}"


def _fmt_coefs(arr, label):
    """One-line display of a coefficient array."""
    v = np.round(np.asarray(arr).ravel(), 6)
    return f"{label:<24s}: {v}"


class Result:
    def __init__(self, baseline=None, tau=None, covariance_tau=None, std_tau=None,
                 return_tau_scalar=False, inference_method=None):
        self.baseline = baseline
        self.tau = tau
        self.covariance_tau = covariance_tau
        self.std_tau = std_tau
        self.inference_method = inference_method
        # Attached by each solver's fit() after construction; enables all diagnostics.
        self.O = None
        self.Z = None
        if return_tau_scalar:
            # tau may be 0-d (numpy scalar); tau[0] raises "invalid index to scalar variable".
            if tau is not None:
                self.tau = float(np.asarray(tau).ravel()[0])
            if self.covariance_tau is not None:
                self.covariance_tau = float(np.asarray(self.covariance_tau).ravel()[0])
            if self.std_tau is not None:
                self.std_tau = float(np.asarray(self.std_tau).ravel()[0])

    # ------------------------------------------------------------------ #
    #  Core derived arrays                                                 #
    # ------------------------------------------------------------------ #

    @property
    def residuals(self):
        """O - baseline for all cells; None if O or baseline are unavailable."""
        if self.O is None or self.baseline is None:
            return None
        return self.O - self.baseline

    @property
    def effect_matrix(self):
        """Residuals restricted to treated cells: (O - baseline) * (Z > 0)."""
        r = self.residuals
        if r is None or self.Z is None:
            return None
        return r * (self.Z > 0)

    # ------------------------------------------------------------------ #
    #  Treatment pattern                                                   #
    # ------------------------------------------------------------------ #

    @property
    def z_pattern(self):
        """Detected treatment pattern: 'block', 'staggered', 'non-monotone', or None.

        Block
            All treated units share the same treatment start period (irreversible).
        Staggered
            Treated units have unit-specific start periods, all irreversible.
        Non-monotone
            Treatment can turn off; typical of IID or adaptive assignment.
        """
        if self.Z is None:
            return None
        Z2d = np.asarray(self.Z)
        if Z2d.ndim != 2:
            Z2d = Z2d.squeeze()
        if Z2d.ndim != 2:
            return None
        treated_rows = np.where(np.any(Z2d > 0, axis=1))[0]
        if len(treated_rows) == 0:
            return None
        # Monotone = once Z=1, it stays 1 for the rest of that row.
        for i in treated_rows:
            diffs = np.diff(Z2d[i].astype(float))
            if np.any(diffs < -1e-9):
                return "non-monotone"
        # All treated rows are monotone; distinguish block vs staggered.
        first_periods = [int(np.where(Z2d[i] > 0)[0][0]) for i in treated_rows]
        if len(set(first_periods)) == 1:
            return "block"
        return "staggered"

    # ------------------------------------------------------------------ #
    #  Fit diagnostics                                                     #
    # ------------------------------------------------------------------ #

    @property
    def untreated_rmse(self):
        """RMSE of (O - baseline) on all cells where Z == 0.

        Covers both pure control units and pre-treatment cells of treated units.
        Accessible on the result object; use control_rmse / pre_exposure_rmse
        for a more informative decomposition.
        """
        if self.O is None or self.baseline is None or self.Z is None:
            return None
        mask = (self.Z == 0)
        if not np.any(mask):
            return None
        r = (self.O - self.baseline)[mask]
        return float(np.sqrt(np.mean(r ** 2)))

    @property
    def control_rmse(self):
        """RMSE on pure control units (never treated) across all time periods.

        Tests how well the model captures the dynamics of units that are
        never treated.  Particularly informative for donor-based methods
        (OLS SC, RSC) where the counterfactual is explicitly built from donors.
        Returns None when every unit has at least one treated period (e.g. IID).
        """
        if self.O is None or self.baseline is None or self.Z is None:
            return None
        Z2d = np.asarray(self.Z)
        control_units = np.where(np.all(Z2d == 0, axis=1))[0]
        if len(control_units) == 0:
            return None
        r = (self.O - self.baseline)[control_units, :]
        return float(np.sqrt(np.mean(r ** 2)))

    @property
    def untreated_r2(self):
        """R-squared on untreated cells: 1 - SS_res / SS_tot where Z == 0.

        Scale-free version of untreated_rmse.  Directly interpretable (0-1)
        and comparable across datasets with different outcome scales.
        """
        if self.O is None or self.baseline is None or self.Z is None:
            return None
        mask = (self.Z == 0)
        if not np.any(mask):
            return None
        y = self.O[mask]
        r = (self.O - self.baseline)[mask]
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return None
        return float(1.0 - np.sum(r ** 2) / ss_tot)

    def _pre_exposure_info(self):
        """Return (rmse, n_cells) for pre-exposure periods across ALL units, or (None, 0).

        Pre-exposure is defined per unit as all periods strictly before that
        unit's first treated period:

        - Treated units: periods 0 .. t_i - 1  (standard pre-treatment window)
        - Control units (never treated): all T periods (always "pre-exposure")

        For non-monotone patterns (IID, adaptive) this only captures the period
        before the *first* treatment episode per unit; later gaps between episodes
        are not included.  Use untreated_r2 for a fuller picture in those cases.
        """
        if self.O is None or self.baseline is None or self.Z is None:
            return None, 0
        Z2d = np.asarray(self.Z)
        if Z2d.ndim != 2:
            Z2d = Z2d.squeeze()
        if Z2d.ndim != 2:
            return None, 0
        pre_mask = np.zeros_like(Z2d, dtype=bool)
        for i in range(Z2d.shape[0]):
            treated_cols = np.where(Z2d[i] > 0)[0]
            if len(treated_cols) == 0:
                # Control unit: all periods are pre-exposure.
                pre_mask[i, :] = True
            else:
                t0 = int(treated_cols[0])
                if t0 > 0:
                    pre_mask[i, :t0] = True
        n_cells = int(np.sum(pre_mask))
        if n_cells == 0:
            return None, 0
        r = (self.O - self.baseline)[pre_mask]
        return float(np.sqrt(np.mean(r ** 2))), n_cells

    @property
    def pre_exposure_rmse(self):
        """RMSE on pre-treatment periods for each treated unit.

        See _pre_exposure_info() for details on how pre-exposure is defined
        and its limitations for non-monotone (IID/adaptive) patterns.
        """
        rmse, _ = self._pre_exposure_info()
        return rmse

    @property
    def rmspe_ratio(self):
        """Signal-to-noise ratio: |ATT| / pre_exposure_rmse.

        A ratio >> 1 indicates the estimated effect is large relative to the
        model's pre-treatment fit error.
        """
        pre = self.pre_exposure_rmse
        if pre is None or pre == 0:
            return None
        tau_scalar = float(np.mean(self.tau)) if np.ndim(self.tau) > 0 else float(self.tau)
        return float(abs(tau_scalar) / pre)

    # ------------------------------------------------------------------ #
    #  Estimator-internals hook (overridden by subclasses)                 #
    # ------------------------------------------------------------------ #

    def _summary_internals(self):
        """Return a list of strings for the 'Model internals' section of summary().

        Subclasses override this to expose weights, rank, lambda, num_factors,
        covariate coefficients, convergence info, etc.  Each string is one
        printed line (with two extra spaces of indent prepended by summary()).
        An empty string inserts a blank separator.  Return [] to show nothing.
        """
        return []

    # ------------------------------------------------------------------ #
    #  Summary                                                             #
    # ------------------------------------------------------------------ #

    def summary(self):
        """Print a concise table of ATT, fit diagnostics, and model internals.

        Returns
        -------
        self
            Enables method chaining.
        """
        W = 56

        def _fmt(v, prec=6):
            if v is None:
                return "N/A"
            if np.ndim(v) == 0:
                return f"{float(v):.{prec}g}"
            arr = np.asarray(v)
            if arr.size == 1:
                return f"{float(arr.flat[0]):.{prec}g}"
            return str(np.round(arr, prec))

        SEP = "-" * W
        lines = ["=" * W, f"  {type(self).__name__}", "=" * W]

        # ---- Panel info header ----------------------------------------
        if self.O is not None and self.Z is not None:
            N, T = self.O.shape
            n_treated = int(np.sum(self.Z > 0))
            n_total = N * T
            n_treated_units = int(np.sum(np.any(self.Z > 0, axis=1)))
            pct = 100.0 * n_treated / n_total
            lines.append(f"  {'Panel':<26s}: {N} units x {T} periods")
            lines.append(f"  {'Treated cells':<26s}: {n_treated} / {n_total}  ({pct:.1f}%)")
            lines.append(f"  {'Treated units':<26s}: {n_treated_units}")
            pattern = self.z_pattern
            if pattern == "block":
                Z2d = np.asarray(self.Z)
                t0 = int(np.where(Z2d[np.where(np.any(Z2d > 0, axis=1))[0][0]] > 0)[0][0])
                lines.append(f"  {'Treatment pattern':<26s}: Block  (T0 = {t0})")
            elif pattern == "staggered":
                Z2d = np.asarray(self.Z)
                treated_rows = np.where(np.any(Z2d > 0, axis=1))[0]
                fps = [int(np.where(Z2d[i] > 0)[0][0]) for i in treated_rows]
                lines.append(
                    f"  {'Treatment pattern':<26s}: Staggered  (T0: {min(fps)}..{max(fps)})"
                )
            elif pattern == "non-monotone":
                lines.append(f"  {'Treatment pattern':<26s}: Non-monotone (IID/adaptive)")
            lines.append(SEP)

        # ---- ATT --------------------------------------------------------
        lines.append(f"  {'ATT (tau)':<26s}: {_fmt(self.tau)}")
        if self.std_tau is not None:
            lines.append(f"  {'Std dev (tau)':<26s}: {_fmt(self.std_tau)}")
        if self.inference_method is not None:
            lines.append(f"  {'Inference method':<26s}: {self.inference_method}")

        # ---- Fit diagnostics --------------------------------------------
        lines += ["", "  --- Fit diagnostics ---"]
        lines.append(f"  {'Untreated R2':<26s}: {_fmt(self.untreated_r2, prec=4)}")

        ctrl = self.control_rmse
        if ctrl is not None:
            lines.append(f"  {'Control RMSE':<26s}: {_fmt(ctrl)}")
        else:
            lines.append(f"  {'Control RMSE':<26s}: N/A  (no pure control units)")

        pre_rmse, pre_n = self._pre_exposure_info()
        if pre_rmse is not None:
            lines.append(
                f"  {'Pre-exposure RMSE':<26s}: {_fmt(pre_rmse)}  ({pre_n} cells)"
            )
            if self.z_pattern == "non-monotone":
                lines.append(
                    f"  {'  [non-monotone note]':<26s}: pre-period before first treatment only;"
                )
                lines.append(
                    f"  {'':26s}  prefer Untreated R2 for full picture"
                )
        else:
            lines.append(f"  {'Pre-exposure RMSE':<26s}: N/A")

        lines.append(f"  {'RMSPE ratio':<26s}: {_fmt(self.rmspe_ratio, prec=4)}")

        # ---- Model internals --------------------------------------------
        internals = self._summary_internals()
        if internals:
            lines += ["", "  --- Model internals ---"]
            for item in internals:
                lines.append(f"  {item}" if item else "")

        lines.append("=" * W)
        print("\n".join(lines))
        return self

    # ------------------------------------------------------------------ #
    #  Plot helpers                                                        #
    # ------------------------------------------------------------------ #

    def plot_actual_vs_counterfactual(
        self,
        unit,
        unit_label=None,
        time_labels=None,
        title=None,
    ):
        """Interactive actual-vs-counterfactual plot for a single unit.

        Single-panel Plotly figure showing actual outcome (solid blue) vs
        counterfactual baseline (dashed orange).

        Treatment periods (Z=1) are highlighted with a green background band
        and a green tick strip along the x-axis.  For monotone (block/staggered)
        treatment a dashed vertical line marks T0 and the shaded region extends
        to the end of the panel; for non-monotone patterns each treated period
        gets its own highlighted column.

        An annotation box shows the unit-specific ATT, pre-treatment RMSE,
        overall panel ATT, and Std(tau) where available.

        Parameters
        ----------
        unit : int
            Row index of the unit to plot.
        unit_label : str, optional
            Human-readable name for the unit (e.g. a state or country name).
            Defaults to ``'Unit {unit}'``.
        time_labels : list, optional
            X-axis tick labels (length T).  Defaults to integer indices 0..T-1.
        title : str, optional
            Override the figure title.

        Returns
        -------
        plotly.graph_objects.Figure
            Call ``.show()`` to display or ``.write_html()`` to save.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError(
                "plotly is required for this plot.  "
                "Install it with: pip install plotly"
            )

        if self.O is None or self.baseline is None:
            raise ValueError(
                "O and baseline must be available on the result object "
                "(call solver.fit() first)."
            )
        if self.Z is None:
            raise ValueError(
                "Z (treatment mask) must be available on the result object "
                "(call solver.fit() first)."
            )

        N, T = self.O.shape
        if not (0 <= unit < N):
            raise ValueError(f"unit={unit} is out of range [0, {N - 1}].")

        actual = self.O[unit, :]
        cf     = self.baseline[unit, :]
        gap    = actual - cf
        z_unit = np.asarray(self.Z)[unit, :]

        # Always use integer x internally so vrect/vline stay simple.
        x = list(range(T))
        tick_text = [str(v) for v in (time_labels if time_labels is not None else x)]

        label = unit_label or f"Unit {unit}"
        treated_idx = np.where(z_unit > 0)[0]
        is_treated  = len(treated_idx) > 0
        is_monotone = is_treated and np.all(np.diff(z_unit.astype(float)) >= -1e-9)

        # Unit-level diagnostics
        unit_att = float(np.mean(gap[z_unit > 0])) if is_treated else None
        if is_treated and int(treated_idx[0]) > 0:
            pre_cols = np.arange(int(treated_idx[0]))
            unit_pre_rmse = float(np.sqrt(np.mean(gap[pre_cols] ** 2)))
        else:
            unit_pre_rmse = None

        # ------------------------------------------------------------------ #
        #  Single-panel figure                                                 #
        # ------------------------------------------------------------------ #
        fig = go.Figure()

        # Colours — green theme for treatment
        SHADE_BG    = "rgba(34, 160, 80, 0.15)"
        VLINE_COLOR = "rgba(20, 120, 55, 0.90)"

        # ---- Treatment highlighting ----------------------------------------
        if is_treated:
            if is_monotone:
                t0 = int(treated_idx[0])
                fig.add_vrect(
                    x0=t0 - 0.5, x1=T - 0.5,
                    fillcolor=SHADE_BG, layer="below", line_width=0,
                )
                fig.add_vline(
                    x=t0,
                    line=dict(color=VLINE_COLOR, width=2, dash="dash"),
                )
                fig.add_annotation(
                    x=t0 + 0.3, xref="x",
                    y=1.0, yref="paper",
                    text="Treatment",
                    showarrow=False,
                    font=dict(size=10, color=VLINE_COLOR, family="Arial"),
                    xanchor="left", yanchor="top",
                )
            else:
                # Non-monotone: one band per treated column
                for t_idx in treated_idx:
                    fig.add_vrect(
                        x0=t_idx - 0.5, x1=t_idx + 0.5,
                        fillcolor="rgba(34, 160, 80, 0.22)",
                        layer="below", line_width=0,
                    )


        # ---- Main traces ---------------------------------------------------
        fig.add_trace(
            go.Scatter(
                x=x, y=actual,
                mode="lines+markers",
                name="Actual",
                line=dict(color="#1f77b4", width=2.5),
                marker=dict(size=4),
                hovertemplate="t=%{x}  actual=%{y:.4g}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=cf,
                mode="lines",
                name="Counterfactual",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
                hovertemplate="t=%{x}  counterfactual=%{y:.4g}<extra></extra>",
            )
        )

        # ---- Annotation box -----------------------------------------------
        ann_parts = []
        if unit_att is not None:
            ann_parts.append(f"<b>Unit ATT:</b>  {unit_att:.4g}")
        if unit_pre_rmse is not None:
            ann_parts.append(f"<b>Pre-RMSE:</b>  {unit_pre_rmse:.4g}")
        if self.tau is not None:
            t_val = float(np.mean(self.tau)) if np.ndim(self.tau) > 0 else float(self.tau)
            ann_parts.append(f"<b>Overall ATT:</b> {t_val:.4g}")
        if self.std_tau is not None:
            std_val = float(np.mean(self.std_tau)) if np.ndim(self.std_tau) > 0 else float(self.std_tau)
            ann_parts.append(f"<b>Std(tau):</b>  {std_val:.4g}")

        if ann_parts:
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.01, y=0.97,
                text="<br>".join(ann_parts),
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.90)",
                bordercolor="#aaaaaa",
                borderwidth=1,
                font=dict(size=11),
                xanchor="left",
                yanchor="top",
            )

        # ---- Layout -------------------------------------------------------
        plot_title = title or f"{type(self).__name__}  -  {label}"
        fig.update_layout(
            title=dict(text=plot_title, font=dict(size=14), x=0.5, xanchor="center"),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.04,
                xanchor="right", x=1,
                font=dict(size=11),
            ),
            height=420,
            margin=dict(l=65, r=30, t=100, b=55),
            plot_bgcolor="white",
            paper_bgcolor="white",
            # Primary y-axis for the outcome lines
            yaxis=dict(
                title="Outcome",
                showgrid=True,
                gridcolor="#eeeeee",
            ),
            xaxis=dict(
                title="Time period",
                tickvals=x,
                ticktext=tick_text,
                showgrid=True,
                gridcolor="#eeeeee",
            ),
        )

        return fig
