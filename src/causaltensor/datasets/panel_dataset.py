# src/causaltensor/datasets/panel_dataset.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Dict, Any, List, Literal, Union
import numpy as np
import pandas as pd
from pathlib import Path

ArrayLike = Union[np.ndarray, pd.DataFrame]

# Options (small, typed containers)
@dataclass(frozen=True)
class PanelAlignOptions:
    sort_units: bool = True
    sort_times: bool = True
    drop_unknown_units: bool = False
    drop_unknown_times: bool = False
    strict: bool = False  # if True, raise on misalignment

@dataclass(frozen=True)
class ImputeOptions:
    strategy: Literal["none", "ffill", "bfill", "median"] = "none"
    axis: Literal["unit", "time"] = "time"  # for ffill/bfill
    fill_value: Optional[float] = None

@dataclass(frozen=True)
class ScaleOptions:
    strategy: Literal["none", "standard", "minmax", "robust", "log1p"] = "none"
    with_mean: bool = True  # only for "standard"
    clip_minmax: Optional[Tuple[float, float]] = None  # safety clipping after scale

# TODO: Issue (24): ValidationReport class
# TODO: Z and X can't be lists or 3D arrays for now; future work
# TODO: Add basic properties like O, outcome, treatment, Z, units, times, etc.
class PanelDataset:
    """
    A thin, optional data container for panel matrices used by CausalTensor.
    Holds O (outcome, N×T), optional Z (treatment, N×T), and optional X (covars, K×N×T or N×T×K).
    Preserves unit/time indexing to round-trip to DataFrames when needed.
    """
    @classmethod
    def from_builtin(
        cls,
        dataset_name: str,
        datasets_path: str | Path | None = None,
    ) -> PanelDataset:
        """
        Construct a PanelDataset from one of the built-in example datasets.

        Parameters
        ----------
        dataset_name : str
            Name understood by `dataset_loader.load_dataset` (e.g., "smoking", "basque", ...).
        datasets_path : str | Path | None, optional
            Filesystem path to the raw datasets folder. If None, resolves to
            `<package_root>/datasets/raw`.

        Returns
        -------
        PanelDataset
            O: N×T outcome matrix
            Z: N×T treatment matrix (or None)
            X: N×K entity-level covariates (or None)
        """
        # Resolve default datasets path relative to the installed package
        if datasets_path is None:
            datasets_path = str(Path(__file__).resolve().parent / "raw") + "/"
        else:
            datasets_path = Path(datasets_path)

        from .dataset_loader import load_dataset

        O_df, Z_df, X_df = load_dataset(dataset_name, datasets_path=str(datasets_path))

        if not isinstance(O_df, pd.DataFrame):
            raise TypeError("load_dataset must return O_df as a DataFrame.")
        if Z_df is not None and not isinstance(Z_df, pd.DataFrame):
            raise TypeError("Z_df must be a DataFrame or None.")
        if X_df is not None and not isinstance(X_df, pd.DataFrame):
            raise TypeError("X_df must be a DataFrame (entities as index) or None.")

        O_df = _coerce_time_index(O_df)
        if Z_df is not None:
            Z_df = _coerce_time_index(Z_df)
            Z_df = Z_df.reindex(index=O_df.index, columns=O_df.columns)

        if O_df.index.duplicated().any():
            raise ValueError("O_df has duplicated entities.")
        if O_df.columns.duplicated().any():
            raise ValueError("O_df has duplicated time labels.")
        O_df = O_df.sort_index().sort_index(axis=1)
        if Z_df is not None:
            if Z_df.index.duplicated().any() or Z_df.columns.duplicated().any():
                raise ValueError("Z_df has duplicated labels after reindex.")
            Z_df = Z_df.sort_index().sort_index(axis=1)

        return cls.from_dataframes(O_df=O_df, Z_df=Z_df, X_entity_df=X_df)

    @classmethod
    def from_long(
        cls,
        data: pd.DataFrame,
        unit_col: str,
        time_col: str,
        outcome_col: str,
        treat_col: Optional[str] = None,
        covar_cols: Optional[Sequence[str]] = None,
    ) -> PanelDataset:
        """
        Construct a PanelDataset from a long-form DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            Long-format panel dataset containing unit, time, and outcome columns.
            Example columns: ['unit_id', 'time_id', 'y', 'z', 'x1', 'x2'].
        unit_col : str
            Column name identifying panel entities (e.g., store, country).
        time_col : str
            Column name identifying time periods (e.g., year, week).
        outcome_col : str
            Column name for the outcome variable (O).
        treat_col : str, optional
            Column name for the treatment/intervention variable (Z).
            If omitted or not found, Z will be None.
        covar_cols : Sequence[str], optional
            List of entity-level covariate columns (X). These will be aggregated
            across time (mean per unit) to form an N×K matrix.

        Returns
        -------
        PanelDataset
            A new instance with:
            - O: outcome matrix (N×T)
            - Z: treatment matrix (N×T) or None
            - X: entity-level covariates (N×K) or None
            - unit_index: array of unique unit IDs
            - time_index: array of unique, sorted time IDs
        """
        df = data.copy()

        df[time_col] = _as_orderable_time(df[time_col])

        O_df = df.pivot(index=unit_col, columns=time_col, values=outcome_col)

        Z_df = None
        if treat_col and treat_col in df.columns:
            Z_df = df.pivot(index=unit_col, columns=time_col, values=treat_col)

        X_entity_df = None
        if covar_cols:
            # drop duplicates if unit_col + covars contain repeated rows
            grp = df[[unit_col] + list(covar_cols)].drop_duplicates()
            X_entity_df = grp.groupby(unit_col, as_index=True).mean()

        obj = cls.from_dataframes(O_df=O_df, Z_df=Z_df, X_entity_df=X_entity_df)
        obj.unit_name = unit_col
        obj.time_name = time_col

        return obj

    @classmethod
    def from_arrays(
        cls,
        O: ArrayLike,
        Z: Optional[ArrayLike] = None,
        X: Optional[ArrayLike] = None,
        unit_index: Optional[Sequence[Any]] = None,
        time_index: Optional[Sequence[Any]] = None,
    ) -> PanelDataset:
        """
        Construct a PanelDataset from wide matrices.

        Parameters
        ----------
        O : ArrayLike
            Outcome matrix of shape (N, T). Must be 2-D.
        Z : ArrayLike, optional
            Treatment matrix of shape (N, T). Must be 2-D and the same shape as O, if provided.
        X : ArrayLike, optional
            Entity-level covariates of shape (N, K). Must be 2-D and have the same N as O.
        unit_index : sequence, optional
            Labels for the N entities. If None, defaults to np.arange(N).
        time_index : sequence, optional
            Labels for the T periods. If None, defaults to np.arange(T).

        Returns
        -------
        PanelDataset
            With O (N×T), Z (N×T or None), X (N×K or None), and aligned indices.
        """
        # Normalize O and its indices (require 2-D)
        O_arr, units, times = _as_matrix(O, unit_index, time_index, name="O")
        O_arr = np.asarray(O_arr, dtype=float)
        if O_arr.ndim != 2:
            raise ValueError(f"O must be 2D (N×T); got shape {O_arr.shape}.")
        N, T = O_arr.shape

        # Normalize Z (must match O)
        Z_arr = None
        if Z is not None:
            Z_arr, zu, zt = _as_matrix(Z, unit_index, time_index, name="Z")
            Z_arr = np.asarray(Z_arr, dtype=float)
            if Z_arr.ndim != 2:
                raise ValueError(f"Z must be 2D (N×T); got shape {Z_arr.shape}.")
            if Z_arr.shape != (N, T):
                raise ValueError(f"Z must have the same shape as O {(N,T)}; got {Z_arr.shape}.")
            _assert_same_index(units, times, zu, zt, "Z")

        # Normalize X (entity-level only, N×K)
        X_arr = None
        if X is not None:
            X_arr = _as_ndarray(X)
            X_arr = np.asarray(X_arr, dtype=float)
            if X_arr.ndim != 2:
                raise ValueError(f"X must be 2D (N×K) with N={N}; got shape {X_arr.shape}.")
            if X_arr.shape[0] != N:
                raise ValueError(f"X must have the same number of rows as O (N={N}); got {X_arr.shape[0]}.")

        return cls(
            O=O_arr,
            Z=Z_arr,
            X=X_arr,
            unit_index=np.asarray(units),
            time_index=np.asarray(times),
            fitted_={}
        )

    @classmethod
    def from_dataframes(
        cls,
        O_df: pd.DataFrame,
        Z_df: Optional[pd.DataFrame] = None,
        X_entity_df: Optional[pd.DataFrame] = None,
    ) -> PanelDataset:
        """
        Construct a PanelDataset from wide-form DataFrames.

        Parameters
        ----------
        O_df : pd.DataFrame
            Outcome matrix (N×T) with entities as index and time as columns.
        Z_df : pd.DataFrame, optional
            Treatment matrix (N×T), aligned with O_df.
        X_entity_df : pd.DataFrame, optional
            Entity-level covariates (N×K), index matching O_df.

        Returns
        -------
        PanelDataset
            With O (N×T), Z (N×T or None), X (N×K or None), and aligned indices.
        """
        O_df = _coerce_time_index(O_df)
        if Z_df is not None:
            Z_df = _coerce_time_index(Z_df)

        if O_df.index.duplicated().any():
            raise ValueError("O_df has duplicated entity labels.")
        if O_df.columns.duplicated().any():
            raise ValueError("O_df has duplicated time labels.")
        O_df = O_df.sort_index().sort_index(axis=1)
        if Z_df is not None:
            Z_df = Z_df.reindex(index=O_df.index, columns=O_df.columns)
            if Z_df.index.duplicated().any() or Z_df.columns.duplicated().any():
                raise ValueError("Z_df has duplicated labels after reindex.")
            Z_df = Z_df.sort_index().sort_index(axis=1)

        units = O_df.index.to_numpy()
        times = O_df.columns.to_numpy()
        O = O_df.to_numpy(dtype=float)

        Z = None if Z_df is None else Z_df.to_numpy(dtype=float)
        X = None
        if X_entity_df is not None:
            X = (
                X_entity_df.reindex(index=O_df.index)
                .to_numpy(dtype=float)
            )

        return cls(O=O, Z=Z, X=X, 
                   unit_index=units, time_index=times, 
                   unit_name=O_df.index.name, time_name=O_df.columns.name)

    def __init__(
        self,
        O: np.ndarray,
        Z: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        unit_index: Optional[np.ndarray] = None,
        time_index: Optional[np.ndarray] = None,
        unit_name: Optional[str] = None,
        time_name: Optional[str] = None,
        fitted_: Optional[Dict[str, Any]] = None,
        report: Optional[Any] = None,   # placeholder for ValidationReport in Issue (24)
    ):
        O = np.asarray(O, dtype=float).copy(order="C")
        if O.ndim != 2:
            raise ValueError(f"O must be 2D (N×T); got {O.shape}.")
        
        N, T = O.shape

        if Z is not None:
            Z = np.asarray(Z, dtype=float)
            if Z.ndim != 2 or Z.shape != (N, T):
                raise ValueError(f"Z must be 2D with shape {(N, T)}; got {Z.shape}.")
            
        if X is not None:
            X = np.asarray(X, dtype=float)
            if X.ndim != 2 or X.shape[0] != N:
                raise ValueError(f"X must be 2D (N×K) with N={N}; got {X.shape}.")
            
        units = np.arange(N) if unit_index is None else np.asarray(unit_index)
        times = np.arange(T) if time_index is None else np.asarray(time_index)
        if units.shape[0] != N or times.shape[0] != T:
            raise ValueError("unit_index/time_index must match O.shape.")
        if len(np.unique(units)) != units.shape[0]:
            raise ValueError("unit_index contains duplicates.")
        if len(np.unique(times)) != times.shape[0]:
            raise ValueError("time_index contains duplicates.")

        self.O = O
        self.Z = Z
        self.X = X
        self.unit_index = units
        self.time_index = times
        self.unit_name = unit_name
        self.time_name = time_name
        self.fitted_: Dict[str, Any] = {} if fitted_ is None else dict(fitted_)
        self.report = report  # (24) will populate this
        self.history: List[str] = []  # optional log of operations

    # ---- Public ops
    def align(self, units: Optional[Sequence[Any]] = None, times: Optional[Sequence[Any]] = None,
              options: PanelAlignOptions = PanelAlignOptions()) -> "PanelDataset":
        """
        Align to a target unit/time index; by default just sorts current indices.
        """
        units_cur = self.unit_index
        times_cur = self.time_index

        # Determine target indices
        target_units = np.array(units_cur if units is None else units)
        target_times = np.array(times_cur if times is None else times)

        if options.sort_units:
            target_units = np.array(sorted(target_units, key=lambda x: (x is None, x)))
        if options.sort_times:
            target_times = np.array(sorted(target_times, key=lambda x: (x is None, x)))

        # Build index maps
        unit_pos = _build_position_map(units_cur)
        time_pos = _build_position_map(times_cur)

        missing_u = [u for u in target_units if u not in unit_pos]
        missing_t = [t for t in target_times if t not in time_pos]
        if (missing_u or missing_t) and options.strict:
            raise KeyError(f"Missing units={missing_u}, times={missing_t} in strict mode.")
        # Filter if allowed
        kept_units = [u for u in target_units if (u in unit_pos) or (not options.drop_unknown_units)]
        kept_times = [t for t in target_times if (t in time_pos) or (not options.drop_unknown_times)]

        # Index arrays (default to keeping originals if unknowns)
        u_idx = np.array([unit_pos.get(u, None) for u in kept_units])
        t_idx = np.array([time_pos.get(t, None) for t in kept_times])

        # Drop Nones (unknowns) by necessity
        u_mask = u_idx != None  # noqa: E711
        t_mask = t_idx != None  # noqa: E711
        u_idx = u_idx[u_mask].astype(int)
        t_idx = t_idx[t_mask].astype(int)
        new_units = np.array(kept_units)[u_mask]
        new_times = np.array(kept_times)[t_mask]

        O2 = self.O[np.ix_(u_idx, t_idx)]
        Z2 = self.Z[np.ix_(u_idx, t_idx)] if self.Z is not None else None
        X2 = self._align_X(u_idx, t_idx)

        return PanelDataset(
            O=O2,
            Z=Z2,
            X=X2,
            unit_index=new_units,
            time_index=new_times,
            unit_name=self.unit_name,
            time_name=self.time_name,
            fitted_=self.fitted_.copy(),
            report=self.report,
        )

    def balance(self, min_pre_period: Optional[int] = None) -> PanelDataset:
        """
        Simple balancer: removes units or times that are fully missing in O (or below threshold).
        min_pre_period is a placeholder for future causal-aware balancing.
        """
        mask_units = ~np.all(np.isnan(self.O), axis=1)
        mask_times = ~np.all(np.isnan(self.O), axis=0)

        O2 = self.O[mask_units][:, mask_times]
        Z2 = self.Z[mask_units][:, mask_times] if self.Z is not None else None
        X2 = self._align_X(np.where(mask_units)[0], np.where(mask_times)[0])

        return PanelDataset(
            O=O2,
            Z=Z2,
            X=X2,
            unit_index=self.unit_index[mask_units],
            time_index=self.time_index[mask_times],
            unit_name=self.unit_name,
            time_name=self.time_name,
            fitted_=self.fitted_.copy(),
            report=self.report,
        )

    def impute(self, options: ImputeOptions = ImputeOptions()) -> PanelDataset:
        if options.strategy == "none":
            return self._clone()
        O2 = self.O.copy()
        if options.strategy in {"ffill", "bfill"}:
            axis = 1 if options.axis == "time" else 0
            O2 = _ffill_bfill(O2, axis=axis, direction="forward" if options.strategy == "ffill" else "backward")
        elif options.strategy == "median":
            # per-unit median imputation by default (axis=1 over time)
            med = np.nanmedian(O2, axis=1, keepdims=True)
            inds = np.where(np.isnan(O2))
            O2[inds] = np.take_along_axis(med, inds[0][:, None], axis=0).ravel()
        else:
            raise NotImplementedError(f"Impute strategy {options.strategy} not implemented.")

        Z2 = self.Z if self.Z is not None else None
        X2 = self.X if self.X is not None else None
        fitted = self.fitted_.copy()
        fitted["impute"] = options
        return PanelDataset(
            O=O2,
            Z=Z2,
            X=X2,
            unit_index=self.unit_index,
            time_index=self.time_index,
            unit_name=self.unit_name,
            time_name=self.time_name,
            fitted_=fitted,
            report=self.report,
        )

    def scale(self, options: ScaleOptions = ScaleOptions()) -> PanelDataset:
        if options.strategy == "none":
            return self._clone()

        O2 = self.O.astype(float).copy()
        params: Dict[str, Any] = {}

        if options.strategy == "log1p":
            O2 = np.log1p(np.clip(O2, a_min=0, a_max=None))
        elif options.strategy == "standard":
            # Mean/Std per time column (T-wise) or per unit (we standardize per-time by default)
            mean = np.nanmean(O2, axis=0, keepdims=True)
            std = np.nanstd(O2, axis=0, ddof=0, keepdims=True)
            std[std == 0] = 1.0
            if options.with_mean:
                O2 = (O2 - mean) / std
            else:
                O2 = O2 / std
            params.update({"mean": mean, "std": std})
        elif options.strategy == "minmax":
            mn = np.nanmin(O2, axis=0, keepdims=True)
            mx = np.nanmax(O2, axis=0, keepdims=True)
            denom = (mx - mn)
            denom[denom == 0] = 1.0
            O2 = (O2 - mn) / denom
            params.update({"min": mn, "max": mx})
        elif options.strategy == "robust":
            q1 = np.nanquantile(O2, 0.25, axis=0, keepdims=True)
            q3 = np.nanquantile(O2, 0.75, axis=0, keepdims=True)
            iqr = q3 - q1
            iqr[iqr == 0] = 1.0
            O2 = (O2 - q1) / iqr
            params.update({"q1": q1, "q3": q3})
        else:
            raise NotImplementedError(f"Scale strategy {options.strategy} not implemented.")

        if options.clip_minmax is not None:
            lo, hi = options.clip_minmax
            O2 = np.clip(O2, lo, hi)

        fitted = self.fitted_.copy()
        fitted["scale"] = {"options": options, "params": params}
        return PanelDataset(
            O=O2,
            Z=self.Z,
            X=self.X,
            unit_index=self.unit_index,
            time_index=self.time_index,
            unit_name=self.unit_name,
            time_name=self.time_name,
            fitted_=fitted,
            report=self.report,
        )

    def validate(self, **kwargs) -> PanelDataset:
        """
        Hook for Issue (24). For now, store a stub; later, we call ct.validate_panel(...)
        and attach the returned report.
        """
        # Lazy import when (24) lands:
        # from causaltensor.diagnostics.validation import validate_panel, ValidationOptions
        # rep = validate_panel(self.O, Z=self.Z, X=self.X, unit_ids=self.unit_index, time_ids=self.time_index, options=...)
        rep = {"status": "not_implemented_yet", "note": "Attach Issue (24) report here."}
        new = self._clone()
        new.report = rep
        return new

    def _clone(self) -> PanelDataset:
        return PanelDataset(
            O=self.O.copy(),
            Z=None if self.Z is None else self.Z.copy(),
            X=None if self.X is None else self.X.copy(),
            unit_index=self.unit_index.copy(),
            time_index=self.time_index.copy(),
            unit_name=self.unit_name,
            time_name=self.time_name,
            fitted_=self.fitted_.copy(),
            report=None if self.report is None else dict(self.report),
        )

    def _align_X(self, u_idx: np.ndarray, t_idx: np.ndarray) -> Optional[np.ndarray]:
        """
        Align entity-level covariates X (N×K) by unit indices.
        For meantime, X has no time dimension so t_idx is ignored.
        """
        if self.X is None:
            return None
        u_idx = np.asarray(u_idx, dtype=int)
        return self.X[u_idx, :]

    # ---- Outputs
    def to_matrices(
        self,
        *,
        copy: bool = False,
        dtype: Optional[np.dtype] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Return (O, Z, X) as NumPy arrays.

        Parameters
        ----------
        copy : bool, default False
            If True, return copies to avoid aliasing user mutations.
        dtype : np.dtype, optional
            If provided, cast arrays to this dtype (e.g., float).

        Notes
        -----
        - Shapes: O (N×T), Z (N×T or None), X (N×K or None).
        - This method does not modify internal arrays.
        """
        O = self.O
        Z = self.Z
        X = self.X

        if dtype is not None:
            O = O.astype(dtype, copy=False)
            if Z is not None:
                Z = Z.astype(dtype, copy=False)
            if X is not None:
                X = X.astype(dtype, copy=False)

        if copy:
            O = O.copy(order="C")
            if Z is not None:
                Z = Z.copy(order="C")
            if X is not None:
                X = X.copy(order="C")

        return O, Z, X

    def to_dataframes(
        self,
        *,
        x_colnames: Optional[Sequence[str]] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Return (O_df, Z_df, X_df) as pandas DataFrames.

        Parameters
        ----------
        x_colnames : sequence of str, optional
            Column names for X_df (N×K). If None, uses self.x_colnames if available,
            otherwise falls back to 'x0', 'x1', ...

        Returns
        -------
        O_df : (N×T) DataFrame with entities as index and times as columns
        Z_df : (N×T) DataFrame or None
        X_df : (N×K) DataFrame or None
        """
        # Preserve index names if we set them in constructors (not for now); otherwise None is fine
        unit_name = getattr(self, "unit_name", None)
        time_name = getattr(self, "time_name", None)
        stored_x_names = getattr(self, "x_colnames", None)

        O_df = pd.DataFrame(
            self.O,
            index=pd.Index(self.unit_index, name=unit_name),
            columns=pd.Index(self.time_index, name=time_name),
        )

        Z_df = None
        if self.Z is not None:
            Z_df = pd.DataFrame(
                self.Z,
                index=pd.Index(self.unit_index, name=unit_name),
                columns=pd.Index(self.time_index, name=time_name),
            )

        X_df = None
        if self.X is not None:
            k = self.X.shape[1]
            cols = None
            if x_colnames is not None:
                if len(x_colnames) != k:
                    raise ValueError(f"x_colnames length {len(x_colnames)} != K={k}.")
                cols = list(x_colnames)
            elif stored_x_names is not None and len(stored_x_names) == k:
                cols = list(stored_x_names)
            else:
                cols = [f"x{j}" for j in range(k)]

            X_df = pd.DataFrame(
                self.X,
                index=pd.Index(self.unit_index, name=unit_name),
                columns=cols,
            )

        return O_df, Z_df, X_df


# Utils
def _as_ndarray(x: ArrayLike) -> np.ndarray:
    return x.to_numpy() if isinstance(x, pd.DataFrame) else np.asarray(x)

def _as_matrix(M: ArrayLike, uidx: Optional[Sequence[Any]], tidx: Optional[Sequence[Any]], name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(M, pd.DataFrame):
        arr = M.to_numpy()
        units = M.index.to_numpy() if uidx is None else np.asarray(uidx)
        times = M.columns.to_numpy() if tidx is None else np.asarray(tidx)
        return arr, units, times
    arr = np.asarray(M)
    n, t = arr.shape[:2]
    units = np.arange(n) if uidx is None else np.asarray(uidx)
    times = np.arange(t) if tidx is None else np.asarray(tidx)
    if units.shape[0] != n or times.shape[0] != t:
        raise ValueError(f"{name}: provided unit/time indices do not match shape.")
    return arr, units, times

def _assert_same_index(u1, t1, u2, t2, name: str) -> None:
    if len(u1) != len(u2) or len(t1) != len(t2) or np.any(u1 != u2) or np.any(t1 != t2):
        raise ValueError(f"{name} index mismatch with O. Use PanelDataset.align(...) first if needed.")

def _build_position_map(values: Sequence[Any]) -> Dict[Any, int]:
    return {v: i for i, v in enumerate(values)}

def _ffill_bfill(A: np.ndarray, axis: int, direction: Literal["forward", "backward"]) -> np.ndarray:
    B = A.copy()
    if axis == 0:
        B = B.T  # operate over rows (time) for each unit
    # Forward/backward fill per row
    for i in range(B.shape[0]):
        row = B[i, :]
        mask = np.isnan(row)
        if mask.all():
            continue
        if direction == "forward":
            last = np.nan
            for j in range(row.size):
                if not np.isnan(row[j]):
                    last = row[j]
                elif not np.isnan(last):
                    row[j] = last
        else:
            nxt = np.nan
            for j in range(row.size - 1, -1, -1):
                if not np.isnan(row[j]):
                    nxt = row[j]
                elif not np.isnan(nxt):
                    row[j] = nxt
        B[i, :] = row
    if axis == 0:
        B = B.T
    return B

def _as_orderable_time(x: pd.Series) -> pd.Series:
    # Accept ints, years, datetimes; convert to an orderable type but keep display stable
    if np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.floating):
        return x.astype(int)

    # Try to coerce string-like numeric years (e.g. "2019") to integers.
    # pd.to_numeric will raise on non-numeric values when errors='raise'.
    try:
        # Pre-clean strings: strip whitespace and remove common thousands separators
        # (commas, underscores, and internal spaces). This lets values like
        # " 2,019 ", "2_019", or "2019.0" be interpreted as numeric years.
        if x.dtype == object or np.issubdtype(x.dtype, np.str_):
            s = x.astype(str).str.strip()
            s_clean = s.str.replace(r"[,_\s]+", "", regex=True)
            num = pd.to_numeric(s_clean, errors="raise")
        else:
            num = pd.to_numeric(x, errors="raise")
        # If numeric coercion succeeded, prefer integer representation when possible.
        # Check whether all values are whole numbers (e.g., '2019' -> 2019.0)
        if np.issubdtype(num.dtype, np.integer) or np.all(np.equal(np.mod(num, 1), 0)):
            return num.astype(int)
        return num
    except Exception:
        # fall through to datetime / categorical handling
        pass

    if np.issubdtype(x.dtype, np.datetime64):
        return pd.to_datetime(x)

    # As fallback: category codes preserving order of appearance
    return x.astype("category").cat.as_ordered().cat.codes

def _coerce_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the time axis (columns) is numeric or datetime-like when possible.

    Tries to coerce string year labels (e.g. "1970") to integers.
    If conversion fails, returns the DataFrame unchanged.

    This avoids misalignment between O_df and Z_df loaded from different sources.
    """
    cols = df.columns
    name = df.columns.name
    # Try integer coercion first (e.g. "1970" -> 1970)
    try:
        if all(str(c).isdigit() for c in cols):
            new_cols = pd.Index([int(c) for c in cols], name=name)
            return df.set_axis(new_cols, axis=1)
    except Exception:
        pass

    # Try datetime coercion (e.g. "2020-01-01")
    try:
        new_cols = pd.to_datetime(cols, errors="raise")
        new_cols = new_cols.rename(name)
        return df.set_axis(new_cols, axis=1)
    except Exception:
        return df
