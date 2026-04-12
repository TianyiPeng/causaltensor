import numpy as np


def sample_treatment_parameters(n, T, rng):
    """
    Sample treatment pattern parameters with safe integer bounds.
    
    Args:
        n: Number of entities
        T: Number of time periods
        rng: Random number generator
    
    Returns:
        Tuple of (m1, m2, lookback_a, duration_b)
    """
    # m1 in [ceil(0.05*n), floor(0.2*n)], ensure >=1
    m1_low = max(1, int(np.ceil(0.05 * n)))
    m1_high = max(m1_low, int(np.floor(0.2 * n)))
    m1 = int(rng.integers(m1_low, m1_high + 1))
    
    # m2 in [ceil(0.6*T), floor(0.8*T)] capped to [0, T-1]
    m2_low = min(int(np.ceil(0.6 * T)), T - 1)
    m2_high = min(max(m2_low, int(np.floor(0.8 * T))), T - 1)
    m2 = int(rng.integers(m2_low, m2_high + 1))
    
    # lookback_a in [ceil(0.05*T), floor(0.1*T)], clamp to [1, T-1]
    lb_low = max(1, int(np.ceil(0.05 * T)))
    lb_high = max(lb_low, int(np.floor(0.1 * T)))
    lookback_a = int(rng.integers(lb_low, lb_high + 1))
    lookback_a = min(lookback_a, max(1, T - 1))
    
    # duration_b in [ceil(0.1*T), floor(0.2*T)], clamp to [1, T-1]
    dur_low = max(1, int(np.ceil(0.1 * T)))
    dur_high = max(dur_low, int(np.floor(0.2 * T)))
    duration_b = int(rng.integers(dur_low, dur_high + 1))
    duration_b = min(duration_b, max(1, T - 1))
    
    return m1, m2, lookback_a, duration_b

def build_baseline_M(O, treated_states, treat_start_years, type = 'control'):
    if type == 'control':
        # Create boolean mask from list of treated state indices.
        # If treated_states is empty (no observed treatment), M = all of O.
        treated_mask = np.zeros(O.shape[0], dtype=bool)
        treated_mask[treated_states] = True
        M = O[~treated_mask, :]
    elif type == 'pre-treatment':
        if not treat_start_years:
            raise ValueError(
                "Cannot build a pre-treatment baseline when treat_start_years "
                "is empty (dataset has no observed treatment). Use "
                "baseline_type='control' instead."
            )
        M = O[:, :min(treat_start_years)]

    return M, M.shape[0], M.shape[1]



def inject_treatment_centered(M, Z, *,
                              treatment_level=0.2,
                              sigma_unit_scale=1.0,
                              sigma_time_scale=0.5,
                              rng=None):
    """
    Inject treatment with unit- and time-heterogeneity, centered over treated cells:
        O = M + (tau* + delta_i + eta_t) ∘ Z

    - tau* = mean(M) * treatment_level
    - delta_i ~ N(0, sigma_unit_scale * tau*)
    - eta_t   ~ N(0, sigma_time_scale * tau*)
    - Both delta_i and eta_t are re-centered (weighted by treated counts) so that
      the ATT over treated cells equals tau*

    Returns:
        O            : observed panel
        att_true     : ground-truth ATT over treated cells
        tau_star     : intended average effect
    """
    rng = np.random.default_rng(rng)
    n, T = M.shape

    # Base effect and heterogeneity scales
    tau_star = np.mean(M) * treatment_level
    sigma_delta = sigma_unit_scale * tau_star
    sigma_eta   = sigma_time_scale * tau_star

    # Draw heterogeneity
    delta_i = rng.normal(0.0, sigma_delta, size=n)   # unit-fixed shock
    eta_t   = rng.normal(0.0, sigma_eta,   size=T)   # time-fixed shock

    # Treated counts
    treated_per_i = Z.sum(axis=1)        # times each unit is treated
    treated_per_t = Z.sum(axis=0)        # units treated at each time
    treated_total = Z.sum()

    # Center over treated cells (if any treated)
    delta_i_ctr = delta_i.copy()
    eta_t_ctr   = eta_t.copy()
    if treated_total > 0:
        # unit-centering: weighted by #treated times per unit
        if treated_per_i.sum() > 0:
            delta_mean_treated = (delta_i * treated_per_i).sum() / treated_per_i.sum()
            delta_i_ctr -= delta_mean_treated
        # time-centering: weighted by #treated units per time
        if treated_per_t.sum() > 0:
            eta_mean_treated = (eta_t * treated_per_t).sum() / treated_per_t.sum()
            eta_t_ctr -= eta_mean_treated

    # Build treatment matrix and observed panel
    Tmat = tau_star + delta_i_ctr[:, None] + eta_t_ctr[None, :]
    O = M + Tmat * Z

    # Ground-truth ATT - will be same as tau_star after unit and time centering
    att_true = (Tmat * Z).sum() / treated_total if treated_total > 0 else 0.0
    assert np.isclose(att_true, tau_star)

    return O, tau_star


def subsample_panel(O, treated_states, rng, max_units=None, max_time=None):
    """
    Randomly subsample an (n x T) panel to at most max_units rows and
    max_time columns, always keeping any real treated rows intact.

    Parameters
    ----------
    O : np.ndarray  (n x T)
    treated_states : list of int
        Row indices that must be kept (real treated units).
    rng : np.random.Generator
    max_units : int or None
        Cap on number of rows. None means no limit.
    max_time : int or None
        Cap on number of columns. None means no limit.

    Returns
    -------
    O_sub : np.ndarray
    treated_states_sub : list of int
        Treated-unit indices remapped into the subsampled row space.
    """
    n, T = O.shape
    row_idx = np.arange(n)

    if max_units is not None and n > max_units:
        if len(treated_states) < max_units:
            # Preserve all treated units and fill the rest with random controls.
            control_idx = np.array([i for i in row_idx if i not in treated_states])
            n_control_keep = max_units - len(treated_states)
            sampled_control = rng.choice(control_idx, size=n_control_keep, replace=False)
            row_idx = np.concatenate([sorted(treated_states), np.sort(sampled_control)])
        else:
            # More treated units than the cap (e.g. retail promo datasets) —
            # sample uniformly from all rows; treated_states lose their special status.
            row_idx = np.sort(rng.choice(n, size=max_units, replace=False))

    col_idx = np.arange(T, dtype=int)
    if max_time is not None and T > max_time:
        col_idx = np.sort(rng.choice(T, size=max_time, replace=False))

    O_sub = O[np.ix_(row_idx.astype(int), col_idx)]

    # Remap treated_states to new row positions (only those that survived sampling)
    row_idx_list = list(row_idx)
    treated_set = set(treated_states)
    treated_states_sub = [
        row_idx_list.index(i) for i in row_idx_list if i in treated_set
    ]

    return O_sub, treated_states_sub
