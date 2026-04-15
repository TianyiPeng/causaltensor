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

def build_baseline_M(O, treated_states, treat_start_years, baseline_type='control'):
    if baseline_type == 'control':
        # Create boolean mask from list of treated state indices.
        # If treated_states is empty (no observed treatment), M = all of O.
        treated_mask = np.zeros(O.shape[0], dtype=bool)
        treated_mask[treated_states] = True
        M = O[~treated_mask, :]
    elif baseline_type == 'pre-treatment':
        if not treat_start_years:
            raise ValueError(
                "Cannot build a pre-treatment baseline when treat_start_years "
                "is empty (dataset has no observed treatment). Use "
                "baseline_type='control' instead."
            )
        M = O[:, :min(treat_start_years)]
    else:
        raise ValueError(
            f"baseline_type must be 'control' or 'pre-treatment', got {baseline_type!r}"
        )

    return M, M.shape[0], M.shape[1]


def filter_treated_for_pretreatment_baseline(treated_states, treat_start_years):
    """
    Keep parallel lists aligned to treated units with treatment start index > 0.

    Pre-treatment baseline uses M = O[:, :min(starts)]; if any unit has start 0,
    that minimum is 0 and the pre-period is empty. Restricting to starts > 0
    avoids an empty slice while still using all pre-treatment periods common to
    the included units.
    """
    ts = []
    ys = []
    for r, t in zip(treated_states, treat_start_years):
        if t > 0:
            ts.append(r)
            ys.append(t)
    return ts, ys


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
