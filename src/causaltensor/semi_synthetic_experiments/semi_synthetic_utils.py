import numpy as np


def build_baseline_M(O, treated_states, treat_start_years, type = 'control'):
    if type == 'control':
        # Create boolean mask from list of treated state indices
        treated_mask = np.zeros(O.shape[0], dtype=bool)
        treated_mask[treated_states] = True
        M = O[~treated_mask, :]
    elif type == 'pre-treatment':
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
