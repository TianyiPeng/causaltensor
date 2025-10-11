class Result():
    def __init__(self, baseline = None, tau = None, covariance_tau = None, std_tau = None, return_tau_scalar=False):
        self.baseline = baseline # the baseline outcome (i.e, the outcome without treatment and noise)
        self.tau = tau #treatment effect estimator
        self.covariance_tau = covariance_tau #covariance matrix of tau
        self.std_tau = std_tau #standard deviation of tau
        if return_tau_scalar:
            ## if tau is a scalar, then return tau instead of [tau]
            self.tau = tau[0]
            if self.covariance_tau is not None:
                self.covariance_tau = covariance_tau[0, 0]
            if self.std_tau is not None:
                self.std_tau = std_tau[0] 