"""This file answers the question:
Given a candidate parameter set (a,b,ρ,m,σ), how good or bad is the fit?"""
""" We are trying to minimize"""
import numpy as np

from app.models.svi.raw_svi import RawSVIFormula


class RawSVIObjective:
    def __init__(self, penalty_weight: float = 1e6):
        self.penalty_weight = penalty_weight

    def __call__(self, params, k, w_market) -> float:
        a, b, rho, m, sigma = params

        # we add the penalties to the objective function to discourage the optimizer 
        # from exploring parameter regions that violate the no-arbitrage conditions.
        penalty = 0.0

        if b <= 0:
            penalty += self.penalty_weight * (1.0 - b) ** 2

        if sigma <= 0:
            penalty += self.penalty_weight * (1.0 - sigma) ** 2

        if abs(rho) >= 1:
            penalty += self.penalty_weight * (abs(rho) - 0.999) ** 2

        w_fit = RawSVIFormula.evaluate(k, a, b, rho, m, sigma)

        # we take the minimum of w_fit and 0 to find any negative values, 
        # and penalize them heavily to enforce the no-arbitrage 
        # condition that total variance must be non-negative.
        negative_part = np.minimum(w_fit, 0.0)
        penalty += self.penalty_weight * np.sum(negative_part ** 2)

        error = np.sum((w_fit - w_market) ** 2)
        return float(error + penalty)