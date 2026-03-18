"""This file implements the calibration service for the raw SVI model."""
"""Take a market slice, transform it, run optimization, 
evaluate fitted values, and package the result."""
import numpy as np
from scipy.optimize import minimize

from app.models.svi.common import MarketSlice, CalibrationResult, MarketTransform
from app.models.svi.raw_svi import RawSVIFormula, RawSVIObjective


class RawSVICalibrationService:
    def __init__(self, penalty_weight: float = 1e6):
        # The objective function includes penalties to enforce no-arbitrage conditions,
        self.objective = RawSVIObjective(penalty_weight=penalty_weight)

    @staticmethod
    def default_initial_guess(w_market) -> np.ndarray:
        """Provides a default initial guess for the parameters based on the market total variance."""
        w_market = np.asarray(w_market, dtype=float)
        return np.array([
            max(1e-6, np.min(w_market) * 0.8),  # a
            0.1,                                # b
            -0.3,                               # rho
            0.0,                                # m
            0.1,                                # sigma
        ])

    @staticmethod
    def parameter_bounds():
        """Returns bounds for the parameters to ensure they stay in reasonable regions."""
        return [
            (-1.0, 2.0),     # a
            (1e-8, 10.0),    # b
            (-0.999, 0.999), # rho
            (-5.0, 5.0),     # m
            (1e-8, 5.0),     # sigma
        ]

    def calibrate(self, market_slice: MarketSlice, initial_guess=None) -> CalibrationResult:
        """Calibrates the raw SVI model to the given market slice."""
        # Validate the market slice first
        market_slice.validate()

        # Transform market data to the form needed for calibration
        forward = MarketTransform.forward_price(
            spot=market_slice.spot,
            rate=market_slice.rate,
            dividend_yield=market_slice.dividend_yield,
            maturity=market_slice.maturity,
        )

        k_market = MarketTransform.log_moneyness(
            strikes=market_slice.strikes,
            forward=forward,
        )

        w_market = MarketTransform.total_variance_from_iv(
            implied_vols=market_slice.implied_vols,
            maturity=market_slice.maturity,
        )

        if initial_guess is None:
            initial_guess = self.default_initial_guess(w_market)
        # finds the best set of a, b, rho, m, sigma that minimizes 
        # the objective function, which includes both the fit error and 
        # the penalties.
        result = minimize(
            self.objective,
            x0=np.asarray(initial_guess, dtype=float),
            args=(k_market, w_market),
            method="L-BFGS-B",
            bounds=self.parameter_bounds(),
        )

        a_hat, b_hat, rho_hat, m_hat, sigma_hat = result.x

        # After optimization, we evaluate the fitted total variance and implied volatilities
        # using the optimized parameters, so that we can compare them to the market data.
        w_fit_market = RawSVIFormula.evaluate(
            k_market,
            a_hat,
            b_hat,
            rho_hat,
            m_hat,
            sigma_hat,
        )

        iv_fit_market = MarketTransform.implied_vol_from_total_variance(
            total_variance=w_fit_market,
            maturity=market_slice.maturity,
        )

        return CalibrationResult(
            model_name="raw_svi",
            parameters={
                "a": float(a_hat),
                "b": float(b_hat),
                "rho": float(rho_hat),
                "m": float(m_hat),
                "sigma": float(sigma_hat),
            },
            success=bool(result.success),
            message=str(result.message),
            objective_value=float(result.fun),
            strikes=list(np.asarray(market_slice.strikes, dtype=float)),
            log_moneyness=list(np.asarray(k_market, dtype=float)),
            market_iv=list(np.asarray(market_slice.implied_vols, dtype=float)),
            fitted_iv=list(np.asarray(iv_fit_market, dtype=float)),
            market_total_variance=list(np.asarray(w_market, dtype=float)),
            fitted_total_variance=list(np.asarray(w_fit_market, dtype=float)),
        )