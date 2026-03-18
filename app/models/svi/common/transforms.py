"""Financial Preprocessing Transforms: Contains functions to 
transform raw market data (spot, strikes, implied vols) into the format needed for 
SVI calibration (log-moneyness, total variance)."""
import numpy as np


class MarketTransform:
    @staticmethod
    def forward_price(
        spot: float,
        rate: float,
        dividend_yield: float,
        maturity: float,
    ) -> float:
        """Forward price F = S * exp((r - q) * T)"""
        return float(spot * np.exp((rate - dividend_yield) * maturity))

    @staticmethod
    def log_moneyness(strikes, forward: float):
        """Log-moneyness k = ln(K/F)"""
        strikes = np.asarray(strikes, dtype=float)
        return np.log(strikes / forward)

    @staticmethod
    def total_variance_from_iv(implied_vols, maturity: float):
        """Total variance w = (implied_vol)^2 * T"""
        implied_vols = np.asarray(implied_vols, dtype=float)
        return (implied_vols ** 2) * maturity

    @staticmethod
    def implied_vol_from_total_variance(total_variance, maturity: float):
        """Implied volatility from total variance: iv = sqrt(w / T)"""
        return np.sqrt(np.maximum(total_variance, 0.0) / maturity)