import numpy as np
import matplotlib.pyplot as plt

from app.models.svi.common import MarketSlice, CalibrationResult, MarketTransform
from app.models.svi.raw_svi import RawSVIFormula


class RawSVIPlottingService:
    @staticmethod
    def plot_total_variance(
        market_slice: MarketSlice,
        calibration_result: CalibrationResult,
    ) -> None:
        k_market = np.asarray(calibration_result.log_moneyness, dtype=float)
        w_market = np.asarray(calibration_result.market_total_variance, dtype=float)

        params = calibration_result.parameters
        k_grid = np.linspace(k_market.min() - 0.1, k_market.max() + 0.1, 300)

        w_fit_grid = RawSVIFormula.evaluate(
            k_grid,
            params["a"],
            params["b"],
            params["rho"],
            params["m"],
            params["sigma"],
        )

        plt.figure(figsize=(8, 5))
        plt.scatter(k_market, w_market, label="Market total variance")
        plt.plot(k_grid, w_fit_grid, label="Raw SVI fit")
        plt.xlabel("Log-moneyness k = ln(K/F)")
        plt.ylabel("Total variance w")
        plt.title("Raw SVI Calibration in Total Variance Space")
        plt.legend()
        plt.grid(True)

    @staticmethod
    def plot_implied_vol(
        market_slice: MarketSlice,
        calibration_result: CalibrationResult,
    ) -> None:
        k_market = np.asarray(calibration_result.log_moneyness, dtype=float)
        market_iv = np.asarray(calibration_result.market_iv, dtype=float)

        params = calibration_result.parameters
        k_grid = np.linspace(k_market.min() - 0.1, k_market.max() + 0.1, 300)

        w_fit_grid = RawSVIFormula.evaluate(
            k_grid,
            params["a"],
            params["b"],
            params["rho"],
            params["m"],
            params["sigma"],
        )

        iv_fit_grid = MarketTransform.implied_vol_from_total_variance(
            total_variance=w_fit_grid,
            maturity=market_slice.maturity,
        )

        plt.figure(figsize=(8, 5))
        plt.scatter(k_market, market_iv, label="Market implied vol")
        plt.plot(k_grid, iv_fit_grid, label="Raw SVI implied vol fit")
        plt.xlabel("Log-moneyness k = ln(K/F)")
        plt.ylabel("Implied volatility")
        plt.title("Raw SVI Implied Volatility Smile")
        plt.legend()
        plt.grid(True)