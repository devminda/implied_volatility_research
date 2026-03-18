"""Defines the raw SVI model"""
"""Given log-moneyness k and parameters (a,b,ρ,m,σ), compute fitted total variance w(k)."""
"""Only computes the formula"""
import numpy as np


class RawSVIFormula:
    @staticmethod
    def evaluate(k, a: float, b: float, rho: float, m: float, sigma: float):
        """
        Raw SVI total variance parameterization:

            w(k) = a + b * [ rho*(k - m) + sqrt((k - m)^2 + sigma^2) ]

        Parameters
        ----------
        k : array-like
            Log-moneyness values.
        a, b, rho, m, sigma : float
            Raw SVI parameters.

        Returns
        -------
        np.ndarray
            Total implied variance values.
        """
        k = np.asarray(k, dtype=float)
        return a + b * (
            rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2)
        )