"""Defines a container for market data corresponding to a single maturity, along with validation logic."""
"""A slice means one maturity, multiple strikes, and corresponding implied vols. This is the input to the calibration routine."""
"""Mainly stores the data and also has some validation logic to ensure the data is consistent and reasonable."""
from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass
class MarketSlice:
    spot: float
    rate: float
    dividend_yield: float
    maturity: float
    strikes: Sequence[float]
    implied_vols: Sequence[float]

    def validate(self) -> None:
        # ensures spot price is positive, maturity is positive
        if self.spot <= 0:
            raise ValueError("Spot must be positive.")
        # Maturity is positive
        if self.maturity <= 0:
            raise ValueError("Maturity must be positive.")
        # strikes and implied vols are non-empty, same length, and all positive
        strikes = np.asarray(self.strikes, dtype=float)
        implied_vols = np.asarray(self.implied_vols, dtype=float)

        if strikes.size == 0:
            raise ValueError("Strikes cannot be empty.")

        if strikes.size != implied_vols.size:
            raise ValueError("Strikes and implied_vols must have the same length.")
        # strikes and implied vols must be positive
        if np.any(strikes <= 0):
            raise ValueError("All strikes must be positive.")

        if np.any(implied_vols <= 0):
            raise ValueError("All implied vols must be positive.")