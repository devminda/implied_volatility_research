"""Defines a container for the results of calibrating an SVI model to market data, 
including the calibrated parameters, success status, and details of the fit."""
"""Basically stores the output of the calibration"""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CalibrationResult:
    model_name: str
    parameters: Dict[str, float]
    success: bool
    message: str
    objective_value: float
    strikes: List[float]
    log_moneyness: List[float]
    market_iv: List[float]
    fitted_iv: List[float]
    market_total_variance: List[float]
    fitted_total_variance: List[float]

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "parameters": self.parameters,
            "success": self.success,
            "message": self.message,
            "objective_value": self.objective_value,
            "market_points": {
                "strikes": self.strikes,
                "log_moneyness": self.log_moneyness,
                "market_iv": self.market_iv,
                "fitted_iv": self.fitted_iv,
                "market_total_variance": self.market_total_variance,
                "fitted_total_variance": self.fitted_total_variance,
            },
        }