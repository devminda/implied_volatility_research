"""Defines the data structure for raw SVI parameters."""
from dataclasses import dataclass

@dataclass
class RawSVIParameters:
    a: float
    b: float
    rho: float
    m: float
    sigma: float