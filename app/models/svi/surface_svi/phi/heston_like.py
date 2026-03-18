import numpy as np


class HestonLikePhi:
    def __init__(self, lam: float):
        self.lam = lam

    def __call__(self, theta):
        theta = np.asarray(theta, dtype=float)
        x = self.lam * theta

        eps = 1e-12
        x = np.maximum(x, eps)

        return (1.0 / x) * (1.0 - (1.0 - np.exp(-x)) / x)