import numpy as np


class SSVIFormula:
    def __init__(self, rho: float, phi):
        self.rho = rho
        self.phi = phi

    def evaluate(self, k, theta):
        k = np.asarray(k, dtype=float)
        theta = np.asarray(theta, dtype=float)

        phi_theta = self.phi(theta)

        term = phi_theta * k + self.rho

        return (theta / 2.0) * (
            1.0
            + self.rho * phi_theta * k
            + np.sqrt(term**2 + (1 - self.rho**2))
        )