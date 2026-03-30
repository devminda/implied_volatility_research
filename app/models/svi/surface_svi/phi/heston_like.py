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

class PowerLawPhi:
    def __init__(self, eta: float, gamma: float):
        self.eta = eta
        self.gamma = gamma

    def __call__(self, theta):
        theta = np.asarray(theta, dtype=float)
        eps = 1e-12
        theta = np.maximum(theta, eps)
        return self.eta * theta ** (-self.gamma)

class StabilizedPowerLawPhi:
    def __init__(self, eta: float, gamma: float):
        self.eta = eta
        self.gamma = gamma

    def __call__(self, theta):
        theta = np.asarray(theta, dtype=float)
        eps = 1e-12
        theta = np.maximum(theta, eps)

        return self.eta / (theta**self.gamma * (1 + theta)**(1 - self.gamma))