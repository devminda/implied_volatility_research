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

        return 0.5 * theta * (
            1.0
            + self.rho * phi_theta * k
            + np.sqrt(term**2 + (1.0 - self.rho**2))
        )
    @staticmethod
    def theta_of_T(T, sigma_atm=0.20, mean_revert=False, kappa=1.5, theta_inf=0.04):
        T = np.asarray(T, dtype=float)
        if mean_revert:
            return theta_inf * (1.0 - np.exp(-kappa * T))
        return sigma_atm**2 * T

    @staticmethod
    def build_iv_surface(k_grid, T_grid, model, theta_func):
        k_grid = np.asarray(k_grid, dtype=float)
        T_grid = np.asarray(T_grid, dtype=float)

        if np.any(T_grid <= 0):
            raise ValueError("All maturities must be strictly positive (T > 0).")        

        kk, TT = np.meshgrid(k_grid, T_grid)
        theta_T = theta_func(TT)

        w = model.evaluate(kk, theta_T)
        iv = np.sqrt(np.maximum(w, 0.0) / TT)

        return kk, TT, w, iv