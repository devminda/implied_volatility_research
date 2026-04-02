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
    
    def evaluate_d1(self, k, theta):
        """First derivative of w with respect to k."""
        k = np.asarray(k, dtype=float)
        theta = np.asarray(theta, dtype=float)

        p = self.phi(theta)
        denom = np.sqrt(p**2 * k**2 + 2.0 * p * self.rho * k + 1.0)

        return 0.5 * theta * p * (
            (p * k + self.rho * denom + self.rho * denom) / denom
        )

    def evaluate_d1(self, k, theta):
        """First derivative of w with respect to k — matches SSVI1."""
        k = np.asarray(k, dtype=float)
        theta = np.asarray(theta, dtype=float)

        p = self.phi(theta)
        inner = p**2 * k**2 + 2.0 * p * self.rho * k + 1.0
        sqrt_inner = np.sqrt(inner)

        return 0.5 * theta * p * (p * k + self.rho * sqrt_inner + self.rho) / sqrt_inner

    def evaluate_d2(self, k, theta):
        """Second derivative of w with respect to k — matches SSVI2."""
        k = np.asarray(k, dtype=float)
        theta = np.asarray(theta, dtype=float)

        p = self.phi(theta)
        inner = p**2 * k**2 + 2.0 * p * self.rho * k + 1.0

        return 0.5 * theta * p**2 * (1.0 - self.rho**2) / (inner * np.sqrt(inner))

    def evaluate_dt(self, k, T, theta_func, eps: float = 1e-4):
        """
        First derivative of w with respect to T by central difference — matches SSVIt.
        Needs theta_func because theta depends on T.
        """
        k = np.asarray(k, dtype=float)
        T = np.asarray(T, dtype=float)

        w_plus  = self.evaluate(k, theta_func(T + eps))
        w_minus = self.evaluate(k, theta_func(T - eps))

        return (w_plus - w_minus) / (2.0 * eps)

    def g(self, k, T, theta_func):
        """
        Butterfly arbitrage function from equation 2.1 of the paper.
        g >= 0 everywhere is required for no butterfly arbitrage.
        """
        k = np.asarray(k, dtype=float)
        T = np.asarray(T, dtype=float)

        theta = theta_func(T)
        w  = self.evaluate(k, theta)
        w1 = self.evaluate_d1(k, theta)
        w2 = self.evaluate_d2(k, theta)

        return (
            (1.0 - 0.5 * k * w1 / w)**2
            - 0.25 * w1**2 * (0.25 + 1.0 / w)
            + 0.5 * w2
        )

    def density(self, k, T, theta_func):
        """
        Risk-neutral probability density — matches densitySSVI.
        Positive density everywhere confirms no butterfly arbitrage.
        """
        k = np.asarray(k, dtype=float)
        T = np.asarray(T, dtype=float)

        theta = theta_func(T)
        w  = self.evaluate(k, theta)
        g  = self.g(k, T, theta_func)

        d_minus = -k / np.sqrt(w) - 0.5 * np.sqrt(w)

        return g * np.exp(-0.5 * d_minus**2) / np.sqrt(2.0 * np.pi * w)

    def local_variance(self, k, T, theta_func):
        """
        Dupire local variance from SSVI — matches SSVI_LocalVarg.
            sigma_local^2(k, T) = dw/dT / g(k, T)
        This is what produces the smoother surface.
        """
        k = np.asarray(k, dtype=float)
        T = np.asarray(T, dtype=float)

        dw_dt = self.evaluate_dt(k, T, theta_func)
        g     = self.g(k, T, theta_func)

        return dw_dt / g

    def local_vol(self, k, T, theta_func):
        """Local volatility — square root of local variance."""
        return np.sqrt(np.maximum(self.local_variance(k, T, theta_func), 0.0))
    
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
    
    @staticmethod
    def build_local_vol_surface(k_grid, T_grid, model, theta_func):
        """
        Build the local vol surface — this is what produces the smoother plot.
        """
        k_grid = np.asarray(k_grid, dtype=float)
        T_grid = np.asarray(T_grid, dtype=float)

        if np.any(T_grid <= 0):
            raise ValueError("All maturities must be strictly positive (T > 0).")

        kk, TT = np.meshgrid(k_grid, T_grid)

        lv = model.local_vol(kk, TT, theta_func)

        return kk, TT, lv